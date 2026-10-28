#include "hexels3d_forward.h"
#include "hexels3d_backward.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <math.h>
#include <torch/extension.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace {

constexpr int kNumChannels = 3;
constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
constexpr int kBlockSize = kBlockX * kBlockY;
constexpr float kNearPlane = 0.2f;
constexpr float kFarPlane = 100.0f;
constexpr float kFilterSize = 0.707106f;  // sqrt(2) / 2
constexpr float kFilterInvSquare = 2.0f;
constexpr float kHexRadius = 1.15470053838f;  // 2 / sqrt(3)
constexpr float kLog254 = 5.537334267f;  // log(254)
constexpr int kColorModelFlat = 0;
constexpr int kColorModelLinear = 1;
constexpr int kColorModelQuadratic = 2;
constexpr int kColorModelQuadraticBounded = 3;
constexpr int kViewModelNone = 0;
constexpr int kViewModelLinear = 1;
constexpr int kViewModelLobe = 2;
constexpr int kExactSortCenterReplay = 0;
constexpr int kExactSortHybrid = 1;
constexpr int kExactSortSelectedAll = 2;
constexpr int kExactSortCropAll = 3;
constexpr int kExactSortBoundaryTail = 4;
constexpr int kExactSortPixelMask = 5;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    do {              \
        CHECK_CUDA(x); \
        CHECK_CONTIGUOUS(x); \
    } while (0)

#define CUDA_CALL(expr) \
    do {                \
        const cudaError_t err = (expr); \
        TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err)); \
    } while (0)

__forceinline__ __device__ float3 transform_point_4x3(const float3 &p, const float *matrix) {
    return {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    };
}

__forceinline__ __device__ float3 cross3(const float3 a, const float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__forceinline__ __device__ float3 dnormvdv3(const float3 v, const float3 dv) {
    const float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + 1e-8f;
    const float invsum32 = rsqrtf(sum2 * sum2 * sum2);

    float3 out;
    out.x = ((sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
    out.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
    out.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
    return out;
}

__forceinline__ __device__ void get_rect(
    const float2 p, const int2 max_radius, uint2 &rect_min, uint2 &rect_max, dim3 grid
) {
    rect_min = {
        min(grid.x, max(0, (int)((p.x - (float)max_radius.x) / (float)kBlockX))),
        min(grid.y, max(0, (int)((p.y - (float)max_radius.y) / (float)kBlockY))),
    };
    rect_max = {
        min(grid.x, max(0, (int)((p.x + (float)max_radius.x + (float)kBlockX - 1.f) / (float)kBlockX))),
        min(grid.y, max(0, (int)((p.y + (float)max_radius.y + (float)kBlockY - 1.f) / (float)kBlockY))),
    };
}

__forceinline__ __device__ bool in_frustum(
    const int idx, const float *orig_points, const float *viewmatrix, float3 &p_view
) {
    const float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
    p_view = transform_point_4x3(p_orig, viewmatrix);
    return p_view.z > kNearPlane;
}

inline __device__ glm::mat3 quat_to_rotmat(const glm::vec4 quat) {
    // Matches 2DGS surfel rasterizer convention: input quaternion packed as (w, x, y, z)
    // in the float4 lanes (i.e., w stored in quat.x).
    const float s = rsqrtf(quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z);
    const float w = quat.x * s;
    const float x = quat.y * s;
    const float y = quat.z * s;
    const float z = quat.w * s;
    return glm::mat3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

inline __device__ glm::vec4 quat_to_rotmat_vjp(const glm::vec4 quat, const glm::mat3 v_R) {
    const float s = rsqrtf(quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z);
    const float w = quat.x * s;
    const float x = quat.y * s;
    const float y = quat.z * s;
    const float z = quat.w * s;

    glm::vec4 v_quat;
    v_quat.x =
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) + z * (v_R[0][1] - v_R[1][0]));
    v_quat.y = 2.f * (
        -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) + z * (v_R[0][2] + v_R[2][0]) +
        w * (v_R[1][2] - v_R[2][1])
    );
    v_quat.z = 2.f * (
        x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) + z * (v_R[1][2] + v_R[2][1]) +
        w * (v_R[2][0] - v_R[0][2])
    );
    v_quat.w = 2.f * (
        x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) - 2.f * z * (v_R[0][0] + v_R[1][1]) +
        w * (v_R[0][1] - v_R[1][0])
    );

    const glm::vec4 q_unit = quat * s;
    return s * (v_quat - q_unit * glm::dot(q_unit, v_quat));
}

inline __device__ glm::mat3 scale_to_mat(const glm::vec2 scale, const float glob_scale) {
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    return S;
}

__device__ void compute_transmat(
    const float3 &p_orig,
    const glm::vec2 scale,
    const float mod,
    const glm::vec4 rot,
    const float *projmatrix,
    const int W,
    const int H,
    glm::mat3 &T
) {
    const glm::mat3 R = quat_to_rotmat(rot);
    const glm::mat3 S = scale_to_mat(scale, mod);
    const glm::mat3 L = R * S;

    const glm::mat3x4 splat2world = glm::mat3x4(
        glm::vec4(L[0], 0.0),
        glm::vec4(L[1], 0.0),
        glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1.0)
    );

    const glm::mat4 world2ndc = glm::mat4(
        projmatrix[0],
        projmatrix[4],
        projmatrix[8],
        projmatrix[12],
        projmatrix[1],
        projmatrix[5],
        projmatrix[9],
        projmatrix[13],
        projmatrix[2],
        projmatrix[6],
        projmatrix[10],
        projmatrix[14],
        projmatrix[3],
        projmatrix[7],
        projmatrix[11],
        projmatrix[15]
    );

    const glm::mat3x4 ndc2pix = glm::mat3x4(
        glm::vec4((float)W / 2.0f, 0.0f, 0.0f, ((float)W - 1.0f) / 2.0f),
        glm::vec4(0.0f, (float)H / 2.0f, 0.0f, ((float)H - 1.0f) / 2.0f),
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );

    T = glm::transpose(splat2world) * world2ndc * ndc2pix;
}

__device__ bool compute_aabb(const glm::mat3 T, const float cutoff, float2 &point_image, float2 &extent, bool &bounded) {
    const glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
    const float d = glm::dot(t, T[2] * T[2]);
    bounded = d < 0.0f;
    if (d == 0.0f) return false;
    const glm::vec3 f = (1.0f / d) * t;

    const glm::vec2 p = glm::vec2(glm::dot(f, T[0] * T[2]), glm::dot(f, T[1] * T[2]));

    const glm::vec2 h0 = p * p - glm::vec2(glm::dot(f, T[0] * T[0]), glm::dot(f, T[1] * T[1]));
    const glm::vec2 h = sqrt(max(glm::vec2(1e-4f, 1e-4f), h0));

    point_image = {p.x, p.y};
    extent = {h.x, h.y};
    return true;
}

__device__ bool compute_hex_bounds(
    const glm::mat3 T, float apothem, const int W, const int H,
    float2 &center, int2 &radius
) {
    // Pad before rejecting a degenerate support at the alpha cutoff.
    apothem += 4e-5f * fmaxf(1.f, fabsf(apothem));
    if (apothem <= 0.f) return false;
    const float r = kHexRadius * apothem;
    const glm::vec2 vertices[6] = {
        {r, 0.f}, {0.5f * r, apothem}, {-0.5f * r, apothem},
        {-r, 0.f}, {-0.5f * r, -apothem}, {0.5f * r, -apothem}
    };
    glm::vec2 lower(FLT_MAX), upper(-FLT_MAX);
    for (int i = 0; i < 6; ++i) {
        const glm::vec3 a(vertices[i], 1.f);
        const glm::vec3 b(vertices[(i + 1) % 6], 1.f);
        const float za = glm::dot(a, T[2]);
        const float zb = glm::dot(b, T[2]);
        if (za >= kNearPlane) {
            const glm::vec2 p(glm::dot(a, T[0]) / za, glm::dot(a, T[1]) / za);
            lower = glm::min(lower, p);
            upper = glm::max(upper, p);
        }
        if ((za < kNearPlane) != (zb < kNearPlane)) {
            const glm::vec3 edge = a + ((kNearPlane - za) / (zb - za)) * (b - a);
            const glm::vec2 p(glm::dot(edge, T[0]) / kNearPlane, glm::dot(edge, T[1]) / kNearPlane);
            lower = glm::min(lower, p);
            upper = glm::max(upper, p);
        }
    }
    if (lower.x > W || lower.y > H || upper.x < -1.f || upper.y < -1.f) return false;
    lower = glm::max(lower - glm::vec2(1.f), glm::vec2(-kBlockX, -kBlockY));
    upper = glm::min(upper + glm::vec2(1.f), glm::vec2(W + kBlockX, H + kBlockY));
    center = {(lower.x + upper.x) * 0.5f, (lower.y + upper.y) * 0.5f};
    radius = {(int)ceilf((upper.x - lower.x) * 0.5f) + 1,
              (int)ceilf((upper.y - lower.y) * 0.5f) + 1};
    return true;
}

__forceinline__ __device__ float sigmoid_neg_dist(const float dist, const float softness) {
    const float x = dist * softness;
    return 1.f / (1.f + __expf(x));
}

__forceinline__ __device__ float hex_sdf_edge(const float u, const float v, int &edge_id) {
    // Regular pointy-top hexagon SDF expressed as the max of 3 oriented half-plane distances.
    // This is equivalent to the classic (abs-folded) form:
    //   dist = max(|v|, 0.5*(sqrt(3)*|u| + |v|)) - 1
    // because max(|a+b|, |a-b|) == |a| + |b|.
    //
    // We keep the 3-way max so we can attach a different softness per edge orientation.
    constexpr float k = 1.7320508f;  // sqrt(3)
    constexpr float a = 0.5f * k;    // 0.8660254
    constexpr float b = 0.5f;

    const float d0 = fabsf(v);
    const float t1 = a * u + b * v;
    const float t2 = a * u - b * v;
    const float d1 = fabsf(t1);
    const float d2 = fabsf(t2);

    // Tie-break matches the previous 2-way max implementation: prefer slanted edges on ties with |v|.
    float max_slant = d1;
    edge_id = 1;
    if (d2 >= max_slant) {
        max_slant = d2;
        edge_id = 2;
    }
    if (d0 > max_slant) {
        edge_id = 0;
        return d0 - 1.f;
    }
    return max_slant - 1.f;
}

__forceinline__ __device__ void hex_sdf_grad(
    const float u, const float v, float &dist, float &d_dist_du, float &d_dist_dv, int &edge_id
) {
    constexpr float k = 1.7320508f;  // sqrt(3)
    constexpr float a = 0.5f * k;    // 0.8660254
    constexpr float b = 0.5f;

    const float d0 = fabsf(v);
    const float t1 = a * u + b * v;
    const float t2 = a * u - b * v;
    const float d1 = fabsf(t1);
    const float d2 = fabsf(t2);

    float max_slant = d1;
    edge_id = 1;
    if (d2 >= max_slant) {
        max_slant = d2;
        edge_id = 2;
    }

    if (d0 > max_slant) {
        edge_id = 0;
        dist = d0 - 1.f;
        d_dist_du = 0.f;
        d_dist_dv = (v >= 0.f) ? 1.f : -1.f;
        return;
    }

    dist = max_slant - 1.f;
    if (edge_id == 1) {
        const float s = (t1 >= 0.f) ? 1.f : -1.f;
        d_dist_du = a * s;
        d_dist_dv = b * s;
        return;
    }
    const float s = (t2 >= 0.f) ? 1.f : -1.f;
    d_dist_du = a * s;
    d_dist_dv = -b * s;
}

__forceinline__ __device__ void bounded_quadratic_coordinates(
    const float u, const float v, float &u_bounded, float &v_bounded
) {
    int edge_id;
    const float rho = hex_sdf_edge(u, v, edge_id) + 1.f;
    const float inv_scale = (rho > 1.f) ? (1.f / rho) : 1.f;
    u_bounded = u * inv_scale;
    v_bounded = v * inv_scale;
}

__forceinline__ __device__ void bounded_quadratic_coordinate_vjp(
    const float u,
    const float v,
    const float v_u_bounded,
    const float v_v_bounded,
    float &v_u,
    float &v_v
) {
    float dist;
    float d_rho_du;
    float d_rho_dv;
    int edge_id;
    hex_sdf_grad(u, v, dist, d_rho_du, d_rho_dv, edge_id);
    const float rho = dist + 1.f;
    if (rho <= 1.f) {
        v_u += v_u_bounded;
        v_v += v_v_bounded;
        return;
    }

    const float inv_rho = 1.f / rho;
    const float radial_vjp = (v_u_bounded * u + v_v_bounded * v) * inv_rho * inv_rho;
    v_u += v_u_bounded * inv_rho - radial_vjp * d_rho_du;
    v_v += v_v_bounded * inv_rho - radial_vjp * d_rho_dv;
}

__global__ void preprocess_hexels3d(
    const int P,
    const float *orig_points,
    const float2 *scales,
    const float scale_modifier,
    const float4 *rotations,
    const float *opacities,
    const float *softnesses_in,
    const float softness_default,
    const int softness_stride,
    const float *viewmatrix,
    const float *projmatrix,
    const float *campos,
    const int W,
    const int H,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    int *radii,
    int2 *tile_radii,
    float2 *points_xy_image,
    float2 *points_xy_center,
    float *depths,
    float *transMats,
    int *tiles_touched,
    float *opacities_out,
    float *softnesses_out,
    float3 *view_dirs,
    float3 *normals_out,
    const dim3 tile_grid
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= P) return;

    radii[idx] = 0;
    tile_radii[idx] = {0, 0};
    tiles_touched[idx] = 0;

    const float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
    const float3 c = {campos[0], campos[1], campos[2]};
    const float3 d = {p_orig.x - c.x, p_orig.y - c.y, p_orig.z - c.z};
    const float inv_norm = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z + 1e-8f);
    view_dirs[idx] = {d.x * inv_norm, d.y * inv_norm, d.z * inv_norm};

    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, p_view)) return;

    glm::mat3 T;
    const glm::vec4 rot = {rotations[idx].x, rotations[idx].y, rotations[idx].z, rotations[idx].w};
    compute_transmat(
        p_orig,
        {scales[idx].x, scales[idx].y},
        scale_modifier,
        rot,
        projmatrix,
        W,
        H,
        T
    );

    // Softness can be:
    // - per-primitive scalar (softness_stride=1)
    // - per-primitive (3,) anisotropic (softness_stride=3), aligned with the 3 oriented SDF terms
    float3 softness3 = {softness_default, softness_default, softness_default};
    if (softnesses_in != nullptr) {
        if (softness_stride == 1) {
            const float s = softnesses_in[idx];
            softness3 = {s, s, s};
        } else if (softness_stride == 3) {
            const float *s = softnesses_in + idx * 3;
            softness3 = {s[0], s[1], s[2]};
        }
    }
    softness3.x = fmaxf(softness3.x, 1e-4f);
    softness3.y = fmaxf(softness3.y, 1e-4f);
    softness3.z = fmaxf(softness3.z, 1e-4f);

    const float softness_min = fminf(softness3.x, fminf(softness3.y, softness3.z));
    const float cutoff = kHexRadius * (1.0f + (kLog254 / softness_min));

    float2 point_image = {0.f, 0.f};
    float2 extent = {0.f, 0.f};
    bool bounded;
    const bool old_bound = compute_aabb(T, cutoff, point_image, extent, bounded);
    float radius_base = fmaxf(fmaxf(extent.x, extent.y), cutoff * kFilterSize);
    if (prefilter_edge_on) radius_base = fmaxf(radius_base, 3.f * kFilterSize);
    int radius = old_bound ? (int)ceilf(radius_base) : 0;
    uint2 rect_min, rect_max;
    get_rect(point_image, {radius, radius}, rect_min, rect_max, tile_grid);
    const bool old_visible = old_bound && rect_min.x < rect_max.x && rect_min.y < rect_max.y;
    const bool corrected = conservative_bounds && !prefilter_edge_on;
    if (!old_visible && !corrected) return;
    radii[idx] = old_visible ? radius : 0;

    int2 tile_radius = {radius, radius};
    if (!prefilter_edge_on && bounded && opacities[idx] <= 1.f) {
        tile_radius.x = min(radius, (int)ceilf(fmaxf(extent.x, cutoff * kFilterSize)) + 1);
        tile_radius.y = min(radius, (int)ceilf(fmaxf(extent.y, cutoff * kFilterSize)) + 1);
        get_rect(point_image, tile_radius, rect_min, rect_max, tile_grid);
    }
    const uint2 legacy_min = rect_min, legacy_max = rect_max;
    if (!prefilter_edge_on) {
        const float opacity = opacities[idx];
        if (opacity < 1.f / 255.f) return;
        // Equality can still pass after sigmoid rounding; retain the old support.
        float margin = kLog254 / softness_min;
        if (opacity > 1.f / 255.f) {
            const float log_cutoff = logf((255.f * (1.f + 8.f * FLT_EPSILON)) * opacity - 1.f);
            const float softness_max = fmaxf(softness3.x, fmaxf(softness3.y, softness3.z));
            margin = log_cutoff / (log_cutoff < 0.f ? softness_max : softness_min);
        }
        if (!compute_hex_bounds(T, 1.f + margin, W, H, point_image, tile_radius)) return;
        get_rect(point_image, tile_radius, rect_min, rect_max, tile_grid);
        if (!corrected) {
            // Intersect with historical tile coverage so existing checkpoints render identically.
            rect_min = {max(rect_min.x, legacy_min.x), max(rect_min.y, legacy_min.y)};
            rect_max = {min(rect_max.x, legacy_max.x), min(rect_max.y, legacy_max.y)};
            if (rect_min.x >= rect_max.x || rect_min.y >= rect_max.y) return;
            point_image = {(rect_min.x + rect_max.x) * (kBlockX * 0.5f),
                           (rect_min.y + rect_max.y) * (kBlockY * 0.5f)};
            tile_radius = {(int)(rect_max.x - rect_min.x) * (kBlockX / 2),
                           (int)(rect_max.y - rect_min.y) * (kBlockY / 2)};
        }
        if (corrected && !old_visible) radii[idx] = max(tile_radius.x, tile_radius.y);
    }
    tile_radii[idx] = tile_radius;
    depths[idx] = p_view.z;
    points_xy_image[idx] = point_image;
    if (points_xy_center != nullptr) {
        const glm::mat4 world2ndc = glm::mat4(
            projmatrix[0],
            projmatrix[4],
            projmatrix[8],
            projmatrix[12],
            projmatrix[1],
            projmatrix[5],
            projmatrix[9],
            projmatrix[13],
            projmatrix[2],
            projmatrix[6],
            projmatrix[10],
            projmatrix[14],
            projmatrix[3],
            projmatrix[7],
            projmatrix[11],
            projmatrix[15]
        );
        const glm::vec4 clip = glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1.0f) * world2ndc;
        const float inv_w = 1.0f / (clip.w + 1e-8f);
        const float x_pix = ((float)W / 2.0f) * (clip.x * inv_w) + ((float)W - 1.0f) / 2.0f;
        const float y_pix = ((float)H / 2.0f) * (clip.y * inv_w) + ((float)H - 1.0f) / 2.0f;
        points_xy_center[idx] = {x_pix, y_pix};
    }
    opacities_out[idx] = opacities[idx];
    if (softness_stride == 3) {
        float *out = softnesses_out + idx * 3;
        out[0] = softness3.x;
        out[1] = softness3.y;
        out[2] = softness3.z;
    } else {
        softnesses_out[idx] = softness3.x;
    }
    tiles_touched[idx] = (int)((rect_max.y - rect_min.y) * (rect_max.x - rect_min.x));

    float *T_ptr = transMats + idx * 9;
    T_ptr[0] = T[0][0];
    T_ptr[1] = T[0][1];
    T_ptr[2] = T[0][2];
    T_ptr[3] = T[1][0];
    T_ptr[4] = T[1][1];
    T_ptr[5] = T[1][2];
    T_ptr[6] = T[2][0];
    T_ptr[7] = T[2][1];
    T_ptr[8] = T[2][2];

    if (normals_out != nullptr) {
        const glm::mat3 R = quat_to_rotmat(rot);
        const glm::vec3 n = R[2];
        normals_out[idx] = {n.x, n.y, n.z};
    }
}

__global__ void duplicate_with_keys(
    const int P,
    const float2 *points_xy,
    const float *depths,
    const int *offsets,
    uint64_t *keys_unsorted,
    uint32_t *values_unsorted,
    const int *radii,
    const int2 *tile_radii,
    const dim3 tile_grid
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= P) return;

    if (radii[idx] <= 0 || tile_radii[idx].x <= 0 || tile_radii[idx].y <= 0) return;

    uint32_t off = (idx == 0) ? 0 : (uint32_t)offsets[idx - 1];

    uint2 rect_min, rect_max;
    get_rect(points_xy[idx], tile_radii[idx], rect_min, rect_max, tile_grid);

    for (uint32_t y = rect_min.y; y < rect_max.y; ++y) {
        for (uint32_t x = rect_min.x; x < rect_max.x; ++x) {
            uint64_t key = (uint64_t)(y * tile_grid.x + x);
            key <<= 32;
            key |= *((const uint32_t *)&depths[idx]);
            keys_unsorted[off] = key;
            values_unsorted[off] = (uint32_t)idx;
            off++;
        }
    }
}

__global__ void identify_tile_ranges(const int L, const uint64_t *point_list_keys, uint2 *ranges) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= L) return;

    const uint64_t key = point_list_keys[idx];
    const uint32_t currtile = (uint32_t)(key >> 32);
    if (idx == 0) {
        ranges[currtile].x = 0;
    } else {
        const uint32_t prevtile = (uint32_t)(point_list_keys[idx - 1] >> 32);
        if (currtile != prevtile) {
            ranges[prevtile].y = (uint32_t)idx;
            ranges[currtile].x = (uint32_t)idx;
        }
    }
    if (idx == L - 1) {
        ranges[currtile].y = (uint32_t)L;
    }
}

__device__ __forceinline__ float surface_normal_sign(const float3 normal, const float *view_dir) {
    return normal.x * view_dir[0] + normal.y * view_dir[1] + normal.z * view_dir[2] > 0.f ? -1.f : 1.f;
}

template <bool kReturnSurface, bool kPrefilter, bool kReturnOwner = false>
__global__ void render_hexels3d(
    const uint2 *ranges,
    const uint32_t *point_list,
    const float2 *points_xy_center,
    const int W,
    const int H,
    const float *colors,
    const float *color_grads,
    const float *color_quads,
    const float *view_grads,
    const float *specular_ks,
    const float *specular_log_s,
    const float *view_dirs,
    const float *transMats,
    const float *opacities,
    const float *softnesses,
    const int softness_stride,
    const float *bg_color,
    float *out_color,
    float *primitive_contrib,
    float *out_overdraw,
    float *out_overdraw_hits,
    const int overdraw_base_count,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const float3 *normals,
    float *out_depth,
    float *out_alpha,
    float *out_surface,
    int64_t *out_owner_id,
    float *out_owner_weight,
    float *out_owner_uv,
    float *out_owner_sdf,
    float *out_owner_depth,
    float *out_total_alpha,
    const bool return_median_depth = false,
    float *final_transmittance = nullptr,
    int *last_index = nullptr
) {
    const uint32_t horizontal_blocks = (W + kBlockX - 1) / kBlockX;

    const uint2 pix_min = {blockIdx.x * kBlockX, blockIdx.y * kBlockY};
    const uint2 pix = {pix_min.x + (uint32_t)threadIdx.x, pix_min.y + (uint32_t)threadIdx.y};
    const uint32_t pix_id = (uint32_t)(W * (int)pix.y + (int)pix.x);
    const float2 pixf = {(float)pix.x, (float)pix.y};

    const bool inside = pix.x < (uint32_t)W && pix.y < (uint32_t)H;
    bool done_color = !inside;
    bool done_surf = !inside;
    bool done = done_color;
    if constexpr (!kReturnSurface) {
        done_surf = true;
        done = done_color;
    } else {
        done = done_color && done_surf;
    }

    const uint2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];
    const int rounds = (int)((range.y - range.x + kBlockSize - 1) / kBlockSize);
    int to_do = (int)(range.y - range.x);

    __shared__ float collected_opacity[kBlockSize];
    __shared__ float3 collected_softness[kBlockSize];
    __shared__ float3 collected_Tu[kBlockSize];
    __shared__ float3 collected_Tv[kBlockSize];
    __shared__ float3 collected_Tw[kBlockSize];
    __shared__ float3 collected_color[kBlockSize];
    __shared__ float3 collected_color_grad_u[kBlockSize];
    __shared__ float3 collected_color_grad_v[kBlockSize];
    __shared__ float3 collected_view_dir[kBlockSize];
    __shared__ float3 collected_view_grad_r[kBlockSize];
    __shared__ float3 collected_view_grad_g[kBlockSize];
    __shared__ float3 collected_view_grad_b[kBlockSize];
    __shared__ float3 collected_specular_ks[kBlockSize];
    __shared__ float collected_specular_log_s[kBlockSize];
    __shared__ float3 collected_normal[kBlockSize];
    __shared__ uint32_t collected_id[kBlockSize];
    __shared__ float collected_contrib[kBlockSize];
    __shared__ float2 collected_center[kPrefilter ? kBlockSize : 1];

    int n_visited = 0;
    float T = 1.0f;
    float C[kNumChannels] = {0.f, 0.f, 0.f};
    float overdraw = 0.0f;
    float overdraw_hits = 0.0f;
    float T_surf = 1.0f;
    float D_surf = 0.0f;
    float median_depth = 0.0f;
    float M1_surf = 0.0f;
    float M2_surf = 0.0f;
    float distortion_surf = 0.0f;
    float3 N_surf = {0.f, 0.f, 0.f};
    int64_t owner_id = -1;
    float owner_weight = 0.0f;
    float2 owner_uv = {0.0f, 0.0f};
    float owner_sdf = 0.0f;
    float owner_depth = 0.0f;

    for (int i = 0; i < rounds; ++i, to_do -= kBlockSize) {
        const int num_done = __syncthreads_count(done);
        if (num_done == kBlockSize) break;

        const int shared_idx = threadIdx.y * kBlockX + threadIdx.x;
        const int progress = i * kBlockSize + shared_idx;
        collected_contrib[shared_idx] = 0.0f;
        if ((int)range.x + progress < (int)range.y) {
            const uint32_t coll_id = point_list[range.x + (uint32_t)progress];
            collected_id[shared_idx] = coll_id;
            if constexpr (kPrefilter) {
                collected_center[shared_idx] = points_xy_center[coll_id];
            }
            collected_opacity[shared_idx] = opacities[coll_id];
            if (softness_stride == 3) {
                const float *s = softnesses + (int)coll_id * 3;
                collected_softness[shared_idx] = {s[0], s[1], s[2]};
            } else {
                const float s = softnesses[coll_id];
                collected_softness[shared_idx] = {s, s, s};
            }

            const float *Tm = transMats + (int)coll_id * 9;
            collected_Tu[shared_idx] = {Tm[0], Tm[1], Tm[2]};
            collected_Tv[shared_idx] = {Tm[3], Tm[4], Tm[5]};
            collected_Tw[shared_idx] = {Tm[6], Tm[7], Tm[8]};

            const float *c = colors + (int)coll_id * 3;
            collected_color[shared_idx] = {c[0], c[1], c[2]};

            if (color_model >= 1) {
                const float *cg = color_grads + (int)coll_id * 6;
                collected_color_grad_u[shared_idx] = {cg[0], cg[1], cg[2]};
                collected_color_grad_v[shared_idx] = {cg[3], cg[4], cg[5]};
            } else {
                collected_color_grad_u[shared_idx] = {0.f, 0.f, 0.f};
                collected_color_grad_v[shared_idx] = {0.f, 0.f, 0.f};
            }

            if (use_view_grads) {
                const float *vd = view_dirs + (int)coll_id * 3;
                collected_view_dir[shared_idx] = {vd[0], vd[1], vd[2]};

                const float *vg = view_grads + (int)coll_id * 9;
                collected_view_grad_r[shared_idx] = {vg[0], vg[1], vg[2]};
                collected_view_grad_g[shared_idx] = {vg[3], vg[4], vg[5]};
                collected_view_grad_b[shared_idx] = {vg[6], vg[7], vg[8]};
            } else {
                if (view_model == kViewModelLobe) {
                    const float *vd = view_dirs + (int)coll_id * 3;
                    collected_view_dir[shared_idx] = {vd[0], vd[1], vd[2]};
                } else {
                    collected_view_dir[shared_idx] = {0.f, 0.f, 0.f};
                }
                collected_view_grad_r[shared_idx] = {0.f, 0.f, 0.f};
                collected_view_grad_g[shared_idx] = {0.f, 0.f, 0.f};
                collected_view_grad_b[shared_idx] = {0.f, 0.f, 0.f};
            }

            if (view_model == kViewModelLobe) {
                const float *ks = specular_ks + (int)coll_id * 3;
                collected_specular_ks[shared_idx] = {ks[0], ks[1], ks[2]};
                collected_specular_log_s[shared_idx] = specular_log_s[coll_id];
            } else {
                collected_specular_ks[shared_idx] = {0.f, 0.f, 0.f};
                collected_specular_log_s[shared_idx] = 0.f;
            }

            if (normals != nullptr) {
                collected_normal[shared_idx] = normals[(int)coll_id];
            } else {
                collected_normal[shared_idx] = {0.f, 0.f, 0.f};
            }
        }
        __syncthreads();

        const int batch_size = min(kBlockSize, to_do);
        for (int j = 0; !done && j < batch_size; ++j) {
            const uint32_t g_id = collected_id[j];
            const bool is_base = (int)g_id < overdraw_base_count;
            if constexpr (kReturnSurface) {
                if (done_color && !is_base) {
                    continue;
                }
            }
            const float3 Tu = collected_Tu[j];
            const float3 Tv = collected_Tv[j];
            const float3 Tw = collected_Tw[j];

            const float3 k = {pixf.x * Tw.x - Tu.x, pixf.x * Tw.y - Tu.y, pixf.x * Tw.z - Tu.z};
            const float3 l = {pixf.y * Tw.x - Tv.x, pixf.y * Tw.y - Tv.y, pixf.y * Tw.z - Tv.z};
            const float3 p = cross3(k, l);
            if (p.z == 0.0f) continue;

            const float2 s = {p.x / p.z, p.y / p.z};

            const float depth_sdf = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
            if (depth_sdf < kNearPlane) continue;

            int edge_id;
            const float dist = hex_sdf_edge(s.x, s.y, edge_id);
            const float3 soft3 = collected_softness[j];
            const float softness_eff = (edge_id == 0) ? soft3.x : ((edge_id == 1) ? soft3.y : soft3.z);
            const float boundary = sigmoid_neg_dist(dist, softness_eff);
            const float alpha_raw_sdf = collected_opacity[j] * boundary;
            const float alpha_sdf = fminf(alpha_raw_sdf, 0.99f);
            float alpha = alpha_sdf;
            bool use_filter = false;
            if constexpr (kPrefilter) {
                const float2 center = collected_center[j];
                const float dx = pixf.x - center.x;
                const float dy = pixf.y - center.y;
                const float screen_dist_sq = dx * dx + dy * dy;
                const float G = __expf(-0.5f * kFilterInvSquare * screen_dist_sq);
                const float alpha_raw_filter = collected_opacity[j] * G;
                const float alpha_filter = fminf(alpha_raw_filter, 0.99f);
                use_filter = alpha_filter > alpha_sdf;
                alpha = use_filter ? alpha_filter : alpha_sdf;
            }
            if (alpha < 1.0f / 255.0f) continue;
            const float depth = use_filter ? Tw.z : depth_sdf;

            if (!done_color) {
                if (out_overdraw != nullptr && is_base && alpha_sdf >= 1.0f / 255.0f) {
                    overdraw += alpha_raw_sdf;
                }
                if (out_overdraw_hits != nullptr && is_base && alpha_sdf >= 1.0f / 255.0f) {
                    overdraw_hits += 1.0f;
                }

                const float test_T = T * (1.f - alpha);
                if (test_T < 0.0001f) {
                    done_color = true;
                } else {
                    const float w = alpha * T;
                    const float u = use_filter ? 0.f : s.x;
                    const float v = use_filter ? 0.f : s.y;

                    if constexpr (kReturnOwner) {
                        if (w > owner_weight) {
                            owner_id = (int64_t)g_id;
                            owner_weight = w;
                            owner_uv = s;
                            owner_sdf = dist;
                            owner_depth = depth;
                        }
                    }

                    const float3 c_base = collected_color[j];
                    const float3 grad_u = collected_color_grad_u[j];
                    const float3 grad_v = collected_color_grad_v[j];
                    float3 quad_uu = {0.f, 0.f, 0.f};
                    float3 quad_uv = {0.f, 0.f, 0.f};
                    float3 quad_vv = {0.f, 0.f, 0.f};
                    if (
                        color_model == kColorModelQuadratic ||
                        color_model == kColorModelQuadraticBounded
                    ) {
                        const float *cq = color_quads + (int)g_id * 9;
                        quad_uu = {cq[0], cq[1], cq[2]};
                        quad_uv = {cq[3], cq[4], cq[5]};
                        quad_vv = {cq[6], cq[7], cq[8]};
                    }
                    const float3 view_dir = collected_view_dir[j];
                    const float3 vgrad_r = collected_view_grad_r[j];
                    const float3 vgrad_g = collected_view_grad_g[j];
                    const float3 vgrad_b = collected_view_grad_b[j];
                    const float3 ks = collected_specular_ks[j];
                    const float log_s = collected_specular_log_s[j];
                    const float3 n = collected_normal[j];

                    float3 c = c_base;
                    if (color_model >= 1) {
                        c.x += u * grad_u.x + v * grad_v.x;
                        c.y += u * grad_u.y + v * grad_v.y;
                        c.z += u * grad_u.z + v * grad_v.z;
                    }
                    if (color_model == kColorModelQuadratic) {
                        const float u2 = u * u;
                        const float uv = u * v;
                        const float v2 = v * v;
                        c.x += u2 * quad_uu.x + uv * quad_uv.x + v2 * quad_vv.x;
                        c.y += u2 * quad_uu.y + uv * quad_uv.y + v2 * quad_vv.y;
                        c.z += u2 * quad_uu.z + uv * quad_uv.z + v2 * quad_vv.z;
                    } else if (color_model == kColorModelQuadraticBounded) {
                        float u_bounded;
                        float v_bounded;
                        bounded_quadratic_coordinates(u, v, u_bounded, v_bounded);
                        const float u2 = u_bounded * u_bounded;
                        const float uv = u_bounded * v_bounded;
                        const float v2 = v_bounded * v_bounded;
                        c.x += u2 * quad_uu.x + uv * quad_uv.x + v2 * quad_vv.x;
                        c.y += u2 * quad_uu.y + uv * quad_uv.y + v2 * quad_vv.y;
                        c.z += u2 * quad_uu.z + uv * quad_uv.z + v2 * quad_vv.z;
                    }
                    if (use_view_grads) {
                        c.x += (view_dir.x * vgrad_r.x + view_dir.y * vgrad_r.y + view_dir.z * vgrad_r.z);
                        c.y += (view_dir.x * vgrad_g.x + view_dir.y * vgrad_g.y + view_dir.z * vgrad_g.z);
                        c.z += (view_dir.x * vgrad_b.x + view_dir.y * vgrad_b.y + view_dir.z * vgrad_b.z);
                    }
                    if (view_model == kViewModelLobe) {
                        const float cos_nv = fmaxf(-(n.x * view_dir.x + n.y * view_dir.y + n.z * view_dir.z), 0.f);
                        const float log_s_clamped = fminf(fmaxf(log_s, 0.f), 5.0f);
                        const float s_exp = __expf(log_s_clamped);
                        const float spec = (cos_nv > 0.f) ? __powf(cos_nv, s_exp) : 0.f;
                        c.x += ks.x * spec;
                        c.y += ks.y * spec;
                        c.z += ks.z * spec;
                    }

                    C[0] += c.x * w;
                    C[1] += c.y * w;
                    C[2] += c.z * w;
                    T = test_T;
                    n_visited = i * kBlockSize + j + 1;

                    if (primitive_contrib != nullptr) {
                        atomicAdd(&collected_contrib[j], w);
                    }
                }
            }

            if constexpr (kReturnSurface) {
                if (!done_surf && is_base) {
                    const float test_T_surf = T_surf * (1.f - alpha);
                    if (test_T_surf < 0.0001f) {
                        done_surf = true;
                    } else {
                        if (return_median_depth && T_surf > 0.5f) {
                            median_depth = depth;
                        }
                        const float w_surf = alpha * T_surf;
                        const float m =
                            kFarPlane / (kFarPlane - kNearPlane) *
                            (1.0f - kNearPlane / depth);
                        const float accumulated_alpha = 1.0f - T_surf;
                        distortion_surf +=
                            (m * m * accumulated_alpha + M2_surf - 2.0f * m * M1_surf) *
                            w_surf;
                        M1_surf += m * w_surf;
                        M2_surf += m * m * w_surf;
                        D_surf += depth * w_surf;
                        float3 n = normals[(int)g_id];
                        const float normal_sign = surface_normal_sign(n, view_dirs + (int)g_id * 3);
                        n = {normal_sign * n.x, normal_sign * n.y, normal_sign * n.z};
                        N_surf.x += n.x * w_surf;
                        N_surf.y += n.y * w_surf;
                        N_surf.z += n.z * w_surf;
                        T_surf = test_T_surf;
                    }
                }
                done = done_color && done_surf;
            } else {
                done = done_color;
            }
        }

        if (primitive_contrib != nullptr) {
            __syncthreads();
            if (shared_idx < batch_size) {
                atomicAdd(&primitive_contrib[(int)collected_id[shared_idx]], collected_contrib[shared_idx]);
            }
            __syncthreads();
        }
    }

    if (inside) {
        if (final_transmittance != nullptr) {
            final_transmittance[pix_id] = T;
            last_index[pix_id] = n_visited;
        }
        out_color[0 * H * W + pix_id] = C[0] + T * bg_color[0];
        out_color[1 * H * W + pix_id] = C[1] + T * bg_color[1];
        out_color[2 * H * W + pix_id] = C[2] + T * bg_color[2];
        if (out_overdraw != nullptr) {
            out_overdraw[pix_id] = overdraw;
        }
        if (out_overdraw_hits != nullptr) {
            out_overdraw_hits[pix_id] = overdraw_hits;
        }
        if constexpr (kReturnSurface) {
            out_depth[pix_id] = D_surf;
            out_alpha[pix_id] = 1.0f - T_surf;
            out_surface[0 * H * W + pix_id] = N_surf.x;
            out_surface[1 * H * W + pix_id] = N_surf.y;
            out_surface[2 * H * W + pix_id] = N_surf.z;
            out_surface[3 * H * W + pix_id] = distortion_surf;
            if (return_median_depth) {
                out_surface[4 * H * W + pix_id] = median_depth;
            }
        }
        if constexpr (kReturnOwner) {
            out_owner_id[pix_id] = owner_id;
            out_owner_weight[pix_id] = owner_weight;
            out_owner_uv[0 * H * W + pix_id] = owner_uv.x;
            out_owner_uv[1 * H * W + pix_id] = owner_uv.y;
            out_owner_sdf[pix_id] = owner_sdf;
            out_owner_depth[pix_id] = owner_depth;
            out_total_alpha[pix_id] = 1.0f - T;
        }
    }
}

uint32_t get_higher_msb(uint32_t n) {
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb) {
            msb += step;
        } else {
            msb -= step;
        }
    }
    if (n >> msb) {
        msb++;
    }
    return msb;
}

__forceinline__ __device__ bool compute_hexel_sdf(
    const float2 pixf,
    const float3 Tu,
    const float3 Tv,
    const float3 Tw,
    const float opacity,
    const float3 softness3,
    float &u,
    float &v,
    float &depth,
    float &dist,
    int &edge_id,
    float &softness_eff,
    float &boundary,
    float &alpha_raw,
    float &alpha,
    bool &clamped
) {
    const float3 k = {pixf.x * Tw.x - Tu.x, pixf.x * Tw.y - Tu.y, pixf.x * Tw.z - Tu.z};
    const float3 l = {pixf.y * Tw.x - Tv.x, pixf.y * Tw.y - Tv.y, pixf.y * Tw.z - Tv.z};
    const float3 p = cross3(k, l);
    if (p.z == 0.0f) return false;

    u = p.x / p.z;
    v = p.y / p.z;

    depth = (u * Tw.x + v * Tw.y) + Tw.z;
    if (depth < kNearPlane) return false;

    dist = hex_sdf_edge(u, v, edge_id);
    softness_eff = (edge_id == 0) ? softness3.x : ((edge_id == 1) ? softness3.y : softness3.z);
    boundary = sigmoid_neg_dist(dist, softness_eff);

    alpha_raw = opacity * boundary;
    clamped = alpha_raw > 0.99f;
    alpha = clamped ? 0.99f : alpha_raw;
    return true;
}

__forceinline__ __device__ float3 exact_sort_appearance(
    const uint32_t g_id,
    const float u,
    const float v,
    const float *colors,
    const float *color_grads,
    const float *color_quads,
    const float *view_grads,
    const float *specular_ks,
    const float *specular_log_s,
    const float3 *view_dirs,
    const float3 *normals,
    const int color_model,
    const int view_model,
    const bool use_view_grads
) {
    const float *base = colors + (int)g_id * 3;
    float3 color = {base[0], base[1], base[2]};

    if (color_model >= 1) {
        const float *grad = color_grads + (int)g_id * 6;
        color.x += u * grad[0] + v * grad[3];
        color.y += u * grad[1] + v * grad[4];
        color.z += u * grad[2] + v * grad[5];
    }
    if (color_model == kColorModelQuadratic) {
        const float *quad = color_quads + (int)g_id * 9;
        const float u2 = u * u;
        const float uv = u * v;
        const float v2 = v * v;
        color.x += u2 * quad[0] + uv * quad[3] + v2 * quad[6];
        color.y += u2 * quad[1] + uv * quad[4] + v2 * quad[7];
        color.z += u2 * quad[2] + uv * quad[5] + v2 * quad[8];
    } else if (color_model == kColorModelQuadraticBounded) {
        const float *quad = color_quads + (int)g_id * 9;
        float u_bounded;
        float v_bounded;
        bounded_quadratic_coordinates(u, v, u_bounded, v_bounded);
        const float u2 = u_bounded * u_bounded;
        const float uv = u_bounded * v_bounded;
        const float v2 = v_bounded * v_bounded;
        color.x += u2 * quad[0] + uv * quad[3] + v2 * quad[6];
        color.y += u2 * quad[1] + uv * quad[4] + v2 * quad[7];
        color.z += u2 * quad[2] + uv * quad[5] + v2 * quad[8];
    }
    if (use_view_grads) {
        const float3 direction = view_dirs[g_id];
        const float *grad = view_grads + (int)g_id * 9;
        color.x += direction.x * grad[0] + direction.y * grad[1] + direction.z * grad[2];
        color.y += direction.x * grad[3] + direction.y * grad[4] + direction.z * grad[5];
        color.z += direction.x * grad[6] + direction.y * grad[7] + direction.z * grad[8];
    }
    if (view_model == kViewModelLobe) {
        const float3 direction = view_dirs[g_id];
        const float3 normal = normals[g_id];
        const float cos_nv = fmaxf(
            -(normal.x * direction.x + normal.y * direction.y + normal.z * direction.z),
            0.f
        );
        const float log_s = fminf(fmaxf(specular_log_s[g_id], 0.f), 5.f);
        const float exponent = __expf(log_s);
        const float spec = (cos_nv > 0.f) ? __powf(cos_nv, exponent) : 0.f;
        const float *ks = specular_ks + (int)g_id * 3;
        color.x += ks[0] * spec;
        color.y += ks[1] * spec;
        color.z += ks[2] * spec;
    }
    return color;
}

__forceinline__ __device__ float alpha_contour_proximity(
    const float2 pixf,
    const float u,
    const float v,
    const int edge_id,
    const float opacity,
    const float softness_eff,
    const float3 Tu,
    const float3 Tv,
    const float3 Tw
) {
    constexpr float a0 = 1.f / 255.f;
    if (opacity <= a0) return 0.f;

    constexpr float a = 0.8660254f;
    constexpr float b = 0.5f;
    float normal_u;
    float normal_v;
    if (edge_id == 0) {
        normal_u = 0.f;
        normal_v = (v >= 0.f) ? 1.f : -1.f;
    } else if (edge_id == 1) {
        const float sign = (a * u + b * v >= 0.f) ? 1.f : -1.f;
        normal_u = sign * a;
        normal_v = sign * b;
    } else {
        const float sign = (a * u - b * v >= 0.f) ? 1.f : -1.f;
        normal_u = sign * a;
        normal_v = -sign * b;
    }

    const float d_star = logf((opacity - a0) / a0) / softness_eff;
    const float level = 1.f + d_star;
    const float3 local_point = {normal_u * level, normal_v * level, 1.f};
    const float3 local_tangent = {-normal_v, normal_u, 0.f};
    const float3 screen_point = {
        Tu.x * local_point.x + Tu.y * local_point.y + Tu.z,
        Tv.x * local_point.x + Tv.y * local_point.y + Tv.z,
        Tw.x * local_point.x + Tw.y * local_point.y + Tw.z,
    };
    const float3 screen_tangent = {
        Tu.x * local_tangent.x + Tu.y * local_tangent.y,
        Tv.x * local_tangent.x + Tv.y * local_tangent.y,
        Tw.x * local_tangent.x + Tw.y * local_tangent.y,
    };
    const float3 line = cross3(screen_point, screen_tangent);
    const float norm = sqrtf(line.x * line.x + line.y * line.y);
    if (norm == 0.f) return 0.f;
    const float distance = fabsf(line.x * pixf.x + line.y * pixf.y + line.z) / norm;
    return fmaxf(0.f, 1.f - distance);
}

__global__ void compute_pixel_inversion_risk(
    const uint2 *ranges,
    const uint32_t *point_list,
    const int W,
    const int H,
    const int crop_x0,
    const int crop_y0,
    const int crop_x1,
    const int crop_y1,
    const float *colors,
    const float *color_grads,
    const float *color_quads,
    const float *view_grads,
    const float *specular_ks,
    const float *specular_log_s,
    const float3 *view_dirs,
    const float3 *normals,
    const float *transMats,
    const float *opacities,
    const float *softnesses,
    const int softness_stride,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    float *risk_map
) {
    const uint32_t horizontal_blocks = (W + kBlockX - 1) / kBlockX;
    const uint2 pix_min = {blockIdx.x * kBlockX, blockIdx.y * kBlockY};
    if (
        (int)pix_min.x >= crop_x1 || (int)pix_min.y >= crop_y1 ||
        (int)pix_min.x + kBlockX <= crop_x0 || (int)pix_min.y + kBlockY <= crop_y0
    ) {
        return;
    }

    const uint2 pix = {pix_min.x + (uint32_t)threadIdx.x, pix_min.y + (uint32_t)threadIdx.y};
    const uint32_t pix_id = (uint32_t)(W * (int)pix.y + (int)pix.x);
    const float2 pixf = {(float)pix.x, (float)pix.y};
    const bool inside =
        pix.x < (uint32_t)W && pix.y < (uint32_t)H &&
        (int)pix.x >= crop_x0 && (int)pix.x < crop_x1 &&
        (int)pix.y >= crop_y0 && (int)pix.y < crop_y1;
    bool done = !inside;

    const uint2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];
    const int rounds = (int)((range.y - range.x + kBlockSize - 1) / kBlockSize);
    int to_do = (int)(range.y - range.x);

    __shared__ float collected_opacity[kBlockSize];
    __shared__ float3 collected_softness[kBlockSize];
    __shared__ float3 collected_Tu[kBlockSize];
    __shared__ float3 collected_Tv[kBlockSize];
    __shared__ float3 collected_Tw[kBlockSize];
    __shared__ uint32_t collected_id[kBlockSize];

    float transmittance = 1.f;
    float risk = 0.f;
    bool have_previous = false;
    float previous_depth = 0.f;
    float previous_alpha = 0.f;
    float previous_transmittance = 1.f;
    float previous_boundary = 0.f;
    float3 previous_color = {0.f, 0.f, 0.f};

    const int shared_idx = threadIdx.y * kBlockX + threadIdx.x;
    for (int round = 0; round < rounds; ++round, to_do -= kBlockSize) {
        if (__syncthreads_count(done) == kBlockSize) break;

        const int progress = round * kBlockSize + shared_idx;
        if ((int)range.x + progress < (int)range.y) {
            const uint32_t g_id = point_list[range.x + (uint32_t)progress];
            collected_id[shared_idx] = g_id;
            collected_opacity[shared_idx] = opacities[g_id];
            if (softness_stride == 3) {
                const float *soft = softnesses + (int)g_id * 3;
                collected_softness[shared_idx] = {soft[0], soft[1], soft[2]};
            } else {
                const float soft = softnesses[g_id];
                collected_softness[shared_idx] = {soft, soft, soft};
            }
            const float *transform = transMats + (int)g_id * 9;
            collected_Tu[shared_idx] = {transform[0], transform[1], transform[2]};
            collected_Tv[shared_idx] = {transform[3], transform[4], transform[5]};
            collected_Tw[shared_idx] = {transform[6], transform[7], transform[8]};
        }
        __syncthreads();

        const int batch_size = min(kBlockSize, to_do);
        for (int j = 0; !done && j < batch_size; ++j) {
            float u, v, depth, dist, softness_eff, boundary, alpha_raw, alpha;
            int edge_id;
            bool clamped;
            if (!compute_hexel_sdf(
                    pixf,
                    collected_Tu[j],
                    collected_Tv[j],
                    collected_Tw[j],
                    collected_opacity[j],
                    collected_softness[j],
                    u,
                    v,
                    depth,
                    dist,
                    edge_id,
                    softness_eff,
                    boundary,
                    alpha_raw,
                    alpha,
                    clamped
                )) {
                continue;
            }
            if (alpha < 1.f / 255.f) continue;

            const float test_transmittance = transmittance * (1.f - alpha);
            if (test_transmittance < 0.0001f) {
                done = true;
                continue;
            }

            const uint32_t g_id = collected_id[j];
            const float3 color = exact_sort_appearance(
                g_id,
                u,
                v,
                colors,
                color_grads,
                color_quads,
                view_grads,
                specular_ks,
                specular_log_s,
                view_dirs,
                normals,
                color_model,
                view_model,
                use_view_grads
            );
            const float contour = alpha_contour_proximity(
                pixf,
                u,
                v,
                edge_id,
                collected_opacity[j],
                softness_eff,
                collected_Tu[j],
                collected_Tv[j],
                collected_Tw[j]
            );

            if (have_previous) {
                const float tolerance = 8.f * FLT_EPSILON * fmaxf(
                    1.f,
                    fmaxf(fabsf(previous_depth), fabsf(depth))
                );
                if (previous_depth > depth + tolerance) {
                    const float contrast = fminf(
                        1.f,
                        fmaxf(
                            fabsf(previous_color.x - color.x),
                            fmaxf(
                                fabsf(previous_color.y - color.y),
                                fabsf(previous_color.z - color.z)
                            )
                        )
                    );
                    const float term = previous_transmittance * previous_alpha * alpha * contrast *
                                       fmaxf(previous_boundary, contour);
                    risk = fmaxf(risk, term);
                }
            }

            have_previous = true;
            previous_depth = depth;
            previous_alpha = alpha;
            previous_transmittance = transmittance;
            previous_boundary = contour;
            previous_color = color;
            transmittance = test_transmittance;
        }
        __syncthreads();
    }

    if (inside) risk_map[pix_id] = risk;
}

__global__ void count_exact_sort_hits(
    const uint2 *ranges,
    const uint32_t *point_list,
    const int W,
    const int H,
    const int crop_x0,
    const int crop_y0,
    const int crop_x1,
    const int crop_y1,
    const float *transMats,
    const float *opacities,
    const float *softnesses,
    const int softness_stride,
    const bool *selected_mask,
    const bool *pixel_mask,
    const int exact_mode,
    int *hit_counts,
    int64_t *record_counts,
    bool *enabled_mask
) {
    const uint2 pix_min = {blockIdx.x * kBlockX, blockIdx.y * kBlockY};
    if (
        (int)pix_min.x >= crop_x1 || (int)pix_min.y >= crop_y1 ||
        (int)pix_min.x + kBlockX <= crop_x0 || (int)pix_min.y + kBlockY <= crop_y0
    ) {
        return;
    }

    const uint32_t horizontal_blocks = (W + kBlockX - 1) / kBlockX;
    const uint2 pix = {pix_min.x + (uint32_t)threadIdx.x, pix_min.y + (uint32_t)threadIdx.y};
    const uint32_t pix_id = (uint32_t)(W * (int)pix.y + (int)pix.x);
    const float2 pixf = {(float)pix.x, (float)pix.y};
    const bool inside =
        pix.x < (uint32_t)W && pix.y < (uint32_t)H &&
        (int)pix.x >= crop_x0 && (int)pix.x < crop_x1 &&
        (int)pix.y >= crop_y0 && (int)pix.y < crop_y1;

    const uint2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];
    const int rounds = (int)((range.y - range.x + kBlockSize - 1) / kBlockSize);
    int to_do = (int)(range.y - range.x);
    int hits = 0;
    bool selected_touch = false;

    __shared__ float collected_opacity[kBlockSize];
    __shared__ float3 collected_softness[kBlockSize];
    __shared__ float3 collected_Tu[kBlockSize];
    __shared__ float3 collected_Tv[kBlockSize];
    __shared__ float3 collected_Tw[kBlockSize];
    __shared__ uint32_t collected_id[kBlockSize];
    __shared__ bool collected_selected[kBlockSize];

    const int shared_idx = threadIdx.y * kBlockX + threadIdx.x;
    for (int round = 0; round < rounds; ++round, to_do -= kBlockSize) {
        const int progress = round * kBlockSize + shared_idx;
        if ((int)range.x + progress < (int)range.y) {
            const uint32_t g_id = point_list[range.x + (uint32_t)progress];
            collected_id[shared_idx] = g_id;
            collected_opacity[shared_idx] = opacities[g_id];
            if (softness_stride == 3) {
                const float *soft = softnesses + (int)g_id * 3;
                collected_softness[shared_idx] = {soft[0], soft[1], soft[2]};
            } else {
                const float soft = softnesses[g_id];
                collected_softness[shared_idx] = {soft, soft, soft};
            }
            const float *transform = transMats + (int)g_id * 9;
            collected_Tu[shared_idx] = {transform[0], transform[1], transform[2]};
            collected_Tv[shared_idx] = {transform[3], transform[4], transform[5]};
            collected_Tw[shared_idx] = {transform[6], transform[7], transform[8]};
            collected_selected[shared_idx] = selected_mask != nullptr && selected_mask[g_id];
        }
        __syncthreads();

        if (inside) {
            const int batch_size = min(kBlockSize, to_do);
            for (int j = 0; j < batch_size; ++j) {
                float u, v, depth, dist, softness_eff, boundary, alpha_raw, alpha;
                int edge_id;
                bool clamped;
                if (!compute_hexel_sdf(
                        pixf,
                        collected_Tu[j],
                        collected_Tv[j],
                        collected_Tw[j],
                        collected_opacity[j],
                        collected_softness[j],
                        u,
                        v,
                        depth,
                        dist,
                        edge_id,
                        softness_eff,
                        boundary,
                        alpha_raw,
                        alpha,
                        clamped
                    )) {
                    continue;
                }
                if (alpha < 1.f / 255.f) continue;
                ++hits;
                selected_touch = selected_touch || collected_selected[j];
            }
        }
        __syncthreads();
    }

    if (inside) {
        const bool enabled = exact_mode == kExactSortPixelMask ? pixel_mask[pix_id] :
            exact_mode == kExactSortCenterReplay || exact_mode == kExactSortCropAll ||
            exact_mode == kExactSortBoundaryTail || selected_touch;
        hit_counts[pix_id] = hits;
        record_counts[pix_id] = enabled ? (int64_t)hits : 0;
        enabled_mask[pix_id] = enabled;
    }
}

__global__ void fill_exact_sort_records(
    const uint2 *ranges,
    const uint32_t *point_list,
    const int W,
    const int H,
    const int crop_x0,
    const int crop_y0,
    const int crop_x1,
    const int crop_y1,
    const float *center_depths,
    const float *transMats,
    const float *opacities,
    const float *softnesses,
    const int softness_stride,
    const bool *selected_mask,
    const bool *enabled_mask,
    const int64_t *record_offsets,
    const int exact_mode,
    uint64_t *keys_unsorted,
    uint32_t *values_unsorted
) {
    const uint2 pix_min = {blockIdx.x * kBlockX, blockIdx.y * kBlockY};
    if (
        (int)pix_min.x >= crop_x1 || (int)pix_min.y >= crop_y1 ||
        (int)pix_min.x + kBlockX <= crop_x0 || (int)pix_min.y + kBlockY <= crop_y0
    ) {
        return;
    }

    const uint32_t horizontal_blocks = (W + kBlockX - 1) / kBlockX;
    const uint2 pix = {pix_min.x + (uint32_t)threadIdx.x, pix_min.y + (uint32_t)threadIdx.y};
    const uint32_t pix_id = (uint32_t)(W * (int)pix.y + (int)pix.x);
    const float2 pixf = {(float)pix.x, (float)pix.y};
    const bool inside =
        pix.x < (uint32_t)W && pix.y < (uint32_t)H &&
        (int)pix.x >= crop_x0 && (int)pix.x < crop_x1 &&
        (int)pix.y >= crop_y0 && (int)pix.y < crop_y1;
    const bool enabled = inside && enabled_mask[pix_id];

    const uint2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];
    const int rounds = (int)((range.y - range.x + kBlockSize - 1) / kBlockSize);
    int to_do = (int)(range.y - range.x);
    int64_t write_index = enabled ? ((pix_id == 0) ? 0 : record_offsets[pix_id - 1]) : 0;

    __shared__ float collected_center_depth[kBlockSize];
    __shared__ float collected_opacity[kBlockSize];
    __shared__ float3 collected_softness[kBlockSize];
    __shared__ float3 collected_Tu[kBlockSize];
    __shared__ float3 collected_Tv[kBlockSize];
    __shared__ float3 collected_Tw[kBlockSize];
    __shared__ uint32_t collected_id[kBlockSize];
    __shared__ bool collected_selected[kBlockSize];

    const int shared_idx = threadIdx.y * kBlockX + threadIdx.x;
    for (int round = 0; round < rounds; ++round, to_do -= kBlockSize) {
        const int progress = round * kBlockSize + shared_idx;
        if ((int)range.x + progress < (int)range.y) {
            const uint32_t g_id = point_list[range.x + (uint32_t)progress];
            collected_id[shared_idx] = g_id;
            collected_center_depth[shared_idx] = center_depths[g_id];
            collected_opacity[shared_idx] = opacities[g_id];
            if (softness_stride == 3) {
                const float *soft = softnesses + (int)g_id * 3;
                collected_softness[shared_idx] = {soft[0], soft[1], soft[2]};
            } else {
                const float soft = softnesses[g_id];
                collected_softness[shared_idx] = {soft, soft, soft};
            }
            const float *transform = transMats + (int)g_id * 9;
            collected_Tu[shared_idx] = {transform[0], transform[1], transform[2]};
            collected_Tv[shared_idx] = {transform[3], transform[4], transform[5]};
            collected_Tw[shared_idx] = {transform[6], transform[7], transform[8]};
            collected_selected[shared_idx] = selected_mask != nullptr && selected_mask[g_id];
        }
        __syncthreads();

        if (enabled) {
            const int batch_size = min(kBlockSize, to_do);
            for (int j = 0; j < batch_size; ++j) {
                float u, v, depth, dist, softness_eff, boundary, alpha_raw, alpha;
                int edge_id;
                bool clamped;
                if (!compute_hexel_sdf(
                        pixf,
                        collected_Tu[j],
                        collected_Tv[j],
                        collected_Tw[j],
                        collected_opacity[j],
                        collected_softness[j],
                        u,
                        v,
                        depth,
                        dist,
                        edge_id,
                        softness_eff,
                        boundary,
                        alpha_raw,
                        alpha,
                        clamped
                    )) {
                    continue;
                }
                if (alpha < 1.f / 255.f) continue;

                float sort_depth = collected_center_depth[j];
                if (
                    exact_mode == kExactSortSelectedAll || exact_mode == kExactSortCropAll ||
                    exact_mode == kExactSortPixelMask ||
                    (exact_mode == kExactSortHybrid && collected_selected[j]) ||
                    (exact_mode == kExactSortBoundaryTail && dist >= 0.f)
                ) {
                    sort_depth = depth;
                }
                uint64_t key = (uint64_t)pix_id << 32;
                key |= (uint64_t)__float_as_uint(sort_depth);
                keys_unsorted[write_index] = key;
                values_unsorted[write_index] = collected_id[j];
                ++write_index;
            }
        }
        __syncthreads();
    }
}

__global__ void composite_exact_sort_records(
    const int W,
    const int H,
    const int crop_x0,
    const int crop_y0,
    const int crop_x1,
    const int crop_y1,
    const float *colors,
    const float *color_grads,
    const float *color_quads,
    const float *view_grads,
    const float *specular_ks,
    const float *specular_log_s,
    const float3 *view_dirs,
    const float3 *normals,
    const float *transMats,
    const float *opacities,
    const float *softnesses,
    const int softness_stride,
    const float *background,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const bool *enabled_mask,
    const int64_t *record_counts,
    const int64_t *record_offsets,
    const uint32_t *values_unsorted,
    const uint32_t *values_sorted,
    float *out_color,
    bool *order_changed
) {
    const uint2 pix_min = {blockIdx.x * kBlockX, blockIdx.y * kBlockY};
    if (
        (int)pix_min.x >= crop_x1 || (int)pix_min.y >= crop_y1 ||
        (int)pix_min.x + kBlockX <= crop_x0 || (int)pix_min.y + kBlockY <= crop_y0
    ) {
        return;
    }

    const uint2 pix = {pix_min.x + (uint32_t)threadIdx.x, pix_min.y + (uint32_t)threadIdx.y};
    if (
        pix.x >= (uint32_t)W || pix.y >= (uint32_t)H ||
        (int)pix.x < crop_x0 || (int)pix.x >= crop_x1 ||
        (int)pix.y < crop_y0 || (int)pix.y >= crop_y1
    ) {
        return;
    }

    const uint32_t pix_id = (uint32_t)(W * (int)pix.y + (int)pix.x);
    if (!enabled_mask[pix_id]) return;

    const float2 pixf = {(float)pix.x, (float)pix.y};
    const int64_t count = record_counts[pix_id];
    const int64_t start = (pix_id == 0) ? 0 : record_offsets[pix_id - 1];
    bool changed = false;
    for (int64_t index = 0; index < count; ++index) {
        changed = changed || values_unsorted[start + index] != values_sorted[start + index];
    }
    order_changed[pix_id] = changed;

    float transmittance = 1.f;
    float3 accumulated = {0.f, 0.f, 0.f};
    for (int64_t index = 0; index < count; ++index) {
        const uint32_t g_id = values_sorted[start + index];
        const float *transform = transMats + (int)g_id * 9;
        const float3 Tu = {transform[0], transform[1], transform[2]};
        const float3 Tv = {transform[3], transform[4], transform[5]};
        const float3 Tw = {transform[6], transform[7], transform[8]};
        float3 softness3;
        if (softness_stride == 3) {
            const float *soft = softnesses + (int)g_id * 3;
            softness3 = {soft[0], soft[1], soft[2]};
        } else {
            const float soft = softnesses[g_id];
            softness3 = {soft, soft, soft};
        }

        float u, v, depth, dist, softness_eff, boundary, alpha_raw, alpha;
        int edge_id;
        bool clamped;
        if (!compute_hexel_sdf(
                pixf,
                Tu,
                Tv,
                Tw,
                opacities[g_id],
                softness3,
                u,
                v,
                depth,
                dist,
                edge_id,
                softness_eff,
                boundary,
                alpha_raw,
                alpha,
                clamped
            )) {
            continue;
        }
        if (alpha < 1.f / 255.f) continue;

        const float test_transmittance = transmittance * (1.f - alpha);
        if (test_transmittance < 0.0001f) break;
        const float weight = alpha * transmittance;
        const float3 color = exact_sort_appearance(
            g_id,
            u,
            v,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            view_dirs,
            normals,
            color_model,
            view_model,
            use_view_grads
        );
        accumulated.x += color.x * weight;
        accumulated.y += color.y * weight;
        accumulated.z += color.z * weight;
        transmittance = test_transmittance;
    }

    out_color[0 * H * W + pix_id] = accumulated.x + transmittance * background[0];
    out_color[1 * H * W + pix_id] = accumulated.y + transmittance * background[1];
    out_color[2 * H * W + pix_id] = accumulated.z + transmittance * background[2];
}

template <bool kHasSurfaceGrads, bool kPrefilter>
__global__ void render_hexels3d_backward(
    const uint2 *ranges,
    const uint32_t *point_list,
    const float2 *points_xy_center,
    float2 *grad_points_xy_center,
    const int W,
    const int H,
    const float *colors,
    const float *color_grads,
    const float *color_quads,
    const float *view_grads,
    const float *specular_ks,
    const float *specular_log_s,
    const int view_model,
    const float *view_dirs,
    const float *transMats,
    const float *opacities,
    const float *softnesses,
    const int softness_stride,
    const float *bg_color,
    const float *grad_out_color,
    const float *grad_out_overdraw,
    const float *grad_out_surf_depth,
    const float *grad_out_surf_alpha,
    const float *grad_out_surf_normals,
    const float3 *normals,
    float *grad_colors,
    float *grad_color_grads,
    float *grad_color_quads,
    float *grad_view_grads,
    float *grad_specular_ks,
    float *grad_specular_log_s,
    float *grad_opacities,
    float *grad_softnesses,
    float *grad_transMats,
    float *abs_means2d,
    float *grad_view_dirs,
    float *grad_normals,
    const int overdraw_base_count,
    const int color_model,
    const bool use_view_grads,
    const bool has_median_gradient = false,
    const float *final_transmittance = nullptr,
    const int *last_index = nullptr
) {
    const uint32_t horizontal_blocks = (W + kBlockX - 1) / kBlockX;

    const uint2 pix_min = {blockIdx.x * kBlockX, blockIdx.y * kBlockY};
    const uint2 pix = {pix_min.x + (uint32_t)threadIdx.x, pix_min.y + (uint32_t)threadIdx.y};
    const uint32_t pix_id = (uint32_t)(W * (int)pix.y + (int)pix.x);
    const float2 pixf = {(float)pix.x, (float)pix.y};

    const bool inside = pix.x < (uint32_t)W && pix.y < (uint32_t)H;
    if (!inside) return;

    const uint2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];
    const int range_len = (int)(range.y - range.x);
    if (range_len <= 0) return;

    const float v_out_r = inside ? grad_out_color[0 * H * W + pix_id] : 0.f;
    const float v_out_g = inside ? grad_out_color[1 * H * W + pix_id] : 0.f;
    const float v_out_b = inside ? grad_out_color[2 * H * W + pix_id] : 0.f;
    const float v_overdraw = (grad_out_overdraw != nullptr) ? grad_out_overdraw[pix_id] : 0.f;

    float v_surf_depth = 0.f;
    float v_surf_alpha = 0.f;
    float v_surf_distortion = 0.f;
    float v_surf_median = 0.f;
    float3 v_surf_normal = {0.f, 0.f, 0.f};
    if constexpr (kHasSurfaceGrads) {
        if (grad_out_surf_depth != nullptr) {
            v_surf_depth = grad_out_surf_depth[pix_id];
        }
        if (grad_out_surf_alpha != nullptr) {
            v_surf_alpha = grad_out_surf_alpha[pix_id];
        }
        if (grad_out_surf_normals != nullptr) {
            v_surf_normal = {
                grad_out_surf_normals[0 * H * W + pix_id],
                grad_out_surf_normals[1 * H * W + pix_id],
                grad_out_surf_normals[2 * H * W + pix_id],
            };
            v_surf_distortion =
                grad_out_surf_normals[3 * H * W + pix_id];
            if (has_median_gradient) {
                v_surf_median = grad_out_surf_normals[4 * H * W + pix_id];
            }
        }
    }

    float T_color = 1.0f;
    float T_surf = 1.0f;
    float M1_surf = 0.0f;
    float M2_surf = 0.0f;
    int n_visited_color = range_len;
    int n_visited_surf = range_len;
    int median_index = -1;
    bool done_color = false;
    bool done_surf = false;

    if (final_transmittance != nullptr) {
        T_color = inside ? final_transmittance[pix_id] : 1.f;
        n_visited_color = inside ? last_index[pix_id] : 0;
    } else {
        for (int idx = 0; idx < range_len; ++idx) {
            const uint32_t g_id = point_list[range.x + (uint32_t)idx];
            const bool is_base = (int)g_id < overdraw_base_count;

            if constexpr (kHasSurfaceGrads) {
                if (done_color && !is_base) {
                    continue;
                }
            }

            const float opacity = opacities[(int)g_id];
            float3 softness3;
            if (softness_stride == 3) {
                const float *s = softnesses + (int)g_id * 3;
                softness3 = {s[0], s[1], s[2]};
            } else {
                const float s = softnesses[(int)g_id];
                softness3 = {s, s, s};
            }
            const float *Tm = transMats + (int)g_id * 9;
            const float3 Tu = {Tm[0], Tm[1], Tm[2]};
            const float3 Tv = {Tm[3], Tm[4], Tm[5]};
            const float3 Tw = {Tm[6], Tm[7], Tm[8]};

            float u;
            float v;
            float depth_sdf;
            float dist;
            int edge_id;
            float softness_eff;
            float boundary;
            float alpha_raw_sdf;
            float alpha_sdf;
            bool clamped_sdf;
            if (!compute_hexel_sdf(
                    pixf,
                    Tu,
                    Tv,
                    Tw,
                    opacity,
                    softness3,
                    u,
                    v,
                    depth_sdf,
                    dist,
                    edge_id,
                    softness_eff,
                    boundary,
                    alpha_raw_sdf,
                    alpha_sdf,
                    clamped_sdf
                )) {
                continue;
            }
            float alpha = alpha_sdf;
            float depth = depth_sdf;
            if constexpr (kPrefilter) {
                const float2 center = points_xy_center[(int)g_id];
                const float dx = pixf.x - center.x;
                const float dy = pixf.y - center.y;
                const float screen_dist_sq = dx * dx + dy * dy;
                const float G = __expf(-0.5f * kFilterInvSquare * screen_dist_sq);
                const float alpha_raw_filter = opacity * G;
                const float alpha_filter = (alpha_raw_filter > 0.99f) ? 0.99f : alpha_raw_filter;
                if (alpha_filter > alpha_sdf) {
                    alpha = alpha_filter;
                    depth = Tw.z;
                }
            }
            if (alpha < 1.0f / 255.0f) {
                continue;
            }

            if (!done_color) {
                const float test_T = T_color * (1.f - alpha);
                if (test_T < 0.0001f) {
                    n_visited_color = idx;
                    if constexpr (!kHasSurfaceGrads) {
                        break;
                    }
                    done_color = true;
                } else {
                    T_color = test_T;
                }
            }

            if constexpr (kHasSurfaceGrads) {
                if (!done_surf && is_base) {
                    const float test_T_surf = T_surf * (1.f - alpha);
                    if (test_T_surf < 0.0001f) {
                        n_visited_surf = idx;
                        done_surf = true;
                    } else {
                        if (has_median_gradient && T_surf > 0.5f) {
                            median_index = idx;
                        }
                        const float m =
                            kFarPlane / (kFarPlane - kNearPlane) *
                            (1.0f - kNearPlane / depth);
                        const float w_surf = alpha * T_surf;
                        M1_surf += m * w_surf;
                        M2_surf += m * m * w_surf;
                        T_surf = test_T_surf;
                    }
                }
                if (done_color && done_surf) {
                    break;
                }
            }
        }
    }

    int n_visited_max = n_visited_color;
    if constexpr (kHasSurfaceGrads) {
        if (n_visited_surf > n_visited_max) {
            n_visited_max = n_visited_surf;
        }
    }

    float T_after = T_color;
    float3 behind = {bg_color[0], bg_color[1], bg_color[2]};

    float T_after_surf = T_surf;
    float behind_depth = 0.f;
    float behind_alpha = 0.f;
    float3 behind_normal = {0.f, 0.f, 0.f};
    float last_distortion_transmittance_vjp = 0.f;
    const float final_alpha_surf = 1.0f - T_surf;

    for (int idx = n_visited_max - 1; idx >= 0; --idx) {
        const uint32_t g_id = point_list[range.x + (uint32_t)idx];
        const bool is_base = (int)g_id < overdraw_base_count;
        const bool active_color = idx < n_visited_color;
        bool active_surf = false;
        if constexpr (kHasSurfaceGrads) {
            active_surf = is_base && (idx < n_visited_surf);
        }
        if (!active_color && !active_surf) {
            continue;
        }

        const float opacity = opacities[(int)g_id];
        float3 softness3;
        if (softness_stride == 3) {
            const float *s = softnesses + (int)g_id * 3;
            softness3 = {s[0], s[1], s[2]};
        } else {
            const float s = softnesses[(int)g_id];
            softness3 = {s, s, s};
        }
        const float *Tm = transMats + (int)g_id * 9;
        const float3 Tu = {Tm[0], Tm[1], Tm[2]};
        const float3 Tv = {Tm[3], Tm[4], Tm[5]};
        const float3 Tw = {Tm[6], Tm[7], Tm[8]};

        float u;
        float v;
        float depth_sdf;
        float dist;
        int edge_id;
        float softness_eff;
        float boundary;
        float alpha_raw_sdf;
        float alpha_sdf;
        bool clamped_sdf;
        if (!compute_hexel_sdf(
                pixf,
                Tu,
                Tv,
                Tw,
                opacity,
                softness3,
                u,
                v,
                depth_sdf,
                dist,
                edge_id,
                softness_eff,
                boundary,
                alpha_raw_sdf,
                alpha_sdf,
                clamped_sdf
            )) {
            continue;
        }

        float alpha = alpha_sdf;
        bool use_filter = false;
        float G_filter = 0.f;
        float dx_filter = 0.f;
        float dy_filter = 0.f;
        bool clamped_filter = false;
        if constexpr (kPrefilter) {
            const float2 center = points_xy_center[(int)g_id];
            dx_filter = pixf.x - center.x;
            dy_filter = pixf.y - center.y;
            const float screen_dist_sq = dx_filter * dx_filter + dy_filter * dy_filter;
            G_filter = __expf(-0.5f * kFilterInvSquare * screen_dist_sq);
            const float alpha_raw_filter = opacity * G_filter;
            clamped_filter = alpha_raw_filter > 0.99f;
            const float alpha_filter = clamped_filter ? 0.99f : alpha_raw_filter;
            use_filter = alpha_filter > alpha_sdf;
            alpha = use_filter ? alpha_filter : alpha_sdf;
        }
        if (alpha < 1.0f / 255.0f) {
            continue;
        }

        const float depth = use_filter ? Tw.z : depth_sdf;
        const float inv_one_minus_alpha = 1.f / (1.f - alpha);

        float3 c = {0.f, 0.f, 0.f};
        float3 grad_u = {0.f, 0.f, 0.f};
        float3 grad_v = {0.f, 0.f, 0.f};
        float3 quad_uu = {0.f, 0.f, 0.f};
        float3 quad_uv = {0.f, 0.f, 0.f};
        float3 quad_vv = {0.f, 0.f, 0.f};
        float3 view_dir = {0.f, 0.f, 0.f};
        float3 vgrad_r = {0.f, 0.f, 0.f};
        float3 vgrad_g = {0.f, 0.f, 0.f};
        float3 vgrad_b = {0.f, 0.f, 0.f};
        float3 ks = {0.f, 0.f, 0.f};
        float log_s = 0.f;
        float3 n = {0.f, 0.f, 0.f};
        float cos_nv = 0.f;
        float spec = 0.f;
        float s_exp = 0.f;
        bool log_s_is_clamped = false;

        float v_alpha_comp_total = 0.f;
        float v_u = 0.f;
        float v_v = 0.f;
        float v_depth_direct = 0.f;
        float T_before = 0.f;
        float w = 0.f;

        if (active_color) {
            T_before = T_after * inv_one_minus_alpha;
            w = alpha * T_before;

            const float *c_base_ptr = colors + (int)g_id * 3;
            const float3 c_base = {c_base_ptr[0], c_base_ptr[1], c_base_ptr[2]};

            if (color_model >= 1) {
                const float *cg = color_grads + (int)g_id * 6;
                grad_u = {cg[0], cg[1], cg[2]};
                grad_v = {cg[3], cg[4], cg[5]};
            }
            if (
                color_model == kColorModelQuadratic ||
                color_model == kColorModelQuadraticBounded
            ) {
                const float *cq = color_quads + (int)g_id * 9;
                quad_uu = {cq[0], cq[1], cq[2]};
                quad_uv = {cq[3], cq[4], cq[5]};
                quad_vv = {cq[6], cq[7], cq[8]};
            }
            if (use_view_grads || view_model == kViewModelLobe) {
                const float *vd = view_dirs + (int)g_id * 3;
                view_dir = {vd[0], vd[1], vd[2]};
                if (use_view_grads) {
                    const float *vg = view_grads + (int)g_id * 9;
                    vgrad_r = {vg[0], vg[1], vg[2]};
                    vgrad_g = {vg[3], vg[4], vg[5]};
                    vgrad_b = {vg[6], vg[7], vg[8]};
                }
            }
            if (view_model == kViewModelLobe) {
                const float *ks_ptr = specular_ks + (int)g_id * 3;
                ks = {ks_ptr[0], ks_ptr[1], ks_ptr[2]};
                log_s = specular_log_s[(int)g_id];
                if (normals != nullptr) {
                    n = normals[(int)g_id];
                }
            }

            const float u_app = use_filter ? 0.f : u;
            const float v_app = use_filter ? 0.f : v;

            c = c_base;
            if (color_model >= 1) {
                c.x += u_app * grad_u.x + v_app * grad_v.x;
                c.y += u_app * grad_u.y + v_app * grad_v.y;
                c.z += u_app * grad_u.z + v_app * grad_v.z;
            }
            if (color_model == kColorModelQuadratic) {
                const float u2 = u_app * u_app;
                const float uv = u_app * v_app;
                const float v2 = v_app * v_app;
                c.x += u2 * quad_uu.x + uv * quad_uv.x + v2 * quad_vv.x;
                c.y += u2 * quad_uu.y + uv * quad_uv.y + v2 * quad_vv.y;
                c.z += u2 * quad_uu.z + uv * quad_uv.z + v2 * quad_vv.z;
            } else if (color_model == kColorModelQuadraticBounded) {
                float u_bounded;
                float v_bounded;
                bounded_quadratic_coordinates(u_app, v_app, u_bounded, v_bounded);
                const float u2 = u_bounded * u_bounded;
                const float uv = u_bounded * v_bounded;
                const float v2 = v_bounded * v_bounded;
                c.x += u2 * quad_uu.x + uv * quad_uv.x + v2 * quad_vv.x;
                c.y += u2 * quad_uu.y + uv * quad_uv.y + v2 * quad_vv.y;
                c.z += u2 * quad_uu.z + uv * quad_uv.z + v2 * quad_vv.z;
            }
            if (use_view_grads) {
                c.x += (view_dir.x * vgrad_r.x + view_dir.y * vgrad_r.y + view_dir.z * vgrad_r.z);
                c.y += (view_dir.x * vgrad_g.x + view_dir.y * vgrad_g.y + view_dir.z * vgrad_g.z);
                c.z += (view_dir.x * vgrad_b.x + view_dir.y * vgrad_b.y + view_dir.z * vgrad_b.z);
            }
            if (view_model == kViewModelLobe) {
                cos_nv = fmaxf(-(n.x * view_dir.x + n.y * view_dir.y + n.z * view_dir.z), 0.f);
                const float log_s_clamped = fminf(fmaxf(log_s, 0.f), 5.0f);
                log_s_is_clamped = (log_s < 0.f) || (log_s > 5.0f);
                s_exp = __expf(log_s_clamped);
                spec = (cos_nv > 0.f) ? __powf(cos_nv, s_exp) : 0.f;
                c.x += ks.x * spec;
                c.y += ks.y * spec;
                c.z += ks.z * spec;
            }

            atomicAdd(grad_colors + (int)g_id * 3 + 0, w * v_out_r);
            atomicAdd(grad_colors + (int)g_id * 3 + 1, w * v_out_g);
            atomicAdd(grad_colors + (int)g_id * 3 + 2, w * v_out_b);

            if (color_model >= 1 && grad_color_grads != nullptr) {
                atomicAdd(grad_color_grads + (int)g_id * 6 + 0, w * v_out_r * u_app);
                atomicAdd(grad_color_grads + (int)g_id * 6 + 1, w * v_out_g * u_app);
                atomicAdd(grad_color_grads + (int)g_id * 6 + 2, w * v_out_b * u_app);
                atomicAdd(grad_color_grads + (int)g_id * 6 + 3, w * v_out_r * v_app);
                atomicAdd(grad_color_grads + (int)g_id * 6 + 4, w * v_out_g * v_app);
                atomicAdd(grad_color_grads + (int)g_id * 6 + 5, w * v_out_b * v_app);
            }
            if (color_model == kColorModelQuadratic && grad_color_quads != nullptr) {
                const float u2 = u_app * u_app;
                const float uv = u_app * v_app;
                const float v2 = v_app * v_app;
                atomicAdd(grad_color_quads + (int)g_id * 9 + 0, w * v_out_r * u2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 1, w * v_out_g * u2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 2, w * v_out_b * u2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 3, w * v_out_r * uv);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 4, w * v_out_g * uv);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 5, w * v_out_b * uv);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 6, w * v_out_r * v2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 7, w * v_out_g * v2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 8, w * v_out_b * v2);
            } else if (
                color_model == kColorModelQuadraticBounded && grad_color_quads != nullptr
            ) {
                float u_bounded;
                float v_bounded;
                bounded_quadratic_coordinates(u_app, v_app, u_bounded, v_bounded);
                const float u2 = u_bounded * u_bounded;
                const float uv = u_bounded * v_bounded;
                const float v2 = v_bounded * v_bounded;
                atomicAdd(grad_color_quads + (int)g_id * 9 + 0, w * v_out_r * u2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 1, w * v_out_g * u2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 2, w * v_out_b * u2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 3, w * v_out_r * uv);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 4, w * v_out_g * uv);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 5, w * v_out_b * uv);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 6, w * v_out_r * v2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 7, w * v_out_g * v2);
                atomicAdd(grad_color_quads + (int)g_id * 9 + 8, w * v_out_b * v2);
            }

            if (use_view_grads && grad_view_grads != nullptr) {
                atomicAdd(grad_view_grads + (int)g_id * 9 + 0, w * v_out_r * view_dir.x);
                atomicAdd(grad_view_grads + (int)g_id * 9 + 1, w * v_out_r * view_dir.y);
                atomicAdd(grad_view_grads + (int)g_id * 9 + 2, w * v_out_r * view_dir.z);
                atomicAdd(grad_view_grads + (int)g_id * 9 + 3, w * v_out_g * view_dir.x);
                atomicAdd(grad_view_grads + (int)g_id * 9 + 4, w * v_out_g * view_dir.y);
                atomicAdd(grad_view_grads + (int)g_id * 9 + 5, w * v_out_g * view_dir.z);
                atomicAdd(grad_view_grads + (int)g_id * 9 + 6, w * v_out_b * view_dir.x);
                atomicAdd(grad_view_grads + (int)g_id * 9 + 7, w * v_out_b * view_dir.y);
                atomicAdd(grad_view_grads + (int)g_id * 9 + 8, w * v_out_b * view_dir.z);

                if (!use_filter) {
                    const float3 v_view_dir = {
                        w * (v_out_r * vgrad_r.x + v_out_g * vgrad_g.x + v_out_b * vgrad_b.x),
                        w * (v_out_r * vgrad_r.y + v_out_g * vgrad_g.y + v_out_b * vgrad_b.y),
                        w * (v_out_r * vgrad_r.z + v_out_g * vgrad_g.z + v_out_b * vgrad_b.z),
                    };
                    atomicAdd(grad_view_dirs + (int)g_id * 3 + 0, v_view_dir.x);
                    atomicAdd(grad_view_dirs + (int)g_id * 3 + 1, v_view_dir.y);
                    atomicAdd(grad_view_dirs + (int)g_id * 3 + 2, v_view_dir.z);
                }
            }
            if (view_model == kViewModelLobe) {
                if (grad_specular_ks != nullptr) {
                    atomicAdd(grad_specular_ks + (int)g_id * 3 + 0, w * v_out_r * spec);
                    atomicAdd(grad_specular_ks + (int)g_id * 3 + 1, w * v_out_g * spec);
                    atomicAdd(grad_specular_ks + (int)g_id * 3 + 2, w * v_out_b * spec);
                }

                const float v_spec =
                    w * (v_out_r * ks.x + v_out_g * ks.y + v_out_b * ks.z);
                if (grad_specular_log_s != nullptr && !log_s_is_clamped && cos_nv > 0.f) {
                    atomicAdd(
                        grad_specular_log_s + (int)g_id,
                        v_spec * (spec * logf(cos_nv) * s_exp)
                    );
                }

                if (!use_filter && v_spec != 0.f && cos_nv > 0.f) {
                    const float v_cos = v_spec * (s_exp * spec / cos_nv);
                    atomicAdd(grad_view_dirs + (int)g_id * 3 + 0, -v_cos * n.x);
                    atomicAdd(grad_view_dirs + (int)g_id * 3 + 1, -v_cos * n.y);
                    atomicAdd(grad_view_dirs + (int)g_id * 3 + 2, -v_cos * n.z);
                    if (grad_normals != nullptr) {
                        atomicAdd(grad_normals + (int)g_id * 3 + 0, -v_cos * view_dir.x);
                        atomicAdd(grad_normals + (int)g_id * 3 + 1, -v_cos * view_dir.y);
                        atomicAdd(grad_normals + (int)g_id * 3 + 2, -v_cos * view_dir.z);
                    }
                }
            }

            v_alpha_comp_total += T_before *
                (v_out_r * (c.x - behind.x) + v_out_g * (c.y - behind.y) + v_out_b * (c.z - behind.z));

            if (!use_filter && color_model >= 1) {
                v_u += w * (v_out_r * grad_u.x + v_out_g * grad_u.y + v_out_b * grad_u.z);
                v_v += w * (v_out_r * grad_v.x + v_out_g * grad_v.y + v_out_b * grad_v.z);
            }
            if (!use_filter && color_model == kColorModelQuadratic) {
                const float du_q_r = 2.f * u * quad_uu.x + v * quad_uv.x;
                const float du_q_g = 2.f * u * quad_uu.y + v * quad_uv.y;
                const float du_q_b = 2.f * u * quad_uu.z + v * quad_uv.z;
                const float dv_q_r = u * quad_uv.x + 2.f * v * quad_vv.x;
                const float dv_q_g = u * quad_uv.y + 2.f * v * quad_vv.y;
                const float dv_q_b = u * quad_uv.z + 2.f * v * quad_vv.z;
                v_u += w * (v_out_r * du_q_r + v_out_g * du_q_g + v_out_b * du_q_b);
                v_v += w * (v_out_r * dv_q_r + v_out_g * dv_q_g + v_out_b * dv_q_b);
            } else if (!use_filter && color_model == kColorModelQuadraticBounded) {
                float u_bounded;
                float v_bounded;
                bounded_quadratic_coordinates(u, v, u_bounded, v_bounded);
                const float du_q_r = 2.f * u_bounded * quad_uu.x + v_bounded * quad_uv.x;
                const float du_q_g = 2.f * u_bounded * quad_uu.y + v_bounded * quad_uv.y;
                const float du_q_b = 2.f * u_bounded * quad_uu.z + v_bounded * quad_uv.z;
                const float dv_q_r = u_bounded * quad_uv.x + 2.f * v_bounded * quad_vv.x;
                const float dv_q_g = u_bounded * quad_uv.y + 2.f * v_bounded * quad_vv.y;
                const float dv_q_b = u_bounded * quad_uv.z + 2.f * v_bounded * quad_vv.z;
                const float v_u_bounded =
                    w * (v_out_r * du_q_r + v_out_g * du_q_g + v_out_b * du_q_b);
                const float v_v_bounded =
                    w * (v_out_r * dv_q_r + v_out_g * dv_q_g + v_out_b * dv_q_b);
                bounded_quadratic_coordinate_vjp(
                    u, v, v_u_bounded, v_v_bounded, v_u, v_v
                );
            }
        }

        float T_before_surf = 0.f;
        float w_surf = 0.f;
        float3 n_surf = {0.f, 0.f, 0.f};
        if constexpr (kHasSurfaceGrads) {
            if (active_surf) {
                T_before_surf = T_after_surf * inv_one_minus_alpha;
                w_surf = alpha * T_before_surf;
                if (v_surf_depth != 0.f) {
                    v_alpha_comp_total += T_before_surf * v_surf_depth * (depth - behind_depth);
                    if (!use_filter) {
                        v_depth_direct += w_surf * v_surf_depth;
                        v_u += v_depth_direct * Tw.x;
                        v_v += v_depth_direct * Tw.y;
                    }
                }
                if (v_surf_distortion != 0.f) {
                    const float m =
                        kFarPlane / (kFarPlane - kNearPlane) *
                        (1.0f - kNearPlane / depth);
                    const float weight_vjp =
                        (M2_surf + m * m * final_alpha_surf - 2.0f * m * M1_surf) *
                        v_surf_distortion;
                    v_alpha_comp_total += T_before_surf *
                        (weight_vjp - last_distortion_transmittance_vjp);
                    last_distortion_transmittance_vjp =
                        alpha * weight_vjp +
                        (1.0f - alpha) * last_distortion_transmittance_vjp;
                    if (!use_filter) {
                        const float m_vjp =
                            2.0f * w_surf * (m * final_alpha_surf - M1_surf) *
                            v_surf_distortion;
                        const float depth_vjp =
                            m_vjp * (kFarPlane * kNearPlane) /
                            ((kFarPlane - kNearPlane) * depth * depth);
                        v_depth_direct += depth_vjp;
                        v_u += depth_vjp * Tw.x;
                        v_v += depth_vjp * Tw.y;
                    }
                }
                if (idx == median_index && v_surf_median != 0.f) {
                    if (use_filter) {
                        atomicAdd(grad_transMats + (int)g_id * 9 + 8, v_surf_median);
                    } else {
                        v_depth_direct += v_surf_median;
                        v_u += v_surf_median * Tw.x;
                        v_v += v_surf_median * Tw.y;
                    }
                }
                if (v_surf_alpha != 0.f) {
                    v_alpha_comp_total += T_before_surf * v_surf_alpha * (1.f - behind_alpha);
                }
                if (grad_out_surf_normals != nullptr && normals != nullptr) {
                    n_surf = normals[(int)g_id];
                    const float normal_sign = surface_normal_sign(n_surf, view_dirs + (int)g_id * 3);
                    n_surf = {normal_sign * n_surf.x, normal_sign * n_surf.y, normal_sign * n_surf.z};
                    v_alpha_comp_total += T_before_surf * (
                        v_surf_normal.x * (n_surf.x - behind_normal.x) +
                        v_surf_normal.y * (n_surf.y - behind_normal.y) +
                        v_surf_normal.z * (n_surf.z - behind_normal.z)
                    );
                    if (!use_filter && grad_normals != nullptr) {
                        atomicAdd(grad_normals + (int)g_id * 3 + 0, normal_sign * w_surf * v_surf_normal.x);
                        atomicAdd(grad_normals + (int)g_id * 3 + 1, normal_sign * w_surf * v_surf_normal.y);
                        atomicAdd(grad_normals + (int)g_id * 3 + 2, normal_sign * w_surf * v_surf_normal.z);
                    }
                }
            }
        }

        if (use_filter) {
            if (!clamped_filter) {
                atomicAdd(grad_opacities + (int)g_id, v_alpha_comp_total * G_filter);
                if constexpr (kPrefilter) {
                    if (grad_points_xy_center != nullptr) {
                        const float factor = v_alpha_comp_total * opacity * G_filter * kFilterInvSquare;
                        atomicAdd(&grad_points_xy_center[(int)g_id].x, factor * dx_filter);
                        atomicAdd(&grad_points_xy_center[(int)g_id].y, factor * dy_filter);
                    }
                }
            }
        } else {
            if (!clamped_sdf) {
                atomicAdd(grad_opacities + (int)g_id, v_alpha_comp_total * boundary);
            }
        }
        const bool alpha_sdf_nontrivial = alpha_sdf >= (1.0f / 255.0f);
        const bool has_overdraw = (v_overdraw != 0.f) && active_color && is_base && alpha_sdf_nontrivial;
        if (has_overdraw) {
            atomicAdd(grad_opacities + (int)g_id, v_overdraw * boundary);
        }
        if (grad_softnesses != nullptr) {
            const float sig = boundary;
            float v_softness = 0.f;
            if (!use_filter && !clamped_sdf) {
                v_softness += v_alpha_comp_total * opacity * (-dist) * sig * (1.f - sig);
            }
            if (has_overdraw) {
                v_softness += v_overdraw * opacity * (-dist) * sig * (1.f - sig);
            }
            if (v_softness != 0.f) {
                if (softness_stride == 3) {
                    atomicAdd(grad_softnesses + (int)g_id * 3 + edge_id, v_softness);
                } else {
                    atomicAdd(grad_softnesses + (int)g_id, v_softness);
                }
            }
        }

        if ((!use_filter && !clamped_sdf) || has_overdraw) {
            float dist_g;
            float d_dist_du;
            float d_dist_dv;
            int edge_id_g;
            hex_sdf_grad(u, v, dist_g, d_dist_du, d_dist_dv, edge_id_g);
            const float softness_eff_g =
                (edge_id_g == 0) ? softness3.x : ((edge_id_g == 1) ? softness3.y : softness3.z);
            const float d_sig_ddist = -softness_eff_g * boundary * (1.f - boundary);
            float v_alpha_dist = has_overdraw ? (v_overdraw * opacity * d_sig_ddist) : 0.f;
            if (!use_filter && !clamped_sdf) {
                v_alpha_dist += v_alpha_comp_total * opacity * d_sig_ddist;
            }
            v_u += v_alpha_dist * d_dist_du;
            v_v += v_alpha_dist * d_dist_dv;
        }

        const float3 k = {pixf.x * Tw.x - Tu.x, pixf.x * Tw.y - Tu.y, pixf.x * Tw.z - Tu.z};
        const float3 l = {pixf.y * Tw.x - Tv.x, pixf.y * Tw.y - Tv.y, pixf.y * Tw.z - Tv.z};
        const float3 p = cross3(k, l);
        if (p.z != 0.f) {
            const float inv_pz = 1.f / p.z;
            const float dsx_pz = v_u * inv_pz;
            const float dsy_pz = v_v * inv_pz;
            const float3 v_p = {dsx_pz, dsy_pz, -(dsx_pz * u + dsy_pz * v)};
            const float3 v_k = cross3(l, v_p);
            const float3 v_l = cross3(v_p, k);

            const float3 v_Tu = {-v_k.x, -v_k.y, -v_k.z};
            const float3 v_Tv = {-v_l.x, -v_l.y, -v_l.z};
            float3 v_Tw = {
                pixf.x * v_k.x + pixf.y * v_l.x,
                pixf.x * v_k.y + pixf.y * v_l.y,
                pixf.x * v_k.z + pixf.y * v_l.z,
            };
            if constexpr (kHasSurfaceGrads) {
                if (active_surf && v_depth_direct != 0.f) {
                    v_Tw.x += u * v_depth_direct;
                    v_Tw.y += v * v_depth_direct;
                    v_Tw.z += v_depth_direct;
                }
            }

            if (abs_means2d != nullptr) {
                const float pixel_grad_x = v_Tu.z * Tw.z * 0.5f * (float)W;
                const float pixel_grad_y = v_Tv.z * Tw.z * 0.5f * (float)H;
                atomicAdd(abs_means2d + (int)g_id * 4 + 3,
                          sqrtf(pixel_grad_x * pixel_grad_x + pixel_grad_y * pixel_grad_y));
            }
            atomicAdd(grad_transMats + (int)g_id * 9 + 0, v_Tu.x);
            atomicAdd(grad_transMats + (int)g_id * 9 + 1, v_Tu.y);
            atomicAdd(grad_transMats + (int)g_id * 9 + 2, v_Tu.z);
            atomicAdd(grad_transMats + (int)g_id * 9 + 3, v_Tv.x);
            atomicAdd(grad_transMats + (int)g_id * 9 + 4, v_Tv.y);
            atomicAdd(grad_transMats + (int)g_id * 9 + 5, v_Tv.z);
            atomicAdd(grad_transMats + (int)g_id * 9 + 6, v_Tw.x);
            atomicAdd(grad_transMats + (int)g_id * 9 + 7, v_Tw.y);
            atomicAdd(grad_transMats + (int)g_id * 9 + 8, v_Tw.z);
        }

        if (active_color) {
            behind = {alpha * c.x + (1.f - alpha) * behind.x,
                      alpha * c.y + (1.f - alpha) * behind.y,
                      alpha * c.z + (1.f - alpha) * behind.z};
            T_after = T_before;
        }
        if constexpr (kHasSurfaceGrads) {
            if (active_surf) {
                behind_depth = alpha * depth + (1.f - alpha) * behind_depth;
                behind_alpha = alpha * 1.0f + (1.f - alpha) * behind_alpha;
                if (grad_out_surf_normals != nullptr && normals != nullptr) {
                    behind_normal = {alpha * n_surf.x + (1.f - alpha) * behind_normal.x,
                                     alpha * n_surf.y + (1.f - alpha) * behind_normal.y,
                                     alpha * n_surf.z + (1.f - alpha) * behind_normal.z};
                }
                T_after_surf = T_before_surf;
            }
        }
    }
}

__global__ void transmat_to_geometry_backward(
    const int P,
    const float *means3d,
    const float2 *scales,
    const float scale_modifier,
    const float4 *rotations,
    const float *projmatrix,
    const float *campos,
    const int W,
    const int H,
    const int *radii,
    const float *grad_transMats,
    const float *grad_view_dirs,
    const float *grad_normals,
    float *grad_means3d,
    float2 *grad_scales,
    float4 *grad_rotations
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= P) return;
    if (radii[idx] <= 0) return;

    const float3 p_orig = {means3d[3 * idx + 0], means3d[3 * idx + 1], means3d[3 * idx + 2]};
    const glm::vec4 rot = {rotations[idx].x, rotations[idx].y, rotations[idx].z, rotations[idx].w};
    const glm::vec2 scale = {scales[idx].x, scales[idx].y};

    const glm::mat3 R = quat_to_rotmat(rot);
    const glm::mat3 S = scale_to_mat(scale, scale_modifier);
    const glm::mat3 L = R * S;

    const glm::mat3x4 splat2world = glm::mat3x4(
        glm::vec4(L[0], 0.0),
        glm::vec4(L[1], 0.0),
        glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1.0)
    );

    const glm::mat4 world2ndc = glm::mat4(
        projmatrix[0],
        projmatrix[4],
        projmatrix[8],
        projmatrix[12],
        projmatrix[1],
        projmatrix[5],
        projmatrix[9],
        projmatrix[13],
        projmatrix[2],
        projmatrix[6],
        projmatrix[10],
        projmatrix[14],
        projmatrix[3],
        projmatrix[7],
        projmatrix[11],
        projmatrix[15]
    );

    const glm::mat3x4 ndc2pix = glm::mat3x4(
        glm::vec4((float)W / 2.0f, 0.0f, 0.0f, ((float)W - 1.0f) / 2.0f),
        glm::vec4(0.0f, (float)H / 2.0f, 0.0f, ((float)H - 1.0f) / 2.0f),
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );

    const glm::mat3x4 Pmat = world2ndc * ndc2pix;

    const glm::mat3 v_T = glm::mat3(
        grad_transMats[idx * 9 + 0],
        grad_transMats[idx * 9 + 1],
        grad_transMats[idx * 9 + 2],
        grad_transMats[idx * 9 + 3],
        grad_transMats[idx * 9 + 4],
        grad_transMats[idx * 9 + 5],
        grad_transMats[idx * 9 + 6],
        grad_transMats[idx * 9 + 7],
        grad_transMats[idx * 9 + 8]
    );
    const glm::mat3x4 v_M = Pmat * glm::transpose(v_T);
    glm::vec3 v_mean = glm::vec3(v_M[2]);

    const float3 c = {campos[0], campos[1], campos[2]};
    const float3 d = {p_orig.x - c.x, p_orig.y - c.y, p_orig.z - c.z};
    const float3 v_view_dir = {
        grad_view_dirs[idx * 3 + 0],
        grad_view_dirs[idx * 3 + 1],
        grad_view_dirs[idx * 3 + 2],
    };
    const float3 v_d = dnormvdv3(d, v_view_dir);
    v_mean += glm::vec3(v_d.x, v_d.y, v_d.z);

    grad_means3d[idx * 3 + 0] = v_mean.x;
    grad_means3d[idx * 3 + 1] = v_mean.y;
    grad_means3d[idx * 3 + 2] = v_mean.z;

    const glm::mat3 v_RS = glm::mat3(glm::vec3(v_M[0]), glm::vec3(v_M[1]), glm::vec3(0.f));
    glm::vec3 v_R2 = glm::vec3(v_RS[2]);
    if (grad_normals != nullptr) {
        v_R2 += glm::vec3(
            grad_normals[idx * 3 + 0],
            grad_normals[idx * 3 + 1],
            grad_normals[idx * 3 + 2]
        );
    }
    const glm::mat3 v_R = glm::mat3(
        v_RS[0] * glm::vec3(scale_modifier * scale.x),
        v_RS[1] * glm::vec3(scale_modifier * scale.y),
        v_R2
    );
    const glm::vec4 v_rot = quat_to_rotmat_vjp(rot, v_R);
    grad_rotations[idx] = {v_rot.x, v_rot.y, v_rot.z, v_rot.w};

    grad_scales[idx] = {
        (float)glm::dot(v_RS[0], R[0]) * scale_modifier,
        (float)glm::dot(v_RS[1], R[1]) * scale_modifier,
    };
}

__global__ void transmat_to_means2d_backward(
    const int P,
    const int W,
    const int H,
    const int stride,
    const int *radii,
    const float *transMats,
    const float *grad_transMats,
    float *grad_means2d
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= P) return;
    if (radii[idx] <= 0) return;

    const float depth = transMats[idx * 9 + 8];
    grad_means2d[idx * stride + 0] = grad_transMats[idx * 9 + 2] * depth * 0.5f * (float)W;
    grad_means2d[idx * stride + 1] = grad_transMats[idx * 9 + 5] * depth * 0.5f * (float)H;
    grad_means2d[idx * stride + 2] = 0.f;
}

__global__ void points_xy_center_to_means3d_backward(
    const int P,
    const float *means3d,
    const float *projmatrix,
    const int W,
    const int H,
    const float2 *grad_points_xy_center,
    float *grad_means3d
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= P) return;

    const float2 v_pix = grad_points_xy_center[idx];
    if (v_pix.x == 0.f && v_pix.y == 0.f) return;

    const float x = means3d[3 * idx + 0];
    const float y = means3d[3 * idx + 1];
    const float z = means3d[3 * idx + 2];

    const float clip_x = projmatrix[0] * x + projmatrix[4] * y + projmatrix[8] * z + projmatrix[12];
    const float clip_y = projmatrix[1] * x + projmatrix[5] * y + projmatrix[9] * z + projmatrix[13];
    const float clip_w = projmatrix[3] * x + projmatrix[7] * y + projmatrix[11] * z + projmatrix[15];

    const float inv_w = 1.0f / (clip_w + 1e-8f);
    const float inv_w2 = inv_w * inv_w;

    const float v_xndc = v_pix.x * ((float)W / 2.0f);
    const float v_yndc = v_pix.y * ((float)H / 2.0f);
    const float common = (v_xndc * clip_x + v_yndc * clip_y) * inv_w2;

    grad_means3d[3 * idx + 0] +=
        (v_xndc * inv_w * projmatrix[0]) + (v_yndc * inv_w * projmatrix[1]) - (common * projmatrix[3]);
    grad_means3d[3 * idx + 1] +=
        (v_xndc * inv_w * projmatrix[4]) + (v_yndc * inv_w * projmatrix[5]) - (common * projmatrix[7]);
    grad_means3d[3 * idx + 2] +=
        (v_xndc * inv_w * projmatrix[8]) + (v_yndc * inv_w * projmatrix[9]) - (common * projmatrix[11]);
}

}  // namespace

namespace {

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_hexels3d_forward_impl(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug,
    const bool return_contrib,
    const bool return_overdraw,
    const bool return_overdraw_hits,
    const bool return_surface,
    const int overdraw_base_count,
    int64_t *out_owner_id = nullptr,
    float *out_owner_weight = nullptr,
    float *out_owner_uv = nullptr,
    float *out_owner_sdf = nullptr,
    float *out_owner_depth = nullptr,
    float *out_total_alpha = nullptr,
    const bool return_median_depth = false,
    std::vector<torch::Tensor> *saved = nullptr
) {
    (void)tan_fovx;
    (void)tan_fovy;

    CHECK_INPUT(means3d);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(colors);
    CHECK_INPUT(color_grads);
    CHECK_INPUT(color_quads);
    CHECK_INPUT(view_grads);
    CHECK_INPUT(specular_ks);
    CHECK_INPUT(specular_log_s);
    CHECK_INPUT(opacities);
    CHECK_INPUT(softnesses);
    CHECK_INPUT(viewmatrix);
    CHECK_INPUT(projmatrix);
    CHECK_INPUT(campos);
    CHECK_INPUT(background);

    TORCH_CHECK(means3d.ndimension() == 2 && means3d.size(1) == 3, "means3d must have shape (P,3)");
    TORCH_CHECK(scales.ndimension() == 2 && scales.size(1) == 2, "scales must have shape (P,2)");
    TORCH_CHECK(rotations.ndimension() == 2 && rotations.size(1) == 4, "rotations must have shape (P,4)");
    TORCH_CHECK(colors.ndimension() == 2 && colors.size(1) == 3, "colors must have shape (P,3)");
    TORCH_CHECK(
        color_model >= kColorModelFlat && color_model <= kColorModelQuadraticBounded,
        "color_model must be one of: 0 (flat), 1 (linear), 2 (quadratic), 3 (quadratic_bounded)"
    );
    TORCH_CHECK(
        view_model >= 0 && view_model <= 2,
        "view_model must be one of: 0 (none), 1 (linear), 2 (lobe)"
    );
    const bool want_linear = color_model >= kColorModelLinear;
    const bool want_quadratic =
        color_model == kColorModelQuadratic || color_model == kColorModelQuadraticBounded;
    if (want_linear) {
        TORCH_CHECK(
            color_grads.ndimension() == 3 && color_grads.size(1) == 2 && color_grads.size(2) == 3,
            "color_grads must have shape (P,2,3)"
        );
    } else {
        TORCH_CHECK(
            color_grads.numel() == 0 ||
                (color_grads.ndimension() == 3 && color_grads.size(1) == 2 && color_grads.size(2) == 3),
            "color_grads must have shape (P,2,3) or be empty"
        );
    }
    if (want_quadratic) {
        TORCH_CHECK(
            color_quads.ndimension() == 3 && color_quads.size(1) == 3 && color_quads.size(2) == 3,
            "color_quads must have shape (P,3,3)"
        );
    } else {
        TORCH_CHECK(
            color_quads.numel() == 0 ||
                (color_quads.ndimension() == 3 && color_quads.size(1) == 3 && color_quads.size(2) == 3),
            "color_quads must have shape (P,3,3) or be empty"
        );
    }
    if (use_view_grads) {
        TORCH_CHECK(
            view_grads.ndimension() == 3 && view_grads.size(1) == 3 && view_grads.size(2) == 3,
            "view_grads must have shape (P,3,3)"
        );
    } else {
        TORCH_CHECK(
            view_grads.numel() == 0 || (view_grads.ndimension() == 3 && view_grads.size(1) == 3 && view_grads.size(2) == 3),
            "view_grads must have shape (P,3,3) or be empty"
        );
    }
    if (view_model == kViewModelLobe) {
        TORCH_CHECK(specular_ks.ndimension() == 2 && specular_ks.size(1) == 3, "specular_ks must have shape (P,3)");
        TORCH_CHECK(specular_log_s.ndimension() == 1, "specular_log_s must have shape (P,)");
    } else {
        TORCH_CHECK(
            specular_ks.numel() == 0 || (specular_ks.ndimension() == 2 && specular_ks.size(1) == 3),
            "specular_ks must have shape (P,3) or be empty"
        );
        TORCH_CHECK(
            specular_log_s.numel() == 0 || specular_log_s.ndimension() == 1,
            "specular_log_s must have shape (P,) or be empty"
        );
    }
    TORCH_CHECK(opacities.ndimension() == 1, "opacities must have shape (P,)");
    TORCH_CHECK(
        softnesses.ndimension() == 1 || softnesses.ndimension() == 2,
        "softnesses must have shape (P,), (P,3) or (0,)"
    );
    TORCH_CHECK(viewmatrix.numel() == 16, "viewmatrix must have 16 elements");
    TORCH_CHECK(projmatrix.numel() == 16, "projmatrix must have 16 elements");
    TORCH_CHECK(campos.numel() == 3, "campos must have 3 elements");
    TORCH_CHECK(background.numel() == 3, "background must have 3 elements");
    TORCH_CHECK(softness > 0.0f, "softness must be > 0");
    TORCH_CHECK(image_width > 0 && image_height > 0, "invalid image dimensions");
    TORCH_CHECK(means3d.scalar_type() == torch::kFloat32, "means3d must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat32, "rotations must be float32");
    TORCH_CHECK(colors.scalar_type() == torch::kFloat32, "colors must be float32");
    TORCH_CHECK(color_grads.scalar_type() == torch::kFloat32, "color_grads must be float32");
    TORCH_CHECK(color_quads.scalar_type() == torch::kFloat32, "color_quads must be float32");
    TORCH_CHECK(view_grads.scalar_type() == torch::kFloat32, "view_grads must be float32");
    TORCH_CHECK(specular_ks.scalar_type() == torch::kFloat32, "specular_ks must be float32");
    TORCH_CHECK(specular_log_s.scalar_type() == torch::kFloat32, "specular_log_s must be float32");
    TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(softnesses.scalar_type() == torch::kFloat32, "softnesses must be float32");
    TORCH_CHECK(viewmatrix.scalar_type() == torch::kFloat32, "viewmatrix must be float32");
    TORCH_CHECK(projmatrix.scalar_type() == torch::kFloat32, "projmatrix must be float32");
    TORCH_CHECK(campos.scalar_type() == torch::kFloat32, "campos must be float32");
    TORCH_CHECK(background.scalar_type() == torch::kFloat32, "background must be float32");

    const int P = (int)means3d.size(0);
    int overdraw_base = overdraw_base_count;
    if (overdraw_base <= 0 || overdraw_base > P) {
        overdraw_base = P;
    }
    if (want_linear) {
        TORCH_CHECK(color_grads.size(0) == P, "color_grads must have shape (P,2,3)");
    }
    if (want_quadratic) {
        TORCH_CHECK(color_quads.size(0) == P, "color_quads must have shape (P,3,3)");
    }
    if (use_view_grads) {
        TORCH_CHECK(view_grads.size(0) == P, "view_grads must have shape (P,3,3)");
    }
    if (view_model == kViewModelLobe) {
        TORCH_CHECK(specular_ks.size(0) == P, "specular_ks must have shape (P,3)");
        TORCH_CHECK(specular_log_s.numel() == P, "specular_log_s must have shape (P,)");
    }
    int softness_stride = 1;
    if (softnesses.numel() == 0) {
        softness_stride = 1;
    } else if (softnesses.ndimension() == 1) {
        TORCH_CHECK(softnesses.numel() == P, "softnesses must have shape (P,), (P,3) or (0,)");
        softness_stride = 1;
    } else {
        TORCH_CHECK(softnesses.size(0) == P && softnesses.size(1) == 3, "softnesses must have shape (P,3)");
        softness_stride = 3;
    }

    auto float_opts = means3d.options().dtype(torch::kFloat32);
    auto int_opts = means3d.options().dtype(torch::kInt32);

    torch::Tensor final_transmittance = saved ? torch::empty({image_height, image_width}, float_opts)
                                             : torch::empty({0}, float_opts);
    torch::Tensor last_index = saved ? torch::empty({image_height, image_width}, int_opts)
                                    : torch::empty({0}, int_opts);
    float *final_transmittance_ptr = saved ? final_transmittance.data_ptr<float>() : nullptr;
    int *last_index_ptr = saved ? last_index.data_ptr<int>() : nullptr;
    torch::Tensor values_sorted = torch::empty({0}, int_opts);
    torch::Tensor out_color = torch::zeros({kNumChannels, image_height, image_width}, float_opts);
    torch::Tensor radii = torch::zeros({P}, int_opts);
    torch::Tensor tile_radii = torch::empty({P, 2}, int_opts);
    torch::Tensor contrib = torch::empty({0}, float_opts);
    float *primitive_contrib = nullptr;
    if (return_contrib) {
        contrib = torch::zeros({P}, float_opts);
        primitive_contrib = contrib.contiguous().data_ptr<float>();
    }
    torch::Tensor overdraw = torch::empty({0}, float_opts);
    float *out_overdraw = nullptr;
    if (return_overdraw) {
        overdraw = torch::zeros({image_height, image_width}, float_opts);
        out_overdraw = overdraw.contiguous().data_ptr<float>();
    }
    torch::Tensor overdraw_hits = torch::empty({0}, float_opts);
    float *out_overdraw_hits = nullptr;
    if (return_overdraw_hits) {
        overdraw_hits = torch::zeros({image_height, image_width}, float_opts);
        out_overdraw_hits = overdraw_hits.contiguous().data_ptr<float>();
    }

    torch::Tensor surf_depth = torch::empty({0}, float_opts);
    torch::Tensor surf_alpha = torch::empty({0}, float_opts);
    torch::Tensor surf_normals = torch::empty({0}, float_opts);
    float *out_surf_depth = nullptr;
    float *out_surf_alpha = nullptr;
    float *out_surf_normals = nullptr;
    const bool need_normals = return_surface || (view_model == kViewModelLobe);
    torch::Tensor normals = torch::empty({0}, float_opts);
    float3 *normals_ptr = nullptr;
    if (return_surface) {
        surf_depth = torch::zeros({image_height, image_width}, float_opts);
        surf_alpha = torch::zeros({image_height, image_width}, float_opts);
        surf_normals = torch::zeros({kNumChannels + 1 + int(return_median_depth), image_height, image_width}, float_opts);
        out_surf_depth = surf_depth.contiguous().data_ptr<float>();
        out_surf_alpha = surf_alpha.contiguous().data_ptr<float>();
        out_surf_normals = surf_normals.contiguous().data_ptr<float>();
    }
    if (need_normals) {
        normals = torch::zeros({P, 3}, float_opts);
        normals_ptr = (float3 *)normals.contiguous().data_ptr<float>();
    }

    torch::Tensor depths = torch::zeros({P}, float_opts);
    torch::Tensor points_xy_image = torch::zeros({P, 2}, float_opts);
    torch::Tensor points_xy_center = torch::empty({0}, float_opts);
    float2 *points_xy_center_ptr = nullptr;
    if (prefilter_edge_on) {
        points_xy_center = torch::zeros({P, 2}, float_opts);
        points_xy_center_ptr = (float2 *)points_xy_center.contiguous().data_ptr<float>();
    }
    torch::Tensor transMats = torch::zeros({P, 9}, float_opts);
    torch::Tensor tiles_touched = torch::zeros({P}, int_opts);
    torch::Tensor opacities_out = torch::zeros({P}, float_opts);
    torch::Tensor softnesses_out = (softness_stride == 3) ? torch::zeros({P, 3}, float_opts)
                                                          : torch::zeros({P}, float_opts);
    torch::Tensor view_dirs = torch::zeros({P, 3}, float_opts);

    const dim3 tile_grid(
        (image_width + kBlockX - 1) / kBlockX,
        (image_height + kBlockY - 1) / kBlockY,
        1
    );

    const float *softnesses_in_ptr = nullptr;
    if (softnesses.numel() > 0) {
        softnesses_in_ptr = softnesses.contiguous().data_ptr<float>();
    }

    if (P > 0) {
        preprocess_hexels3d<<<(P + 255) / 256, 256>>>(
            P,
            means3d.contiguous().data_ptr<float>(),
            (const float2 *)scales.contiguous().data_ptr<float>(),
            scale_modifier,
            (const float4 *)rotations.contiguous().data_ptr<float>(),
            opacities.contiguous().data_ptr<float>(),
            softnesses_in_ptr,
            softness,
            softness_stride,
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            image_width,
            image_height,
            prefilter_edge_on,
            conservative_bounds,
            radii.contiguous().data_ptr<int>(),
            (int2 *)tile_radii.data_ptr<int>(),
            (float2 *)points_xy_image.contiguous().data_ptr<float>(),
            points_xy_center_ptr,
            depths.contiguous().data_ptr<float>(),
            transMats.contiguous().data_ptr<float>(),
            tiles_touched.contiguous().data_ptr<int>(),
            opacities_out.contiguous().data_ptr<float>(),
            softnesses_out.contiguous().data_ptr<float>(),
            (float3 *)view_dirs.contiguous().data_ptr<float>(),
            normals_ptr,
            tile_grid
        );
    }

    torch::Tensor offsets = torch::zeros({P}, int_opts);
    size_t scan_bytes = 0;
    if (P > 0) {
        cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, tiles_touched.data_ptr<int>(), offsets.data_ptr<int>(), P);
    }
    torch::Tensor scan_tmp = torch::empty({(long long)scan_bytes}, means3d.options().dtype(torch::kUInt8));
    if (P > 0) {
        cub::DeviceScan::InclusiveSum(
            scan_tmp.data_ptr<uint8_t>(), scan_bytes, tiles_touched.data_ptr<int>(), offsets.data_ptr<int>(), P
        );
    }

    int num_rendered = 0;
    if (P > 0) {
        CUDA_CALL(cudaMemcpy(&num_rendered, offsets.data_ptr<int>() + (P - 1), sizeof(int), cudaMemcpyDeviceToHost));
    }

    const int num_tiles = (int)(tile_grid.x * tile_grid.y);

    torch::Tensor ranges = torch::zeros({num_tiles, 2}, means3d.options().dtype(torch::kInt32));

    if (num_rendered > 0) {
        torch::Tensor keys_unsorted = torch::empty({num_rendered}, means3d.options().dtype(torch::kInt64));
        torch::Tensor values_unsorted = torch::empty({num_rendered}, means3d.options().dtype(torch::kInt32));
        torch::Tensor keys_sorted = torch::empty({num_rendered}, means3d.options().dtype(torch::kInt64));
        values_sorted = torch::empty({num_rendered}, means3d.options().dtype(torch::kInt32));

        duplicate_with_keys<<<(P + 255) / 256, 256>>>(
            P,
            (const float2 *)points_xy_image.contiguous().data_ptr<float>(),
            depths.contiguous().data_ptr<float>(),
            offsets.contiguous().data_ptr<int>(),
            (uint64_t *)keys_unsorted.contiguous().data_ptr<int64_t>(),
            (uint32_t *)values_unsorted.contiguous().data_ptr<int>(),
            radii.contiguous().data_ptr<int>(),
            (const int2 *)tile_radii.data_ptr<int>(),
            tile_grid
        );

        const int bit = (int)get_higher_msb((uint32_t)num_tiles);
        size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_bytes,
            (uint64_t *)keys_unsorted.data_ptr<int64_t>(),
            (uint64_t *)keys_sorted.data_ptr<int64_t>(),
            (uint32_t *)values_unsorted.data_ptr<int>(),
            (uint32_t *)values_sorted.data_ptr<int>(),
            num_rendered,
            0,
            32 + bit
        );
        torch::Tensor sort_tmp = torch::empty({(long long)sort_bytes}, means3d.options().dtype(torch::kUInt8));
        cub::DeviceRadixSort::SortPairs(
            sort_tmp.data_ptr<uint8_t>(),
            sort_bytes,
            (uint64_t *)keys_unsorted.data_ptr<int64_t>(),
            (uint64_t *)keys_sorted.data_ptr<int64_t>(),
            (uint32_t *)values_unsorted.data_ptr<int>(),
            (uint32_t *)values_sorted.data_ptr<int>(),
            num_rendered,
            0,
            32 + bit
        );

        CUDA_CALL(cudaMemset(ranges.data_ptr<int>(), 0, num_tiles * sizeof(uint2)));

        identify_tile_ranges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            (const uint64_t *)keys_sorted.contiguous().data_ptr<int64_t>(),
            (uint2 *)ranges.contiguous().data_ptr<int>()
        );

        if (return_surface) {
            if (prefilter_edge_on) {
                render_hexels3d<true, true><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                    (const float2 *)points_xy_center_ptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    out_color.contiguous().data_ptr<float>(),
                    primitive_contrib,
                    out_overdraw,
                    out_overdraw_hits,
                    overdraw_base,
                    color_model,
                    view_model,
                    use_view_grads,
                    normals_ptr,
                    out_surf_depth,
                    out_surf_alpha,
                    out_surf_normals,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    return_median_depth,
                    final_transmittance_ptr,
                    last_index_ptr
                );
            } else {
                render_hexels3d<true, false><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                    (const float2 *)points_xy_center_ptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    out_color.contiguous().data_ptr<float>(),
                    primitive_contrib,
                    out_overdraw,
                    out_overdraw_hits,
                    overdraw_base,
                    color_model,
                    view_model,
                    use_view_grads,
                    normals_ptr,
                    out_surf_depth,
                    out_surf_alpha,
                    out_surf_normals,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    return_median_depth,
                    final_transmittance_ptr,
                    last_index_ptr
                );
            }
        } else {
            if (prefilter_edge_on) {
                if (out_owner_id != nullptr) {
                    render_hexels3d<false, true, true><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                        (const uint2 *)ranges.contiguous().data_ptr<int>(),
                        (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                        (const float2 *)points_xy_center_ptr,
                        image_width,
                        image_height,
                        colors.contiguous().data_ptr<float>(),
                        want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                        want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                        use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                        specular_ks.contiguous().data_ptr<float>(),
                        specular_log_s.contiguous().data_ptr<float>(),
                        view_dirs.contiguous().data_ptr<float>(),
                        transMats.contiguous().data_ptr<float>(),
                        opacities_out.contiguous().data_ptr<float>(),
                        softnesses_out.contiguous().data_ptr<float>(),
                        softness_stride,
                        background.contiguous().data_ptr<float>(),
                        out_color.contiguous().data_ptr<float>(),
                        primitive_contrib,
                        out_overdraw,
                        out_overdraw_hits,
                        overdraw_base,
                        color_model,
                        view_model,
                        use_view_grads,
                        normals_ptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        out_owner_id,
                        out_owner_weight,
                        out_owner_uv,
                        out_owner_sdf,
                        out_owner_depth,
                        out_total_alpha,
                        false,
                        final_transmittance_ptr,
                        last_index_ptr
                    );
                } else {
                    render_hexels3d<false, true><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                        (const uint2 *)ranges.contiguous().data_ptr<int>(),
                        (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                        (const float2 *)points_xy_center_ptr,
                        image_width,
                        image_height,
                        colors.contiguous().data_ptr<float>(),
                        want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                        want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                        use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                        specular_ks.contiguous().data_ptr<float>(),
                        specular_log_s.contiguous().data_ptr<float>(),
                        view_dirs.contiguous().data_ptr<float>(),
                        transMats.contiguous().data_ptr<float>(),
                        opacities_out.contiguous().data_ptr<float>(),
                        softnesses_out.contiguous().data_ptr<float>(),
                        softness_stride,
                        background.contiguous().data_ptr<float>(),
                        out_color.contiguous().data_ptr<float>(),
                        primitive_contrib,
                        out_overdraw,
                        out_overdraw_hits,
                        overdraw_base,
                        color_model,
                        view_model,
                        use_view_grads,
                        normals_ptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        false,
                        final_transmittance_ptr,
                        last_index_ptr
                    );
                }
            } else {
                if (out_owner_id != nullptr) {
                    render_hexels3d<false, false, true><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                        (const uint2 *)ranges.contiguous().data_ptr<int>(),
                        (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                        (const float2 *)points_xy_center_ptr,
                        image_width,
                        image_height,
                        colors.contiguous().data_ptr<float>(),
                        want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                        want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                        use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                        specular_ks.contiguous().data_ptr<float>(),
                        specular_log_s.contiguous().data_ptr<float>(),
                        view_dirs.contiguous().data_ptr<float>(),
                        transMats.contiguous().data_ptr<float>(),
                        opacities_out.contiguous().data_ptr<float>(),
                        softnesses_out.contiguous().data_ptr<float>(),
                        softness_stride,
                        background.contiguous().data_ptr<float>(),
                        out_color.contiguous().data_ptr<float>(),
                        primitive_contrib,
                        out_overdraw,
                        out_overdraw_hits,
                        overdraw_base,
                        color_model,
                        view_model,
                        use_view_grads,
                        normals_ptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        out_owner_id,
                        out_owner_weight,
                        out_owner_uv,
                        out_owner_sdf,
                        out_owner_depth,
                        out_total_alpha,
                        false,
                        final_transmittance_ptr,
                        last_index_ptr
                    );
                } else {
                    render_hexels3d<false, false><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                        (const uint2 *)ranges.contiguous().data_ptr<int>(),
                        (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                        (const float2 *)points_xy_center_ptr,
                        image_width,
                        image_height,
                        colors.contiguous().data_ptr<float>(),
                        want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                        want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                        use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                        specular_ks.contiguous().data_ptr<float>(),
                        specular_log_s.contiguous().data_ptr<float>(),
                        view_dirs.contiguous().data_ptr<float>(),
                        transMats.contiguous().data_ptr<float>(),
                        opacities_out.contiguous().data_ptr<float>(),
                        softnesses_out.contiguous().data_ptr<float>(),
                        softness_stride,
                        background.contiguous().data_ptr<float>(),
                        out_color.contiguous().data_ptr<float>(),
                        primitive_contrib,
                        out_overdraw,
                        out_overdraw_hits,
                        overdraw_base,
                        color_model,
                        view_model,
                        use_view_grads,
                        normals_ptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        false,
                        final_transmittance_ptr,
                        last_index_ptr
                    );
                }
            }
        }
    } else {
        if (return_surface) {
            if (prefilter_edge_on) {
                render_hexels3d<true, true><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    nullptr,
                    (const float2 *)points_xy_center_ptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    out_color.contiguous().data_ptr<float>(),
                    primitive_contrib,
                    out_overdraw,
                    out_overdraw_hits,
                    overdraw_base,
                    color_model,
                    view_model,
                    use_view_grads,
                    normals_ptr,
                    out_surf_depth,
                    out_surf_alpha,
                    out_surf_normals,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    return_median_depth,
                    final_transmittance_ptr,
                    last_index_ptr
                );
            } else {
                render_hexels3d<true, false><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    nullptr,
                    (const float2 *)points_xy_center_ptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    out_color.contiguous().data_ptr<float>(),
                    primitive_contrib,
                    out_overdraw,
                    out_overdraw_hits,
                    overdraw_base,
                    color_model,
                    view_model,
                    use_view_grads,
                    normals_ptr,
                    out_surf_depth,
                    out_surf_alpha,
                    out_surf_normals,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    return_median_depth,
                    final_transmittance_ptr,
                    last_index_ptr
                );
            }
        } else {
            if (prefilter_edge_on) {
                render_hexels3d<false, true><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    nullptr,
                    (const float2 *)points_xy_center_ptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    out_color.contiguous().data_ptr<float>(),
                    primitive_contrib,
                    out_overdraw,
                    out_overdraw_hits,
                    overdraw_base,
                    color_model,
                    view_model,
                    use_view_grads,
                    normals_ptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    false,
                    final_transmittance_ptr,
                    last_index_ptr
                );
            } else {
                render_hexels3d<false, false><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    nullptr,
                    (const float2 *)points_xy_center_ptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    out_color.contiguous().data_ptr<float>(),
                    primitive_contrib,
                    out_overdraw,
                    out_overdraw_hits,
                    overdraw_base,
                    color_model,
                    view_model,
                    use_view_grads,
                    normals_ptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    false,
                    final_transmittance_ptr,
                    last_index_ptr
                );
            }
        }
    }

    if (debug) {
        CUDA_CALL(cudaDeviceSynchronize());
    }

    if (saved != nullptr) {
        *saved = {radii, transMats, opacities_out, softnesses_out, view_dirs, normals,
                  points_xy_center, ranges, values_sorted, final_transmittance, last_index};
    }
    return {out_color, radii, contrib, overdraw, overdraw_hits, surf_depth, surf_alpha, surf_normals};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> rasterize_hexels3d_forward(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    auto result = rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        false,
        false,
        false,
        false,
        -1
    );
    return {std::get<0>(result), std::get<1>(result)};
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> rasterize_hexels3d_forward_cached(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    std::vector<torch::Tensor> saved;
    auto result = rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        false,
        false,
        false,
        false,
        -1,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false, &saved
    );
    return {std::get<0>(result), std::get<1>(result), saved};
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_hexels3d_owner_forward(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    const auto float_opts = means3d.options().dtype(torch::kFloat32);
    torch::Tensor owner_id = torch::full(
        {image_height, image_width},
        -1,
        means3d.options().dtype(torch::kInt64)
    );
    torch::Tensor owner_weight = torch::zeros({image_height, image_width}, float_opts);
    torch::Tensor owner_uv = torch::zeros({2, image_height, image_width}, float_opts);
    torch::Tensor owner_sdf = torch::zeros({image_height, image_width}, float_opts);
    torch::Tensor owner_depth = torch::zeros({image_height, image_width}, float_opts);
    torch::Tensor total_alpha = torch::zeros({image_height, image_width}, float_opts);

    auto result = rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        false,
        false,
        false,
        false,
        -1,
        owner_id.contiguous().data_ptr<int64_t>(),
        owner_weight.contiguous().data_ptr<float>(),
        owner_uv.contiguous().data_ptr<float>(),
        owner_sdf.contiguous().data_ptr<float>(),
        owner_depth.contiguous().data_ptr<float>(),
        total_alpha.contiguous().data_ptr<float>()
    );
    return {
        std::get<0>(result),
        std::get<1>(result),
        owner_id,
        owner_weight,
        owner_uv,
        owner_sdf,
        owner_depth,
        total_alpha,
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_hexels3d_forward_with_contrib(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    auto result = rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        true,
        false,
        false,
        false,
        -1
    );
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_hexels3d_forward_with_overdraw(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int overdraw_base_count,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    auto result = rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        false,
        true,
        false,
        false,
        overdraw_base_count
    );
    return {std::get<0>(result), std::get<1>(result), std::get<3>(result)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_hexels3d_forward_with_contrib_overdraw(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int overdraw_base_count,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    auto result = rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        true,
        true,
        false,
        false,
        overdraw_base_count
    );
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result), std::get<3>(result)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_hexels3d_forward_with_overdraw_hits(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int overdraw_base_count,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    auto result = rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        false,
        true,
        true,
        false,
        overdraw_base_count
    );
    return {std::get<0>(result), std::get<1>(result), std::get<3>(result), std::get<4>(result)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_hexels3d_forward_with_contrib_overdraw_hits(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int overdraw_base_count,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    auto result = rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        true,
        true,
        true,
        false,
        overdraw_base_count
    );
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result), std::get<3>(result), std::get<4>(result)};
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_hexels3d_forward_all(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const bool return_contrib,
    const bool return_overdraw,
    const bool return_overdraw_hits,
    const bool return_surface,
    const int overdraw_base_count,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    return rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        return_contrib,
        return_overdraw,
        return_overdraw_hits,
        return_surface,
        overdraw_base_count
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_hexels3d_forward_median(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const bool return_contrib,
    const bool return_overdraw,
    const bool return_overdraw_hits,
    const bool return_surface,
    const int overdraw_base_count,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    TORCH_CHECK(return_surface, "Median depth requires surface outputs");
    return rasterize_hexels3d_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        prefilter_edge_on,
        conservative_bounds,
        debug,
        return_contrib,
        return_overdraw,
        return_overdraw_hits,
        return_surface,
        overdraw_base_count,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, true
    );
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_hexels3d_exact_sort_forward_impl(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const torch::Tensor &native_image,
    const torch::Tensor &selected_mask,
    const torch::Tensor &pixel_mask,
    const int image_width,
    const int image_height,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int exact_mode,
    const int crop_x0,
    const int crop_y0,
    const int crop_x1,
    const int crop_y1,
    const bool compute_risk,
    const bool debug
) {
    CHECK_INPUT(means3d);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(colors);
    CHECK_INPUT(color_grads);
    CHECK_INPUT(color_quads);
    CHECK_INPUT(view_grads);
    CHECK_INPUT(specular_ks);
    CHECK_INPUT(specular_log_s);
    CHECK_INPUT(opacities);
    CHECK_INPUT(softnesses);
    CHECK_INPUT(viewmatrix);
    CHECK_INPUT(projmatrix);
    CHECK_INPUT(campos);
    CHECK_INPUT(background);
    CHECK_INPUT(native_image);
    CHECK_INPUT(selected_mask);
    CHECK_INPUT(pixel_mask);

    const int P = (int)means3d.size(0);
    const torch::Device device = means3d.device();
    TORCH_CHECK(
        scales.device() == device && rotations.device() == device && colors.device() == device &&
            color_grads.device() == device && color_quads.device() == device && view_grads.device() == device &&
            specular_ks.device() == device && specular_log_s.device() == device && opacities.device() == device &&
            softnesses.device() == device && viewmatrix.device() == device && projmatrix.device() == device &&
            campos.device() == device && background.device() == device && native_image.device() == device &&
            selected_mask.device() == device && pixel_mask.device() == device,
        "all exact-sort tensors must be on the same CUDA device"
    );
    TORCH_CHECK(means3d.ndimension() == 2 && means3d.size(1) == 3, "means3d must have shape (P,3)");
    TORCH_CHECK(scales.sizes() == torch::IntArrayRef({P, 2}), "scales must have shape (P,2)");
    TORCH_CHECK(rotations.sizes() == torch::IntArrayRef({P, 4}), "rotations must have shape (P,4)");
    TORCH_CHECK(colors.sizes() == torch::IntArrayRef({P, 3}), "colors must have shape (P,3)");
    TORCH_CHECK(opacities.ndimension() == 1 && opacities.numel() == P, "opacities must have shape (P,)");
    TORCH_CHECK(selected_mask.scalar_type() == torch::kBool, "selected_mask must be bool");
    TORCH_CHECK(pixel_mask.scalar_type() == torch::kBool, "pixel_mask must be bool");
    if (exact_mode == kExactSortPixelMask) {
        TORCH_CHECK(selected_mask.numel() == 0, "selected_mask must be empty in pixel-mask mode");
        TORCH_CHECK(
            pixel_mask.ndimension() == 2 && pixel_mask.size(0) == image_height && pixel_mask.size(1) == image_width,
            "pixel_mask must have shape (H,W)"
        );
    } else {
        TORCH_CHECK(
            selected_mask.ndimension() == 1 && selected_mask.numel() == P,
            "selected_mask must have shape (P,)"
        );
        TORCH_CHECK(pixel_mask.numel() == 0, "pixel_mask must be empty outside pixel-mask mode");
    }
    TORCH_CHECK(
        native_image.ndimension() == 3 && native_image.size(0) == 3 && native_image.size(1) == image_height &&
            native_image.size(2) == image_width,
        "native_image must have shape (3,H,W)"
    );
    TORCH_CHECK(viewmatrix.numel() == 16, "viewmatrix must have 16 elements");
    TORCH_CHECK(projmatrix.numel() == 16, "projmatrix must have 16 elements");
    TORCH_CHECK(campos.numel() == 3, "campos must have 3 elements");
    TORCH_CHECK(background.numel() == 3, "background must have 3 elements");
    TORCH_CHECK(image_width > 0 && image_height > 0, "invalid image dimensions");
    TORCH_CHECK(
        (uint64_t)image_width * (uint64_t)image_height <= UINT32_MAX,
        "image has too many pixels for exact-sort keys"
    );
    TORCH_CHECK(softness > 0.f, "softness must be > 0");
    TORCH_CHECK(
        color_model >= kColorModelFlat && color_model <= kColorModelQuadraticBounded,
        "invalid color_model"
    );
    TORCH_CHECK(view_model >= 0 && view_model <= 2, "invalid view_model");
    TORCH_CHECK(
        exact_mode >= kExactSortCenterReplay && exact_mode <= kExactSortPixelMask,
        "invalid exact sort mode"
    );
    TORCH_CHECK(
        crop_x0 >= 0 && crop_y0 >= 0 && crop_x0 < crop_x1 && crop_y0 < crop_y1 &&
            crop_x1 <= image_width && crop_y1 <= image_height,
        "crop must satisfy 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H"
    );

    const torch::ScalarType float_type = torch::kFloat32;
    TORCH_CHECK(means3d.scalar_type() == float_type, "means3d must be float32");
    TORCH_CHECK(scales.scalar_type() == float_type, "scales must be float32");
    TORCH_CHECK(rotations.scalar_type() == float_type, "rotations must be float32");
    TORCH_CHECK(colors.scalar_type() == float_type, "colors must be float32");
    TORCH_CHECK(color_grads.scalar_type() == float_type, "color_grads must be float32");
    TORCH_CHECK(color_quads.scalar_type() == float_type, "color_quads must be float32");
    TORCH_CHECK(view_grads.scalar_type() == float_type, "view_grads must be float32");
    TORCH_CHECK(specular_ks.scalar_type() == float_type, "specular_ks must be float32");
    TORCH_CHECK(specular_log_s.scalar_type() == float_type, "specular_log_s must be float32");
    TORCH_CHECK(opacities.scalar_type() == float_type, "opacities must be float32");
    TORCH_CHECK(softnesses.scalar_type() == float_type, "softnesses must be float32");
    TORCH_CHECK(viewmatrix.scalar_type() == float_type, "viewmatrix must be float32");
    TORCH_CHECK(projmatrix.scalar_type() == float_type, "projmatrix must be float32");
    TORCH_CHECK(campos.scalar_type() == float_type, "campos must be float32");
    TORCH_CHECK(background.scalar_type() == float_type, "background must be float32");
    TORCH_CHECK(native_image.scalar_type() == float_type, "native_image must be float32");

    const bool want_linear = color_model >= kColorModelLinear;
    const bool want_quadratic =
        color_model == kColorModelQuadratic || color_model == kColorModelQuadraticBounded;
    if (want_linear) {
        TORCH_CHECK(
            color_grads.ndimension() == 3 && color_grads.size(0) == P && color_grads.size(1) == 2 &&
                color_grads.size(2) == 3,
            "color_grads must have shape (P,2,3)"
        );
    }
    if (want_quadratic) {
        TORCH_CHECK(
            color_quads.ndimension() == 3 && color_quads.size(0) == P && color_quads.size(1) == 3 &&
                color_quads.size(2) == 3,
            "color_quads must have shape (P,3,3)"
        );
    }
    if (use_view_grads) {
        TORCH_CHECK(
            view_model == kViewModelLinear && view_grads.ndimension() == 3 && view_grads.size(0) == P &&
                view_grads.size(1) == 3 && view_grads.size(2) == 3,
            "linear view_grads must have shape (P,3,3)"
        );
    }
    if (view_model == kViewModelLobe) {
        TORCH_CHECK(specular_ks.sizes() == torch::IntArrayRef({P, 3}), "specular_ks must have shape (P,3)");
        TORCH_CHECK(
            specular_log_s.ndimension() == 1 && specular_log_s.numel() == P,
            "specular_log_s must have shape (P,)"
        );
    }

    int softness_stride = 1;
    if (softnesses.numel() == 0) {
        softness_stride = 1;
    } else if (softnesses.ndimension() == 1) {
        TORCH_CHECK(softnesses.numel() == P, "softnesses must have shape (P,), (P,3) or (0,)");
    } else {
        TORCH_CHECK(
            softnesses.ndimension() == 2 && softnesses.size(0) == P && softnesses.size(1) == 3,
            "softnesses must have shape (P,3)"
        );
        softness_stride = 3;
    }

    auto float_opts = means3d.options().dtype(torch::kFloat32);
    auto int_opts = means3d.options().dtype(torch::kInt32);
    auto long_opts = means3d.options().dtype(torch::kInt64);
    auto bool_opts = means3d.options().dtype(torch::kBool);
    torch::Tensor out_color = native_image.clone();
    torch::Tensor order_changed = torch::zeros({image_height, image_width}, bool_opts);
    torch::Tensor hit_counts = torch::zeros({image_height, image_width}, int_opts);
    torch::Tensor enabled_mask = torch::zeros({image_height, image_width}, bool_opts);
    torch::Tensor risk_map = torch::zeros({image_height, image_width}, float_opts);

    torch::Tensor radii = torch::zeros({P}, int_opts);
    torch::Tensor tile_radii = torch::empty({P, 2}, int_opts);
    torch::Tensor points_xy_image = torch::zeros({P, 2}, float_opts);
    torch::Tensor center_depths = torch::zeros({P}, float_opts);
    torch::Tensor transMats = torch::zeros({P, 9}, float_opts);
    torch::Tensor tiles_touched = torch::zeros({P}, int_opts);
    torch::Tensor opacities_out = torch::zeros({P}, float_opts);
    torch::Tensor softnesses_out = (softness_stride == 3) ? torch::zeros({P, 3}, float_opts)
                                                          : torch::zeros({P}, float_opts);
    torch::Tensor view_dirs = torch::zeros({P, 3}, float_opts);
    torch::Tensor normals = torch::empty({0}, float_opts);
    float3 *normals_ptr = nullptr;
    if (view_model == kViewModelLobe) {
        normals = torch::zeros({P, 3}, float_opts);
        normals_ptr = (float3 *)normals.data_ptr<float>();
    }

    const dim3 tile_grid(
        (image_width + kBlockX - 1) / kBlockX,
        (image_height + kBlockY - 1) / kBlockY,
        1
    );
    const float *softnesses_in = softnesses.numel() > 0 ? softnesses.data_ptr<float>() : nullptr;
    if (P > 0) {
        preprocess_hexels3d<<<(P + 255) / 256, 256>>>(
            P,
            means3d.data_ptr<float>(),
            (const float2 *)scales.data_ptr<float>(),
            scale_modifier,
            (const float4 *)rotations.data_ptr<float>(),
            opacities.data_ptr<float>(),
            softnesses_in,
            softness,
            softness_stride,
            viewmatrix.data_ptr<float>(),
            projmatrix.data_ptr<float>(),
            campos.data_ptr<float>(),
            image_width,
            image_height,
            false,
            false,
            radii.data_ptr<int>(),
            (int2 *)tile_radii.data_ptr<int>(),
            (float2 *)points_xy_image.data_ptr<float>(),
            nullptr,
            center_depths.data_ptr<float>(),
            transMats.data_ptr<float>(),
            tiles_touched.data_ptr<int>(),
            opacities_out.data_ptr<float>(),
            softnesses_out.data_ptr<float>(),
            (float3 *)view_dirs.data_ptr<float>(),
            normals_ptr,
            tile_grid
        );
    }

    torch::Tensor primitive_offsets = torch::zeros({P}, int_opts);
    size_t scan_bytes = 0;
    if (P > 0) {
        cub::DeviceScan::InclusiveSum(
            nullptr,
            scan_bytes,
            tiles_touched.data_ptr<int>(),
            primitive_offsets.data_ptr<int>(),
            P
        );
    }
    torch::Tensor scan_tmp = torch::empty({(long long)scan_bytes}, means3d.options().dtype(torch::kUInt8));
    if (P > 0) {
        cub::DeviceScan::InclusiveSum(
            scan_tmp.data_ptr<uint8_t>(),
            scan_bytes,
            tiles_touched.data_ptr<int>(),
            primitive_offsets.data_ptr<int>(),
            P
        );
    }

    int num_rendered = 0;
    if (P > 0) {
        CUDA_CALL(cudaMemcpy(
            &num_rendered,
            primitive_offsets.data_ptr<int>() + (P - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));
    }

    const int num_tiles = (int)(tile_grid.x * tile_grid.y);
    torch::Tensor ranges = torch::zeros({num_tiles, 2}, int_opts);
    torch::Tensor tile_values_sorted = torch::empty({0}, int_opts);
    if (num_rendered > 0) {
        torch::Tensor keys_unsorted = torch::empty({num_rendered}, long_opts);
        torch::Tensor values_unsorted = torch::empty({num_rendered}, int_opts);
        torch::Tensor keys_sorted = torch::empty({num_rendered}, long_opts);
        tile_values_sorted = torch::empty({num_rendered}, int_opts);

        duplicate_with_keys<<<(P + 255) / 256, 256>>>(
            P,
            (const float2 *)points_xy_image.data_ptr<float>(),
            center_depths.data_ptr<float>(),
            primitive_offsets.data_ptr<int>(),
            (uint64_t *)keys_unsorted.data_ptr<int64_t>(),
            (uint32_t *)values_unsorted.data_ptr<int>(),
            radii.data_ptr<int>(),
            (const int2 *)tile_radii.data_ptr<int>(),
            tile_grid
        );

        const int bit = (int)get_higher_msb((uint32_t)num_tiles);
        size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_bytes,
            (uint64_t *)keys_unsorted.data_ptr<int64_t>(),
            (uint64_t *)keys_sorted.data_ptr<int64_t>(),
            (uint32_t *)values_unsorted.data_ptr<int>(),
            (uint32_t *)tile_values_sorted.data_ptr<int>(),
            num_rendered,
            0,
            32 + bit
        );
        torch::Tensor sort_tmp = torch::empty({(long long)sort_bytes}, means3d.options().dtype(torch::kUInt8));
        cub::DeviceRadixSort::SortPairs(
            sort_tmp.data_ptr<uint8_t>(),
            sort_bytes,
            (uint64_t *)keys_unsorted.data_ptr<int64_t>(),
            (uint64_t *)keys_sorted.data_ptr<int64_t>(),
            (uint32_t *)values_unsorted.data_ptr<int>(),
            (uint32_t *)tile_values_sorted.data_ptr<int>(),
            num_rendered,
            0,
            32 + bit
        );
        identify_tile_ranges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            (const uint64_t *)keys_sorted.data_ptr<int64_t>(),
            (uint2 *)ranges.data_ptr<int>()
        );
    }

    if (compute_risk) {
        compute_pixel_inversion_risk<<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
            (const uint2 *)ranges.data_ptr<int>(),
            num_rendered > 0 ? (const uint32_t *)tile_values_sorted.data_ptr<int>() : nullptr,
            image_width,
            image_height,
            crop_x0,
            crop_y0,
            crop_x1,
            crop_y1,
            colors.data_ptr<float>(),
            want_linear ? color_grads.data_ptr<float>() : nullptr,
            want_quadratic ? color_quads.data_ptr<float>() : nullptr,
            use_view_grads ? view_grads.data_ptr<float>() : nullptr,
            view_model == kViewModelLobe ? specular_ks.data_ptr<float>() : nullptr,
            view_model == kViewModelLobe ? specular_log_s.data_ptr<float>() : nullptr,
            (const float3 *)view_dirs.data_ptr<float>(),
            normals_ptr,
            transMats.data_ptr<float>(),
            opacities_out.data_ptr<float>(),
            softnesses_out.data_ptr<float>(),
            softness_stride,
            color_model,
            view_model,
            use_view_grads,
            risk_map.data_ptr<float>()
        );
    }

    const int64_t num_pixels = (int64_t)image_width * image_height;
    torch::Tensor record_counts = torch::zeros({image_height, image_width}, long_opts);
    count_exact_sort_hits<<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
        (const uint2 *)ranges.data_ptr<int>(),
        num_rendered > 0 ? (const uint32_t *)tile_values_sorted.data_ptr<int>() : nullptr,
        image_width,
        image_height,
        crop_x0,
        crop_y0,
        crop_x1,
        crop_y1,
        transMats.data_ptr<float>(),
        opacities_out.data_ptr<float>(),
        softnesses_out.data_ptr<float>(),
        softness_stride,
        selected_mask.numel() > 0 ? selected_mask.data_ptr<bool>() : nullptr,
        pixel_mask.numel() > 0 ? pixel_mask.data_ptr<bool>() : nullptr,
        exact_mode,
        hit_counts.data_ptr<int>(),
        record_counts.data_ptr<int64_t>(),
        enabled_mask.data_ptr<bool>()
    );

    torch::Tensor record_offsets = torch::zeros({image_height, image_width}, long_opts);
    size_t record_scan_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr,
        record_scan_bytes,
        record_counts.data_ptr<int64_t>(),
        record_offsets.data_ptr<int64_t>(),
        num_pixels
    );
    torch::Tensor record_scan_tmp = torch::empty(
        {(long long)record_scan_bytes},
        means3d.options().dtype(torch::kUInt8)
    );
    cub::DeviceScan::InclusiveSum(
        record_scan_tmp.data_ptr<uint8_t>(),
        record_scan_bytes,
        record_counts.data_ptr<int64_t>(),
        record_offsets.data_ptr<int64_t>(),
        num_pixels
    );

    int64_t num_records = 0;
    CUDA_CALL(cudaMemcpy(
        &num_records,
        record_offsets.data_ptr<int64_t>() + (num_pixels - 1),
        sizeof(int64_t),
        cudaMemcpyDeviceToHost
    ));
    TORCH_CHECK(num_records <= INT_MAX, "exact-sort crop has too many hit records; use smaller crop strips");

    torch::Tensor record_keys_unsorted = torch::empty({num_records}, long_opts);
    torch::Tensor record_values_unsorted = torch::empty({num_records}, int_opts);
    torch::Tensor record_keys_sorted = torch::empty({num_records}, long_opts);
    torch::Tensor record_values_sorted = torch::empty({num_records}, int_opts);
    if (num_records > 0) {
        fill_exact_sort_records<<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
            (const uint2 *)ranges.data_ptr<int>(),
            (const uint32_t *)tile_values_sorted.data_ptr<int>(),
            image_width,
            image_height,
            crop_x0,
            crop_y0,
            crop_x1,
            crop_y1,
            center_depths.data_ptr<float>(),
            transMats.data_ptr<float>(),
            opacities_out.data_ptr<float>(),
            softnesses_out.data_ptr<float>(),
            softness_stride,
            selected_mask.numel() > 0 ? selected_mask.data_ptr<bool>() : nullptr,
            enabled_mask.data_ptr<bool>(),
            record_offsets.data_ptr<int64_t>(),
            exact_mode,
            (uint64_t *)record_keys_unsorted.data_ptr<int64_t>(),
            (uint32_t *)record_values_unsorted.data_ptr<int>()
        );

        // CUB radix sort is stable, so equal hit-depth keys retain native center-order ties.
        size_t record_sort_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            record_sort_bytes,
            (uint64_t *)record_keys_unsorted.data_ptr<int64_t>(),
            (uint64_t *)record_keys_sorted.data_ptr<int64_t>(),
            (uint32_t *)record_values_unsorted.data_ptr<int>(),
            (uint32_t *)record_values_sorted.data_ptr<int>(),
            (int)num_records,
            0,
            64
        );
        torch::Tensor record_sort_tmp = torch::empty(
            {(long long)record_sort_bytes},
            means3d.options().dtype(torch::kUInt8)
        );
        cub::DeviceRadixSort::SortPairs(
            record_sort_tmp.data_ptr<uint8_t>(),
            record_sort_bytes,
            (uint64_t *)record_keys_unsorted.data_ptr<int64_t>(),
            (uint64_t *)record_keys_sorted.data_ptr<int64_t>(),
            (uint32_t *)record_values_unsorted.data_ptr<int>(),
            (uint32_t *)record_values_sorted.data_ptr<int>(),
            (int)num_records,
            0,
            64
        );
    }

    composite_exact_sort_records<<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
        image_width,
        image_height,
        crop_x0,
        crop_y0,
        crop_x1,
        crop_y1,
        colors.data_ptr<float>(),
        want_linear ? color_grads.data_ptr<float>() : nullptr,
        want_quadratic ? color_quads.data_ptr<float>() : nullptr,
        use_view_grads ? view_grads.data_ptr<float>() : nullptr,
        view_model == kViewModelLobe ? specular_ks.data_ptr<float>() : nullptr,
        view_model == kViewModelLobe ? specular_log_s.data_ptr<float>() : nullptr,
        (const float3 *)view_dirs.data_ptr<float>(),
        normals_ptr,
        transMats.data_ptr<float>(),
        opacities_out.data_ptr<float>(),
        softnesses_out.data_ptr<float>(),
        softness_stride,
        background.data_ptr<float>(),
        color_model,
        view_model,
        use_view_grads,
        enabled_mask.data_ptr<bool>(),
        record_counts.data_ptr<int64_t>(),
        record_offsets.data_ptr<int64_t>(),
        num_records > 0 ? (const uint32_t *)record_values_unsorted.data_ptr<int>() : nullptr,
        num_records > 0 ? (const uint32_t *)record_values_sorted.data_ptr<int>() : nullptr,
        out_color.data_ptr<float>(),
        order_changed.data_ptr<bool>()
    );

    if (debug) {
        CUDA_CALL(cudaDeviceSynchronize());
    }
    return {out_color, order_changed, hit_counts, enabled_mask, risk_map};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_hexels3d_exact_sort_forward(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const torch::Tensor &native_image,
    const torch::Tensor &selected_mask,
    const int image_width,
    const int image_height,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int exact_mode,
    const int crop_x0,
    const int crop_y0,
    const int crop_x1,
    const int crop_y1,
    const bool debug
) {
    const torch::Tensor pixel_mask = torch::empty({0}, selected_mask.options());
    const auto result = rasterize_hexels3d_exact_sort_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        native_image,
        selected_mask,
        pixel_mask,
        image_width,
        image_height,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        exact_mode,
        crop_x0,
        crop_y0,
        crop_x1,
        crop_y1,
        false,
        debug
    );
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result), std::get<3>(result)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_hexels3d_pixel_risk_forward(
    const torch::Tensor &means3d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const torch::Tensor &native_image,
    const torch::Tensor &pixel_mask,
    const int image_width,
    const int image_height,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int crop_x0,
    const int crop_y0,
    const int crop_x1,
    const int crop_y1,
    const bool debug
) {
    const torch::Tensor selected_mask = torch::empty({0}, pixel_mask.options());
    const auto result = rasterize_hexels3d_exact_sort_forward_impl(
        means3d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        native_image,
        selected_mask,
        pixel_mask,
        image_width,
        image_height,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        kExactSortPixelMask,
        crop_x0,
        crop_y0,
        crop_x1,
        crop_y1,
        true,
        debug
    );
    return {
        std::get<0>(result),
        std::get<4>(result),
        std::get<1>(result),
        std::get<2>(result),
        std::get<3>(result),
    };
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_hexels3d_backward_impl(
    const torch::Tensor &means3d,
    const torch::Tensor &means2d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const torch::Tensor &grad_out_color,
    const torch::Tensor &grad_out_overdraw,
    const torch::Tensor &grad_out_surf_depth,
    const torch::Tensor &grad_out_surf_alpha,
    const torch::Tensor &grad_out_surf_normals,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int overdraw_base_count,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug,
    const std::vector<torch::Tensor> &saved
) {
    (void)tan_fovx;
    (void)tan_fovy;

    CHECK_INPUT(means3d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(colors);
    CHECK_INPUT(color_grads);
    CHECK_INPUT(color_quads);
    CHECK_INPUT(view_grads);
    CHECK_INPUT(specular_ks);
    CHECK_INPUT(specular_log_s);
    CHECK_INPUT(opacities);
    CHECK_INPUT(softnesses);
    CHECK_INPUT(viewmatrix);
    CHECK_INPUT(projmatrix);
    CHECK_INPUT(campos);
    CHECK_INPUT(background);
    CHECK_INPUT(grad_out_color);
    CHECK_INPUT(grad_out_overdraw);
    CHECK_INPUT(grad_out_surf_depth);
    CHECK_INPUT(grad_out_surf_alpha);
    CHECK_INPUT(grad_out_surf_normals);

    TORCH_CHECK(means3d.ndimension() == 2 && means3d.size(1) == 3, "means3d must have shape (P,3)");
    TORCH_CHECK(
        means2d.numel() == 0 || (means2d.ndimension() == 2 && means2d.size(0) == means3d.size(0)
                               && (means2d.size(1) == 3 || means2d.size(1) == 4)),
        "means2d must have shape (P,3), (P,4) or be empty"
    );
    TORCH_CHECK(scales.ndimension() == 2 && scales.size(1) == 2, "scales must have shape (P,2)");
    TORCH_CHECK(rotations.ndimension() == 2 && rotations.size(1) == 4, "rotations must have shape (P,4)");
    TORCH_CHECK(colors.ndimension() == 2 && colors.size(1) == 3, "colors must have shape (P,3)");
    TORCH_CHECK(
        color_model >= kColorModelFlat && color_model <= kColorModelQuadraticBounded,
        "color_model must be one of: 0 (flat), 1 (linear), 2 (quadratic), 3 (quadratic_bounded)"
    );
    TORCH_CHECK(
        view_model >= 0 && view_model <= 2,
        "view_model must be one of: 0 (none), 1 (linear), 2 (lobe)"
    );
    const bool want_linear = color_model >= kColorModelLinear;
    const bool want_quadratic =
        color_model == kColorModelQuadratic || color_model == kColorModelQuadraticBounded;
    if (want_linear) {
        TORCH_CHECK(
            color_grads.ndimension() == 3 && color_grads.size(1) == 2 && color_grads.size(2) == 3,
            "color_grads must have shape (P,2,3)"
        );
    } else {
        TORCH_CHECK(
            color_grads.numel() == 0 ||
                (color_grads.ndimension() == 3 && color_grads.size(1) == 2 && color_grads.size(2) == 3),
            "color_grads must have shape (P,2,3) or be empty"
        );
    }
    if (want_quadratic) {
        TORCH_CHECK(
            color_quads.ndimension() == 3 && color_quads.size(1) == 3 && color_quads.size(2) == 3,
            "color_quads must have shape (P,3,3)"
        );
    } else {
        TORCH_CHECK(
            color_quads.numel() == 0 ||
                (color_quads.ndimension() == 3 && color_quads.size(1) == 3 && color_quads.size(2) == 3),
            "color_quads must have shape (P,3,3) or be empty"
        );
    }
    if (use_view_grads) {
        TORCH_CHECK(
            view_grads.ndimension() == 3 && view_grads.size(1) == 3 && view_grads.size(2) == 3,
            "view_grads must have shape (P,3,3)"
        );
    } else {
        TORCH_CHECK(
            view_grads.numel() == 0 || (view_grads.ndimension() == 3 && view_grads.size(1) == 3 && view_grads.size(2) == 3),
            "view_grads must have shape (P,3,3) or be empty"
        );
    }
    if (view_model == kViewModelLobe) {
        TORCH_CHECK(specular_ks.ndimension() == 2 && specular_ks.size(1) == 3, "specular_ks must have shape (P,3)");
        TORCH_CHECK(specular_log_s.ndimension() == 1, "specular_log_s must have shape (P,)");
    } else {
        TORCH_CHECK(
            specular_ks.numel() == 0 || (specular_ks.ndimension() == 2 && specular_ks.size(1) == 3),
            "specular_ks must have shape (P,3) or be empty"
        );
        TORCH_CHECK(
            specular_log_s.numel() == 0 || specular_log_s.ndimension() == 1,
            "specular_log_s must have shape (P,) or be empty"
        );
    }
    TORCH_CHECK(opacities.ndimension() == 1, "opacities must have shape (P,)");
    TORCH_CHECK(
        softnesses.ndimension() == 1 || softnesses.ndimension() == 2,
        "softnesses must have shape (P,), (P,3) or (0,)"
    );
    TORCH_CHECK(viewmatrix.numel() == 16, "viewmatrix must have 16 elements");
    TORCH_CHECK(projmatrix.numel() == 16, "projmatrix must have 16 elements");
    TORCH_CHECK(campos.numel() == 3, "campos must have 3 elements");
    TORCH_CHECK(background.numel() == 3, "background must have 3 elements");
    TORCH_CHECK(
        grad_out_color.ndimension() == 3 && grad_out_color.size(0) == 3 &&
            grad_out_color.size(1) == image_height && grad_out_color.size(2) == image_width,
        "grad_out_color must have shape (3,H,W)"
    );
    TORCH_CHECK(
        grad_out_overdraw.numel() == 0 ||
            (grad_out_overdraw.ndimension() == 2 && grad_out_overdraw.size(0) == image_height &&
                grad_out_overdraw.size(1) == image_width),
        "grad_out_overdraw must have shape (H,W) or be empty"
    );
    TORCH_CHECK(
        grad_out_surf_depth.numel() == 0 ||
            (grad_out_surf_depth.ndimension() == 2 && grad_out_surf_depth.size(0) == image_height &&
                grad_out_surf_depth.size(1) == image_width),
        "grad_out_surf_depth must have shape (H,W) or be empty"
    );
    TORCH_CHECK(
        grad_out_surf_alpha.numel() == 0 ||
            (grad_out_surf_alpha.ndimension() == 2 && grad_out_surf_alpha.size(0) == image_height &&
                grad_out_surf_alpha.size(1) == image_width),
        "grad_out_surf_alpha must have shape (H,W) or be empty"
    );
    TORCH_CHECK(
        grad_out_surf_normals.numel() == 0 ||
            (grad_out_surf_normals.ndimension() == 3 &&
                (grad_out_surf_normals.size(0) == 4 || grad_out_surf_normals.size(0) == 5) &&
                grad_out_surf_normals.size(1) == image_height && grad_out_surf_normals.size(2) == image_width),
        "grad_out_surf_normals must have shape (4,H,W), (5,H,W) or be empty"
    );
    TORCH_CHECK(softness > 0.0f, "softness must be > 0");
    TORCH_CHECK(image_width > 0 && image_height > 0, "invalid image dimensions");

    TORCH_CHECK(means3d.scalar_type() == torch::kFloat32, "means3d must be float32");
    TORCH_CHECK(means2d.scalar_type() == torch::kFloat32, "means2d must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat32, "rotations must be float32");
    TORCH_CHECK(colors.scalar_type() == torch::kFloat32, "colors must be float32");
    TORCH_CHECK(color_grads.scalar_type() == torch::kFloat32, "color_grads must be float32");
    TORCH_CHECK(color_quads.scalar_type() == torch::kFloat32, "color_quads must be float32");
    TORCH_CHECK(view_grads.scalar_type() == torch::kFloat32, "view_grads must be float32");
    TORCH_CHECK(specular_ks.scalar_type() == torch::kFloat32, "specular_ks must be float32");
    TORCH_CHECK(specular_log_s.scalar_type() == torch::kFloat32, "specular_log_s must be float32");
    TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(softnesses.scalar_type() == torch::kFloat32, "softnesses must be float32");
    TORCH_CHECK(viewmatrix.scalar_type() == torch::kFloat32, "viewmatrix must be float32");
    TORCH_CHECK(projmatrix.scalar_type() == torch::kFloat32, "projmatrix must be float32");
    TORCH_CHECK(campos.scalar_type() == torch::kFloat32, "campos must be float32");
    TORCH_CHECK(background.scalar_type() == torch::kFloat32, "background must be float32");
    TORCH_CHECK(grad_out_color.scalar_type() == torch::kFloat32, "grad_out_color must be float32");
    TORCH_CHECK(grad_out_overdraw.scalar_type() == torch::kFloat32, "grad_out_overdraw must be float32");
    TORCH_CHECK(grad_out_surf_depth.scalar_type() == torch::kFloat32, "grad_out_surf_depth must be float32");
    TORCH_CHECK(grad_out_surf_alpha.scalar_type() == torch::kFloat32, "grad_out_surf_alpha must be float32");
    TORCH_CHECK(grad_out_surf_normals.scalar_type() == torch::kFloat32, "grad_out_surf_normals must be float32");

    const int P = (int)means3d.size(0);
    int overdraw_base = overdraw_base_count;
    if (overdraw_base <= 0 || overdraw_base > P) {
        overdraw_base = P;
    }
    if (want_linear) {
        TORCH_CHECK(color_grads.size(0) == P, "color_grads must have shape (P,2,3)");
    }
    if (want_quadratic) {
        TORCH_CHECK(color_quads.size(0) == P, "color_quads must have shape (P,3,3)");
    }
    if (use_view_grads) {
        TORCH_CHECK(view_grads.size(0) == P, "view_grads must have shape (P,3,3)");
    }
    if (view_model == kViewModelLobe) {
        TORCH_CHECK(specular_ks.size(0) == P, "specular_ks must have shape (P,3)");
        TORCH_CHECK(specular_log_s.numel() == P, "specular_log_s must have shape (P,)");
    }
    int softness_stride = 1;
    if (softnesses.numel() == 0) {
        softness_stride = 1;
    } else if (softnesses.ndimension() == 1) {
        TORCH_CHECK(softnesses.numel() == P, "softnesses must have shape (P,), (P,3) or (0,)");
        softness_stride = 1;
    } else {
        TORCH_CHECK(softnesses.size(0) == P && softnesses.size(1) == 3, "softnesses must have shape (P,3)");
        softness_stride = 3;
    }

    const bool cached = !saved.empty();
    if (cached) {
        TORCH_CHECK(saved.size() == 11, "Expected eleven saved rasterizer buffers");
        auto check = [&](int i, at::IntArrayRef shape, at::ScalarType dtype) {
            CHECK_INPUT(saved[i]);
            TORCH_CHECK(saved[i].device() == means3d.device(), "Saved buffer device mismatch");
            TORCH_CHECK(saved[i].scalar_type() == dtype && saved[i].sizes() == shape,
                        "Invalid saved rasterizer buffer ", i);
        };
        check(0, {P}, torch::kInt32);
        check(1, {P, 9}, torch::kFloat32);
        check(2, {P}, torch::kFloat32);
        if (softness_stride == 3) check(3, {P, 3}, torch::kFloat32);
        else check(3, {P}, torch::kFloat32);
        check(4, {P, 3}, torch::kFloat32);
        if (view_model == kViewModelLobe) check(5, {P, 3}, torch::kFloat32);
        else check(5, {0}, torch::kFloat32);
        if (prefilter_edge_on) check(6, {P, 2}, torch::kFloat32);
        else check(6, {0}, torch::kFloat32);
        check(7, {((image_width + 15) / 16) * ((image_height + 15) / 16), 2}, torch::kInt32);
        check(8, {saved[8].numel()}, torch::kInt32);
        check(9, {image_height, image_width}, torch::kFloat32);
        check(10, {image_height, image_width}, torch::kInt32);
        TORCH_CHECK(grad_out_overdraw.numel() == 0 && grad_out_surf_depth.numel() == 0 &&
                    grad_out_surf_alpha.numel() == 0 && grad_out_surf_normals.numel() == 0,
                    "Cached RGB backward does not accept auxiliary gradients");
    }

    auto float_opts = means3d.options().dtype(torch::kFloat32);
    auto int_opts = means3d.options().dtype(torch::kInt32);

    torch::Tensor grad_means3d = torch::zeros({P, 3}, float_opts);
    torch::Tensor grad_means2d = torch::zeros_like(means2d);
    float *grad_means2d_ptr = means2d.numel() > 0 ? grad_means2d.contiguous().data_ptr<float>() : nullptr;
    const int means2d_stride = means2d.numel() > 0 ? (int)means2d.size(1) : 3;
    float *abs_means2d_ptr = means2d_stride == 4 ? grad_means2d_ptr : nullptr;
    torch::Tensor grad_scales = torch::zeros({P, 2}, float_opts);
    torch::Tensor grad_rotations = torch::zeros({P, 4}, float_opts);
    torch::Tensor grad_colors = torch::zeros({P, 3}, float_opts);
    torch::Tensor grad_color_grads = torch::zeros_like(color_grads);
    torch::Tensor grad_color_quads = torch::zeros_like(color_quads);
    torch::Tensor grad_view_grads = torch::zeros_like(view_grads);
    torch::Tensor grad_specular_ks = torch::zeros_like(specular_ks);
    torch::Tensor grad_specular_log_s = torch::zeros_like(specular_log_s);
    float *grad_specular_ks_ptr =
        view_model == kViewModelLobe ? grad_specular_ks.contiguous().data_ptr<float>() : nullptr;
    float *grad_specular_log_s_ptr =
        view_model == kViewModelLobe ? grad_specular_log_s.contiguous().data_ptr<float>() : nullptr;
    torch::Tensor grad_opacities = torch::zeros({P}, float_opts);
    torch::Tensor grad_softnesses = torch::zeros_like(softnesses, float_opts);
    float *grad_softnesses_ptr = softnesses.numel() > 0 ? grad_softnesses.contiguous().data_ptr<float>() : nullptr;

    torch::Tensor radii = cached ? saved[0] : torch::zeros({P}, int_opts);
    torch::Tensor tile_radii = cached ? torch::empty({0}, int_opts) : torch::empty({P, 2}, int_opts);
    torch::Tensor depths = cached ? torch::empty({0}, float_opts) : torch::zeros({P}, float_opts);
    torch::Tensor points_xy_image = cached ? torch::empty({0}, float_opts) : torch::zeros({P, 2}, float_opts);
    torch::Tensor points_xy_center = torch::empty({0}, float_opts);
    torch::Tensor grad_points_xy_center = torch::empty({0}, float_opts);
    float2 *points_xy_center_ptr = nullptr;
    float2 *grad_points_xy_center_ptr = nullptr;
    if (prefilter_edge_on) {
        points_xy_center = cached ? saved[6] : torch::zeros({P, 2}, float_opts);
        grad_points_xy_center = torch::zeros({P, 2}, float_opts);
        points_xy_center_ptr = (float2 *)points_xy_center.contiguous().data_ptr<float>();
        grad_points_xy_center_ptr = (float2 *)grad_points_xy_center.contiguous().data_ptr<float>();
    }
    torch::Tensor transMats = cached ? saved[1] : torch::zeros({P, 9}, float_opts);
    torch::Tensor tiles_touched = cached ? torch::empty({0}, int_opts) : torch::zeros({P}, int_opts);
    torch::Tensor opacities_out = cached ? saved[2] : torch::zeros({P}, float_opts);
    torch::Tensor softnesses_out = cached ? saved[3] : (softness_stride == 3) ? torch::zeros({P, 3}, float_opts)
                                                          : torch::zeros({P}, float_opts);
    torch::Tensor view_dirs = cached ? saved[4] : torch::zeros({P, 3}, float_opts);
    torch::Tensor grad_transMats = torch::zeros({P, 9}, float_opts);
    torch::Tensor grad_view_dirs = torch::zeros({P, 3}, float_opts);

    const bool has_surface_grads = grad_out_surf_depth.numel() > 0 || grad_out_surf_alpha.numel() > 0 ||
        grad_out_surf_normals.numel() > 0;
    torch::Tensor normals = torch::zeros({0}, float_opts);
    torch::Tensor grad_normals = torch::zeros({0}, float_opts);
    float3 *normals_ptr = nullptr;
    float *grad_normals_ptr = nullptr;
    const bool need_normals = view_model == kViewModelLobe || grad_out_surf_normals.numel() > 0;
    if (need_normals) {
        normals = cached ? saved[5] : torch::zeros({P, 3}, float_opts);
        grad_normals = torch::zeros({P, 3}, float_opts);
        normals_ptr = (float3 *)normals.contiguous().data_ptr<float>();
        grad_normals_ptr = grad_normals.contiguous().data_ptr<float>();
    }

    const dim3 tile_grid(
        (image_width + kBlockX - 1) / kBlockX,
        (image_height + kBlockY - 1) / kBlockY,
        1
    );

    const float *softnesses_in_ptr = nullptr;
    if (softnesses.numel() > 0) {
        softnesses_in_ptr = softnesses.contiguous().data_ptr<float>();
    }

    const int num_tiles = (int)(tile_grid.x * tile_grid.y);
    torch::Tensor ranges = cached ? saved[7] : torch::zeros({num_tiles, 2}, int_opts);
    torch::Tensor values_sorted = cached ? saved[8] : torch::empty({0}, int_opts);
    int num_rendered = (int)values_sorted.numel();
    if (!cached) {
        if (P > 0) {
            preprocess_hexels3d<<<(P + 255) / 256, 256>>>(
                P,
                means3d.contiguous().data_ptr<float>(),
                (const float2 *)scales.contiguous().data_ptr<float>(),
                scale_modifier,
                (const float4 *)rotations.contiguous().data_ptr<float>(),
                opacities.contiguous().data_ptr<float>(),
                softnesses_in_ptr,
                softness,
                softness_stride,
                viewmatrix.contiguous().data_ptr<float>(),
                projmatrix.contiguous().data_ptr<float>(),
                campos.contiguous().data_ptr<float>(),
                image_width,
                image_height,
                prefilter_edge_on,
                conservative_bounds,
                radii.contiguous().data_ptr<int>(),
                (int2 *)tile_radii.data_ptr<int>(),
                (float2 *)points_xy_image.contiguous().data_ptr<float>(),
                points_xy_center_ptr,
                depths.contiguous().data_ptr<float>(),
                transMats.contiguous().data_ptr<float>(),
                tiles_touched.contiguous().data_ptr<int>(),
                opacities_out.contiguous().data_ptr<float>(),
                softnesses_out.contiguous().data_ptr<float>(),
                (float3 *)view_dirs.contiguous().data_ptr<float>(),
                normals_ptr,
                tile_grid
            );
        }

        torch::Tensor offsets = torch::zeros({P}, int_opts);
        size_t scan_bytes = 0;
        if (P > 0) {
            cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, tiles_touched.data_ptr<int>(), offsets.data_ptr<int>(), P);
        }
        torch::Tensor scan_tmp = torch::empty({(long long)scan_bytes}, means3d.options().dtype(torch::kUInt8));
        if (P > 0) {
            cub::DeviceScan::InclusiveSum(
                scan_tmp.data_ptr<uint8_t>(), scan_bytes, tiles_touched.data_ptr<int>(), offsets.data_ptr<int>(), P
            );
        }

        if (P > 0) {
            CUDA_CALL(cudaMemcpy(&num_rendered, offsets.data_ptr<int>() + (P - 1), sizeof(int), cudaMemcpyDeviceToHost));
        }

        if (num_rendered > 0) {
            torch::Tensor keys_unsorted = torch::empty({num_rendered}, means3d.options().dtype(torch::kInt64));
            torch::Tensor values_unsorted = torch::empty({num_rendered}, means3d.options().dtype(torch::kInt32));
            torch::Tensor keys_sorted = torch::empty({num_rendered}, means3d.options().dtype(torch::kInt64));
            values_sorted = torch::empty({num_rendered}, means3d.options().dtype(torch::kInt32));

            duplicate_with_keys<<<(P + 255) / 256, 256>>>(
                P,
                (const float2 *)points_xy_image.contiguous().data_ptr<float>(),
                depths.contiguous().data_ptr<float>(),
                offsets.contiguous().data_ptr<int>(),
                (uint64_t *)keys_unsorted.contiguous().data_ptr<int64_t>(),
                (uint32_t *)values_unsorted.contiguous().data_ptr<int>(),
                radii.contiguous().data_ptr<int>(),
                (const int2 *)tile_radii.data_ptr<int>(),
                tile_grid
            );

            const int bit = (int)get_higher_msb((uint32_t)num_tiles);
            size_t sort_bytes = 0;
            cub::DeviceRadixSort::SortPairs(
                nullptr,
                sort_bytes,
                (uint64_t *)keys_unsorted.data_ptr<int64_t>(),
                (uint64_t *)keys_sorted.data_ptr<int64_t>(),
                (uint32_t *)values_unsorted.data_ptr<int>(),
                (uint32_t *)values_sorted.data_ptr<int>(),
                num_rendered,
                0,
                32 + bit
            );
            torch::Tensor sort_tmp = torch::empty({(long long)sort_bytes}, means3d.options().dtype(torch::kUInt8));
            cub::DeviceRadixSort::SortPairs(
                sort_tmp.data_ptr<uint8_t>(),
                sort_bytes,
                (uint64_t *)keys_unsorted.data_ptr<int64_t>(),
                (uint64_t *)keys_sorted.data_ptr<int64_t>(),
                (uint32_t *)values_unsorted.data_ptr<int>(),
                (uint32_t *)values_sorted.data_ptr<int>(),
                num_rendered,
                0,
                32 + bit
            );

            CUDA_CALL(cudaMemset(ranges.data_ptr<int>(), 0, num_tiles * sizeof(uint2)));
            identify_tile_ranges<<<(num_rendered + 255) / 256, 256>>>(
                num_rendered,
                (const uint64_t *)keys_sorted.contiguous().data_ptr<int64_t>(),
                (uint2 *)ranges.contiguous().data_ptr<int>()
            );

        }
    }
    if (num_rendered > 0) {
        const float *grad_out_overdraw_ptr = grad_out_overdraw.numel() > 0 ? grad_out_overdraw.contiguous().data_ptr<float>() : nullptr;
        const float *grad_out_surf_depth_ptr =
            grad_out_surf_depth.numel() > 0 ? grad_out_surf_depth.contiguous().data_ptr<float>() : nullptr;
        const float *grad_out_surf_alpha_ptr =
            grad_out_surf_alpha.numel() > 0 ? grad_out_surf_alpha.contiguous().data_ptr<float>() : nullptr;
        const float *grad_out_surf_normals_ptr =
            grad_out_surf_normals.numel() > 0 ? grad_out_surf_normals.contiguous().data_ptr<float>() : nullptr;

        if (has_surface_grads) {
            if (prefilter_edge_on) {
                render_hexels3d_backward<true, true><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                    (const float2 *)points_xy_center_ptr,
                    grad_points_xy_center_ptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_model,
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    grad_out_color.contiguous().data_ptr<float>(),
                    grad_out_overdraw_ptr,
                    grad_out_surf_depth_ptr,
                    grad_out_surf_alpha_ptr,
                    grad_out_surf_normals_ptr,
                    (const float3 *)normals_ptr,
                    grad_colors.contiguous().data_ptr<float>(),
                    want_linear ? grad_color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? grad_color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? grad_view_grads.contiguous().data_ptr<float>() : nullptr,
                    grad_specular_ks_ptr,
                    grad_specular_log_s_ptr,
                    grad_opacities.contiguous().data_ptr<float>(),
                    grad_softnesses_ptr,
                    grad_transMats.contiguous().data_ptr<float>(),
                    abs_means2d_ptr,
                    grad_view_dirs.contiguous().data_ptr<float>(),
                    grad_normals_ptr,
                    overdraw_base,
                    color_model,
                    use_view_grads,
                    grad_out_surf_normals.numel() > 0 && grad_out_surf_normals.size(0) == 5
                );
            } else {
                render_hexels3d_backward<true, false><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                    nullptr,
                    nullptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_model,
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    grad_out_color.contiguous().data_ptr<float>(),
                    grad_out_overdraw_ptr,
                    grad_out_surf_depth_ptr,
                    grad_out_surf_alpha_ptr,
                    grad_out_surf_normals_ptr,
                    (const float3 *)normals_ptr,
                    grad_colors.contiguous().data_ptr<float>(),
                    want_linear ? grad_color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? grad_color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? grad_view_grads.contiguous().data_ptr<float>() : nullptr,
                    grad_specular_ks_ptr,
                    grad_specular_log_s_ptr,
                    grad_opacities.contiguous().data_ptr<float>(),
                    grad_softnesses_ptr,
                    grad_transMats.contiguous().data_ptr<float>(),
                    abs_means2d_ptr,
                    grad_view_dirs.contiguous().data_ptr<float>(),
                    grad_normals_ptr,
                    overdraw_base,
                    color_model,
                    use_view_grads,
                    grad_out_surf_normals.numel() > 0 && grad_out_surf_normals.size(0) == 5
                );
            }
        } else {
            if (prefilter_edge_on) {
                render_hexels3d_backward<false, true><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                    (const float2 *)points_xy_center_ptr,
                    grad_points_xy_center_ptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_model,
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    grad_out_color.contiguous().data_ptr<float>(),
                    grad_out_overdraw_ptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    (const float3 *)normals_ptr,
                    grad_colors.contiguous().data_ptr<float>(),
                    want_linear ? grad_color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? grad_color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? grad_view_grads.contiguous().data_ptr<float>() : nullptr,
                    grad_specular_ks_ptr,
                    grad_specular_log_s_ptr,
                    grad_opacities.contiguous().data_ptr<float>(),
                    grad_softnesses_ptr,
                    grad_transMats.contiguous().data_ptr<float>(),
                    abs_means2d_ptr,
                    grad_view_dirs.contiguous().data_ptr<float>(),
                    grad_normals_ptr,
                    overdraw_base,
                    color_model,
                    use_view_grads,
                    false,
                    cached ? saved[9].data_ptr<float>() : nullptr,
                    cached ? saved[10].data_ptr<int>() : nullptr
                );
            } else {
                render_hexels3d_backward<false, false><<<tile_grid, dim3(kBlockX, kBlockY, 1)>>>(
                    (const uint2 *)ranges.contiguous().data_ptr<int>(),
                    (const uint32_t *)values_sorted.contiguous().data_ptr<int>(),
                    nullptr,
                    nullptr,
                    image_width,
                    image_height,
                    colors.contiguous().data_ptr<float>(),
                    want_linear ? color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? view_grads.contiguous().data_ptr<float>() : nullptr,
                    specular_ks.contiguous().data_ptr<float>(),
                    specular_log_s.contiguous().data_ptr<float>(),
                    view_model,
                    view_dirs.contiguous().data_ptr<float>(),
                    transMats.contiguous().data_ptr<float>(),
                    opacities_out.contiguous().data_ptr<float>(),
                    softnesses_out.contiguous().data_ptr<float>(),
                    softness_stride,
                    background.contiguous().data_ptr<float>(),
                    grad_out_color.contiguous().data_ptr<float>(),
                    grad_out_overdraw_ptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    (const float3 *)normals_ptr,
                    grad_colors.contiguous().data_ptr<float>(),
                    want_linear ? grad_color_grads.contiguous().data_ptr<float>() : nullptr,
                    want_quadratic ? grad_color_quads.contiguous().data_ptr<float>() : nullptr,
                    use_view_grads ? grad_view_grads.contiguous().data_ptr<float>() : nullptr,
                    grad_specular_ks_ptr,
                    grad_specular_log_s_ptr,
                    grad_opacities.contiguous().data_ptr<float>(),
                    grad_softnesses_ptr,
                    grad_transMats.contiguous().data_ptr<float>(),
                    abs_means2d_ptr,
                    grad_view_dirs.contiguous().data_ptr<float>(),
                    grad_normals_ptr,
                    overdraw_base,
                    color_model,
                    use_view_grads,
                    false,
                    cached ? saved[9].data_ptr<float>() : nullptr,
                    cached ? saved[10].data_ptr<int>() : nullptr
                );
            }
        }

        if (grad_means2d_ptr != nullptr) {
            transmat_to_means2d_backward<<<(P + 255) / 256, 256>>>(
                P,
                image_width,
                image_height,
                means2d_stride,
                radii.contiguous().data_ptr<int>(),
                transMats.contiguous().data_ptr<float>(),
                grad_transMats.contiguous().data_ptr<float>(),
                grad_means2d_ptr
            );
        }

        transmat_to_geometry_backward<<<(P + 255) / 256, 256>>>(
            P,
            means3d.contiguous().data_ptr<float>(),
            (const float2 *)scales.contiguous().data_ptr<float>(),
            scale_modifier,
            (const float4 *)rotations.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            image_width,
            image_height,
            radii.contiguous().data_ptr<int>(),
            grad_transMats.contiguous().data_ptr<float>(),
            grad_view_dirs.contiguous().data_ptr<float>(),
            grad_normals_ptr,
            grad_means3d.contiguous().data_ptr<float>(),
            (float2 *)grad_scales.contiguous().data_ptr<float>(),
            (float4 *)grad_rotations.contiguous().data_ptr<float>()
        );
        if (prefilter_edge_on) {
            points_xy_center_to_means3d_backward<<<(P + 255) / 256, 256>>>(
                P,
                means3d.contiguous().data_ptr<float>(),
                projmatrix.contiguous().data_ptr<float>(),
                image_width,
                image_height,
                (const float2 *)grad_points_xy_center.contiguous().data_ptr<float>(),
                grad_means3d.contiguous().data_ptr<float>()
            );
        }
    }

    if (debug) {
        CUDA_CALL(cudaDeviceSynchronize());
    }

    return {
        grad_means3d,
        grad_means2d,
        grad_scales,
        grad_rotations,
        grad_colors,
        grad_color_grads,
        grad_color_quads,
        grad_view_grads,
        grad_specular_ks,
        grad_specular_log_s,
        grad_opacities,
        grad_softnesses,
    };
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_hexels3d_backward(
    const torch::Tensor &means3d,
    const torch::Tensor &means2d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const torch::Tensor &grad_out_color,
    const torch::Tensor &grad_out_overdraw,
    const torch::Tensor &grad_out_surf_depth,
    const torch::Tensor &grad_out_surf_alpha,
    const torch::Tensor &grad_out_surf_normals,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int overdraw_base_count,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug
) {
    return rasterize_hexels3d_backward_impl(
        means3d,
        means2d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        grad_out_color,
        grad_out_overdraw,
        grad_out_surf_depth,
        grad_out_surf_alpha,
        grad_out_surf_normals,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        overdraw_base_count,
        prefilter_edge_on,
        conservative_bounds,
        debug, {}
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_hexels3d_backward_cached(
    const torch::Tensor &means3d,
    const torch::Tensor &means2d,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const torch::Tensor &color_grads,
    const torch::Tensor &color_quads,
    const torch::Tensor &view_grads,
    const torch::Tensor &specular_ks,
    const torch::Tensor &specular_log_s,
    const torch::Tensor &opacities,
    const torch::Tensor &softnesses,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const torch::Tensor &background,
    const torch::Tensor &grad_out_color,
    const torch::Tensor &grad_out_overdraw,
    const torch::Tensor &grad_out_surf_depth,
    const torch::Tensor &grad_out_surf_alpha,
    const torch::Tensor &grad_out_surf_normals,
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float softness,
    const float scale_modifier,
    const int color_model,
    const int view_model,
    const bool use_view_grads,
    const int overdraw_base_count,
    const bool prefilter_edge_on,
    const bool conservative_bounds,
    const bool debug,
    const std::vector<torch::Tensor> &saved
) {
    return rasterize_hexels3d_backward_impl(
        means3d,
        means2d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        grad_out_color,
        grad_out_overdraw,
        grad_out_surf_depth,
        grad_out_surf_alpha,
        grad_out_surf_normals,
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        softness,
        scale_modifier,
        color_model,
        view_model,
        use_view_grads,
        overdraw_base_count,
        prefilter_edge_on,
        conservative_bounds,
        debug, saved
    );
}
