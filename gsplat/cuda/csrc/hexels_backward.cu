#include "hexels.cuh"

#include "helpers.cuh"

#include <cooperative_groups.h>
#include <cstdint>
#include <math.h>

namespace cg = cooperative_groups;

namespace {

constexpr float ENTROPY_EPS = 1e-7f;

__device__ __forceinline__ float sign_subgrad(const float x) {
    // Subgradient for abs(): sign(0) -> 0.
    return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

__device__ __forceinline__ float sigmoid_neg_dist(const float dist, const float softness) {
    // sigmoid(-dist * softness) = 1 / (1 + exp(dist * softness))
    const float x = dist * softness;
    return 1.f / (1.f + __expf(x));
}

__device__ __forceinline__ float softness_from_triplet(const float3 softness3, const int edge_id) {
    return (edge_id == 0) ? softness3.x : ((edge_id == 1) ? softness3.y : softness3.z);
}

__device__ __forceinline__ float3 load_softness_triplet(
    const float* __restrict__ softnesses,
    const int softness_stride,
    const float softness_default,
    const int g_id
) {
    float3 s3 = {softness_default, softness_default, softness_default};
    if (softnesses == nullptr || softness_stride <= 0) {
        return s3;
    }
    if (softness_stride == 3) {
        const float* s = softnesses + g_id * 3;
        s3 = {s[0], s[1], s[2]};
        return s3;
    }
    const float s = softnesses[g_id];
    return {s, s, s};
}

__device__ __forceinline__ void hexel_dist_and_partials(
    const float2 delta,
    const float2 scale,
    const float rotation,
    const float aa_kernel_size,
    float& dist_out,
    float& du_out,
    float& dv_out,
    float& u_rot_out,
    float& v_rot_out,
    float& u_base_out,
    float& v_base_out,
    float& s_u_out,
    float& s_v_out,
    float& sin_t_out,
    float& cos_t_out,
    float& u_out,  // Final scaled u coordinate (for peak gradient)
    float& v_out,  // Final scaled v coordinate (for peak gradient)
    int& edge_id_out
) {
    sincosf(rotation, &sin_t_out, &cos_t_out);

    u_rot_out = delta.x * cos_t_out + delta.y * sin_t_out;
    v_rot_out = -delta.x * sin_t_out + delta.y * cos_t_out;

    const float sx = scale.x + 1e-8f;
    const float sy = scale.y + 1e-8f;

    u_base_out = u_rot_out / sx;
    v_base_out = v_rot_out / sy;

    s_u_out = 1.f;
    s_v_out = 1.f;
    if (aa_kernel_size > 0.f) {
        const float inv_sx2 = 1.f / (sx * sx);
        const float inv_sy2 = 1.f / (sy * sy);
        s_u_out = rsqrtf(1.f + aa_kernel_size * inv_sx2);
        s_v_out = rsqrtf(1.f + aa_kernel_size * inv_sy2);
    }

    u_out = u_base_out * s_u_out;
    v_out = v_base_out * s_v_out;

    constexpr float k = 1.7320508f; // sqrt(3)
    constexpr float a = 0.5f * k;   // 0.8660254
    constexpr float b = 0.5f;

    const float d0 = fabsf(v_out);
    const float t1 = a * u_out + b * v_out;
    const float t2 = a * u_out - b * v_out;
    const float d1 = fabsf(t1);
    const float d2 = fabsf(t2);

    float max_slant = d1;
    int edge_id = 1;
    if (d2 >= max_slant) {
        max_slant = d2;
        edge_id = 2;
    }
    if (d0 > max_slant) {
        edge_id = 0;
        dist_out = d0 - 1.f;
        du_out = 0.f;
        dv_out = sign_subgrad(v_out);
        edge_id_out = edge_id;
        return;
    }

    dist_out = max_slant - 1.f;
    if (edge_id == 1) {
        const float s = sign_subgrad(t1);
        du_out = a * s;
        dv_out = b * s;
    } else {
        const float s = sign_subgrad(t2);
        du_out = a * s;
        dv_out = -b * s;
    }
    edge_id_out = edge_id;
}

__device__ __forceinline__ void hexel_dist_and_partials_affine(
    const float2 delta,
    const float4 affine,
    const float aa_kernel_size,
    float& dist_out,
    float& du_out,
    float& dv_out,
    float& u_base_out,
    float& v_base_out,
    float& s_u_out,
    float& s_v_out,
    float& u_out,  // Final scaled u coordinate (for peak gradient)
    float& v_out,  // Final scaled v coordinate (for peak gradient)
    int& edge_id_out
) {
    // Row-vector convention: [dx, dy] @ [[a00, a01], [a10, a11]]
    u_base_out = delta.x * affine.x + delta.y * affine.z;
    v_base_out = delta.x * affine.y + delta.y * affine.w;

    s_u_out = 1.f;
    s_v_out = 1.f;
    if (aa_kernel_size > 0.f) {
        const float inv_len_u2 = affine.x * affine.x + affine.z * affine.z;
        const float inv_len_v2 = affine.y * affine.y + affine.w * affine.w;
        s_u_out = rsqrtf(1.f + aa_kernel_size * inv_len_u2);
        s_v_out = rsqrtf(1.f + aa_kernel_size * inv_len_v2);
    }

    u_out = u_base_out * s_u_out;
    v_out = v_base_out * s_v_out;

    constexpr float k = 1.7320508f; // sqrt(3)
    constexpr float a = 0.5f * k;   // 0.8660254
    constexpr float b = 0.5f;

    const float d0 = fabsf(v_out);
    const float t1 = a * u_out + b * v_out;
    const float t2 = a * u_out - b * v_out;
    const float d1 = fabsf(t1);
    const float d2 = fabsf(t2);

    float max_slant = d1;
    int edge_id = 1;
    if (d2 >= max_slant) {
        max_slant = d2;
        edge_id = 2;
    }
    if (d0 > max_slant) {
        edge_id = 0;
        dist_out = d0 - 1.f;
        du_out = 0.f;
        dv_out = sign_subgrad(v_out);
        edge_id_out = edge_id;
        return;
    }

    dist_out = max_slant - 1.f;
    if (edge_id == 1) {
        const float s = sign_subgrad(t1);
        du_out = a * s;
        dv_out = b * s;
    } else {
        const float s = sign_subgrad(t2);
        du_out = a * s;
        dv_out = -b * s;
    }
    edge_id_out = edge_id;
}

__device__ __forceinline__ void add_overdraw_and_entropy_grads(
    const int32_t topk[TOP_K],
    const float d_local[TOP_K],
    const float denom,
    const float v_weights_sum,
    const float v_entropy,
    float v_d_local[TOP_K]
) {
    if (v_weights_sum != 0.f) {
        for (int k = 0; k < TOP_K; ++k) {
            if (topk[k] < 0) continue;
            v_d_local[k] += v_weights_sum;
        }
    }
    if (v_entropy != 0.f) {
        float dHdp[TOP_K] = {0.f};
        float avg = 0.f;
        for (int k = 0; k < TOP_K; ++k) {
            if (topk[k] < 0) continue;
            const float p = d_local[k] / denom;
            const float dhdp = -(logf(p + ENTROPY_EPS) + 1.f);
            dHdp[k] = dhdp;
            avg += p * dhdp;
        }
        for (int k = 0; k < TOP_K; ++k) {
            if (topk[k] < 0) continue;
            const float dhdw = (dHdp[k] - avg) / denom;
            v_d_local[k] += v_entropy * dhdw;
        }
    }
}

} // namespace

__global__ void nd_rasterize_backward_topk_norm_hexels_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float2* __restrict__ scales,
    const float* __restrict__ rotation,
    const float* __restrict__ colors,
    const float* __restrict__ emission,
    const float* __restrict__ v_output,
    const float* __restrict__ v_weights_sum,
    const float* __restrict__ v_entropy,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_scale,
    float* __restrict__ v_rot,
    float* __restrict__ v_colors,
    float* __restrict__ v_emission,
    int* __restrict__ pixel_topk,
    const float aa_kernel_size,
    const float* __restrict__ softnesses,
    const int softness_stride,
    const float softness_default,
    const float* __restrict__ sharpness,
    float* __restrict__ v_sharpness,
    float* __restrict__ v_softnesses
) {
    auto block = cg::this_thread_block();
    const unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = min((int)(i * img_size.x + j), (int)(img_size.x * img_size.y - 1));

    const bool inside = (i < img_size.y && j < img_size.x);
    if (!inside) return;

    const float* v_out = &(v_output[channels * pix_id]);
    const int* topk = &pixel_topk[pix_id * TOP_K];
    if (topk[0] < 0) return;

    int32_t topk_ids[TOP_K];
    for (int k = 0; k < TOP_K; ++k) {
        topk_ids[k] = topk[k];
    }

    float d_local[TOP_K] = {0.0f};
    float u_coord[TOP_K] = {0.0f};  // Store for peak gradient
    float v_coord[TOP_K] = {0.0f};  // Store for peak gradient
    float prim_sharpness[TOP_K] = {0.0f};  // Store per-primitive sharpness
    float prim_peak[TOP_K] = {0.0f};  // Store per-primitive peak weight (emission uses peak only)
    float v_peak_emit_local[TOP_K] = {0.0f};
    float denom = EPS;
    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;
        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};

        const float peak_sharpness = (sharpness != nullptr) ? sharpness[g_id] : 0.f;
        prim_sharpness[k] = peak_sharpness;

        float dist;
        float du;
        float dv;
        float u_rot_coord;
        float v_rot_coord;
        float u_base;
        float v_base;
        float s_u;
        float s_v;
        float sin_t;
        float cos_t;
        float u_final;
        float v_final;
        int edge_id;
        hexel_dist_and_partials(
            delta,
            scales[g_id],
            rotation[g_id],
            aa_kernel_size,
            dist,
            du,
            dv,
            u_rot_coord,
            v_rot_coord,
            u_base,
            v_base,
            s_u,
            s_v,
            sin_t,
            cos_t,
            u_final,
            v_final,
            edge_id
        );

        const float3 softness3 = load_softness_triplet(softnesses, softness_stride, softness_default, g_id);
        const float softness_eff = softness_from_triplet(softness3, edge_id);
        const float sig = sigmoid_neg_dist(dist, softness_eff);
        float alpha = sig;
        if (peak_sharpness > 0.f) {
            const float r2 = u_final * u_final + v_final * v_final;
            const float peak = __expf(-peak_sharpness * r2);
            prim_peak[k] = peak;
            alpha *= peak;
        }
        denom += alpha;
        d_local[k] = alpha;
        u_coord[k] = u_final;
        v_coord[k] = v_final;
    }

    float v_d_local[TOP_K] = {0.f};
    float v_d_sum = 0.f;

    // compute each primitive's contribution to the gradient (image channels)
    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;

        const float alpha = d_local[k];
        const float norm_d = alpha / denom;

        // update v_colors for this primitive, and compute v_norm_d
        float v_norm_d = 0.f;
        const float* rgb = &colors[channels * g_id];
        for (int c = 0; c < channels; ++c) {
            atomicAdd(v_colors + channels * g_id + c, norm_d * v_out[c]);
            v_norm_d += rgb[c] * v_out[c];
        }
        if (emission != nullptr && v_emission != nullptr) {
            const float peak = prim_peak[k];
            if (peak > 0.f) {
                const float* emit = &emission[channels * g_id];
                float v_peak_emit = 0.f;
                for (int c = 0; c < channels; ++c) {
                    atomicAdd(v_emission + channels * g_id + c, peak * v_out[c]);
                    v_peak_emit += emit[c] * v_out[c];
                }
                v_peak_emit_local[k] = v_peak_emit;
            }
        }

        v_d_local[k] = v_norm_d / denom;
        v_d_sum += -alpha / (denom * denom) * v_norm_d;
    }
    for (int k = 0; k < TOP_K; ++k) {
        v_d_local[k] += v_d_sum;
    }

    // Add contributions from auxiliary regularizers.
    const float v_ws = (v_weights_sum != nullptr) ? v_weights_sum[pix_id] : 0.f;
    const float v_ent = (v_entropy != nullptr) ? v_entropy[pix_id] : 0.f;
    add_overdraw_and_entropy_grads(topk_ids, d_local, denom, v_ws, v_ent, v_d_local);

    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;

        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};

        float dist;
        float du;
        float dv;
        float u_rot_coord;
        float v_rot_coord;
        float u_base;
        float v_base;
        float s_u;
        float s_v;
        float sin_t;
        float cos_t;
        float u_final;
        float v_final;
        int edge_id;
        hexel_dist_and_partials(
            delta,
            scales[g_id],
            rotation[g_id],
            aa_kernel_size,
            dist,
            du,
            dv,
            u_rot_coord,
            v_rot_coord,
            u_base,
            v_base,
            s_u,
            s_v,
            sin_t,
            cos_t,
            u_final,
            v_final,
            edge_id
        );

        const float v_d = v_d_local[k];
        const float3 softness3 = load_softness_triplet(softnesses, softness_stride, softness_default, g_id);
        const float softness_eff = softness_from_triplet(softness3, edge_id);
        const float sig = sigmoid_neg_dist(dist, softness_eff);
        const float peak_sharpness = prim_sharpness[k];

        // Backprop through peak mode (when peak_sharpness > 0):
        //   alpha = sig * peak, peak = exp(-peak_sharpness * r²)
        //   v_sig = v_d * peak
        //   v_peak = v_d * sig + v_peak_emit_local[k]   (emission adds to v_peak)
        //   v_r2 = v_peak * (-peak_sharpness) * peak
        float v_sig = v_d;
        float v_u_peak = 0.f;
        float v_v_peak = 0.f;
        if (peak_sharpness > 0.f) {
            const float r2 = u_coord[k] * u_coord[k] + v_coord[k] * v_coord[k];
            const float peak = prim_peak[k];
            v_sig = v_d * peak;
            const float v_peak = v_d * sig + v_peak_emit_local[k];
            const float v_r2 = v_peak * peak * (-peak_sharpness);
            v_u_peak = v_r2 * 2.f * u_coord[k];
            v_v_peak = v_r2 * 2.f * v_coord[k];

            // d(peak)/d(peak_sharpness) = -r² * peak
            if (v_sharpness != nullptr) {
                const float v_sharpness_local = v_peak * peak * (-r2);
                atomicAdd(v_sharpness + g_id, v_sharpness_local);
            }
        }

        if (v_softnesses != nullptr) {
            const float v_soft = v_sig * (-dist) * sig * (1.f - sig);
            if (softness_stride == 3) {
                atomicAdd(v_softnesses + g_id * 3 + edge_id, v_soft);
            } else if (softness_stride == 1) {
                atomicAdd(v_softnesses + g_id, v_soft);
            }
        }

        const float v_dist = v_sig * (-softness_eff) * sig * (1.f - sig);

        const float sx = scales[g_id].x + 1e-8f;
        const float sy = scales[g_id].y + 1e-8f;
        const float inv_sx = 1.f / sx;
        const float inv_sy = 1.f / sy;

        // Gradients through SDF
        float v_u = v_dist * du;
        float v_v = v_dist * dv;

        // Add gradients from peak term
        v_u += v_u_peak;
        v_v += v_v_peak;

        const float v_u_base = v_u * s_u;
        const float v_v_base = v_v * s_v;

        // Gradients through AA coordinate scaling
        const float v_s_u = v_u * u_base;
        const float v_s_v = v_v * v_base;

        const float v_u_rot = v_u_base * inv_sx;
        const float v_v_rot = v_v_base * inv_sy;

        // scale grads (sx, sy) through u_base/v_base
        float v_sx = v_u_base * (-u_rot_coord) * inv_sx * inv_sx;
        float v_sy = v_v_base * (-v_rot_coord) * inv_sy * inv_sy;

        // scale grads through AA scaling s_u/s_v
        if (aa_kernel_size > 0.f) {
            const float ds_u_dsx = aa_kernel_size * (s_u * s_u * s_u) * (inv_sx * inv_sx * inv_sx);
            const float ds_v_dsy = aa_kernel_size * (s_v * s_v * s_v) * (inv_sy * inv_sy * inv_sy);
            v_sx += v_s_u * ds_u_dsx;
            v_sy += v_s_v * ds_v_dsy;
        }

        // delta grads (dx, dy)
        const float v_dx = v_u_rot * cos_t + v_v_rot * (-sin_t);
        const float v_dy = v_u_rot * sin_t + v_v_rot * cos_t;

        // rotation grad
        const float v_rot_angle = v_u_rot * v_rot_coord + v_v_rot * (-u_rot_coord);

        float* v_xy_ptr = (float*)(v_xy);
        atomicAdd(v_xy_ptr + 2 * g_id + 0, v_dx);
        atomicAdd(v_xy_ptr + 2 * g_id + 1, v_dy);

        float* v_scale_ptr = (float*)(v_scale);
        atomicAdd(v_scale_ptr + 2 * g_id + 0, v_sx);
        atomicAdd(v_scale_ptr + 2 * g_id + 1, v_sy);

        atomicAdd(v_rot + g_id, v_rot_angle);
    }
}

__global__ void nd_rasterize_backward_topk_norm_hexels_color_grad_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float2* __restrict__ scales,
    const float* __restrict__ rotation,
    const float* __restrict__ colors,
    const float* __restrict__ color_grads,
    const float color_grad_scale,
    const float* __restrict__ emission,
    const float* __restrict__ v_output,
    const float* __restrict__ v_weights_sum,
    const float* __restrict__ v_entropy,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_scale,
    float* __restrict__ v_rot,
    float* __restrict__ v_colors,
    float* __restrict__ v_color_grads,
    float* __restrict__ v_emission,
    int* __restrict__ pixel_topk,
    const float aa_kernel_size,
    const float* __restrict__ softnesses,
    const int softness_stride,
    const float softness_default,
    const float* __restrict__ sharpness,
    float* __restrict__ v_sharpness,
    float* __restrict__ v_softnesses
) {
    auto block = cg::this_thread_block();
    const unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = min((int)(i * img_size.x + j), (int)(img_size.x * img_size.y - 1));

    const bool inside = (i < img_size.y && j < img_size.x);
    if (!inside) return;

    const float* v_out = &(v_output[channels * pix_id]);
    const int* topk = &pixel_topk[pix_id * TOP_K];
    if (topk[0] < 0) return;

    int32_t topk_ids[TOP_K];
    for (int k = 0; k < TOP_K; ++k) {
        topk_ids[k] = topk[k];
    }

    float d_local[TOP_K] = {0.0f};
    float u_coord[TOP_K] = {0.0f};  // Store for peak gradient
    float v_coord[TOP_K] = {0.0f};  // Store for peak gradient
    float prim_sharpness[TOP_K] = {0.0f};  // Store per-primitive sharpness
    float prim_peak[TOP_K] = {0.0f};
    float v_peak_emit_local[TOP_K] = {0.0f};
    float denom = EPS;
    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;
        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};

        const float peak_sharpness = (sharpness != nullptr) ? sharpness[g_id] : 0.f;
        prim_sharpness[k] = peak_sharpness;

        float dist;
        float du;
        float dv;
        float u_rot_coord;
        float v_rot_coord;
        float u_base;
        float v_base;
        float s_u;
        float s_v;
        float sin_t;
        float cos_t;
        float u_final;
        float v_final;
        int edge_id;
        hexel_dist_and_partials(
            delta,
            scales[g_id],
            rotation[g_id],
            aa_kernel_size,
            dist,
            du,
            dv,
            u_rot_coord,
            v_rot_coord,
            u_base,
            v_base,
            s_u,
            s_v,
            sin_t,
            cos_t,
            u_final,
            v_final,
            edge_id
        );

        const float3 softness3 = load_softness_triplet(softnesses, softness_stride, softness_default, g_id);
        const float softness_eff = softness_from_triplet(softness3, edge_id);
        const float sig = sigmoid_neg_dist(dist, softness_eff);
        float alpha = sig;
        if (peak_sharpness > 0.f) {
            const float r2 = u_final * u_final + v_final * v_final;
            const float peak = __expf(-peak_sharpness * r2);
            prim_peak[k] = peak;
            alpha *= peak;
        }
        denom += alpha;
        d_local[k] = alpha;
        u_coord[k] = u_final;
        v_coord[k] = v_final;
    }

    float v_d_local[TOP_K] = {0.f};
    float v_d_sum = 0.f;

    // compute each primitive's contribution to the gradient (image channels)
    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;

        const float alpha = d_local[k];
        const float norm_d = alpha / denom;

        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};

        // update v_colors and v_color_grads for this primitive, and compute v_norm_d
        float v_norm_d = 0.f;
        for (int c = 0; c < channels; ++c) {
            const float base = colors[channels * g_id + c];
            const float grad_x = color_grads[(g_id * 2 + 0) * channels + c];
            const float grad_y = color_grads[(g_id * 2 + 1) * channels + c];
            const float color = base + color_grad_scale * (delta.x * grad_x + delta.y * grad_y);

            // base color gradient
            atomicAdd(v_colors + channels * g_id + c, norm_d * v_out[c]);
            // color gradient parameters (screen-space delta)
            atomicAdd(
                v_color_grads + (g_id * 2 + 0) * channels + c,
                norm_d * v_out[c] * color_grad_scale * delta.x
            );
            atomicAdd(
                v_color_grads + (g_id * 2 + 1) * channels + c,
                norm_d * v_out[c] * color_grad_scale * delta.y
            );

            // norm_d contribution
            v_norm_d += color * v_out[c];
        }
        if (emission != nullptr && v_emission != nullptr) {
            const float peak = prim_peak[k];
            if (peak > 0.f) {
                const float* emit = &emission[channels * g_id];
                float v_peak_emit = 0.f;
                for (int c = 0; c < channels; ++c) {
                    atomicAdd(v_emission + channels * g_id + c, peak * v_out[c]);
                    v_peak_emit += emit[c] * v_out[c];
                }
                v_peak_emit_local[k] = v_peak_emit;
            }
        }

        // Direct position gradient through color offset (delta depends on xy).
        float v_delta_x_color = 0.f;
        float v_delta_y_color = 0.f;
        for (int c = 0; c < channels; ++c) {
            const float grad_x = color_grads[(g_id * 2 + 0) * channels + c];
            const float grad_y = color_grads[(g_id * 2 + 1) * channels + c];
            v_delta_x_color += norm_d * v_out[c] * color_grad_scale * grad_x;
            v_delta_y_color += norm_d * v_out[c] * color_grad_scale * grad_y;
        }
        float* v_xy_ptr_color = (float*)(v_xy);
        atomicAdd(v_xy_ptr_color + 2 * g_id + 0, v_delta_x_color);
        atomicAdd(v_xy_ptr_color + 2 * g_id + 1, v_delta_y_color);

        v_d_local[k] = v_norm_d / denom;
        v_d_sum += -alpha / (denom * denom) * v_norm_d;
    }
    for (int k = 0; k < TOP_K; ++k) {
        v_d_local[k] += v_d_sum;
    }

    // Add contributions from auxiliary regularizers.
    const float v_ws = (v_weights_sum != nullptr) ? v_weights_sum[pix_id] : 0.f;
    const float v_ent = (v_entropy != nullptr) ? v_entropy[pix_id] : 0.f;
    add_overdraw_and_entropy_grads(topk_ids, d_local, denom, v_ws, v_ent, v_d_local);

    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;

        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};

        float dist;
        float du;
        float dv;
        float u_rot_coord;
        float v_rot_coord;
        float u_base;
        float v_base;
        float s_u;
        float s_v;
        float sin_t;
        float cos_t;
        float u_final;
        float v_final;
        int edge_id;
        hexel_dist_and_partials(
            delta,
            scales[g_id],
            rotation[g_id],
            aa_kernel_size,
            dist,
            du,
            dv,
            u_rot_coord,
            v_rot_coord,
            u_base,
            v_base,
            s_u,
            s_v,
            sin_t,
            cos_t,
            u_final,
            v_final,
            edge_id
        );

        const float v_d = v_d_local[k];
        const float3 softness3 = load_softness_triplet(softnesses, softness_stride, softness_default, g_id);
        const float softness_eff = softness_from_triplet(softness3, edge_id);
        const float sig = sigmoid_neg_dist(dist, softness_eff);
        const float peak_sharpness = prim_sharpness[k];

        // Backprop through peak mode (when peak_sharpness > 0):
        //   alpha = sig * peak, peak = exp(-peak_sharpness * r²)
        //   v_sig = v_d * peak
        //   v_peak = v_d * sig + v_peak_emit_local[k]   (emission adds to v_peak)
        //   v_r2 = v_peak * (-peak_sharpness) * peak
        float v_sig = v_d;
        float v_u_peak = 0.f;
        float v_v_peak = 0.f;
        if (peak_sharpness > 0.f) {
            const float r2 = u_coord[k] * u_coord[k] + v_coord[k] * v_coord[k];
            const float peak = prim_peak[k];
            v_sig = v_d * peak;
            const float v_peak = v_d * sig + v_peak_emit_local[k];
            const float v_r2 = v_peak * peak * (-peak_sharpness);
            v_u_peak = v_r2 * 2.f * u_coord[k];
            v_v_peak = v_r2 * 2.f * v_coord[k];

            // d(peak)/d(peak_sharpness) = -r² * peak
            if (v_sharpness != nullptr) {
                const float v_sharpness_local = v_peak * peak * (-r2);
                atomicAdd(v_sharpness + g_id, v_sharpness_local);
            }
        }

        if (v_softnesses != nullptr) {
            const float v_soft = v_sig * (-dist) * sig * (1.f - sig);
            if (softness_stride == 3) {
                atomicAdd(v_softnesses + g_id * 3 + edge_id, v_soft);
            } else if (softness_stride == 1) {
                atomicAdd(v_softnesses + g_id, v_soft);
            }
        }

        const float v_dist = v_sig * (-softness_eff) * sig * (1.f - sig);

        const float sx = scales[g_id].x + 1e-8f;
        const float sy = scales[g_id].y + 1e-8f;
        const float inv_sx = 1.f / sx;
        const float inv_sy = 1.f / sy;

        // Gradients through SDF
        float v_u = v_dist * du;
        float v_v = v_dist * dv;

        // Add gradients from peak term
        v_u += v_u_peak;
        v_v += v_v_peak;

        const float v_u_base = v_u * s_u;
        const float v_v_base = v_v * s_v;

        // Gradients through AA coordinate scaling
        const float v_s_u = v_u * u_base;
        const float v_s_v = v_v * v_base;

        const float v_u_rot = v_u_base * inv_sx;
        const float v_v_rot = v_v_base * inv_sy;

        // scale grads (sx, sy) through u_base/v_base
        float v_sx = v_u_base * (-u_rot_coord) * inv_sx * inv_sx;
        float v_sy = v_v_base * (-v_rot_coord) * inv_sy * inv_sy;

        // scale grads through AA scaling s_u/s_v
        if (aa_kernel_size > 0.f) {
            const float ds_u_dsx = aa_kernel_size * (s_u * s_u * s_u) * (inv_sx * inv_sx * inv_sx);
            const float ds_v_dsy = aa_kernel_size * (s_v * s_v * s_v) * (inv_sy * inv_sy * inv_sy);
            v_sx += v_s_u * ds_u_dsx;
            v_sy += v_s_v * ds_v_dsy;
        }

        // delta grads (dx, dy)
        const float v_dx = v_u_rot * cos_t + v_v_rot * (-sin_t);
        const float v_dy = v_u_rot * sin_t + v_v_rot * cos_t;

        // rotation grad
        const float v_rot_angle = v_u_rot * v_rot_coord + v_v_rot * (-u_rot_coord);

        float* v_xy_ptr = (float*)(v_xy);
        atomicAdd(v_xy_ptr + 2 * g_id + 0, v_dx);
        atomicAdd(v_xy_ptr + 2 * g_id + 1, v_dy);

        float* v_scale_ptr = (float*)(v_scale);
        atomicAdd(v_scale_ptr + 2 * g_id + 0, v_sx);
        atomicAdd(v_scale_ptr + 2 * g_id + 1, v_sy);

        atomicAdd(v_rot + g_id, v_rot_angle);
    }
}

__global__ void nd_rasterize_backward_topk_norm_hexels_affine_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float4* __restrict__ affines,
    const float* __restrict__ colors,
    const float* __restrict__ emission,
    const float* __restrict__ v_output,
    const float* __restrict__ v_weights_sum,
    const float* __restrict__ v_entropy,
    float2* __restrict__ v_xy,
    float4* __restrict__ v_affines,
    float* __restrict__ v_colors,
    float* __restrict__ v_emission,
    int* __restrict__ pixel_topk,
    const float aa_kernel_size,
    const float* __restrict__ softnesses,
    const int softness_stride,
    const float softness_default,
    const float* __restrict__ sharpness,
    float* __restrict__ v_sharpness,
    float* __restrict__ v_softnesses
) {
    auto block = cg::this_thread_block();
    const unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = min((int)(i * img_size.x + j), (int)(img_size.x * img_size.y - 1));

    const bool inside = (i < img_size.y && j < img_size.x);
    if (!inside) return;

    const float* v_out = &(v_output[channels * pix_id]);
    const int* topk = &pixel_topk[pix_id * TOP_K];
    if (topk[0] < 0) return;

    int32_t topk_ids[TOP_K];
    for (int k = 0; k < TOP_K; ++k) {
        topk_ids[k] = topk[k];
    }

    float d_local[TOP_K] = {0.0f};
    float u_coord[TOP_K] = {0.0f};  // Store for peak gradient
    float v_coord[TOP_K] = {0.0f};  // Store for peak gradient
    float prim_sharpness[TOP_K] = {0.0f};  // Store per-primitive sharpness
    float prim_peak[TOP_K] = {0.0f};
    float v_peak_emit_local[TOP_K] = {0.0f};
    float denom = EPS;
    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;
        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};

        const float peak_sharpness = (sharpness != nullptr) ? sharpness[g_id] : 0.f;
        prim_sharpness[k] = peak_sharpness;

        float dist;
        float du;
        float dv;
        float u_base;
        float v_base;
        float s_u;
        float s_v;
        float u_out;
        float v_out_coord;
        int edge_id;
        hexel_dist_and_partials_affine(
            delta, affines[g_id], aa_kernel_size, dist, du, dv, u_base, v_base, s_u, s_v, u_out, v_out_coord, edge_id
        );

        const float3 softness3 = load_softness_triplet(softnesses, softness_stride, softness_default, g_id);
        const float softness_eff = softness_from_triplet(softness3, edge_id);
        const float sig = sigmoid_neg_dist(dist, softness_eff);
        float alpha = sig;
        if (peak_sharpness > 0.f) {
            const float r2 = u_out * u_out + v_out_coord * v_out_coord;
            const float peak = __expf(-peak_sharpness * r2);
            prim_peak[k] = peak;
            alpha *= peak;
        }
        denom += alpha;
        d_local[k] = alpha;
        u_coord[k] = u_out;
        v_coord[k] = v_out_coord;
    }

    float v_d_local[TOP_K] = {0.f};
    float v_d_sum = 0.f;

    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;

        const float alpha = d_local[k];
        const float norm_d = alpha / denom;

        float v_norm_d = 0.f;
        for (int c = 0; c < channels; ++c) {
            atomicAdd(v_colors + channels * g_id + c, norm_d * v_out[c]);
            v_norm_d += colors[channels * g_id + c] * v_out[c];
        }
        if (emission != nullptr && v_emission != nullptr) {
            const float peak = prim_peak[k];
            if (peak > 0.f) {
                const float* emit = &emission[channels * g_id];
                float v_peak_emit = 0.f;
                for (int c = 0; c < channels; ++c) {
                    atomicAdd(v_emission + channels * g_id + c, peak * v_out[c]);
                    v_peak_emit += emit[c] * v_out[c];
                }
                v_peak_emit_local[k] = v_peak_emit;
            }
        }

        v_d_local[k] = v_norm_d / denom;
        v_d_sum += -alpha / (denom * denom) * v_norm_d;
    }
    for (int k = 0; k < TOP_K; ++k) {
        v_d_local[k] += v_d_sum;
    }

    // Add contributions from auxiliary regularizers.
    const float v_ws = (v_weights_sum != nullptr) ? v_weights_sum[pix_id] : 0.f;
    const float v_ent = (v_entropy != nullptr) ? v_entropy[pix_id] : 0.f;
    add_overdraw_and_entropy_grads(topk_ids, d_local, denom, v_ws, v_ent, v_d_local);

    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;

        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};
        const float4 affine = affines[g_id];

        float dist;
        float du;
        float dv;
        float u_base;
        float v_base;
        float s_u;
        float s_v;
        float u_out;
        float v_out_coord;
        int edge_id;
        hexel_dist_and_partials_affine(
            delta, affine, aa_kernel_size, dist, du, dv, u_base, v_base, s_u, s_v, u_out, v_out_coord, edge_id
        );

        const float v_d = v_d_local[k];
        const float3 softness3 = load_softness_triplet(softnesses, softness_stride, softness_default, g_id);
        const float softness_eff = softness_from_triplet(softness3, edge_id);
        const float sig = sigmoid_neg_dist(dist, softness_eff);
        const float peak_sharpness = prim_sharpness[k];

        // Backprop through peak mode (when peak_sharpness > 0):
        //   alpha = sig * peak, peak = exp(-peak_sharpness * r²)
        //   v_sig = v_d * peak
        //   v_peak = v_d * sig + v_peak_emit_local[k]   (emission adds to v_peak)
        //   v_r2 = v_peak * (-peak_sharpness) * peak
        float v_sig = v_d;
        float v_u_peak = 0.f;
        float v_v_peak = 0.f;
        if (peak_sharpness > 0.f) {
            const float r2 = u_coord[k] * u_coord[k] + v_coord[k] * v_coord[k];
            const float peak = prim_peak[k];
            v_sig = v_d * peak;
            const float v_peak = v_d * sig + v_peak_emit_local[k];
            const float v_r2 = v_peak * peak * (-peak_sharpness);
            v_u_peak = v_r2 * 2.f * u_coord[k];
            v_v_peak = v_r2 * 2.f * v_coord[k];

            // d(peak)/d(peak_sharpness) = -r² * peak
            if (v_sharpness != nullptr) {
                const float v_sharpness_local = v_peak * peak * (-r2);
                atomicAdd(v_sharpness + g_id, v_sharpness_local);
            }
        }

        if (v_softnesses != nullptr) {
            const float v_soft = v_sig * (-dist) * sig * (1.f - sig);
            if (softness_stride == 3) {
                atomicAdd(v_softnesses + g_id * 3 + edge_id, v_soft);
            } else if (softness_stride == 1) {
                atomicAdd(v_softnesses + g_id, v_soft);
            }
        }

        const float v_dist = v_sig * (-softness_eff) * sig * (1.f - sig);

        // Gradients through SDF
        float v_u = v_dist * du;
        float v_v = v_dist * dv;

        // Add gradients from peak term
        v_u += v_u_peak;
        v_v += v_v_peak;

        const float v_u_base = v_u * s_u;
        const float v_v_base = v_v * s_v;
        // Gradients through AA coordinate scaling
        float v_s_u = v_u * u_base;
        float v_s_v = v_v * v_base;
        if (aa_kernel_size <= 0.f) {
            v_s_u = 0.f;
            v_s_v = 0.f;
        }

        // delta grads (dx, dy)
        const float v_dx = v_u_base * affine.x + v_v_base * affine.y;
        const float v_dy = v_u_base * affine.z + v_v_base * affine.w;

        float* v_xy_ptr = (float*)(v_xy);
        atomicAdd(v_xy_ptr + 2 * g_id + 0, v_dx);
        atomicAdd(v_xy_ptr + 2 * g_id + 1, v_dy);

        // affine grads (a00, a01, a10, a11)
        float v_a00 = delta.x * v_u_base;
        float v_a01 = delta.x * v_v_base;
        float v_a10 = delta.y * v_u_base;
        float v_a11 = delta.y * v_v_base;

        // AA scaling gradients w.r.t affine columns.
        if (aa_kernel_size > 0.f) {
            const float s_u_cub = s_u * s_u * s_u;
            const float s_v_cub = s_v * s_v * s_v;
            const float ds_u_da00 = -aa_kernel_size * affine.x * s_u_cub;
            const float ds_u_da10 = -aa_kernel_size * affine.z * s_u_cub;
            const float ds_v_da01 = -aa_kernel_size * affine.y * s_v_cub;
            const float ds_v_da11 = -aa_kernel_size * affine.w * s_v_cub;
            v_a00 += v_s_u * ds_u_da00;
            v_a10 += v_s_u * ds_u_da10;
            v_a01 += v_s_v * ds_v_da01;
            v_a11 += v_s_v * ds_v_da11;
        }

        float* v_aff_ptr = (float*)(v_affines);
        atomicAdd(v_aff_ptr + 4 * g_id + 0, v_a00);
        atomicAdd(v_aff_ptr + 4 * g_id + 1, v_a01);
        atomicAdd(v_aff_ptr + 4 * g_id + 2, v_a10);
        atomicAdd(v_aff_ptr + 4 * g_id + 3, v_a11);
    }
}

__global__ void nd_rasterize_backward_topk_norm_hexels_affine_color_grad_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float4* __restrict__ affines,
    const float* __restrict__ colors,
    const float* __restrict__ color_grads,
    const float color_grad_scale,
    const float* __restrict__ emission,
    const float* __restrict__ v_output,
    const float* __restrict__ v_weights_sum,
    const float* __restrict__ v_entropy,
    float2* __restrict__ v_xy,
    float4* __restrict__ v_affines,
    float* __restrict__ v_colors,
    float* __restrict__ v_color_grads,
    float* __restrict__ v_emission,
    int* __restrict__ pixel_topk,
    const float aa_kernel_size,
    const float* __restrict__ softnesses,
    const int softness_stride,
    const float softness_default,
    const float* __restrict__ sharpness,
    float* __restrict__ v_sharpness,
    float* __restrict__ v_softnesses
) {
    auto block = cg::this_thread_block();
    const unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = min((int)(i * img_size.x + j), (int)(img_size.x * img_size.y - 1));

    const bool inside = (i < img_size.y && j < img_size.x);
    if (!inside) return;

    const float* v_out = &(v_output[channels * pix_id]);
    const int* topk = &pixel_topk[pix_id * TOP_K];
    if (topk[0] < 0) return;

    int32_t topk_ids[TOP_K];
    for (int k = 0; k < TOP_K; ++k) {
        topk_ids[k] = topk[k];
    }

    float d_local[TOP_K] = {0.0f};
    float u_coord[TOP_K] = {0.0f};  // Store for peak gradient
    float v_coord[TOP_K] = {0.0f};  // Store for peak gradient
    float prim_sharpness[TOP_K] = {0.0f};  // Store per-primitive sharpness
    float prim_peak[TOP_K] = {0.0f};
    float v_peak_emit_local[TOP_K] = {0.0f};
    float denom = EPS;
    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;
        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};

        const float peak_sharpness = (sharpness != nullptr) ? sharpness[g_id] : 0.f;
        prim_sharpness[k] = peak_sharpness;

        float dist;
        float du;
        float dv;
        float u_base;
        float v_base;
        float s_u;
        float s_v;
        float u_out;
        float v_out_coord;
        int edge_id;
        hexel_dist_and_partials_affine(
            delta, affines[g_id], aa_kernel_size, dist, du, dv, u_base, v_base, s_u, s_v, u_out, v_out_coord, edge_id
        );

        const float3 softness3 = load_softness_triplet(softnesses, softness_stride, softness_default, g_id);
        const float softness_eff = softness_from_triplet(softness3, edge_id);
        const float sig = sigmoid_neg_dist(dist, softness_eff);
        float alpha = sig;
        if (peak_sharpness > 0.f) {
            const float r2 = u_out * u_out + v_out_coord * v_out_coord;
            const float peak = __expf(-peak_sharpness * r2);
            prim_peak[k] = peak;
            alpha *= peak;
        }
        denom += alpha;
        d_local[k] = alpha;
        u_coord[k] = u_out;
        v_coord[k] = v_out_coord;
    }

    float v_d_local[TOP_K] = {0.f};
    float v_d_sum = 0.f;

    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;

        const float alpha = d_local[k];
        const float norm_d = alpha / denom;

        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};

        // update v_colors and v_color_grads for this primitive, and compute v_norm_d
        float v_norm_d = 0.f;
        for (int c = 0; c < channels; ++c) {
            const float base = colors[channels * g_id + c];
            const float grad_x = color_grads[(g_id * 2 + 0) * channels + c];
            const float grad_y = color_grads[(g_id * 2 + 1) * channels + c];
            const float color = base + color_grad_scale * (delta.x * grad_x + delta.y * grad_y);

            atomicAdd(v_colors + channels * g_id + c, norm_d * v_out[c]);
            atomicAdd(
                v_color_grads + (g_id * 2 + 0) * channels + c,
                norm_d * v_out[c] * color_grad_scale * delta.x
            );
            atomicAdd(
                v_color_grads + (g_id * 2 + 1) * channels + c,
                norm_d * v_out[c] * color_grad_scale * delta.y
            );

            v_norm_d += color * v_out[c];
        }
        if (emission != nullptr && v_emission != nullptr) {
            const float peak = prim_peak[k];
            if (peak > 0.f) {
                const float* emit = &emission[channels * g_id];
                float v_peak_emit = 0.f;
                for (int c = 0; c < channels; ++c) {
                    atomicAdd(v_emission + channels * g_id + c, peak * v_out[c]);
                    v_peak_emit += emit[c] * v_out[c];
                }
                v_peak_emit_local[k] = v_peak_emit;
            }
        }

        // Direct position gradient through color offset (delta depends on xy).
        float v_delta_x_color = 0.f;
        float v_delta_y_color = 0.f;
        for (int c = 0; c < channels; ++c) {
            const float grad_x = color_grads[(g_id * 2 + 0) * channels + c];
            const float grad_y = color_grads[(g_id * 2 + 1) * channels + c];
            v_delta_x_color += norm_d * v_out[c] * color_grad_scale * grad_x;
            v_delta_y_color += norm_d * v_out[c] * color_grad_scale * grad_y;
        }
        float* v_xy_ptr_color = (float*)(v_xy);
        atomicAdd(v_xy_ptr_color + 2 * g_id + 0, v_delta_x_color);
        atomicAdd(v_xy_ptr_color + 2 * g_id + 1, v_delta_y_color);

        v_d_local[k] = v_norm_d / denom;
        v_d_sum += -alpha / (denom * denom) * v_norm_d;
    }
    for (int k = 0; k < TOP_K; ++k) {
        v_d_local[k] += v_d_sum;
    }

    // Add contributions from auxiliary regularizers.
    const float v_ws = (v_weights_sum != nullptr) ? v_weights_sum[pix_id] : 0.f;
    const float v_ent = (v_entropy != nullptr) ? v_entropy[pix_id] : 0.f;
    add_overdraw_and_entropy_grads(topk_ids, d_local, denom, v_ws, v_ent, v_d_local);

    for (int k = 0; k < TOP_K; ++k) {
        const int g_id = topk_ids[k];
        if (g_id < 0) continue;

        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};
        const float4 affine = affines[g_id];

        float dist;
        float du;
        float dv;
        float u_base;
        float v_base;
        float s_u;
        float s_v;
        float u_out;
        float v_out_coord;
        int edge_id;
        hexel_dist_and_partials_affine(
            delta, affine, aa_kernel_size, dist, du, dv, u_base, v_base, s_u, s_v, u_out, v_out_coord, edge_id
        );

        const float v_d = v_d_local[k];
        const float3 softness3 = load_softness_triplet(softnesses, softness_stride, softness_default, g_id);
        const float softness_eff = softness_from_triplet(softness3, edge_id);
        const float sig = sigmoid_neg_dist(dist, softness_eff);
        const float peak_sharpness = prim_sharpness[k];

        // Backprop through peak mode (when peak_sharpness > 0):
        //   alpha = sig * peak, peak = exp(-peak_sharpness * r²)
        //   v_sig = v_d * peak
        //   v_peak = v_d * sig + v_peak_emit_local[k]   (emission adds to v_peak)
        //   v_r2 = v_peak * (-peak_sharpness) * peak
        float v_sig = v_d;
        float v_u_peak = 0.f;
        float v_v_peak = 0.f;
        if (peak_sharpness > 0.f) {
            const float r2 = u_coord[k] * u_coord[k] + v_coord[k] * v_coord[k];
            const float peak = prim_peak[k];
            v_sig = v_d * peak;
            const float v_peak = v_d * sig + v_peak_emit_local[k];
            const float v_r2 = v_peak * peak * (-peak_sharpness);
            v_u_peak = v_r2 * 2.f * u_coord[k];
            v_v_peak = v_r2 * 2.f * v_coord[k];

            // d(peak)/d(peak_sharpness) = -r² * peak
            if (v_sharpness != nullptr) {
                const float v_sharpness_local = v_peak * peak * (-r2);
                atomicAdd(v_sharpness + g_id, v_sharpness_local);
            }
        }

        if (v_softnesses != nullptr) {
            const float v_soft = v_sig * (-dist) * sig * (1.f - sig);
            if (softness_stride == 3) {
                atomicAdd(v_softnesses + g_id * 3 + edge_id, v_soft);
            } else if (softness_stride == 1) {
                atomicAdd(v_softnesses + g_id, v_soft);
            }
        }

        const float v_dist = v_sig * (-softness_eff) * sig * (1.f - sig);

        // Gradients through SDF
        float v_u = v_dist * du;
        float v_v = v_dist * dv;

        // Add gradients from peak term
        v_u += v_u_peak;
        v_v += v_v_peak;

        const float v_u_base = v_u * s_u;
        const float v_v_base = v_v * s_v;
        // Gradients through AA coordinate scaling
        float v_s_u = v_u * u_base;
        float v_s_v = v_v * v_base;
        if (aa_kernel_size <= 0.f) {
            v_s_u = 0.f;
            v_s_v = 0.f;
        }

        // delta grads (dx, dy)
        const float v_dx = v_u_base * affine.x + v_v_base * affine.y;
        const float v_dy = v_u_base * affine.z + v_v_base * affine.w;

        float* v_xy_ptr = (float*)(v_xy);
        atomicAdd(v_xy_ptr + 2 * g_id + 0, v_dx);
        atomicAdd(v_xy_ptr + 2 * g_id + 1, v_dy);

        // affine grads (a00, a01, a10, a11)
        float v_a00 = delta.x * v_u_base;
        float v_a01 = delta.x * v_v_base;
        float v_a10 = delta.y * v_u_base;
        float v_a11 = delta.y * v_v_base;

        if (aa_kernel_size > 0.f) {
            const float s_u_cub = s_u * s_u * s_u;
            const float s_v_cub = s_v * s_v * s_v;
            const float ds_u_da00 = -aa_kernel_size * affine.x * s_u_cub;
            const float ds_u_da10 = -aa_kernel_size * affine.z * s_u_cub;
            const float ds_v_da01 = -aa_kernel_size * affine.y * s_v_cub;
            const float ds_v_da11 = -aa_kernel_size * affine.w * s_v_cub;
            v_a00 += v_s_u * ds_u_da00;
            v_a10 += v_s_u * ds_u_da10;
            v_a01 += v_s_v * ds_v_da01;
            v_a11 += v_s_v * ds_v_da11;
        }

        float* v_aff_ptr = (float*)(v_affines);
        atomicAdd(v_aff_ptr + 4 * g_id + 0, v_a00);
        atomicAdd(v_aff_ptr + 4 * g_id + 1, v_a01);
        atomicAdd(v_aff_ptr + 4 * g_id + 2, v_a10);
        atomicAdd(v_aff_ptr + 4 * g_id + 3, v_a11);
    }
}
