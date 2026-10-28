#include "hexels.cuh"

#include "helpers.cuh"

#include <algorithm>
#include <cooperative_groups.h>
#include <cstdint>
#include <math.h>

namespace cg = cooperative_groups;

namespace {

constexpr float ENTROPY_EPS = 1e-7f;

__device__ __forceinline__ float sigmoid_neg_dist(const float dist, const float softness) {
    // sigmoid(-dist * softness) = 1 / (1 + exp(dist * softness))
    const float x = dist * softness;
    return 1.f / (1.f + __expf(x));
}

__device__ __forceinline__ float hex_sdf_edge(const float u, const float v, int& edge_id) {
    // Regular pointy-top hexagon SDF as max of 3 oriented half-plane distances:
    //   dist = max(|v|, |a*u + b*v|, |a*u - b*v|) - 1
    // with a = sqrt(3)/2, b = 1/2.
    constexpr float k = 1.7320508f; // sqrt(3)
    constexpr float a = 0.5f * k;
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
        return d0 - 1.f;
    }
    return max_slant - 1.f;
}

__device__ __forceinline__ float hexel_alpha(
    const float2 delta,
    const float2 scale,
    const float rotation,
    const float aa_kernel_size,
    const float3 softness3,
    const float peak_sharpness,
    float& peak_out
) {
    float sin_t;
    float cos_t;
    sincosf(rotation, &sin_t, &cos_t);

    const float u_rot = delta.x * cos_t + delta.y * sin_t;
    const float v_rot = -delta.x * sin_t + delta.y * cos_t;

    const float sx = scale.x + 1e-8f;
    const float sy = scale.y + 1e-8f;

    const float u_base = u_rot / sx;
    const float v_base = v_rot / sy;

    float s_u = 1.f;
    float s_v = 1.f;
    if (aa_kernel_size > 0.f) {
        const float inv_sx2 = 1.f / (sx * sx);
        const float inv_sy2 = 1.f / (sy * sy);
        s_u = rsqrtf(1.f + aa_kernel_size * inv_sx2);
        s_v = rsqrtf(1.f + aa_kernel_size * inv_sy2);
    }

    const float u = u_base * s_u;
    const float v = v_base * s_v;

    int edge_id;
    const float dist = hex_sdf_edge(u, v, edge_id);
    const float softness_eff = (edge_id == 0) ? softness3.x : ((edge_id == 1) ? softness3.y : softness3.z);

    // Boundary alpha from sigmoid of SDF (edge-dependent softness)
    float alpha = sigmoid_neg_dist(dist, softness_eff);

    peak_out = 0.f;
    if (peak_sharpness > 0.f) {
        const float r2 = u * u + v * v;
        const float peak = __expf(-peak_sharpness * r2);
        peak_out = peak;
        alpha *= peak;
    }

    return alpha;
}

__device__ __forceinline__ float hexel_alpha_affine(
    const float2 delta,
    const float4 affine,
    const float aa_kernel_size,
    const float3 softness3,
    const float peak_sharpness,
    float& peak_out
) {
    // Row-vector convention: [dx, dy] @ [[a00, a01], [a10, a11]]
    const float u_base = delta.x * affine.x + delta.y * affine.z;
    const float v_base = delta.x * affine.y + delta.y * affine.w;

    float s_u = 1.f;
    float s_v = 1.f;
    if (aa_kernel_size > 0.f) {
        const float inv_len_u2 = affine.x * affine.x + affine.z * affine.z;
        const float inv_len_v2 = affine.y * affine.y + affine.w * affine.w;
        s_u = rsqrtf(1.f + aa_kernel_size * inv_len_u2);
        s_v = rsqrtf(1.f + aa_kernel_size * inv_len_v2);
    }

    const float u = u_base * s_u;
    const float v = v_base * s_v;

    int edge_id;
    const float dist = hex_sdf_edge(u, v, edge_id);
    const float softness_eff = (edge_id == 0) ? softness3.x : ((edge_id == 1) ? softness3.y : softness3.z);

    // Boundary alpha from sigmoid of SDF (edge-dependent softness)
    float alpha = sigmoid_neg_dist(dist, softness_eff);

    peak_out = 0.f;
    if (peak_sharpness > 0.f) {
        const float r2 = u * u + v * v;
        const float peak = __expf(-peak_sharpness * r2);
        peak_out = peak;
        alpha *= peak;
    }

    return alpha;
}

__device__ __forceinline__ void maybe_write_stats(
    const bool inside,
    const int32_t pix_id,
    const int32_t topk[TOP_K],
    const float topk_vals[TOP_K],
    const float denom,
    float* __restrict__ out_weights_sum,
    float* __restrict__ out_entropy
) {
    if (!inside) return;
    if (out_weights_sum != nullptr) {
        out_weights_sum[pix_id] = denom;
    }
    if (out_entropy != nullptr) {
        float entropy = 0.f;
        for (int k = 0; k < TOP_K; ++k) {
            if (topk[k] < 0) continue;
            const float p = topk_vals[k] / denom;
            entropy += -p * logf(p + ENTROPY_EPS);
        }
        out_entropy[pix_id] = entropy;
    }
}

} // namespace

__global__ void nd_rasterize_forward_topk_norm_hexels(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float2* __restrict__ scales,
    const float* __restrict__ rotation,
    const float* __restrict__ colors,
    const float aa_kernel_size,
    const float* __restrict__ softnesses,
    const int softness_stride,
    const float softness_default,
    const float* __restrict__ sharpness,
    const float* __restrict__ emission,
    float* __restrict__ out_img,
    int* __restrict__ pixel_topk,
    float* __restrict__ out_weights_sum,
    float* __restrict__ out_entropy
) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_bounds.x + block.group_index().x;
    const unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = i * img_size.x + j;
    const bool inside = (i < img_size.y && j < img_size.x);
    const bool done = !inside;

    // Which primitives are in this tile
    const int2 range = tile_bins[tile_id];
    if (range.x == range.y) {
        if (inside) {
            for (int c = 0; c < channels; ++c) out_img[pix_id * channels + c] = 0.f;
            for (int k = 0; k < TOP_K; ++k) pixel_topk[pix_id * TOP_K + k] = -1;
            if (out_weights_sum != nullptr) out_weights_sum[pix_id] = EPS;
            if (out_entropy != nullptr) out_entropy[pix_id] = 0.f;
        }
        return;
    }
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float2 scale_batch[BLOCK_SIZE];
    __shared__ float rot_batch[BLOCK_SIZE];
    __shared__ float sharpness_batch[BLOCK_SIZE];
    __shared__ float3 softness_batch[BLOCK_SIZE];

    const int tr = block.thread_rank();

    float pix_out[MAX_CHANNELS] = {0.f};

    int32_t topk[TOP_K];
    float topk_vals[TOP_K] = {0.f};
    float topk_peak[TOP_K] = {0.f};
    for (int k = 0; k < TOP_K; ++k) {
        topk[k] = -1;
    }

    int32_t min_topk_id = 0;
    float min_topk_val = -1.f;

    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        const int batch_start = range.x + BLOCK_SIZE * b;
        const int idx = batch_start + tr;
        if (idx < range.y) {
            const int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            scale_batch[tr] = scales[g_id];
            rot_batch[tr] = rotation[g_id];
            sharpness_batch[tr] = (sharpness != nullptr) ? sharpness[g_id] : 0.f;
            float3 s3 = {softness_default, softness_default, softness_default};
            if (softnesses != nullptr) {
                if (softness_stride == 3) {
                    const float* s = softnesses + g_id * 3;
                    s3 = {s[0], s[1], s[2]};
                } else if (softness_stride == 1) {
                    const float s = softnesses[g_id];
                    s3 = {s, s, s};
                }
            }
            softness_batch[tr] = s3;
        }

        block.sync();

        const int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && inside; ++t) {
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            float peak = 0.f;
            const float alpha = hexel_alpha(
                delta,
                scale_batch[t],
                rot_batch[t],
                aa_kernel_size,
                softness_batch[t],
                sharpness_batch[t],
                peak
            );

            if (alpha > min_topk_val) {
                topk[min_topk_id] = id_batch[t];
                topk_vals[min_topk_id] = alpha;
                topk_peak[min_topk_id] = peak;
                if (min_topk_val < 0.f && min_topk_id + 1 < TOP_K) {
                    ++min_topk_id;
                } else {
                    min_topk_val = 1e30f;
                    for (int32_t k = 0; k < TOP_K; ++k) {
                        if (topk[k] < 0) {
                            min_topk_id = k;
                            min_topk_val = -1.0f;
                            break;
                        }
                        if (topk_vals[k] < min_topk_val) {
                            min_topk_id = k;
                            min_topk_val = topk_vals[k];
                        }
                    }
                }
            }
        }
    }

    float sum_val = 0.f;
    for (int k = 0; k < TOP_K; ++k) {
        if (topk[k] < 0) continue;
        sum_val += topk_vals[k];
    }
    const float denom = sum_val + EPS;

    for (int c = 0; c < channels; ++c) {
        for (int k = 0; k < TOP_K; ++k) {
            const int32_t g = topk[k];
            if (g < 0) continue;
            const float vis = topk_vals[k] / denom;
            pix_out[c] += colors[g * channels + c] * vis;
        }
    }
    if (emission != nullptr) {
        for (int c = 0; c < channels; ++c) {
            for (int k = 0; k < TOP_K; ++k) {
                const int32_t g = topk[k];
                if (g < 0) continue;
                pix_out[c] += emission[g * channels + c] * topk_peak[k];
            }
        }
    }

    if (inside) {
        for (int c = 0; c < channels; ++c) {
            out_img[pix_id * channels + c] = pix_out[c];
        }
        for (int k = 0; k < TOP_K; ++k) {
            pixel_topk[pix_id * TOP_K + k] = topk[k];
        }
    }

    maybe_write_stats(inside, pix_id, topk, topk_vals, denom, out_weights_sum, out_entropy);
}

__global__ void nd_rasterize_forward_topk_norm_hexels_color_grad(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float2* __restrict__ scales,
    const float* __restrict__ rotation,
    const float* __restrict__ colors,
    const float* __restrict__ color_grads,
    const float color_grad_scale,
    const float aa_kernel_size,
    const float* __restrict__ softnesses,
    const int softness_stride,
    const float softness_default,
    const float* __restrict__ sharpness,
    const float* __restrict__ emission,
    float* __restrict__ out_img,
    int* __restrict__ pixel_topk,
    float* __restrict__ out_weights_sum,
    float* __restrict__ out_entropy
) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_bounds.x + block.group_index().x;
    const unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = i * img_size.x + j;
    const bool inside = (i < img_size.y && j < img_size.x);
    const bool done = !inside;

    // Which primitives are in this tile
    const int2 range = tile_bins[tile_id];
    if (range.x == range.y) {
        if (inside) {
            for (int c = 0; c < channels; ++c) out_img[pix_id * channels + c] = 0.f;
            for (int k = 0; k < TOP_K; ++k) pixel_topk[pix_id * TOP_K + k] = -1;
            if (out_weights_sum != nullptr) out_weights_sum[pix_id] = EPS;
            if (out_entropy != nullptr) out_entropy[pix_id] = 0.f;
        }
        return;
    }
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float2 scale_batch[BLOCK_SIZE];
    __shared__ float rot_batch[BLOCK_SIZE];
    __shared__ float sharpness_batch[BLOCK_SIZE];
    __shared__ float3 softness_batch[BLOCK_SIZE];

    const int tr = block.thread_rank();

    float pix_out[MAX_CHANNELS] = {0.f};

    int32_t topk[TOP_K];
    float topk_vals[TOP_K] = {0.f};
    float topk_peak[TOP_K] = {0.f};
    for (int k = 0; k < TOP_K; ++k) {
        topk[k] = -1;
    }

    int32_t min_topk_id = 0;
    float min_topk_val = -1.f;

    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        const int batch_start = range.x + BLOCK_SIZE * b;
        const int idx = batch_start + tr;
        if (idx < range.y) {
            const int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            scale_batch[tr] = scales[g_id];
            rot_batch[tr] = rotation[g_id];
            sharpness_batch[tr] = (sharpness != nullptr) ? sharpness[g_id] : 0.f;
            float3 s3 = {softness_default, softness_default, softness_default};
            if (softnesses != nullptr) {
                if (softness_stride == 3) {
                    const float* s = softnesses + g_id * 3;
                    s3 = {s[0], s[1], s[2]};
                } else if (softness_stride == 1) {
                    const float s = softnesses[g_id];
                    s3 = {s, s, s};
                }
            }
            softness_batch[tr] = s3;
        }

        block.sync();

        const int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && inside; ++t) {
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            float peak = 0.f;
            const float alpha = hexel_alpha(
                delta,
                scale_batch[t],
                rot_batch[t],
                aa_kernel_size,
                softness_batch[t],
                sharpness_batch[t],
                peak
            );

            if (alpha > min_topk_val) {
                topk[min_topk_id] = id_batch[t];
                topk_vals[min_topk_id] = alpha;
                topk_peak[min_topk_id] = peak;
                if (min_topk_val < 0.f && min_topk_id + 1 < TOP_K) {
                    ++min_topk_id;
                } else {
                    min_topk_val = 1e30f;
                    for (int32_t k = 0; k < TOP_K; ++k) {
                        if (topk[k] < 0) {
                            min_topk_id = k;
                            min_topk_val = -1.0f;
                            break;
                        }
                        if (topk_vals[k] < min_topk_val) {
                            min_topk_id = k;
                            min_topk_val = topk_vals[k];
                        }
                    }
                }
            }
        }
    }

    float sum_val = 0.f;
    for (int k = 0; k < TOP_K; ++k) {
        if (topk[k] < 0) continue;
        sum_val += topk_vals[k];
    }
    const float denom = sum_val + EPS;

    for (int c = 0; c < channels; ++c) {
        for (int k = 0; k < TOP_K; ++k) {
            const int32_t g = topk[k];
            if (g < 0) continue;
            const float vis = topk_vals[k] / denom;

            const float2 xy = xys[g];
            const float2 delta = {xy.x - px, xy.y - py};
            const float base = colors[g * channels + c];
            const float grad_x = color_grads[(g * 2 + 0) * channels + c];
            const float grad_y = color_grads[(g * 2 + 1) * channels + c];
            const float color = base + color_grad_scale * (delta.x * grad_x + delta.y * grad_y);
            pix_out[c] += color * vis;
        }
    }
    if (emission != nullptr) {
        for (int c = 0; c < channels; ++c) {
            for (int k = 0; k < TOP_K; ++k) {
                const int32_t g = topk[k];
                if (g < 0) continue;
                pix_out[c] += emission[g * channels + c] * topk_peak[k];
            }
        }
    }

    if (inside) {
        for (int c = 0; c < channels; ++c) {
            out_img[pix_id * channels + c] = pix_out[c];
        }
        for (int k = 0; k < TOP_K; ++k) {
            pixel_topk[pix_id * TOP_K + k] = topk[k];
        }
    }

    maybe_write_stats(inside, pix_id, topk, topk_vals, denom, out_weights_sum, out_entropy);
}

__global__ void nd_rasterize_forward_topk_norm_hexels_affine(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float4* __restrict__ affines,
    const float* __restrict__ colors,
    const float aa_kernel_size,
    const float* __restrict__ softnesses,
    const int softness_stride,
    const float softness_default,
    const float* __restrict__ sharpness,
    const float* __restrict__ emission,
    float* __restrict__ out_img,
    int* __restrict__ pixel_topk,
    float* __restrict__ out_weights_sum,
    float* __restrict__ out_entropy
) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_bounds.x + block.group_index().x;
    const unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = i * img_size.x + j;
    const bool inside = (i < img_size.y && j < img_size.x);
    const bool done = !inside;

    const int2 range = tile_bins[tile_id];
    if (range.x == range.y) {
        if (inside) {
            for (int c = 0; c < channels; ++c) out_img[pix_id * channels + c] = 0.f;
            for (int k = 0; k < TOP_K; ++k) pixel_topk[pix_id * TOP_K + k] = -1;
            if (out_weights_sum != nullptr) out_weights_sum[pix_id] = EPS;
            if (out_entropy != nullptr) out_entropy[pix_id] = 0.f;
        }
        return;
    }
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float4 affine_batch[BLOCK_SIZE];
    __shared__ float sharpness_batch[BLOCK_SIZE];
    __shared__ float3 softness_batch[BLOCK_SIZE];

    const int tr = block.thread_rank();

    float pix_out[MAX_CHANNELS] = {0.f};

    int32_t topk[TOP_K];
    float topk_vals[TOP_K] = {0.f};
    float topk_peak[TOP_K] = {0.f};
    for (int k = 0; k < TOP_K; ++k) {
        topk[k] = -1;
    }

    int32_t min_topk_id = 0;
    float min_topk_val = -1.f;

    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        const int batch_start = range.x + BLOCK_SIZE * b;
        const int idx = batch_start + tr;
        if (idx < range.y) {
            const int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            affine_batch[tr] = affines[g_id];
            sharpness_batch[tr] = (sharpness != nullptr) ? sharpness[g_id] : 0.f;
            float3 s3 = {softness_default, softness_default, softness_default};
            if (softnesses != nullptr) {
                if (softness_stride == 3) {
                    const float* s = softnesses + g_id * 3;
                    s3 = {s[0], s[1], s[2]};
                } else if (softness_stride == 1) {
                    const float s = softnesses[g_id];
                    s3 = {s, s, s};
                }
            }
            softness_batch[tr] = s3;
        }

        block.sync();

        const int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && inside; ++t) {
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            float peak = 0.f;
            const float alpha = hexel_alpha_affine(
                delta,
                affine_batch[t],
                aa_kernel_size,
                softness_batch[t],
                sharpness_batch[t],
                peak
            );

            if (alpha > min_topk_val) {
                topk[min_topk_id] = id_batch[t];
                topk_vals[min_topk_id] = alpha;
                topk_peak[min_topk_id] = peak;
                if (min_topk_val < 0.f && min_topk_id + 1 < TOP_K) {
                    ++min_topk_id;
                } else {
                    min_topk_val = 1e30f;
                    for (int32_t k = 0; k < TOP_K; ++k) {
                        if (topk[k] < 0) {
                            min_topk_id = k;
                            min_topk_val = -1.0f;
                            break;
                        }
                        if (topk_vals[k] < min_topk_val) {
                            min_topk_id = k;
                            min_topk_val = topk_vals[k];
                        }
                    }
                }
            }
        }
    }

    float sum_val = 0.f;
    for (int k = 0; k < TOP_K; ++k) {
        if (topk[k] < 0) continue;
        sum_val += topk_vals[k];
    }
    const float denom = sum_val + EPS;

    for (int c = 0; c < channels; ++c) {
        for (int k = 0; k < TOP_K; ++k) {
            const int32_t g = topk[k];
            if (g < 0) continue;
            const float vis = topk_vals[k] / denom;
            pix_out[c] += colors[g * channels + c] * vis;
        }
    }
    if (emission != nullptr) {
        for (int c = 0; c < channels; ++c) {
            for (int k = 0; k < TOP_K; ++k) {
                const int32_t g = topk[k];
                if (g < 0) continue;
                pix_out[c] += emission[g * channels + c] * topk_peak[k];
            }
        }
    }

    if (inside) {
        for (int c = 0; c < channels; ++c) {
            out_img[pix_id * channels + c] = pix_out[c];
        }
        for (int k = 0; k < TOP_K; ++k) {
            pixel_topk[pix_id * TOP_K + k] = topk[k];
        }
    }

    maybe_write_stats(inside, pix_id, topk, topk_vals, denom, out_weights_sum, out_entropy);
}

__global__ void nd_rasterize_forward_topk_norm_hexels_affine_color_grad(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float4* __restrict__ affines,
    const float* __restrict__ colors,
    const float* __restrict__ color_grads,
    const float color_grad_scale,
    const float aa_kernel_size,
    const float* __restrict__ softnesses,
    const int softness_stride,
    const float softness_default,
    const float* __restrict__ sharpness,
    const float* __restrict__ emission,
    float* __restrict__ out_img,
    int* __restrict__ pixel_topk,
    float* __restrict__ out_weights_sum,
    float* __restrict__ out_entropy
) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_bounds.x + block.group_index().x;
    const unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = i * img_size.x + j;
    const bool inside = (i < img_size.y && j < img_size.x);
    const bool done = !inside;

    const int2 range = tile_bins[tile_id];
    if (range.x == range.y) {
        if (inside) {
            for (int c = 0; c < channels; ++c) out_img[pix_id * channels + c] = 0.f;
            for (int k = 0; k < TOP_K; ++k) pixel_topk[pix_id * TOP_K + k] = -1;
            if (out_weights_sum != nullptr) out_weights_sum[pix_id] = EPS;
            if (out_entropy != nullptr) out_entropy[pix_id] = 0.f;
        }
        return;
    }
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float4 affine_batch[BLOCK_SIZE];
    __shared__ float sharpness_batch[BLOCK_SIZE];
    __shared__ float3 softness_batch[BLOCK_SIZE];

    const int tr = block.thread_rank();

    float pix_out[MAX_CHANNELS] = {0.f};

    int32_t topk[TOP_K];
    float topk_vals[TOP_K] = {0.f};
    float topk_peak[TOP_K] = {0.f};
    for (int k = 0; k < TOP_K; ++k) {
        topk[k] = -1;
    }

    int32_t min_topk_id = 0;
    float min_topk_val = -1.f;

    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        const int batch_start = range.x + BLOCK_SIZE * b;
        const int idx = batch_start + tr;
        if (idx < range.y) {
            const int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            affine_batch[tr] = affines[g_id];
            sharpness_batch[tr] = (sharpness != nullptr) ? sharpness[g_id] : 0.f;
            float3 s3 = {softness_default, softness_default, softness_default};
            if (softnesses != nullptr) {
                if (softness_stride == 3) {
                    const float* s = softnesses + g_id * 3;
                    s3 = {s[0], s[1], s[2]};
                } else if (softness_stride == 1) {
                    const float s = softnesses[g_id];
                    s3 = {s, s, s};
                }
            }
            softness_batch[tr] = s3;
        }

        block.sync();

        const int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && inside; ++t) {
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            float peak = 0.f;
            const float alpha = hexel_alpha_affine(
                delta,
                affine_batch[t],
                aa_kernel_size,
                softness_batch[t],
                sharpness_batch[t],
                peak
            );

            if (alpha > min_topk_val) {
                topk[min_topk_id] = id_batch[t];
                topk_vals[min_topk_id] = alpha;
                topk_peak[min_topk_id] = peak;
                if (min_topk_val < 0.f && min_topk_id + 1 < TOP_K) {
                    ++min_topk_id;
                } else {
                    min_topk_val = 1e30f;
                    for (int32_t k = 0; k < TOP_K; ++k) {
                        if (topk[k] < 0) {
                            min_topk_id = k;
                            min_topk_val = -1.0f;
                            break;
                        }
                        if (topk_vals[k] < min_topk_val) {
                            min_topk_id = k;
                            min_topk_val = topk_vals[k];
                        }
                    }
                }
            }
        }
    }

    float sum_val = 0.f;
    for (int k = 0; k < TOP_K; ++k) {
        if (topk[k] < 0) continue;
        sum_val += topk_vals[k];
    }
    const float denom = sum_val + EPS;

    for (int c = 0; c < channels; ++c) {
        for (int k = 0; k < TOP_K; ++k) {
            const int32_t g = topk[k];
            if (g < 0) continue;
            const float vis = topk_vals[k] / denom;

            const float2 xy = xys[g];
            const float2 delta = {xy.x - px, xy.y - py};
            const float base = colors[g * channels + c];
            const float grad_x = color_grads[(g * 2 + 0) * channels + c];
            const float grad_y = color_grads[(g * 2 + 1) * channels + c];
            const float color = base + color_grad_scale * (delta.x * grad_x + delta.y * grad_y);
            pix_out[c] += color * vis;
        }
    }
    if (emission != nullptr) {
        for (int c = 0; c < channels; ++c) {
            for (int k = 0; k < TOP_K; ++k) {
                const int32_t g = topk[k];
                if (g < 0) continue;
                pix_out[c] += emission[g * channels + c] * topk_peak[k];
            }
        }
    }

    if (inside) {
        for (int c = 0; c < channels; ++c) {
            out_img[pix_id * channels + c] = pix_out[c];
        }
        for (int k = 0; k < TOP_K; ++k) {
            pixel_topk[pix_id * TOP_K + k] = topk[k];
        }
    }

    maybe_write_stats(inside, pix_id, topk, topk_vals, denom, out_weights_sum, out_entropy);
}
