// src/kernels/cuda/kv_fp8.cu - Native FP8 (E4M3 / E5M2) KV storage for QSA layers.
// See include/strata/kernels/kv_fp8.hpp.
#include "strata/kernels/kv_fp8.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_fp8: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_FP8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_fp8: %s: head_dim %lld must be a multiple of %d\n", what, (long long) s.head_dim,
                     KV_FP8_GROUP);
        std::exit(1);
    }
}

// Convert float to FP8 E4M3 byte
__device__ __forceinline__ uint8_t cvt_f32_to_e4m3(float x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_e4m3 v(x);
    return v.__x;
#else
    if (isnan(x)) return 0x7F;
    const uint32_t b = *reinterpret_cast<const uint32_t*>(&x);
    const uint8_t sign = (b >> 24) & 0x80;
    float ax = fabsf(x);
    if (ax > 448.0f) ax = 448.0f;
    if (ax < 0.001953125f) return sign;
    int exp;
    float mant = frexpf(ax, &exp);
    int e = exp + 6;
    if (e <= 0) {
        int m = __float2int_rn(ax * 512.0f);
        return sign | (uint8_t)(m > 7 ? 7 : m);
    }
    if (e >= 15) return sign | 0x7E;
    int m = __float2int_rn((mant * 2.0f - 1.0f) * 8.0f);
    if (m >= 8) { m = 0; ++e; }
    if (e >= 15) return sign | 0x7E;
    return sign | (uint8_t)((e << 3) | m);
#endif
}

// Convert float to FP8 E5M2 byte
__device__ __forceinline__ uint8_t cvt_f32_to_e5m2(float x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_e5m2 v(x);
    return v.__x;
#else
    if (isnan(x)) return 0x7F;
    const uint32_t b = *reinterpret_cast<const uint32_t*>(&x);
    const uint8_t sign = (b >> 24) & 0x80;
    float ax = fabsf(x);
    if (ax > 57344.0f) ax = 57344.0f;
    if (ax < 0.000015258789f) return sign;
    int exp;
    float mant = frexpf(ax, &exp);
    int e = exp + 14;
    if (e <= 0) {
        int m = __float2int_rn(ax * 65536.0f);
        return sign | (uint8_t)(m > 3 ? 3 : m);
    }
    if (e >= 31) return sign | 0x7B;
    int m = __float2int_rn((mant * 2.0f - 1.0f) * 4.0f);
    if (m >= 4) { m = 0; ++e; }
    if (e >= 31) return sign | 0x7B;
    return sign | (uint8_t)((e << 2) | m);
#endif
}

// Convert FP8 E4M3 byte to float
__device__ __forceinline__ float cvt_e4m3_to_f32(uint8_t byte) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_e4m3 v;
    v.__x = byte;
    return float(v);
#else
    const int s = byte >> 7, e = (byte >> 3) & 15, m = byte & 7;
    if (e == 15 && m == 7) return 0.0f;
    float f = (e == 0) ? ((float) m / 8.0f) * (1.0f / 64.0f)
                       : (1.0f + (float) m / 8.0f) * __powf(2.0f, (float)(e - 7));
    return s ? -f : f;
#endif
}

// Convert FP8 E5M2 byte to float
__device__ __forceinline__ float cvt_e5m2_to_f32(uint8_t byte) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_e5m2 v;
    v.__x = byte;
    return float(v);
#else
    const int s = byte >> 7, e = (byte >> 2) & 31, m = byte & 3;
    if (e == 31 && m != 0) return 0.0f;
    float f = (e == 0) ? ((float) m / 4.0f) * (1.0f / 16384.0f)
                       : (1.0f + (float) m / 4.0f) * __powf(2.0f, (float)(e - 15));
    return s ? -f : f;
#endif
}

template <bool E5M2>
__device__ __forceinline__ uint32_t fp8_pair_to_f16x2(const uint8_t* c, float s) {
    float f0, f1;
    if constexpr (E5M2) {
        f0 = cvt_e5m2_to_f32(c[0]) * s;
        f1 = cvt_e5m2_to_f32(c[1]) * s;
    } else {
        f0 = cvt_e4m3_to_f32(c[0]) * s;
        f1 = cvt_e4m3_to_f32(c[1]) * s;
    }
    return (uint32_t) f16_from_f32(f0) | ((uint32_t) f16_from_f32(f1) << 16);
}

// One block = one 64-value group of one KV head of K (blockIdx.z = 0) or V (1); 64 threads, one value each.
// MULTI (S26 STRATA_LFUSE): blockIdx.z = 2 * token + (K / V); token j reads step + j * step_stride
template <bool MULTI = false, bool E5M2 = false>
__global__ void kv_append_fp8_kernel(uint8_t* __restrict__ k_fp8, uint8_t* __restrict__ v_fp8,
                                     uint16_t* __restrict__ k_scale, uint16_t* __restrict__ v_scale,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                     const float* __restrict__ kcur, const float* __restrict__ vcur, int kv_heads,
                                     int head_dim, int page_size, KvHostPools host, int step_stride = 0,
                                     int cur_stride = 0) {
    if constexpr (MULTI) {
        const int j = blockIdx.z >> 1;
        step += (size_t) j * step_stride; kcur += (size_t) j * cur_stride; vcur += (size_t) j * cur_stride;
    }
    const long long pos = (long long) __ldg(step + kStepPos);
    const int h = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    const bool is_v = MULTI ? (blockIdx.z & 1) == 1 : blockIdx.z == 1;
    const int groups = head_dim / KV_FP8_GROUP;
    const float x = (is_v ? vcur : kcur)[h * head_dim + g * KV_FP8_GROUP + t];

    // max |x| over the 64 values: two warps, then combine through shared memory in a fixed order
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float warp_max[2];
    if ((t & 31) == 0) warp_max[t >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(warp_max[0], warp_max[1]);

    constexpr float max_target = E5M2 ? 57344.0f : 448.0f;
    const uint16_t sbits = f16_from_f32(amax / max_target);
    const float sf = f32_from_f16(sbits);
    uint8_t q = 0;
    if (sf > 0.0f) {
        const float scaled_x = x / sf;
        if constexpr (E5M2) q = cvt_f32_to_e5m2(scaled_x);
        else q = cvt_f32_to_e4m3(scaled_x);
    }

    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_fp8 : k_fp8)[row * head_dim + g * KV_FP8_GROUP + t] = q;
        if (t == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (host.k_fp8 != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host.v_fp8 : host.k_fp8)[row * head_dim + g * KV_FP8_GROUP + t] = q;
        if (t == 0) (is_v ? host.v_fp8_scale : host.k_fp8_scale)[row * groups + g] = sbits;
    }
}

// Prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool when given.
template <bool E5M2 = false>
__global__ void kv_append_fp8_batch_kernel(uint8_t* __restrict__ k_fp8, uint8_t* __restrict__ v_fp8,
                                           uint16_t* __restrict__ k_scale, uint16_t* __restrict__ v_scale,
                                           const int32_t* __restrict__ table, int64_t pos0,
                                           const float* __restrict__ K, const float* __restrict__ V,
                                           int kv_heads, int head_dim, int page_size, int is_v_grid,
                                           KvHostPools host, KvHostPools stage) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, g = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const int groups = head_dim / KV_FP8_GROUP;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + g * KV_FP8_GROUP + th];

    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float warp_max[2];
    if ((th & 31) == 0) warp_max[th >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(warp_max[0], warp_max[1]);

    constexpr float max_target = E5M2 ? 57344.0f : 448.0f;
    const uint16_t sbits = f16_from_f32(amax / max_target);
    const float sf = f32_from_f16(sbits);
    uint8_t q = 0;
    if (sf > 0.0f) {
        const float scaled_x = x / sf;
        if constexpr (E5M2) q = cvt_f32_to_e5m2(scaled_x);
        else q = cvt_f32_to_e4m3(scaled_x);
    }

    const long long page = (long long) table[pos / page_size];
    const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) {
        const long long prow = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_fp8 : k_fp8)[prow * head_dim + g * KV_FP8_GROUP + th] = q;
        if (th == 0) (is_v ? v_scale : k_scale)[prow * groups + g] = sbits;
    }
    if (host.k_fp8 != nullptr) {
        (is_v ? host.v_fp8 : host.k_fp8)[row * head_dim + g * KV_FP8_GROUP + th] = q;
        if (th == 0) (is_v ? host.v_fp8_scale : host.k_fp8_scale)[row * groups + g] = sbits;
    }
    if (stage.k_fp8 != nullptr) {
        (is_v ? stage.v_fp8 : stage.k_fp8)[row * head_dim + g * KV_FP8_GROUP + th] = q;
        if (th == 0) (is_v ? stage.v_fp8_scale : stage.k_fp8_scale)[row * groups + g] = sbits;
    }
}

// Gather step[kStepWidth] cells named by ids into FP16 scratch [id][kv_head][head_dim].
template <bool E5M2 = false>
__global__ void kv_gather_fp8_kernel(const uint8_t* __restrict__ k_fp8, const uint8_t* __restrict__ v_fp8,
                                     const uint16_t* __restrict__ k_scale, const uint16_t* __restrict__ v_scale,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                     const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                     uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int per = head_dim / 8;
    const long long total = n_ids * kv_heads * per;
    const long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long) per);
    const int rem = (int) (i % (kv_heads * (long long) per));
    const int h = rem / per, q8 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q8 * 8;
    const int groups = head_dim / KV_FP8_GROUP;
    const float ks = f32_from_f16(k_scale[row * groups + d / KV_FP8_GROUP]);
    const float vs = f32_from_f16(v_scale[row * groups + d / KV_FP8_GROUP]);

    const uint2 kraw = *reinterpret_cast<const uint2*>(k_fp8 + row * head_dim + d);
    const uint2 vraw = *reinterpret_cast<const uint2*>(v_fp8 + row * head_dim + d);
    const uint8_t* kc = reinterpret_cast<const uint8_t*>(&kraw);
    const uint8_t* vc = reinterpret_cast<const uint8_t*>(&vraw);

    uint4 ko, vo;
    ko.x = fp8_pair_to_f16x2<E5M2>(kc + 0, ks); ko.y = fp8_pair_to_f16x2<E5M2>(kc + 2, ks);
    ko.z = fp8_pair_to_f16x2<E5M2>(kc + 4, ks); ko.w = fp8_pair_to_f16x2<E5M2>(kc + 6, ks);
    vo.x = fp8_pair_to_f16x2<E5M2>(vc + 0, vs); vo.y = fp8_pair_to_f16x2<E5M2>(vc + 2, vs);
    vo.z = fp8_pair_to_f16x2<E5M2>(vc + 4, vs); vo.w = fp8_pair_to_f16x2<E5M2>(vc + 6, vs);

    const long long dst = (id * kv_heads + h) * (long long) per + q8;
    reinterpret_cast<uint4*>(k_scratch)[dst] = ko;
    reinterpret_cast<uint4*>(v_scratch)[dst] = vo;
}

}  // namespace

void kv_append_fp8_step(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale,
                        const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur,
                        const QsaShapes& s, void* stream, const KvHostPools* host, bool e5m2) {
    validate(s, "kv_append_fp8_step");
    const int groups = (int) (s.head_dim / KV_FP8_GROUP);
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) groups, 2);
    const KvHostPools h = host != nullptr ? *host : KvHostPools{};
    if (e5m2) {
        kv_append_fp8_kernel<false, true><<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, h);
    } else {
        kv_append_fp8_kernel<false, false><<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, h);
    }
    check("kv_append_fp8_step");
}

void kv_append_fp8_steps(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale,
                         const int32_t* page_table, const int32_t* step, int step_stride, const float* kcur,
                         const float* vcur, int cur_stride, int n_tok, const QsaShapes& s, void* stream,
                         const KvHostPools* host, bool e5m2) {
    if (n_tok <= 0) return;
    validate(s, "kv_append_fp8_steps");
    const int groups = (int) (s.head_dim / KV_FP8_GROUP);
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) groups, (unsigned) (2 * n_tok));
    const KvHostPools h = host != nullptr ? *host : KvHostPools{};
    if (e5m2) {
        kv_append_fp8_kernel<true, true><<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, h, step_stride, cur_stride);
    } else {
        kv_append_fp8_kernel<true, false><<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, h, step_stride, cur_stride);
    }
    check("kv_append_fp8_steps");
}

void kv_append_fp8(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale,
                   const int32_t* page_table, int64_t pos0, int64_t T, const float* K, const float* V,
                   const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage, bool e5m2) {
    if (T <= 0) return;
    validate(s, "kv_append_fp8");
    const int groups = (int) (s.head_dim / KV_FP8_GROUP);
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) groups);
    const KvHostPools h = host != nullptr ? *host : KvHostPools{};
    const KvHostPools stg = stage != nullptr ? *stage : KvHostPools{};

    if (e5m2) {
        kv_append_fp8_batch_kernel<true><<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, 0, h, stg);
        kv_append_fp8_batch_kernel<true><<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, 1, h, stg);
    } else {
        kv_append_fp8_batch_kernel<false><<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, 0, h, stg);
        kv_append_fp8_batch_kernel<false><<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, 1, h, stg);
    }
    check("kv_append_fp8");
}

void kv_gather_fp8_step(const uint8_t* k_fp8, const uint8_t* v_fp8, const uint16_t* k_scale, const uint16_t* v_scale,
                        const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                        const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream, bool e5m2) {
    if (max_ids <= 0) return;
    validate(s, "kv_gather_fp8_step");
    constexpr int block_threads = 256;
    const int per = (int) (s.head_dim / 8);
    const long long total = max_ids * s.n_head_kv * per;
    const unsigned blocks = (unsigned) ((total + block_threads - 1) / block_threads);
    if (e5m2) {
        kv_gather_fp8_kernel<true><<<blocks, block_threads, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, k_scratch, v_scratch);
    } else {
        kv_gather_fp8_kernel<false><<<blocks, block_threads, 0, (cudaStream_t) stream>>>(
            k_fp8, v_fp8, k_scale, v_scale, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, k_scratch, v_scratch);
    }
    check("kv_gather_fp8_step");
}

}  // namespace strata::kernels
