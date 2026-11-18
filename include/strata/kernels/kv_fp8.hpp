// include/strata/kernels/kv_fp8.hpp - Native FP8 (E4M3 / E5M2) KV storage for QSA layers on Blackwell sm_120.
//
// Hardware-accelerated 8-bit floating point Key-Value cache format with per-group-64 FP16 scaling.
// Compared to INT8, FP8 E4M3 provides logarithmic dynamic range with 3 mantissa bits and 4 exponent bits,
// completely eliminating integer quantization truncation in subtle attention tails while consuming
// the exact same compact footprint (1,056 bytes per cell across 2 KV heads, ~12.38 KB / token across all 12 QSA layers).
//
// On NVIDIA Blackwell (sm_120) and Ada (sm_89), conversion to/from float compiles to single-cycle native
// instructions (cvt.rn.satfinite.e4m3x2.f32), with 128-bit vector memory transactions loading 16 values at once.
#pragma once

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int KV_FP8_GROUP = 64;

/// Bytes per cell (one token, one layer): K and V FP8 codes plus their FP16 scales.
inline uint64_t kv_fp8_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * s.head_dim * 2 + (uint64_t) s.n_head_kv * (s.head_dim / KV_FP8_GROUP) * 2 * 2;
}

/// Append the cell at step[kStepPos] (graph-capturable). With a host copy (KV streaming),
/// the cell is mirrored to host memory and only written to VRAM if its block is resident.
void kv_append_fp8_step(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale,
                        const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur,
                        const QsaShapes& s, void* stream, const KvHostPools* host = nullptr, bool e5m2 = false);

/// Batched multi-step append for verification windows (S26 STRATA_LFUSE).
void kv_append_fp8_steps(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale,
                         const int32_t* page_table, const int32_t* step, int step_stride, const float* kcur,
                         const float* vcur, int cur_stride, int n_tok, const QsaShapes& s, void* stream,
                         const KvHostPools* host = nullptr, bool e5m2 = false);

/// The prompt path: T consecutive cells from pos0, K/V [T, n_head_kv, head_dim]; also into staging pool when given.
void kv_append_fp8(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale,
                   const int32_t* page_table, int64_t pos0, int64_t T, const float* K, const float* V,
                   const QsaShapes& s, void* stream, const KvHostPools* host = nullptr,
                   const KvHostPools* stage = nullptr, bool e5m2 = false);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch `[id][kv_head][head_dim]`.
void kv_gather_fp8_step(const uint8_t* k_fp8, const uint8_t* v_fp8, const uint16_t* k_scale, const uint16_t* v_scale,
                        const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                        const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream,
                        bool e5m2 = false);

}  // namespace strata::kernels
