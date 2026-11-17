// src/kernels/ple_cp2_dequant_test.cpp - STRATA_PLE_COMPACT: ple_cp2_dequant (raw CP2 PLE rows to floats on the
// device) against cp2_dequant_row (the host decode), bit for bit. Synthetic bytes, no model.
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
using namespace strata::kernels;

int check(const std::vector<uint8_t>& raw, int64_t rows, const char* what) {
    float* dout = nullptr;
    uint8_t* draw = nullptr;
    if (cudaMalloc((void**) &dout, (size_t) rows * PLE_HEAD_DIM * 4) != cudaSuccess ||
        cudaMalloc((void**) &draw, raw.size()) != cudaSuccess ||
        cudaMemcpy(draw, raw.data(), raw.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::printf("CUDA setup failed for %s\n", what);
        return 1;
    }
    cudaMemset(dout, 0xff, (size_t) rows * PLE_HEAD_DIM * 4);
    ple_cp2_dequant(draw, rows, dout, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("CUDA error for %s: %s\n", what, cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    std::vector<float> got((size_t) rows * PLE_HEAD_DIM), want(got.size());
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(dout);
    cudaFree(draw);
    for (int64_t r = 0; r < rows; ++r) {
        cp2_dequant_row(raw.data() + (size_t) r * PLE_ROW_BYTES_CP2, 1.0f, want.data() + (size_t) r * PLE_HEAD_DIM);
    }
    long long diff = 0, first = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &got[i], 4);
        std::memcpy(&b, &want[i], 4);
        if (std::isnan(got[i]) && std::isnan(want[i])) continue; // a NaN scale
        if (a != b) {
            if (first < 0) first = (long long) i;
            ++diff;
        }
    }
    std::printf("  %-52s %lld rows: %s\n", what, (long long) rows, diff ? "*** DIFFERS ***" : "bitwise equal");
    if (diff) std::printf("      %lld values differ, first at %lld\n", diff, first);
    return diff ? 1 : 0;
}
} // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        std::printf("no CUDA device (skipped)\n");
        return 77;
    }
    int bad = 0;
    std::mt19937 rng(20261118u);
    {   // all 65,536 bf16 scale patterns
        const int64_t rows = 65536;
        std::vector<uint8_t> raw((size_t) rows * PLE_ROW_BYTES_CP2);
        for (auto& b : raw) b = (uint8_t) (rng() & 0xff);
        for (int64_t r = 0; r < rows; ++r) {
            const uint16_t h = (uint16_t) r;
            std::memcpy(&raw[(size_t) r * PLE_ROW_BYTES_CP2 + 40], &h, 2);
        }
        bad += check(raw, rows, "every bf16 scale pattern, random trellis codes");
    }
    {   // random rows (odd count)
        const int64_t rows = 4096 + 37;
        std::vector<uint8_t> raw((size_t) rows * PLE_ROW_BYTES_CP2);
        for (auto& b : raw) b = (uint8_t) (rng() & 0xff);
        bad += check(raw, rows, "random bytes (odd row count)");
    }
    {   // the out-of-table row of PleTable::gather_raw (42 zeros)
        const int64_t rows = 16;
        std::vector<uint8_t> raw((size_t) rows * PLE_ROW_BYTES_CP2, 0);
        bad += check(raw, rows, "out-of-table row (42 zero bytes)");
        std::vector<float> want(PLE_HEAD_DIM);
        cp2_dequant_row(raw.data(), 1.0f, want.data());
        for (int i = 0; i < PLE_HEAD_DIM; ++i) {
            float v = want[i];
            uint32_t a;
            std::memcpy(&a, &v, 4);
            if (a != 0) {
                std::printf("  *** the out-of-table row at %d: val=%e, bits=0x%08x\n", i, v, a);
                ++bad;
                break;
            }
        }
    }
    std::printf("ple_cp2_dequant_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
