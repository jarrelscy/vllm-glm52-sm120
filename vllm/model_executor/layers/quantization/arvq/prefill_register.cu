// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Exact hot prefill: reuse each weight fragment across four token pairs.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// SM120 MMA uses weights as the 16-row operand and P4 planes as columns.
__device__ __forceinline__ void register_mma(
    float& d0, float& d1, float& d2, float& d3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1,
    unsigned sa = 0x38383838u, unsigned sb = 0x38383838u) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, "
      "{%11}, {0,0};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// Reuse the A fragment within a warp across independent token-pair
// accumulators.
template <int PAIRS>
__global__ void register_kernel(const unsigned* hw, const unsigned* hs,
                                const unsigned* x, const unsigned* xs,
                                const int* hot_ids, const int* groups,
                                float* partial, int N, int G, int S,
                                int slots) {
  int group = blockIdx.z;
  const int* desc = groups + (group / 5) * 11;
  int index = group % 5;
  if (index >= desc[0]) return;
  int first = desc[1 + 2 * index], count = desc[2 + 2 * index];
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  int q = lane / 4, c = lane % 4;
  int tile = blockIdx.x * 4 + warp % 4;
  if (tile >= N / 16) return;
  int pair_base = (warp / 4) * PAIRS;
  int expert = hot_ids[first];
  float d0[PAIRS] = {}, d1[PAIRS] = {}, d2[PAIRS] = {}, d3[PAIRS] = {};
  for (int g = G * blockIdx.y / S; g < G * (blockIdx.y + 1) / S; ++g) {
    unsigned a[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      a[j] =
          hw[(((((long long)expert * (N / 16) + tile) * G + g) * 4 + j) * 32) +
             lane];
    unsigned sa =
        hs[((long long)expert * N + tile * 16 + q + 8 * (c & 1)) * G + g];
#pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
      int slot = first + 2 * (pair_base + p);
      bool valid_slot = 2 * (pair_base + p) < count;
      int partner = 2 * (pair_base + p) + 1 < count ? slot + 1 : -1;
      int input_slot = partner >= 0 && q >= 4 ? partner : slot;
      int plane = partner >= 0 ? q % 4 : q;
      bool valid = valid_slot && plane < 4;
      unsigned b0 =
          valid ? x[(((long long)input_slot * 4 + plane) * G + g) * 8 + c] : 0;
      unsigned b1 =
          valid ? x[(((long long)input_slot * 4 + plane) * G + g) * 8 + 4 + c]
                : 0;
      unsigned sb =
          valid ? xs[((long long)input_slot * 4 + plane) * G + g] : 0x38383838u;
      register_mma(d0[p], d1[p], d2[p], d3[p], a[0], a[1], a[2], a[3], b0, b1,
                   sa, sb);
    }
  }
#pragma unroll
  for (int p = 0; p < PAIRS; ++p) {
    int slot = first + 2 * (pair_base + p);
    bool valid_slot = 2 * (pair_base + p) < count;
    int partner = 2 * (pair_base + p) + 1 < count ? slot + 1 : -1;
    if (partner < 0) {
      float a = ldexpf(d0[p], -8 * c) + ldexpf(d1[p], -8 * c - 4);
      float b = ldexpf(d2[p], -8 * c) + ldexpf(d3[p], -8 * c - 4);
      a += __shfl_xor_sync(0xffffffff, a, 1);
      a += __shfl_xor_sync(0xffffffff, a, 2);
      b += __shfl_xor_sync(0xffffffff, b, 1);
      b += __shfl_xor_sync(0xffffffff, b, 2);
      if (valid_slot && c == 0) {
        partial[((long long)slot * N + tile * 16 + q) * S + blockIdx.y] = a;
        partial[((long long)slot * N + tile * 16 + q + 8) * S + blockIdx.y] = b;
      }
    } else {
      float a = ldexpf(d0[p], -8 * (c % 2)) + ldexpf(d1[p], -8 * (c % 2) - 4);
      float b = ldexpf(d2[p], -8 * (c % 2)) + ldexpf(d3[p], -8 * (c % 2) - 4);
      a += __shfl_xor_sync(0xffffffff, a, 1);
      b += __shfl_xor_sync(0xffffffff, b, 1);
      int output_slot = c < 2 ? slot : partner;
      if (c == 0 || c == 2) {
        partial[((long long)output_slot * N + tile * 16 + q) * S + blockIdx.y] =
            a;
        partial[((long long)output_slot * N + tile * 16 + q + 8) * S +
                blockIdx.y] = b;
      }
    }
  }
}
__global__ void register_reduce(const float* p, float* y, const int* cold_ids,
                                const int* hot_ids, const float* hot_global,
                                float cold_global, int N, int slots, int S,
                                int hot_parts) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)N * slots) return;
  int slot = i / N, row = i % N;
  float v = 0;
  for (int s = 0; s < S; s++) v += p[i * S + s];
  float scale = 0;
  if (cold_ids[slot] >= 0)
    scale = cold_global;
  else if (hot_ids[slot] >= 0)
    scale = hot_global[(long long)hot_ids[slot] * hot_parts +
                       row / (N / hot_parts)];
  y[i] = v * scale;
}

extern "C" int wide_launch_register(const void* hw, const void* hs,
                                    const void* global, const void* x,
                                    const void* xs, const void* cold,
                                    const void* hot, const void* groups,
                                    void* partial, void* out, int N, int K,
                                    int slots, int split, int parts, int shared,
                                    void* stream) {
  if (N <= 0 || N % 16 || K <= 0 || K % 128 || slots <= 0 || split <= 0)
    return (int)cudaErrorInvalidValue;
  cudaStream_t s = (cudaStream_t)stream;
  dim3 grid((N + 63) / 64, split, ((slots + 31) / 32) * 5);
  register_kernel<4><<<grid, 128, 0, s>>>(
      (const unsigned*)hw, (const unsigned*)hs, (const unsigned*)x,
      (const unsigned*)xs, (const int*)hot, (const int*)groups, (float*)partial,
      N, K / 64, split, slots);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  register_reduce<<<((long long)slots * N + 255) / 256, 256, 0, s>>>(
      (const float*)partial, (float*)out, (const int*)cold, (const int*)hot,
      (const float*)global, 0.f, N, slots, split, parts);
  return (int)cudaGetLastError();
}
