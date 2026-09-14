// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Decode the same native-fragment NVFP4 representation consumed by hybrid.cu.
// No original BF16 matrix is needed or retained by the caller.
__global__ void nvfp4_dense_dequant_kernel(const uint32_t* packed,
                                           const uint32_t* block_scales,
                                           const float* global_scale,
                                           __nv_bfloat16* output, int N,
                                           int K) {
  int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (int64_t)N * (K / 8)) return;
  int row = index / (K / 8), col = (index % (K / 8)) * 8, groups = K / 64;
  int fragment = (row % 16) / 8 + 2 * ((col % 64) / 32);
  int lane = (row % 8) * 4 + (col % 32) / 8;
  int64_t word_index =
      (((int64_t)(row / 16) * groups + col / 64) * 4 + fragment) * 32 + lane;
  uint32_t word = packed[word_index];
  uint32_t scale = (block_scales[(int64_t)row * groups + col / 64] >>
                    (8 * ((col % 64) / 16))) &
                   255;
  int exponent = (scale >> 3) & 15;
  int mantissa = scale & 7;
  float block_scale =
      exponent == 0
          ? mantissa * 0x1p-9f
          : (1.f + mantissa * .125f) * __uint_as_float((exponent + 120) << 23);
  // Scales produced by the encoder are finite and nonnegative E4M3 values.
  __nv_bfloat16 values[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint32_t code = (word >> (4 * i)) & 15;
    int magnitude = code & 7;
    float value = magnitude < 2
                      ? magnitude * .5f
                      : (1.f + .5f * (magnitude & 1)) *
                            __uint_as_float(((magnitude >> 1) + 126) << 23);
    if (code & 8) value = -value;
    values[i] = __float2bfloat16_rn(value * block_scale * global_scale[0]);
  }
  reinterpret_cast<uint4*>(output)[index] = *reinterpret_cast<uint4*>(values);
}

// C ABI: packed:u32[1,N/16,K/64,4,32], block_scales:u32[N,K/64],
// global_scale:f32[1], output:bf16[N,K], N%16==0, K%64==0, CUDA stream.
extern "C" int nvfp4_dense_dequant(const void* packed, const void* block_scales,
                                   const void* global_scale, void* output,
                                   int N, int K, void* stream) {
  if (N <= 0 || K <= 0 || N % 16 || K % 64) return (int)cudaErrorInvalidValue;
  nvfp4_dense_dequant_kernel<<<((int64_t)N * (K / 8) + 255) / 256, 256, 0,
                               (cudaStream_t)stream>>>(
      (const uint32_t*)packed, (const uint32_t*)block_scales,
      (const float*)global_scale, (__nv_bfloat16*)output, N, K);
  return (int)cudaGetLastError();
}
