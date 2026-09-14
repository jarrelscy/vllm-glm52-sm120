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

__device__ __forceinline__ void dense_paired_mma(
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

// Consecutive dense tokens share the same matrix. Four columns per token retain
// all four residual activation planes; this does not support routed experts.
__global__ void nvfp4_dense_paired_kernel(const unsigned* hw,
                                          const unsigned* hs, const unsigned* x,
                                          const unsigned* xs, float* partial,
                                          int N, int G, int S, int slots) {
  int slot = blockIdx.z * 2;
  int lane = threadIdx.x & 31, q = lane / 4, c = lane % 4;
  int tile = blockIdx.x * 4 + threadIdx.x / 32;
  if (tile >= N / 16) return;
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  for (int g = G * blockIdx.y / S; g < G * (blockIdx.y + 1) / S; g++) {
    int input_slot = slot + q / 4, plane = q % 4;
    bool valid = input_slot < slots;
    int64_t xi = ((int64_t)input_slot * 4 + plane) * G + g;
    unsigned b0 = valid ? x[xi * 8 + c] : 0, b1 = valid ? x[xi * 8 + 4 + c] : 0;
    unsigned sb = valid ? xs[xi] : 0x38383838u;
    unsigned a[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
      a[j] = hw[(((int64_t)tile * G + g) * 4 + j) * 32 + lane];
    unsigned sa = hs[(int64_t)(tile * 16 + q + 8 * (c & 1)) * G + g];
    dense_paired_mma(d0, d1, d2, d3, a[0], a[1], a[2], a[3], b0, b1, sa, sb);
  }
  float a = ldexpf(d0, -8 * (c % 2)) + ldexpf(d1, -8 * (c % 2) - 4);
  float b = ldexpf(d2, -8 * (c % 2)) + ldexpf(d3, -8 * (c % 2) - 4);
  a += __shfl_xor_sync(0xffffffff, a, 1);
  b += __shfl_xor_sync(0xffffffff, b, 1);
  slot += c / 2;
  if ((c == 0 || c == 2) && slot < slots) {
    partial[((int64_t)slot * N + tile * 16 + q) * S + blockIdx.y] = a;
    partial[((int64_t)slot * N + tile * 16 + q + 8) * S + blockIdx.y] = b;
  }
}
__global__ void nvfp4_dense_paired_reduce(const float* partial, float* out,
                                          const float* global, int N, int slots,
                                          int S) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)N * slots) return;
  float v = 0;
  for (int s = 0; s < S; s++) v += partial[i * S + s];
  out[i] = v * global[0];
}
// C ABI: packed weights/scales as dequant above; x:u32[M,4,K/8],
// xs:u8[M,4,K/16], partial:f32[M,N,split], out:f32[M,N].
extern "C" int nvfp4_dense_paired(const void* hw, const void* hs,
                                  const void* global, const void* x,
                                  const void* xs, void* partial, void* out,
                                  int N, int K, int slots, int split,
                                  void* stream) {
  if (N <= 0 || N % 16 || K <= 0 || K % 64 || slots <= 0 || split <= 0)
    return (int)cudaErrorInvalidValue;
  cudaStream_t s = (cudaStream_t)stream;
  nvfp4_dense_paired_kernel<<<dim3((N + 63) / 64, split, (slots + 1) / 2), 128,
                              0, s>>>((const unsigned*)hw, (const unsigned*)hs,
                                      (const unsigned*)x, (const unsigned*)xs,
                                      (float*)partial, N, K / 64, split, slots);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  nvfp4_dense_paired_reduce<<<((int64_t)N * slots + 255) / 256, 256, 0, s>>>(
      (const float*)partial, (float*)out, (const float*)global, N, slots,
      split);
  return (int)cudaGetLastError();
}
