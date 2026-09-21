// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float fp4(unsigned code) {
  unsigned magnitude = code & 7;
  float value = magnitude < 2
                    ? magnitude * .5f
                    : (1.f + .5f * (magnitude & 1)) *
                          __uint_as_float(((magnitude >> 1) + 126) << 23);
  return code & 8 ? -value : value;
}

// RESIDUAL_BOOKS > 1 (mcbook16): codebooks hold one base book plus
// RESIDUAL_BOOKS residual books of 2^RESIDUAL_BITS atoms each; a per-tile
// uint8 selector picks the residual book and book_factors[selector] weights
// its contribution (applied in FP32, exact for arbitrary factors).
template <bool FP16, int RESIDUAL_BITS, int RESIDUAL_BOOKS = 1>
__global__ void arvq_dequant_kernel(const unsigned* packed,
                                    const unsigned* codebooks,
                                    const unsigned char* scales, float global,
                                    void* output, int N, int K,
                                    const unsigned char* sel = nullptr,
                                    const float* bf = nullptr) {
  constexpr int LUT_SIZE = 256 + RESIDUAL_BOOKS * (1 << RESIDUAL_BITS);
  __shared__ unsigned lut[LUT_SIZE];
  __shared__ float factors[RESIDUAL_BOOKS];
  for (int i = threadIdx.x; i < LUT_SIZE; i += blockDim.x)
    lut[i] = codebooks[i];
  if constexpr (RESIDUAL_BOOKS > 1)
    for (int i = threadIdx.x; i < RESIDUAL_BOOKS; i += blockDim.x)
      factors[i] = bf[i];
  __syncthreads();
  int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (int64_t)N * (K / 8)) return;
  int row = index / (K / 8), col = (index % (K / 8)) * 8, G = K / 64;
  int j = (row % 16) / 8 + 2 * ((col % 64) / 32);
  int lane = (row % 8) * 4 + (col % 32) / 8;
  int bit = (j * 32 + lane) * (8 + RESIDUAL_BITS);
  const unsigned* tile =
      packed + ((int64_t)(row / 16) * G + col / 64) * (4 * (8 + RESIDUAL_BITS));
  unsigned pair;
  if constexpr (RESIDUAL_BITS == 8)
    pair = tile[bit / 32] >> (bit % 32);
  else
    pair = __funnelshift_r(tile[bit / 32], tile[bit / 32 + 1], bit % 32);
  unsigned m = 0;
  if constexpr (RESIDUAL_BOOKS > 1)
    m = sel[(int64_t)(row / 16) * G + col / 64];
  unsigned first = lut[pair & 255],
           second = lut[256 + m * (1 << RESIDUAL_BITS) +
                        ((pair >> 8) & ((1 << RESIDUAL_BITS) - 1))];
  float factor = RESIDUAL_BOOKS > 1 ? factors[m] : 1.f;
  float block_scale = __half2float(reinterpret_cast<const half*>(
      scales)[((int64_t)(row / 16) * (K / 128) + col / 128) * 16 + row % 16]);
  uint16_t values[8];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    __half2 a = static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(
        (__nv_fp4x2_storage_t)(first >> (8 * i)), __NV_E2M1));
    __half2 b = static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(
        (__nv_fp4x2_storage_t)(second >> (8 * i)), __NV_E2M1));
    float2 value;
    if constexpr (RESIDUAL_BOOKS > 1) {
      float2 av = __half22float2(a), bv = __half22float2(b);
      value = make_float2(fmaf(factor, bv.x, av.x), fmaf(factor, bv.y, av.y));
    } else {
      value = __half22float2(__hadd2(a, b));
    }
    float scaled_x = value.x * block_scale * global;
    float scaled_y = value.y * block_scale * global;
    if constexpr (FP16) {
      values[2 * i] = __half_as_ushort(__float2half_rn(scaled_x));
      values[2 * i + 1] = __half_as_ushort(__float2half_rn(scaled_y));
    } else {
      values[2 * i] = __bfloat16_as_ushort(__float2bfloat16_rn(scaled_x));
      values[2 * i + 1] = __bfloat16_as_ushort(__float2bfloat16_rn(scaled_y));
    }
  }
  reinterpret_cast<uint4*>(output)[index] = *reinterpret_cast<uint4*>(values);
}

// One expert: packed u32[N/16,K/64,60]+1 readable guard;
// codebooks u32[384], scales f16[N/16,K/128,16], output bf16[N,K].
extern "C" int arvq_dequant(const void* packed, const void* codebooks,
                            const void* scales, float global, void* output,
                            int N, int K, void* stream) {
  if (N <= 0 || K <= 0 || N % 16 || K % 128) return (int)cudaErrorInvalidValue;
  arvq_dequant_kernel<false, 7>
      <<<((int64_t)N * (K / 8) + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned*)packed, (const unsigned*)codebooks,
          (const unsigned char*)scales, global, output, N, K);
  return (int)cudaGetLastError();
}

// Same ABI, with output f16[N,K] to match native activation precision.
extern "C" int arvq_dequant_fp16(const void* packed, const void* codebooks,
                                 const void* scales, float global, void* output,
                                 int N, int K, void* stream) {
  if (N <= 0 || K <= 0 || N % 16 || K % 128) return (int)cudaErrorInvalidValue;
  arvq_dequant_kernel<true, 7>
      <<<((int64_t)N * (K / 8) + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned*)packed, (const unsigned*)codebooks,
          (const unsigned char*)scales, global, output, N, K);
  return (int)cudaGetLastError();
}

// 8+8 variants: 64 words per tile, 512 codebook words, no guard needed.
extern "C" int arvq_dequant_8x8(const void* packed, const void* codebooks,
                                const void* scales, float global, void* output,
                                int N, int K, void* stream) {
  if (N <= 0 || K <= 0 || N % 16 || K % 128) return (int)cudaErrorInvalidValue;
  arvq_dequant_kernel<false, 8>
      <<<((int64_t)N * (K / 8) + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned*)packed, (const unsigned*)codebooks,
          (const unsigned char*)scales, global, output, N, K);
  return (int)cudaGetLastError();
}

// Same ABI, with output f16[N,K] to match native activation precision.
extern "C" int arvq_dequant_fp16_8x8(const void* packed, const void* codebooks,
                                     const void* scales, float global,
                                     void* output, int N, int K, void* stream) {
  if (N <= 0 || K <= 0 || N % 16 || K % 128) return (int)cudaErrorInvalidValue;
  arvq_dequant_kernel<true, 8>
      <<<((int64_t)N * (K / 8) + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned*)packed, (const unsigned*)codebooks,
          (const unsigned char*)scales, global, output, N, K);
  return (int)cudaGetLastError();
}

// v5 mcbook16 (rvq256_mb16_256x8_expert_fp16block): one expert's
// codebooks u32[4352] (base book + 16 residual books), selectors
// u8[N/16,K/64] pick the residual book per tile, book_factors f32[16] weight
// its atoms. Packed/scales/global unchanged. Extra pointers trail the stream.
extern "C" int arvq_dequant_8x8_mb16(const void* packed, const void* codebooks,
                                     const void* scales, float global,
                                     void* output, int N, int K, void* stream,
                                     const void* selectors,
                                     const void* book_factors) {
  if (N <= 0 || K <= 0 || N % 16 || K % 128 || !selectors || !book_factors)
    return (int)cudaErrorInvalidValue;
  arvq_dequant_kernel<false, 8, 16>
      <<<((int64_t)N * (K / 8) + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned*)packed, (const unsigned*)codebooks,
          (const unsigned char*)scales, global, output, N, K,
          (const unsigned char*)selectors, (const float*)book_factors);
  return (int)cudaGetLastError();
}

extern "C" int arvq_dequant_fp16_8x8_mb16(const void* packed,
                                          const void* codebooks,
                                          const void* scales, float global,
                                          void* output, int N, int K,
                                          void* stream, const void* selectors,
                                          const void* book_factors) {
  if (N <= 0 || K <= 0 || N % 16 || K % 128 || !selectors || !book_factors)
    return (int)cudaErrorInvalidValue;
  arvq_dequant_kernel<true, 8, 16>
      <<<((int64_t)N * (K / 8) + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned*)packed, (const unsigned*)codebooks,
          (const unsigned char*)scales, global, output, N, K,
          (const unsigned char*)selectors, (const float*)book_factors);
  return (int)cudaGetLastError();
}

__global__ void scatter_float4(const float4* src, const int64_t* routes,
                               float4* dst, int h4) {
  int row = blockIdx.x;
  int col = blockIdx.y * 256 + threadIdx.x;
  if (col < h4) {
    int64_t target = routes[row];
    float4 bits = src[(int64_t)row * h4 + col];
    dst[target * h4 + col] = bits;
  }
}
extern "C" int arvq_route_scatter(const void* src, const void* routes,
                                  void* dst, int rows, int hidden,
                                  void* stream) {
  if (!src || !routes || !dst || rows < 1 || hidden < 1 || hidden % 4)
    return (int)cudaErrorInvalidValue;
  scatter_float4<<<dim3(rows, (hidden + 1023) / 1024), 256, 0,
                   (cudaStream_t)stream>>>(
      (const float4*)src, (const int64_t*)routes, (float4*)dst, hidden / 4);
  return (int)cudaGetLastError();
}
