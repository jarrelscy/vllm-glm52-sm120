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

template <bool FP16, int RESIDUAL_BITS>
__global__ void arvq_dequant_kernel(const unsigned* packed,
                                    const unsigned* codebooks,
                                    const unsigned char* scales, float global,
                                    void* output, int N, int K) {
  constexpr int LUT_SIZE = 256 + (1 << RESIDUAL_BITS);
  __shared__ unsigned lut[LUT_SIZE];
  for (int i = threadIdx.x; i < LUT_SIZE; i += blockDim.x)
    lut[i] = codebooks[i];
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
  unsigned first = lut[pair & 255],
           second = lut[256 + ((pair >> 8) & ((1 << RESIDUAL_BITS) - 1))];
  float block_scale = __half2float(reinterpret_cast<const half*>(
      scales)[((int64_t)(row / 16) * (K / 128) + col / 128) * 16 + row % 16]);
  uint16_t values[8];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    __half2 a = static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(
        (__nv_fp4x2_storage_t)(first >> (8 * i)), __NV_E2M1));
    __half2 b = static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(
        (__nv_fp4x2_storage_t)(second >> (8 * i)), __NV_E2M1));
    float2 value = __half22float2(__hadd2(a, b));
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

template <bool FP16, int RESIDUAL_BITS>
__global__ void arvq_dequant_gather_kernel(
    const unsigned* packed, const unsigned* codebooks,
    const unsigned char* scales, float global, void* output, int N, int K,
    const __nv_bfloat16* x, const int64_t* routes, __half* rows, int M,
    int top_k, int weight_blocks) {
  // Extra CTAs perform pure gather+BF16->FP16 cast. Weight decode below is
  // byte-for-byte source-identical; existing GEMM shapes and strides unchanged.
  if (blockIdx.x >= weight_blocks) {
    int64_t index =
        (int64_t)(blockIdx.x - weight_blocks) * blockDim.x + threadIdx.x;
    if (index >= (int64_t)M * (K / 8)) return;
    int row = index / (K / 8), col = (index % (K / 8)) * 8;
    int64_t input_row = routes[row] / top_k;
    uint16_t values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      values[i] = __half_as_ushort(
          __float2half_rn(__bfloat162float(x[input_row * K + col + i])));
    reinterpret_cast<uint4*>(rows)[index] = *reinterpret_cast<uint4*>(values);
    return;
  }
  constexpr int LUT_SIZE = 256 + (1 << RESIDUAL_BITS);
  __shared__ unsigned lut[LUT_SIZE];
  for (int i = threadIdx.x; i < LUT_SIZE; i += blockDim.x)
    lut[i] = codebooks[i];
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
  unsigned first = lut[pair & 255],
           second = lut[256 + ((pair >> 8) & ((1 << RESIDUAL_BITS) - 1))];
  float block_scale = __half2float(reinterpret_cast<const half*>(
      scales)[((int64_t)(row / 16) * (K / 128) + col / 128) * 16 + row % 16]);
  uint16_t values[8];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    __half2 a = static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(
        (__nv_fp4x2_storage_t)(first >> (8 * i)), __NV_E2M1));
    __half2 b = static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(
        (__nv_fp4x2_storage_t)(second >> (8 * i)), __NV_E2M1));
    float2 value = __half22float2(__hadd2(a, b));
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

extern "C" int arvq_dequant_gather_fp16_7(
    const void* packed, const void* codebooks, const void* scales, float global,
    void* weight, int N, int K, void* stream, const void* x, const void* routes,
    void* rows, int M, int top_k) {
  if (N <= 0 || K <= 0 || N % 16 || K % 128 || M < 0 || top_k <= 0)
    return (int)cudaErrorInvalidValue;
  int wb = ((int64_t)N * (K / 8) + 255) / 256;
  int rb = ((int64_t)M * (K / 8) + 255) / 256;
  arvq_dequant_gather_kernel<true, 7>
      <<<wb + rb, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned*)packed, (const unsigned*)codebooks,
          (const unsigned char*)scales, global, weight, N, K,
          (const __nv_bfloat16*)x, (const int64_t*)routes, (__half*)rows, M,
          top_k, wb);
  return (int)cudaGetLastError();
}

extern "C" int arvq_dequant_gather_fp16_8(
    const void* packed, const void* codebooks, const void* scales, float global,
    void* weight, int N, int K, void* stream, const void* x, const void* routes,
    void* rows, int M, int top_k) {
  if (N <= 0 || K <= 0 || N % 16 || K % 128 || M < 0 || top_k <= 0)
    return (int)cudaErrorInvalidValue;
  int wb = ((int64_t)N * (K / 8) + 255) / 256;
  int rb = ((int64_t)M * (K / 8) + 255) / 256;
  arvq_dequant_gather_kernel<true, 8>
      <<<wb + rb, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned*)packed, (const unsigned*)codebooks,
          (const unsigned char*)scales, global, weight, N, K,
          (const __nv_bfloat16*)x, (const int64_t*)routes, (__half*)rows, M,
          top_k, wb);
  return (int)cudaGetLastError();
}
