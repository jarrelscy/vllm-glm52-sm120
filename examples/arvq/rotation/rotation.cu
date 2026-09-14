// SPDX-License-Identifier: Apache-2.0
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Each CTA covers 128 contiguous elements within a single routed slot.
// Supplied signs are shared with the weight encoder: W'=(W*signs)@H_B.
// Since H_B is orthogonal and symmetric, x'=(x*signs)@H_B preserves x W^T.
template <int B>
__device__ __forceinline__ float rotate_value(float value, int col, bool cold,
                                              const int8_t* signs,
                                              float* shared) {
  if (B != 0 && cold) {
    value *= signs[col];
#pragma unroll
    for (int distance = 1; distance < 32; distance *= 2) {
      float other = __shfl_xor_sync(0xffffffff, value, distance);
      value = (threadIdx.x & distance) ? other - value : value + other;
    }
    if constexpr (B == 128) {
#pragma unroll
      for (int distance = 32; distance < 128; distance *= 2) {
        shared[threadIdx.x] = value;
        __syncthreads();
        float other = shared[threadIdx.x ^ distance];
        __syncthreads();
        value = (threadIdx.x & distance) ? other - value : value + other;
      }
    }
    value *= B == 32 ? 0.1767766952966369f : 0.08838834764831845f;
  }
  return value;
}

template <int B, bool PACK>
__global__ void rotation_kernel(const half* input, const int8_t* signs,
                                const int* cold_ids, unsigned* packed,
                                unsigned char* scales, float* transformed,
                                int K, int slots) {
  __shared__ float shared[256];
  int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (int64_t)K * slots) return;
  int col = index % K, slot = index / K;
  float value = rotate_value<B>(__half2float(input[index]), col,
                                cold_ids[slot] >= 0, signs, shared);
  if constexpr (!PACK) {
    transformed[index] = value;
  } else {
#pragma unroll
    for (int plane = 0; plane < 4; plane++) {
      float maximum = fabsf(value);
#pragma unroll
      for (int distance = 8; distance; distance /= 2)
        maximum = fmaxf(maximum,
                        __shfl_xor_sync(0xffffffff, maximum, distance));
      int exponent = max(-6, min(8, (int)ceilf(log2f(fmaxf(maximum / 6,
                                                         0x1p-20f)))));
      float scale = exp2f((float)exponent), magnitude = fabsf(value) / scale;
      unsigned code = (magnitude > .25f) + (magnitude > .75f) +
                      (magnitude > 1.25f) + (magnitude > 1.75f) +
                      (magnitude > 2.5f) + (magnitude > 3.5f) + (magnitude > 5.f);
      code |= value < 0 ? 8 : 0;
      unsigned magnitude_code = code & 7;
      float level = magnitude_code < 2
                        ? magnitude_code * .5f
                        : (1.f + .5f * (magnitude_code & 1)) *
                              __uint_as_float(((magnitude_code >> 1) + 126) << 23);
      float decoded = level * scale * ((code & 8) ? -1.f : 1.f);
      value = (value - decoded) * 16;
      unsigned bits = code << (4 * (col & 7));
#pragma unroll
      for (int distance = 4; distance; distance /= 2)
        bits |= __shfl_xor_sync(0xffffffff, bits, distance);
      if ((col & 7) == 0)
        packed[((int64_t)slot * 4 + plane) * (K / 8) + col / 8] = bits;
      if ((col & 15) == 0)
        scales[((int64_t)slot * 4 + plane) * (K / 16) + col / 16] =
            (exponent + 7) << 3;
    }
  }
}

// ABI: input f16[slots,K], signs i8[K], cold_ids i32[slots],
// packed u32[slots,4,K/8], scales u8[slots,4,K/16]; B=0,32,128, K%128=0.
// B=0 ignores signs. Hot slots (cold_ids<0) are always unrotated.
extern "C" int rotation_pack(const void* input, const void* signs,
                              const void* cold_ids, void* packed, void* scales,
                              int K, int slots, int B, void* stream) {
  if (K <= 0 || K % 128 || slots <= 0 || (B != 0 && B != 32 && B != 128))
    return (int)cudaErrorInvalidValue;
  int threads = K % 256 == 0 ? 256 : 128;
  dim3 grid(((int64_t)K * slots + threads - 1) / threads);
#define LAUNCH_ROTATION(BLOCK)                                                \
  rotation_kernel<BLOCK, true><<<grid, threads, 0, (cudaStream_t)stream>>>(          \
      (const half*)input, (const int8_t*)signs, (const int*)cold_ids,            \
      (unsigned*)packed, (unsigned char*)scales, nullptr, K, slots)
  if (B == 0) { LAUNCH_ROTATION(0); }
  else if (B == 32) { LAUNCH_ROTATION(32); }
  else { LAUNCH_ROTATION(128); }
#undef LAUNCH_ROTATION
  return (int)cudaGetLastError();
}

// Diagnostic transform-only ABI: input f16[slots,K], output f32[slots,K].
extern "C" int rotation_transform(const void* input, const void* signs,
                                   const void* cold_ids, void* output,
                                   int K, int slots, int B, void* stream) {
  if (K <= 0 || K % 128 || slots <= 0 || (B != 0 && B != 32 && B != 128))
    return (int)cudaErrorInvalidValue;
  int threads = K % 256 == 0 ? 256 : 128;
  dim3 grid(((int64_t)K * slots + threads - 1) / threads);
#define LAUNCH_TRANSFORM(BLOCK)                                               \
  rotation_kernel<BLOCK, false><<<grid, threads, 0, (cudaStream_t)stream>>>(         \
      (const half*)input, (const int8_t*)signs, (const int*)cold_ids, nullptr,    \
      nullptr, (float*)output, K, slots)
  if (B == 0) { LAUNCH_TRANSFORM(0); }
  else if (B == 32) { LAUNCH_TRANSFORM(32); }
  else { LAUNCH_TRANSFORM(128); }
#undef LAUNCH_TRANSFORM
  return (int)cudaGetLastError();
}
