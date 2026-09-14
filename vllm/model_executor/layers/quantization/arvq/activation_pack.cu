// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// round gate/up to half, SiLU to half, product to half, then original P4.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__device__ __forceinline__ half activation(const float* input, int slot, int k,
                                           int K) {
  float gate =
      __half2float(__float2half_rn(input[(long long)slot * (2 * K) + k]));
  float up =
      __half2float(__float2half_rn(input[(long long)slot * (2 * K) + K + k]));
  // Match PyTorch ActivationSiluKernel.cu opmath_float formula; no fast math.
  half silu = __float2half_rn(gate / (1.0f + ::exp(-gate)));
  return __float2half_rn(__fmul_rn(__half2float(silu), up));
}

__global__ void silu_mul_kernel(const float* input, half* out, int K,
                                int slots) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)slots * K) return;
  out[i] = activation(input, i / K, i % K, K);
}

__global__ void activation_pack_kernel(const float* input, unsigned* out,
                                       unsigned char* sc,
                                       half* debug_activation, int K, int slots,
                                       int P) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)slots * K) return;
  int k = i % K, slot = i / K;
  half act = activation(input, slot, k, K);
  if (debug_activation) debug_activation[i] = act;
  float v = __half2float(act);
  for (int p = 0; p < P; p++) {
    float m = fabsf(v);
    for (int d = 8; d; d /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, d));
    int e = max(-6, min(8, (int)ceilf(log2f(fmaxf(m / 6, 0x1p-20f)))));
    float scale = exp2f((float)e), a = fabsf(v) / scale;
    unsigned q = (a > .25f) + (a > .75f) + (a > 1.25f) + (a > 1.75f) +
                 (a > 2.5f) + (a > 3.5f) + (a > 5.f);
    q |= (v < 0) ? 8 : 0;
    const float tbl[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float dec = tbl[q & 7] * scale * ((q & 8) ? -1.f : 1.f);
    v = (v - dec) * 16;
    unsigned bits = q << (4 * (k & 7));
    for (int d = 4; d; d /= 2) bits |= __shfl_xor_sync(0xffffffff, bits, d);
    if ((k & 7) == 0) out[((long long)slot * P + p) * (K / 8) + k / 8] = bits;
    if ((k & 15) == 0)
      sc[((long long)slot * P + p) * (K / 16) + k / 16] = (e + 7) << 3;
  }
}
extern "C" int fused_silu_mul_f32(const void* input, void* out, int K,
                                  int slots, void* stream) {
  if (K <= 0 || slots <= 0) return (int)cudaErrorInvalidValue;
  silu_mul_kernel<<<((long long)slots * K + 255) / 256, 256, 0,
                    (cudaStream_t)stream>>>((const float*)input, (half*)out, K,
                                            slots);
  return (int)cudaGetLastError();
}
extern "C" int fused_silu_mul_pack(const void* input, void* packed,
                                   void* scales, void* debug_activation, int K,
                                   int slots, void* stream) {
  if (K <= 0 || K % 32 || slots <= 0) return (int)cudaErrorInvalidValue;
  activation_pack_kernel<<<((long long)slots * K + 255) / 256, 256, 0,
                           (cudaStream_t)stream>>>(
      (const float*)input, (unsigned*)packed, (unsigned char*)scales,
      (half*)debug_activation, K, slots, 4);
  return (int)cudaGetLastError();
}
