// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
__device__ int nearest(const float* cb, int n, const float x[8], int lane) {
  float best = 1e30f;
  int idx = 0;
  for (int k = lane; k < n; k += 32) {
    float d = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float v = x[j] - cb[j * n + k];
      d = fmaf(v, v, d);
    }
    if (d < best || (d == best && k < idx)) {
      best = d;
      idx = k;
    }
  }
  for (int off = 16; off; off >>= 1) {
    float d = __shfl_down_sync(0xffffffff, best, off);
    int k = __shfl_down_sync(0xffffffff, idx, off);
    if (d < best || (d == best && k < idx)) {
      best = d;
      idx = k;
    }
  }
  return __shfl_sync(0xffffffff, idx, 0);
}
template <bool HALF>
__global__ void assign_kernel(const void* in, const float* c0, const float* c1,
                              unsigned char* i0, unsigned char* i1, int n,
                              int refine) {
  __shared__ float cb0[256 * 8], cb1[128 * 8];
  for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x)
    cb0[(i % 8) * 256 + i / 8] = c0[i];
  for (int i = threadIdx.x; i < 128 * 8; i += blockDim.x)
    cb1[(i % 8) * 128 + i / 8] = c1[i];
  __syncthreads();
  int lane = threadIdx.x & 31,
      vec = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (vec >= n) return;
  float x[8], r[8];
#pragma unroll
  for (int j = 0; j < 8; j++)
    x[j] = HALF ? __half2float(((const half*)in)[(long long)vec * 8 + j])
                : ((const float*)in)[(long long)vec * 8 + j];
  int a = nearest(cb0, 256, x, lane);
#pragma unroll
  for (int j = 0; j < 8; j++) r[j] = x[j] - cb0[j * 256 + a];
  int b = nearest(cb1, 128, r, lane);
  for (int it = 0; it < refine; it++) {
#pragma unroll
    for (int j = 0; j < 8; j++) r[j] = x[j] - cb1[j * 128 + b];
    a = nearest(cb0, 256, r, lane);
#pragma unroll
    for (int j = 0; j < 8; j++) r[j] = x[j] - cb0[j * 256 + a];
    b = nearest(cb1, 128, r, lane);
  }
  if (lane == 0) {
    i0[vec] = a;
    i1[vec] = b;
  }
}
extern "C" int assign_launch(const void* x, const void* c0, const void* c1,
                             void* i0, void* i1, int n, int refine, int ishalf,
                             int threads, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (ishalf)
    assign_kernel<true>
        <<<(n + threads / 32 - 1) / (threads / 32), threads, 0, s>>>(
            x, (const float*)c0, (const float*)c1, (unsigned char*)i0,
            (unsigned char*)i1, n, refine);
  else
    assign_kernel<false>
        <<<(n + threads / 32 - 1) / (threads / 32), threads, 0, s>>>(
            x, (const float*)c0, (const float*)c1, (unsigned char*)i0,
            (unsigned char*)i1, n, refine);
  return cudaGetLastError();
}
// Natural indices [row,K/8] -> tightly packed native MMA fragment order.
// Each output word extracts up to three15-bit pairs without atomics.
__device__ unsigned pair_at(const unsigned char* a, const unsigned char* b,
                            int N, int K, int tile, int g, int pos) {
  int j = pos / 32, lane = pos % 32, q = lane / 4, c = lane % 4;
  int row = tile * 16 + q + 8 * (j & 1), kg = g * 8 + (j / 2) * 4 + c;
  long long i = (long long)row * (K / 8) + kg;
  return (unsigned)a[i] | ((unsigned)b[i] << 8);
}
__global__ void pack_kernel(const unsigned char* a, const unsigned char* b,
                            unsigned* out, int N, int K) {
  long long z = (long long)blockIdx.x * blockDim.x + threadIdx.x,
            total = (long long)(N / 16) * (K / 64) * 60;
  if (z == total) {
    out[z] = 0;
    return;
  }
  if (z > total) return;
  int word = z % 60, g = (z / 60) % (K / 64), tile = z / (60 * (K / 64)),
      bit = word * 32, pos = bit / 15, shift = bit % 15;
  unsigned long long v = pair_at(a, b, N, K, tile, g, pos);
  if (pos + 1 < 128)
    v |= (unsigned long long)pair_at(a, b, N, K, tile, g, pos + 1) << 15;
  if (pos + 2 < 128)
    v |= (unsigned long long)pair_at(a, b, N, K, tile, g, pos + 2) << 30;
  if (pos + 3 < 128)
    v |= (unsigned long long)pair_at(a, b, N, K, tile, g, pos + 3) << 45;
  out[z] = (unsigned)(v >> shift);
}
extern "C" int pack_launch(const void* a, const void* b, void* out, int N,
                           int K, void* stream) {
  long long total = (long long)(N / 16) * (K / 64) * 60;
  pack_kernel<<<(total + 256) / 256, 256, 0, (cudaStream_t)stream>>>(
      (const unsigned char*)a, (const unsigned char*)b, (unsigned*)out, N, K);
  return cudaGetLastError();
}
__device__ unsigned mapped_pair(const uint16_t* old, const uint16_t* map, int K,
                                int tile, int g, int pos) {
  int j = pos / 32, lane = pos % 32, q = lane / 4, c = lane % 4;
  int row = tile * 16 + q + 8 * (j & 1), kg = g * 8 + (j / 2) * 4 + c;
  return map[old[(long long)row * (K / 8) + kg]];
}
__global__ void translate_pack_kernel(const uint16_t* old, const uint16_t* map,
                                      unsigned* out, int E, int N, int K) {
  long long z = (long long)blockIdx.x * blockDim.x + threadIdx.x,
            expertwords = (long long)(N / 16) * (K / 64) * 60,
            total = E * expertwords;
  if (z == total) {
    out[z] = 0;
    return;
  }
  if (z > total) return;
  int expert = z / expertwords;
  long long local = z % expertwords;
  const uint16_t* src = old + (long long)expert * N * (K / 8);
  int word = local % 60, g = (local / 60) % (K / 64),
      tile = local / (60 * (K / 64)), bit = word * 32, pos = bit / 15,
      shift = bit % 15;
  unsigned long long v = mapped_pair(src, map, K, tile, g, pos);
  if (pos + 1 < 128)
    v |= (unsigned long long)mapped_pair(src, map, K, tile, g, pos + 1) << 15;
  if (pos + 2 < 128)
    v |= (unsigned long long)mapped_pair(src, map, K, tile, g, pos + 2) << 30;
  if (pos + 3 < 128)
    v |= (unsigned long long)mapped_pair(src, map, K, tile, g, pos + 3) << 45;
  out[z] = (unsigned)(v >> shift);
}
extern "C" int translate_pack_launch(const void* old, const void* map,
                                     void* out, int E, int N, int K,
                                     void* stream) {
  long long total = (long long)E * (N / 16) * (K / 64) * 60;
  translate_pack_kernel<<<(total + 256) / 256, 256, 0, (cudaStream_t)stream>>>(
      (const uint16_t*)old, (const uint16_t*)map, (unsigned*)out, E, N, K);
  return cudaGetLastError();
}
