// Local routed-pair prototype: pair map must match exact expert kind and ID.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// SM120 warp MMA: weights occupy the 16-row operand; eight independent
// activation groups (LUT mode) or token columns (direct mode) occupy N.
__device__ __forceinline__ void mma(float& d0, float& d1, float& d2, float& d3,
                                    unsigned a0, unsigned a1, unsigned a2,
                                    unsigned a3, unsigned b0, unsigned b1,
                                    unsigned sa = 0x38383838u,
                                    unsigned sb = 0x38383838u) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, "
      "{%11}, {0,0};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

__global__ void pack_planes(const half* in, unsigned* out, unsigned char* sc,
                            int K, int slots, int P, const int* cold,
                            const int* hot, int* partners) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= slots * K) return;
  if (partners && i < slots) {
    // Sorted selected hot runs contain >=32 routes before chunking. Each
    // 32-slot window intersects at most two runs: <=5 groups of <=8 slots.
    int lane = threadIdx.x & 31;
    unsigned active = __activemask();
    unsigned matches = __match_any_sync(active, hot[i]);
    int before = __popc(matches & ((1u << lane) - 1u));
    bool leader = before % 8 == 0;
    unsigned leaders = __ballot_sync(active, leader);
    int rank = __popc(leaders & ((1u << lane) - 1u));
    int* desc = partners + (i / 32) * 11;
    if (lane == 0) desc[0] = __popc(leaders);
    if (leader && rank < 5) {
      desc[1 + 2 * rank] = i;
      desc[2 + 2 * rank] = min(8, __popc(matches) - before);
    }
  }
  int k = i % K, slot = i / K;
  float v = __half2float(in[i]);
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
extern "C" int hybrid_pack(const void* x, void* q, void* s, int K, int slots,
                           int P, void* stream) {
  pack_planes<<<(slots * K + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
      (const half*)x, (unsigned*)q, (unsigned char*)s, K, slots, P, nullptr,
      nullptr, nullptr);
  return (int)cudaGetLastError();
}
extern "C" int hybrid_pack_pairs(const void* x, void* q, void* s,
                                 const void* cold, const void* hot,
                                 void* partners, int K, int slots, int P,
                                 void* stream) {
  pack_planes<<<(slots * K + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
      (const half*)x, (unsigned*)q, (unsigned char*)s, K, slots, P,
      (const int*)cold, (const int*)hot, (int*)partners);
  return (int)cudaGetLastError();
}
// Four warps now compute four token pairs for one16-row weight tile.
// HOT ONLY: descriptor generation is restricted to sorted same-expert runs.
template <bool SHARED, int ROW_TILES = 1>
__global__ void wide_kernel(const unsigned* hw, const unsigned* hs,
                            const unsigned* x, const unsigned* xs,
                            const int* hot_ids, const int* groups,
                            float* partial, int N, int G, int S, int slots) {
  int group = blockIdx.z;
  const int* desc = groups + (group / 5) * 11;
  int index = group % 5;
  if (index >= desc[0]) return;
  int first = desc[1 + 2 * index], count = desc[2 + 2 * index];
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  int q = lane / 4, c = lane % 4;
  int tile = blockIdx.x * ROW_TILES + warp % ROW_TILES;
  if constexpr (!SHARED) {
    if (tile >= N / 16) return;
  }
  int pair_warp = warp / ROW_TILES;
  int slot = first + 2 * pair_warp;
  bool valid_slot = 2 * pair_warp < count;
  int partner = 2 * pair_warp + 1 < count ? slot + 1 : -1;
  int expert = hot_ids[first];
  __shared__ unsigned sw[128], ss[32];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  for (int g = G * blockIdx.y / S; g < G * (blockIdx.y + 1) / S; ++g) {
    unsigned a[4], sa;
    if constexpr (SHARED) {
      sw[threadIdx.x] =
          hw[(((((long long)expert * (N / 16) + tile) * G + g) * 4 + warp) *
              32) +
             lane];
      if (warp == 0)
        ss[lane] =
            hs[((long long)expert * N + tile * 16 + q + 8 * (c & 1)) * G + g];
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 4; ++j) a[j] = sw[j * 32 + lane];
      sa = ss[lane];
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        a[j] = hw[(((((long long)expert * (N / 16) + tile) * G + g) * 4 + j) *
                   32) +
                  lane];
      sa = hs[((long long)expert * N + tile * 16 + q + 8 * (c & 1)) * G + g];
    }
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
    mma(d0, d1, d2, d3, a[0], a[1], a[2], a[3], b0, b1, sa, sb);
    if constexpr (SHARED) __syncthreads();
  }
  if (partner < 0) {
    float a = ldexpf(d0, -8 * c) + ldexpf(d1, -8 * c - 4);
    float b = ldexpf(d2, -8 * c) + ldexpf(d3, -8 * c - 4);
    a += __shfl_xor_sync(0xffffffff, a, 1);
    a += __shfl_xor_sync(0xffffffff, a, 2);
    b += __shfl_xor_sync(0xffffffff, b, 1);
    b += __shfl_xor_sync(0xffffffff, b, 2);
    if (valid_slot && c == 0) {
      partial[((long long)slot * N + tile * 16 + q) * S + blockIdx.y] = a;
      partial[((long long)slot * N + tile * 16 + q + 8) * S + blockIdx.y] = b;
    }
  } else {
    float a = ldexpf(d0, -8 * (c % 2)) + ldexpf(d1, -8 * (c % 2) - 4);
    float b = ldexpf(d2, -8 * (c % 2)) + ldexpf(d3, -8 * (c % 2) - 4);
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
__global__ void hybrid_reduce(const float* p, float* y, const int* cold_ids,
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

extern "C" int wide_launch(const void* hw, const void* hs, const void* global,
                           const void* x, const void* xs, const void* cold,
                           const void* hot, const void* groups, void* partial,
                           void* out, int N, int K, int slots, int split,
                           int parts, int shared, void* stream) {
  if (N <= 0 || N % 16 || K <= 0 || K % 128 || slots <= 0 || split <= 0)
    return (int)cudaErrorInvalidValue;
  cudaStream_t s = (cudaStream_t)stream;
  dim3 grid(N / 16, split, ((slots + 31) / 32) * 5);
  if (shared == 2)
    wide_kernel<false, 4>
        <<<dim3((N + 63) / 64, split, ((slots + 31) / 32) * 5), 512, 0, s>>>(
            (const unsigned*)hw, (const unsigned*)hs, (const unsigned*)x,
            (const unsigned*)xs, (const int*)hot, (const int*)groups,
            (float*)partial, N, K / 64, split, slots);
  else if (shared == 1)
    wide_kernel<true><<<grid, 128, 0, s>>>(
        (const unsigned*)hw, (const unsigned*)hs, (const unsigned*)x,
        (const unsigned*)xs, (const int*)hot, (const int*)groups,
        (float*)partial, N, K / 64, split, slots);
  else
    wide_kernel<false><<<grid, 128, 0, s>>>(
        (const unsigned*)hw, (const unsigned*)hs, (const unsigned*)x,
        (const unsigned*)xs, (const int*)hot, (const int*)groups,
        (float*)partial, N, K / 64, split, slots);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  hybrid_reduce<<<((long long)slots * N + 255) / 256, 256, 0, s>>>(
      (const float*)partial, (float*)out, (const int*)cold, (const int*)hot,
      (const float*)global, 0.f, N, slots, split, parts);
  return (int)cudaGetLastError();
}
