// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
__global__ void route_pack_planes(const __nv_bfloat16* in, unsigned* out,
                                  unsigned char* sc, int K, int slots, int P,
                                  const int* cold, const int* hot,
                                  int* partners, const int64_t* routes,
                                  int topk) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= slots * K) return;
  if (partners && i < slots) {
    // Sorted selected hot runs contain >=32 routes before chunking. Each
    // 32-slot window intersects at most two runs: <=5 groups of <=8 slots.
    int lane = threadIdx.x & 31;
    unsigned active = __activemask();
    unsigned matches = __match_any_sync(active, hot[i]);
    int before = __popc(matches & ((1u << lane) - 1u));
    bool leader = before % 16 == 0;
    unsigned leaders = __ballot_sync(active, leader);
    int rank = __popc(leaders & ((1u << lane) - 1u));
    int* desc = partners + (i / 32) * 7;
    if (lane == 0) desc[0] = __popc(leaders);
    if (leader && rank < 3) {
      desc[1 + 2 * rank] = i;
      desc[2 + 2 * rank] = min(16, __popc(matches) - before);
    }
  }
  int k = i % K, slot = i / K;
  float v = __half2float(__float2half_rn(
      __bfloat162float(in[(routes[slot] >> 3) * (int64_t)K + k])));
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

extern "C" int hybrid_pack_routes_top8(const void* x, void* q, void* s,
                                       const void* cold, const void* hot,
                                       void* groups, const void* routes, int K,
                                       int slots, int P, int topk,
                                       void* stream) {
  if (topk != 8 || K != 6144 || P != 4 || slots < 1 || slots > 1024 || !x ||
      !q || !s || !cold || !hot || !groups || !routes)
    return (int)cudaErrorInvalidValue;
  route_pack_planes<<<(slots * K + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)x, (unsigned*)q, (unsigned char*)s, K, slots, P,
      (const int*)cold, (const int*)hot, (int*)groups, (const int64_t*)routes,
      topk);
  return (int)cudaGetLastError();
}
