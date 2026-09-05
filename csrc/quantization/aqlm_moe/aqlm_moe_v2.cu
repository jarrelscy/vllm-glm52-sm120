/*
 * aqlm_moe_v2.cu — optimized decode-path kernels for the GLM-5.2 hybrid
 * NVFP4+AQLM fused MoE (see csrc/quantization/aqlm_moe/aqlm_moe.cu for the
 * baseline). Drop-in superset: exports the same four entry points plus a
 * fused hybrid_moe_gemv that covers both storage formats in one launch.
 *
 * Optimizations vs baseline:
 *  - NVFP4: replaces the __constant__ fp4 LUT (divergent indices serialize
 *    the constant cache) with the SM120a hardware cvt.rn.f16x2.e2m1x2
 *    instruction (via cuda_fp4.h) + half2 FMA; optional 256-entry smem LUT
 *    fallback (-DNVFP4_LUT256=1). Adds a software prefetch of the next
 *    weight chunk.
 *  - AQLM: codebook gathers issued in batches of AQLM_MLP (default 8)
 *    schedulable __ldcg loads instead of one serialized `asm volatile`
 *    dependency chain per group; code indices loaded once per int4.
 *    Optional L1-cached codebook loads (-DAQLM_CB_L1=1).
 *  - Fused HybridMatVecMoE kernel: per-slot uniform dispatch between the
 *    AQLM and NVFP4 paths, so one launch per projection replaces
 *    2 gemv launches + 1 eltwise add + masked-slot zero-fill traffic.
 *  - V4 pipelined kernel (AQLM_GEMV_PIPELINE=1, default OFF -> V2/V3
 *    behavior unchanged): full-K activation staging via cp.async (one
 *    barrier pair instead of 2 per K-tile), all per-lane weight/code loads
 *    issued up front (register multi-buffering, 4-6 16B loads in flight),
 *    AQLM codebook gathers double-buffered across batches, fp16 FMA chains
 *    split 8 -> 4x4 with fp32 block combine, and 4-rows-per-warp streaming
 *    for small-K projections (w2). Targets the latency/ILP bound of the
 *    tokens<=8 decode gemv (~25% DRAM util at 90%+ occupancy).
 *
 * Numerics are kept bit-identical in accumulation order to the baseline for
 * the AQLM path; the NVFP4 path accumulates each 16-value scale block in
 * half2 (8 hfma2) before the fp32 block reduction, matching the baseline to
 * ~1e-3 relative (validated in bench.py).
 */

#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef AQLM_MLP
#define AQLM_MLP 8  // codebook gathers kept in flight per code int4 (2,4,8)
#endif
#ifndef AQLM_CB_L1
#define AQLM_CB_L1 0  // 0: __ldcg (L2 only, baseline behavior), 1: __ldca
#endif
#ifndef NVFP4_LUT256
#define NVFP4_LUT256 0  // 1: byte->half2 smem LUT instead of hardware cvt
#endif

namespace aqlm_moe_v2 {

constexpr int THREAD_M = 16;

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ uint4 ld_cb(const uint4* p) {
#if AQLM_CB_L1
  return __ldca(p);
#else
  return __ldcg(p);
#endif
}

// ---------------------------------------------------------------------------
// bf16 activation input (BF16IN template paths, host-selected by x.dtype):
// activations arrive as bf16 raw bytes and are converted to fp16 while (V2/V3,
// register round-trip staging) or right after (V4, cp.async) they land in
// shared memory. __bfloat162float is exact and __float2half is the same
// float->half RN conversion torch uses on CUDA, so the staged fp16 bits are
// identical to the Python-side `x.to(torch.float16)` they replace; every
// downstream FMA then matches the fp16 path bit-for-bit.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int4 bf16x8_to_fp16x8(const int4 v) {
  const __nv_bfloat16* s = reinterpret_cast<const __nv_bfloat16*>(&v);
  int4 o;
  half* d = reinterpret_cast<half*>(&o);
#pragma unroll
  for (int i = 0; i < 8; i++) d[i] = __float2half(__bfloat162float(s[i]));
  return o;
}

template <bool BF16IN>
__device__ __forceinline__ int4 stage_act(const int4 v) {
  return BF16IN ? bf16x8_to_fp16x8(v) : v;
}

// ---------------------------------------------------------------------------
// AQLM: per-(slot,row) warp gemv body. Call with all threads of the block
// (contains __syncthreads); `expert` must be uniform within the block.
// sh_b must hold >= 32*9 int4.
// ---------------------------------------------------------------------------
template <int BOOKS, bool BF16IN = false>
__device__ __forceinline__ void aqlm_slot_gemv(
    const int4* __restrict__ codes, const int4* __restrict__ codebooks,
    const half* __restrict__ scales, const int expert,
    const int4* __restrict__ B, half* __restrict__ C_slot, const int prob_m,
    const int prob_k, int4* sh_b, const int row, const bool pred) {
  const int a_gl_stride = prob_k / 8 / 8;  // int4s per code row
  const int4* a_base[BOOKS];
#pragma unroll
  for (int b = 0; b < BOOKS; b++) {
    a_base[b] = codes + ((int64_t)expert * BOOKS + b) * prob_m * a_gl_stride;
  }

  int b_gl_rd = 0;
  int a_rd = a_gl_stride * row + threadIdx.x % 32;
  const int a_end = a_gl_stride * row + a_gl_stride;
  const uint4* cb = reinterpret_cast<const uint4*>(codebooks);

  float res = 0;
  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) {
        sh_b[9 * (i / 8) + i % 8] = stage_act<BF16IN>(B[b_gl_rd + i]);
      }
    }
    __syncthreads();
    b_gl_rd += 32 * 8;

    if (pred && a_rd < a_end) {
      // The 8 code indices per book arrive in one int4.
      union alignas(16) {
        int4 raw;
        uint16_t u16[8];
      } enc[BOOKS];
#pragma unroll
      for (int b = 0; b < BOOKS; b++) enc[b].raw = __ldg(&a_base[b][a_rd]);

      const int4* bvec = &sh_b[9 * (threadIdx.x % 32)];
#pragma unroll
      for (int i0 = 0; i0 < 8; i0 += AQLM_MLP) {
        // Phase 1: issue all gathers for this batch (independent loads).
        uint4 w[AQLM_MLP][BOOKS];
#pragma unroll
        for (int u = 0; u < AQLM_MLP; u++) {
#pragma unroll
          for (int b = 0; b < BOOKS; b++) {
            w[u][b] = ld_cb(cb + (int64_t)b * 65536 + enc[b].u16[i0 + u]);
          }
        }
        // Phase 2: fp16 FMA against the staged activation chunk.
#pragma unroll
        for (int u = 0; u < AQLM_MLP; u++) {
          half2 wsum[4];
          const half2* a0 = reinterpret_cast<const half2*>(&w[u][0]);
#pragma unroll
          for (int j = 0; j < 4; j++) wsum[j] = a0[j];
          if (BOOKS == 2) {
            const half2* a1 = reinterpret_cast<const half2*>(&w[u][BOOKS - 1]);
#pragma unroll
            for (int j = 0; j < 4; j++) wsum[j] = __hadd2(wsum[j], a1[j]);
          }
          const half2* bb = reinterpret_cast<const half2*>(&bvec[i0 + u]);
          half2 res2 = {};
#pragma unroll
          for (int j = 0; j < 4; j++) res2 = __hfma2(wsum[j], bb[j], res2);
          res += __half2float(res2.x) + __half2float(res2.y);
        }
      }
      a_rd += 32;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = 16; i > 0; i /= 2) res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0) {
      const float s = __half2float(scales[(int64_t)expert * prob_m + row]);
      C_slot[row] = __float2half(res * s);
    }
  }
}

// ---------------------------------------------------------------------------
// NVFP4 helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t v) {
  __nv_fp8_e4m3 f;
  f.__x = v;
  return float(f);
}

#if !NVFP4_LUT256
// SM120a: single cvt.rn.f16x2.e2m1x2 per byte (low nibble -> .x).
__device__ __forceinline__ half2 fp4x2_to_half2(uint8_t v) {
  return half2(__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)v, __NV_E2M1));
}

__device__ __forceinline__ float nvfp4_chunk_dot(const uint4 w,
                                                 const uchar2 bs,
                                                 const half2* __restrict__ bb) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w);
  half2 acc0 = {}, acc1 = {};
#pragma unroll
  for (int i = 0; i < 8; i++)
    acc0 = __hfma2(fp4x2_to_half2(bytes[i]), bb[i], acc0);
#pragma unroll
  for (int i = 8; i < 16; i++)
    acc1 = __hfma2(fp4x2_to_half2(bytes[i]), bb[i], acc1);
  return fp8_e4m3_to_float(bs.x) * (__half2float(acc0.x) + __half2float(acc0.y)) +
         fp8_e4m3_to_float(bs.y) * (__half2float(acc1.x) + __half2float(acc1.y));
}
#define NVFP4_SMEM_EXTRA 0
#else
// Portable variant: byte -> half2 via a 256-entry smem LUT (one 4B load per
// byte, no constant-cache serialization).
#define NVFP4_SMEM_EXTRA 256
__device__ __forceinline__ void nvfp4_fill_lut(half2* lut) {
  const float v[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f,  3.0f,  4.0f,  6.0f,
                       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    lut[i] = __floats2half2_rn(v[i & 0xF], v[i >> 4]);
  }
}

__device__ __forceinline__ float nvfp4_chunk_dot_lut(
    const uint4 w, const uchar2 bs, const half2* __restrict__ bb,
    const half2* __restrict__ lut) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w);
  half2 acc0 = {}, acc1 = {};
#pragma unroll
  for (int i = 0; i < 8; i++) acc0 = __hfma2(lut[bytes[i]], bb[i], acc0);
#pragma unroll
  for (int i = 8; i < 16; i++) acc1 = __hfma2(lut[bytes[i]], bb[i], acc1);
  return fp8_e4m3_to_float(bs.x) * (__half2float(acc0.x) + __half2float(acc0.y)) +
         fp8_e4m3_to_float(bs.y) * (__half2float(acc1.x) + __half2float(acc1.y));
}
#endif

// NVFP4 per-(slot,row) warp gemv body; same calling contract as
// aqlm_slot_gemv. sh_b must hold >= 32*4 int4 (+ LUT extra when enabled).
template <bool BF16IN = false>
__device__ __forceinline__ void nvfp4_slot_gemv(
    const int4* __restrict__ packed, const uchar2* __restrict__ bscale,
    const float* __restrict__ scale2, const int s2n, const int expert,
    const int4* __restrict__ B, half* __restrict__ C_slot, const int prob_m,
    const int prob_k, int4* sh_b, const int row, const bool pred) {
  const int a_gl_stride = prob_k / 32;  // int4s (32 fp4) per row
  const uint4* a_row = reinterpret_cast<const uint4*>(
      packed + ((int64_t)expert * prob_m + row) * a_gl_stride);
  const uchar2* s_row =
      bscale + ((int64_t)expert * prob_m + row) * a_gl_stride;
  const int lane = threadIdx.x % 32;

#if NVFP4_LUT256
  half2* lut = reinterpret_cast<half2*>(sh_b + 32 * 4);
  nvfp4_fill_lut(lut);  // synced by the first staging barrier below
#endif

  float res = 0;
  int iters = (prob_k / 8 + 4 * 32 - 1) / (4 * 32);
  int b_gl_rd = 0;
  int a_rd = lane;

  // Software pipeline: weight chunk for the current iteration is loaded
  // during the previous one.
  bool have = pred && a_rd < a_gl_stride;
  uint4 w_cur = {};
  uchar2 bs_cur = {};
  if (have) {
    w_cur = a_row[a_rd];
    bs_cur = s_row[a_rd];
  }

  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 4; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) sh_b[i] = stage_act<BF16IN>(B[b_gl_rd + i]);
    }
    __syncthreads();
    b_gl_rd += 32 * 4;

    if (have) {
      const int a_nx = a_rd + 32;
      const bool have_nx = a_nx < a_gl_stride;
      uint4 w_nx = {};
      uchar2 bs_nx = {};
      if (have_nx) {
        w_nx = a_row[a_nx];
        bs_nx = s_row[a_nx];
      }
      const half2* bb = reinterpret_cast<const half2*>(&sh_b[lane * 4]);
#if NVFP4_LUT256
      res += nvfp4_chunk_dot_lut(w_cur, bs_cur, bb, lut);
#else
      res += nvfp4_chunk_dot(w_cur, bs_cur, bb);
#endif
      a_rd = a_nx;
      w_cur = w_nx;
      bs_cur = bs_nx;
      have = have_nx;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = 16; i > 0; i /= 2) res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0) {
      const float g = scale2[expert * s2n + (int)(((int64_t)row * s2n) / prob_m)];
      C_slot[row] = __float2half(res * g);
    }
  }
}

// ---------------------------------------------------------------------------
// Standalone kernels (same signatures/semantics as baseline)
// ---------------------------------------------------------------------------
template <int BOOKS>
__global__ void CodeKx16MatVecMoE(const int4* __restrict__ codes,
                                  const int4* __restrict__ B_all,
                                  half* __restrict__ C,
                                  const int4* __restrict__ codebooks,
                                  const half* __restrict__ scales,
                                  const int* __restrict__ expert_ids,
                                  const int prob_m, const int prob_k) {
  const int slot = blockIdx.y;
  const int expert = expert_ids[slot];
  const int row = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  const bool pred = row < prob_m;
  __shared__ int4 sh_b[32 * 9];

  if (expert < 0) {  // slot handled by another format: contribute zeros
    if (pred && threadIdx.x % 32 == 0) {
      C[(int64_t)slot * prob_m + row] = __float2half(0.f);
    }
    return;
  }
  aqlm_slot_gemv<BOOKS>(codes, codebooks, scales, expert,
                        B_all + (int64_t)slot * (prob_k / 8),
                        C + (int64_t)slot * prob_m, prob_m, prob_k, sh_b, row,
                        pred);
}

__global__ void NvFp4MatVecMoE(const int4* __restrict__ packed,
                               const uchar2* __restrict__ bscale,
                               const float* __restrict__ scale2, const int s2n,
                               const int4* __restrict__ B_all,
                               half* __restrict__ C,
                               const int* __restrict__ expert_ids,
                               const int prob_m, const int prob_k) {
  const int slot = blockIdx.y;
  const int expert = expert_ids[slot];
  const int row = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  const bool pred = row < prob_m;
  __shared__ int4 sh_b[32 * 4 + (NVFP4_SMEM_EXTRA + 3) / 4];

  if (expert < 0) {
    if (pred && threadIdx.x % 32 == 0) {
      C[(int64_t)slot * prob_m + row] = __float2half(0.f);
    }
    return;
  }
  nvfp4_slot_gemv(packed, bscale, scale2, s2n, expert,
                  B_all + (int64_t)slot * (prob_k / 8),
                  C + (int64_t)slot * prob_m, prob_m, prob_k, sh_b, row, pred);
}

// ---------------------------------------------------------------------------
// Fused hybrid kernel: one launch covers both storage formats for one
// projection. Exactly one of aqlm_ids[slot] / nv_ids[slot] is >= 0 for an
// active slot; both < 0 writes zeros. The branch is uniform per block
// (blockIdx.y == slot), so the contained __syncthreads is safe.
//
// row_div (AQLM_GEMV_ROWMAP glue, default 1 = previous behavior): slot s
// reads activation row s / row_div, so the caller can pass the compact
// [T, K] token activations with row_div = top_k instead of materializing
// x.repeat_interleave(top_k, dim=0). The staged bytes are identical to the
// expanded tensor's, so outputs are bit-identical; only the per-block base
// pointer changes (one integer divide per block).
// ---------------------------------------------------------------------------
template <int BOOKS, bool BF16IN = false>
__global__ void HybridMatVecMoE(
    const int4* __restrict__ codes, const int4* __restrict__ codebooks,
    const half* __restrict__ scales, const int* __restrict__ aqlm_ids,
    const int4* __restrict__ packed, const uchar2* __restrict__ bscale,
    const float* __restrict__ scale2, const int s2n,
    const int* __restrict__ nv_ids, const int4* __restrict__ B_all,
    half* __restrict__ C, const int prob_m, const int prob_k,
    const int row_div) {
  const int slot = blockIdx.y;
  const int a_id = aqlm_ids[slot];
  const int n_id = nv_ids[slot];
  const int row = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  const bool pred = row < prob_m;
  __shared__ int4 sh_b[32 * 9 + (NVFP4_SMEM_EXTRA + 3) / 4];

  const int4* B = B_all + (int64_t)(slot / row_div) * (prob_k / 8);
  half* C_slot = C + (int64_t)slot * prob_m;

  if (a_id >= 0) {
    aqlm_slot_gemv<BOOKS, BF16IN>(codes, codebooks, scales, a_id, B, C_slot,
                                  prob_m, prob_k, sh_b, row, pred);
  } else if (n_id >= 0) {
    nvfp4_slot_gemv<BF16IN>(packed, bscale, scale2, s2n, n_id, B, C_slot,
                            prob_m, prob_k, sh_b, row, pred);
  } else if (pred && threadIdx.x % 32 == 0) {
    C_slot[row] = __float2half(0.f);
  }
}

// ---------------------------------------------------------------------------
// V3 fused hybrid kernel (DECODE-K, env-gated default-OFF):
//   GLM_MOE_DEDUP=1     cross-slot expert dedup. Slots routed to the same
//     (format, expert) repeat identical codebook gathers / weight reads /
//     fp4 decodes. A leader block (occurrence index % UMAX == 0 in slot
//     order) computes up to UMAX=4 duplicate slots with ONE weight stream
//     and per-slot activation buffers; follower blocks exit. Grid shape is
//     unchanged (graph-safe); election is a 2*n_slots-int scan per block.
//   GLM_MOE_LANE_ROWS=1 rows-per-warp grouping for small-K projections
//     (w2: K=512). When the NVFP4 row stride s = K/32 is a power of two
//     <= 16, lanes are split into R = 32/s groups of G = s lanes, each
//     group owning one row (AQLM uses the first G/2 lanes of a group).
//     Removes the 50-75% idle lanes of the w2 gemv.
// BIT-EXACTNESS: per (slot, row) the per-lane fp16/fp32 accumulation
// content and order are IDENTICAL to the V2 kernel; the within-group
// shuffle tree (offsets G/2..1) has the same fp32 pairing as the V2
// 32-lane tree over zero-padded idle lanes (adds of +0.0 only, and res is
// never -0.0 since it accumulates from +0.0). Dedup only shares loaded
// bytes, never arithmetic between slots.
// ---------------------------------------------------------------------------
template <int BOOKS, int UMAX, bool BF16IN = false>
__device__ __forceinline__ void aqlm_multi_gemv(
    const int4* __restrict__ codes, const int4* __restrict__ codebooks,
    const half* __restrict__ scales, const int expert,
    const int4* __restrict__ B_all, half* __restrict__ C,
    const int* __restrict__ slots_u, const int U, const int prob_m,
    const int prob_k, int4* sh_b, const int row0, const int G,
    const int row_div) {
  // Activation row per slot (ROWMAP: slot / row_div; row_div == 1 -> slot).
  int brow[UMAX];
#pragma unroll
  for (int su = 0; su < UMAX; su++) {
    brow[su] = su < U ? slots_u[su] / row_div : 0;
  }
  const int lane = threadIdx.x % 32;
  const int g = lane / G;        // row group within the warp
  const int tt = lane - g * G;   // in-group lane
  const int row = row0 + g;
  const bool pred = row < prob_m;
  const int a_gl_stride = prob_k / 8 / 8;
  const int4* a_base[BOOKS];
#pragma unroll
  for (int b = 0; b < BOOKS; b++) {
    a_base[b] = codes + ((int64_t)expert * BOOKS + b) * prob_m * a_gl_stride;
  }
  int b_gl_rd = 0;
  int a_rd = a_gl_stride * row + tt;
  const int a_end = a_gl_stride * row + a_gl_stride;
  const uint4* cb = reinterpret_cast<const uint4*>(codebooks);

  float res[UMAX];
#pragma unroll
  for (int su = 0; su < UMAX; su++) res[su] = 0.f;

  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 8 * UMAX; i += blockDim.x) {
      const int su = i / 256, ii = i - su * 256;
      if (su < U && b_gl_rd + ii < prob_k / 8) {
        sh_b[su * 288 + 9 * (ii / 8) + ii % 8] = stage_act<BF16IN>(
            B_all[(int64_t)brow[su] * (prob_k / 8) + b_gl_rd + ii]);
      }
    }
    __syncthreads();
    b_gl_rd += 32 * 8;

    if (pred && a_rd < a_end) {
      union alignas(16) {
        int4 raw;
        uint16_t u16[8];
      } enc[BOOKS];
#pragma unroll
      for (int b = 0; b < BOOKS; b++) enc[b].raw = __ldg(&a_base[b][a_rd]);

#pragma unroll
      for (int i0 = 0; i0 < 8; i0 += AQLM_MLP) {
        uint4 w[AQLM_MLP][BOOKS];
#pragma unroll
        for (int u = 0; u < AQLM_MLP; u++) {
#pragma unroll
          for (int b = 0; b < BOOKS; b++) {
            w[u][b] = ld_cb(cb + (int64_t)b * 65536 + enc[b].u16[i0 + u]);
          }
        }
#pragma unroll
        for (int u = 0; u < AQLM_MLP; u++) {
          half2 wsum[4];
          const half2* a0 = reinterpret_cast<const half2*>(&w[u][0]);
#pragma unroll
          for (int j = 0; j < 4; j++) wsum[j] = a0[j];
          if (BOOKS == 2) {
            const half2* a1 = reinterpret_cast<const half2*>(&w[u][BOOKS - 1]);
#pragma unroll
            for (int j = 0; j < 4; j++) wsum[j] = __hadd2(wsum[j], a1[j]);
          }
#pragma unroll
          for (int su = 0; su < UMAX; su++) {
            if (su < U) {
              const half2* bb = reinterpret_cast<const half2*>(
                  &sh_b[su * 288 + 9 * tt + i0 + u]);
              half2 res2 = {};
#pragma unroll
              for (int j = 0; j < 4; j++) res2 = __hfma2(wsum[j], bb[j], res2);
              res[su] += __half2float(res2.x) + __half2float(res2.y);
            }
          }
        }
      }
      a_rd += 32;
    }
  }

#pragma unroll
  for (int su = 0; su < UMAX; su++) {
    if (su < U) {
      float r = res[su];
      for (int off = G / 2; off > 0; off /= 2) {
        r += __shfl_down_sync(0xffffffff, r, off);
      }
      if (pred && tt == 0) {
        const float s = __half2float(scales[(int64_t)expert * prob_m + row]);
        C[(int64_t)slots_u[su] * prob_m + row] = __float2half(r * s);
      }
    }
  }
}

template <int UMAX, bool BF16IN = false>
__device__ __forceinline__ void nvfp4_multi_gemv(
    const int4* __restrict__ packed, const uchar2* __restrict__ bscale,
    const float* __restrict__ scale2, const int s2n, const int expert,
    const int4* __restrict__ B_all, half* __restrict__ C,
    const int* __restrict__ slots_u, const int U, const int prob_m,
    const int prob_k, int4* sh_b, const int row0, const int G,
    const int row_div) {
  int brow[UMAX];
#pragma unroll
  for (int su = 0; su < UMAX; su++) {
    brow[su] = su < U ? slots_u[su] / row_div : 0;
  }
  const int lane = threadIdx.x % 32;
  const int g = lane / G;
  const int tt = lane - g * G;
  const int row = row0 + g;
  const bool pred = row < prob_m;
  const int a_gl_stride = prob_k / 32;
  const uint4* a_row = reinterpret_cast<const uint4*>(
      packed + ((int64_t)expert * prob_m + row) * a_gl_stride);
  const uchar2* s_row =
      bscale + ((int64_t)expert * prob_m + row) * a_gl_stride;

#if NVFP4_LUT256
  half2* lut = reinterpret_cast<half2*>(sh_b + UMAX * 288);
  nvfp4_fill_lut(lut);  // synced by the first staging barrier below
#endif

  float res[UMAX];
#pragma unroll
  for (int su = 0; su < UMAX; su++) res[su] = 0.f;

  int iters = (prob_k / 8 + 4 * 32 - 1) / (4 * 32);
  int b_gl_rd = 0;
  int a_rd = tt;
  bool have = pred && a_rd < a_gl_stride;
  uint4 w_cur = {};
  uchar2 bs_cur = {};
  if (have) {
    w_cur = a_row[a_rd];
    bs_cur = s_row[a_rd];
  }

  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 4 * UMAX; i += blockDim.x) {
      const int su = i / 128, ii = i - su * 128;
      if (su < U && b_gl_rd + ii < prob_k / 8) {
        sh_b[su * 288 + ii] = stage_act<BF16IN>(
            B_all[(int64_t)brow[su] * (prob_k / 8) + b_gl_rd + ii]);
      }
    }
    __syncthreads();
    b_gl_rd += 32 * 4;

    if (have) {
      const int a_nx = a_rd + 32;
      const bool have_nx = a_nx < a_gl_stride;
      uint4 w_nx = {};
      uchar2 bs_nx = {};
      if (have_nx) {
        w_nx = a_row[a_nx];
        bs_nx = s_row[a_nx];
      }
#pragma unroll
      for (int su = 0; su < UMAX; su++) {
        if (su < U) {
          const half2* bb =
              reinterpret_cast<const half2*>(&sh_b[su * 288 + tt * 4]);
#if NVFP4_LUT256
          res[su] += nvfp4_chunk_dot_lut(w_cur, bs_cur, bb, lut);
#else
          res[su] += nvfp4_chunk_dot(w_cur, bs_cur, bb);
#endif
        }
      }
      a_rd = a_nx;
      w_cur = w_nx;
      bs_cur = bs_nx;
      have = have_nx;
    }
  }

#pragma unroll
  for (int su = 0; su < UMAX; su++) {
    if (su < U) {
      float r = res[su];
      for (int off = G / 2; off > 0; off /= 2) {
        r += __shfl_down_sync(0xffffffff, r, off);
      }
      if (pred && tt == 0) {
        const float gsc =
            scale2[expert * s2n + (int)(((int64_t)row * s2n) / prob_m)];
        C[(int64_t)slots_u[su] * prob_m + row] = __float2half(r * gsc);
      }
    }
  }
}

template <int BOOKS, int UMAX, bool BF16IN = false>
__global__ void HybridMatVecMoEV3(
    const int4* __restrict__ codes, const int4* __restrict__ codebooks,
    const half* __restrict__ scales, const int* __restrict__ aqlm_ids,
    const int4* __restrict__ packed, const uchar2* __restrict__ bscale,
    const float* __restrict__ scale2, const int s2n,
    const int* __restrict__ nv_ids, const int4* __restrict__ B_all,
    half* __restrict__ C, const int prob_m, const int prob_k,
    const int n_slots, const int G, const int row_div) {
  const int slot = blockIdx.y;
  const int my_a = __ldg(&aqlm_ids[slot]);
  const int my_n = __ldg(&nv_ids[slot]);
  const int fmt = my_a >= 0 ? 0 : (my_n >= 0 ? 1 : 2);
  const int key = fmt == 0 ? my_a : (fmt == 1 ? my_n : -1);

  int slots_u[UMAX];
  slots_u[0] = slot;
  int U = 1;
  __shared__ int s_meta[UMAX > 1 ? UMAX : 1];  // [0]=U or -1, [1..]=slots
  if (UMAX > 1) {
    // Leader election in slot order (occurrence index of (fmt, key)),
    // computed ONCE by thread 0 and broadcast via smem: a per-thread scan
    // costs ~400M redundant L1TEX loads across the w2 grid (measured 2x
    // kernel-time regression before this fix).
    if (threadIdx.x == 0) {
      int occ = 0;
      for (int j = 0; j < slot; j++) {
        const int ja = __ldg(&aqlm_ids[j]);
        const int jn = __ldg(&nv_ids[j]);
        const int jf = ja >= 0 ? 0 : (jn >= 0 ? 1 : 2);
        const int jk = jf == 0 ? ja : (jf == 1 ? jn : -1);
        occ += (jf == fmt) & (jk == key);
      }
      if (occ % UMAX) {
        s_meta[0] = -1;  // follower block: leader writes our output
      } else {
        int u = 1;
        for (int j = slot + 1; j < n_slots && u < UMAX; j++) {
          const int ja = __ldg(&aqlm_ids[j]);
          const int jn = __ldg(&nv_ids[j]);
          const int jf = ja >= 0 ? 0 : (jn >= 0 ? 1 : 2);
          const int jk = jf == 0 ? ja : (jf == 1 ? jn : -1);
          if ((jf == fmt) & (jk == key)) s_meta[u++] = j;
        }
        s_meta[0] = u;
      }
    }
    __syncthreads();
    U = s_meta[0];
    if (U < 0) return;  // uniform across the block (same smem value)
#pragma unroll
    for (int su = 1; su < UMAX; su++) {
      if (su < U) slots_u[su] = s_meta[su];
    }
  }

  const int R = 32 / G;
  const int row0 = ((blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32)) * R;
  __shared__ int4 sh_b[UMAX * 288 + (NVFP4_SMEM_EXTRA + 3) / 4];

  if (fmt == 0) {
    aqlm_multi_gemv<BOOKS, UMAX, BF16IN>(codes, codebooks, scales, my_a, B_all,
                                         C, slots_u, U, prob_m, prob_k, sh_b,
                                         row0, G, row_div);
  } else if (fmt == 1) {
    nvfp4_multi_gemv<UMAX, BF16IN>(packed, bscale, scale2, s2n, my_n, B_all, C,
                                   slots_u, U, prob_m, prob_k, sh_b, row0, G,
                                   row_div);
  } else {
    const int lane = threadIdx.x % 32;
    const int g = lane / G, tt = lane - g * G;
    const int row = row0 + g;
    if (row < prob_m && tt == 0) {
#pragma unroll
      for (int su = 0; su < UMAX; su++) {
        if (su < U) {
          C[(int64_t)slots_u[su] * prob_m + row] = __float2half(0.f);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// V4 pipelined kernel (AQLM_GEMV_PIPELINE=1, env-gated, default OFF):
// decode gemv restructured for latency/ILP (measured: V3 runs at ~25% DRAM
// utilization with 90-98% SM occupancy, i.e. bound by dependent-load and
// dependent-FMA chains, not bandwidth).
//
//   1. Full-K activation staging. Production K (6144 / 512) fits entirely in
//      shared memory (<= 12 KiB as int4), so the per-128-int4 staging loop
//      with its 2 __syncthreads per K-tile collapses to ONE cp.async
//      (LDGSTS) stage + one barrier for the whole gemv. cp.async writes
//      smem directly (no register round-trip) and the copy overlaps the
//      weight/code loads issued between commit and wait.
//   2. Load-all-then-compute weight pipelining. Each lane owns at most
//      CPL (<=6) 16B weight chunks per row (w13 NVFP4: exactly 6; AQLM: 3
//      code int4s); ALL of them are issued as independent loads before any
//      FMA, so one memory round-trip covers the row instead of one per
//      chunk. AQLM codebook gathers are double-buffered: batch i+1's 8
//      gathers are in flight while batch i's FMAs run.
//   3. Split accumulators. The 8-deep dependent __hfma2 chains become 4
//      independent 4-deep chains combined in fp32 (shorter fp16 chains:
//      accuracy >= V2/V3), and chunks alternate between two fp32
//      accumulators. Not bit-identical to V2/V3 (fp reassociation) --
//      validated against the fp32 reference in kbench/bench_gemv.py.
//   4. Multi-row warps for small K. w2 (K=512) has ONE chunk per lane, so
//      V3 blocks are pure latency; V4 gives each warp RW=4 row-groups with
//      every row's loads in flight together.
//
// Grid/output contract is identical to V2/V3 (one launch, same C layout);
// the host falls back to the V2/V3 path when the shape doesn't fit
// (prob_k/8 > 1024 int4 of smem or > 6 chunks/lane).
// ---------------------------------------------------------------------------
constexpr int V4_MAX_ACT_INT4 = 1024;  // prob_k <= 8192

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define V4_CP_ASYNC 1
#else
#define V4_CP_ASYNC 0
#endif

__device__ __forceinline__ void cp_async16(void* smem_dst,
                                           const void* gmem_src) {
#if V4_CP_ASYNC
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d),
               "l"(gmem_src));
#else
  *reinterpret_cast<int4*>(smem_dst) =
      *reinterpret_cast<const int4*>(gmem_src);
#endif
}

// L1-allocating variant (.ca) for ROWMAP staging: when row_div > 1 several
// co-resident blocks (different slots of the same token) read the SAME
// activation row, so allocating it in the SM-local L1 turns the duplicate
// L2 fan-in into L1 hits. Pointless at row_div == 1 (each row read once per
// SM) — the default staging stays .cg, byte-identical to the shipped build.
__device__ __forceinline__ void cp_async16_ca(void* smem_dst,
                                              const void* gmem_src) {
#if V4_CP_ASYNC
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(d),
               "l"(gmem_src));
#else
  *reinterpret_cast<int4*>(smem_dst) =
      *reinterpret_cast<const int4*>(gmem_src);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if V4_CP_ASYNC
  asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void cp_async_wait_all_sync() {
#if V4_CP_ASYNC
  asm volatile("cp.async.wait_group 0;\n");
#endif
  __syncthreads();
}

// One 16B NVFP4 chunk (32 fp4) dot 32 staged activations: 4 independent
// 4-deep half2 FMA chains, combined per fp8-scale block in fp32.
__device__ __forceinline__ float nvfp4_chunk_dot_v4(
    const uint4 w, const uchar2 bs, const half2* __restrict__ bb
#if NVFP4_LUT256
    , const half2* __restrict__ lut
#endif
) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w);
#if NVFP4_LUT256
#define V4_CVT(b) lut[b]
#else
#define V4_CVT(b) fp4x2_to_half2(b)
#endif
  half2 a0 = {}, a1 = {}, a2 = {}, a3 = {};
#pragma unroll
  for (int i = 0; i < 4; i++) a0 = __hfma2(V4_CVT(bytes[i]), bb[i], a0);
#pragma unroll
  for (int i = 4; i < 8; i++) a1 = __hfma2(V4_CVT(bytes[i]), bb[i], a1);
#pragma unroll
  for (int i = 8; i < 12; i++) a2 = __hfma2(V4_CVT(bytes[i]), bb[i], a2);
#pragma unroll
  for (int i = 12; i < 16; i++) a3 = __hfma2(V4_CVT(bytes[i]), bb[i], a3);
#undef V4_CVT
  const float lo = __half2float(a0.x) + __half2float(a0.y) +
                   __half2float(a1.x) + __half2float(a1.y);
  const float hi = __half2float(a2.x) + __half2float(a2.y) +
                   __half2float(a3.x) + __half2float(a3.y);
  return fp8_e4m3_to_float(bs.x) * lo + fp8_e4m3_to_float(bs.y) * hi;
}

// AQLM V4 body. sh_b holds the FULL activation row, padded (int4 j at
// sh_b[j + j/8]); staging must have been issued (commit'ed) by the caller,
// this function waits on it AFTER issuing its own global loads.
template <int BOOKS, int RW, int CPL, bool BF16IN = false>
__device__ __forceinline__ void aqlm_v4_gemv(
    const int4* __restrict__ codes, const int4* __restrict__ codebooks,
    const half* __restrict__ scales, const int expert,
    half* __restrict__ C_slot, const int prob_m, const int prob_k,
    int4* __restrict__ sh_b, const int base_row, const int G,
    const int tt) {
  const int R = 32 / G;
  const int stride = prob_k / 64;  // code int4s per row
  const uint4* cb = reinterpret_cast<const uint4*>(codebooks);
  const int4* a_base[BOOKS];
#pragma unroll
  for (int b = 0; b < BOOKS; b++) {
    a_base[b] = codes + ((int64_t)expert * BOOKS + b) * prob_m * stride;
  }

  // Preload ALL code int4s for this warp's rows (independent 16B loads).
  union alignas(16) EncT {
    int4 raw;
    uint16_t u16[8];
  };
  EncT enc[RW][CPL][BOOKS];
  bool val[RW][CPL];
#pragma unroll
  for (int j = 0; j < RW; j++) {
    const int row = base_row + j * R;
    const bool rok = row < prob_m;
#pragma unroll
    for (int c = 0; c < CPL; c++) {
      const int p = tt + c * G;
      val[j][c] = rok && p < stride;
      if (val[j][c]) {
#pragma unroll
        for (int b = 0; b < BOOKS; b++) {
          enc[j][c][b].raw = __ldg(&a_base[b][stride * row + p]);
        }
      }
    }
  }

  float resA[RW], resB[RW];
#pragma unroll
  for (int j = 0; j < RW; j++) resA[j] = resB[j] = 0.f;

  // Codebook-gather pipeline over sub-batches of 4 codes (half a code int4).
  // Sub-batch sb+1's 4*BOOKS gathers are issued while sb's FMAs run, so the
  // L2 round-trip is hidden without the 64-register cost of double-buffering
  // whole 8-gather batches (which drops occupancy to 2 blocks/SM and starves
  // the L2 gather pipeline of warps).
  // Gather width (codes per sub-batch): 4 for the large-K kernel (deeper
  // in-flight gather window wins there: 222us -> 218us aqlm w13), 2 for the
  // small-K multi-row kernel (register/occupancy relief wins: 108.8 -> 106.9
  // aqlm w2). Both measured on 4x RTX PRO 6000 SM120.
  constexpr int GW = (RW == 1) ? 8 : 2;
  constexpr int NSPC = 8 / GW;            // sub-batches per code int4
  constexpr int NSB = RW * CPL * NSPC;    // total sub-batches
  constexpr int GDEPTH = 2;  // depth-3 measured neutral-to-worse
  uint4 gbuf[GDEPTH][GW * BOOKS];

#define AQLM_V4_ISSUE(sb, buf)                                              \
  {                                                                         \
    const int j_ = (sb) / (NSPC * CPL), c_ = ((sb) / NSPC) % CPL;           \
    const int h_ = (sb) % NSPC;                                             \
    if (val[j_][c_]) {                                                      \
      _Pragma("unroll") for (int u = 0; u < GW; u++) {                      \
        _Pragma("unroll") for (int b = 0; b < BOOKS; b++) {                 \
          (buf)[u * BOOKS + b] = ld_cb(                                     \
              cb + (int64_t)b * 65536 + enc[j_][c_][b].u16[GW * h_ + u]);   \
        }                                                                   \
      }                                                                     \
    }                                                                       \
  }

#define AQLM_V4_COMPUTE(sb, buf)                                            \
  {                                                                         \
    const int j_ = (sb) / (NSPC * CPL), c_ = ((sb) / NSPC) % CPL;           \
    const int h_ = (sb) % NSPC;                                             \
    if (val[j_][c_]) {                                                      \
      const int p_ = tt + c_ * G;                                           \
      const int4* bvec = &sh_b[p_ * 9 + GW * h_];                           \
      _Pragma("unroll") for (int u = 0; u < GW; u++) {                      \
        half2 wsum[4];                                                      \
        const half2* a0 = reinterpret_cast<const half2*>(&(buf)[u * BOOKS]);\
        _Pragma("unroll") for (int q = 0; q < 4; q++) wsum[q] = a0[q];      \
        if (BOOKS == 2) {                                                   \
          const half2* a1 =                                                 \
              reinterpret_cast<const half2*>(&(buf)[u * BOOKS + 1]);        \
          _Pragma("unroll") for (int q = 0; q < 4; q++) {                   \
            wsum[q] = __hadd2(wsum[q], a1[q]);                              \
          }                                                                 \
        }                                                                   \
        const half2* bb = reinterpret_cast<const half2*>(&bvec[u]);         \
        half2 r2 = {};                                                      \
        _Pragma("unroll") for (int q = 0; q < 4; q++) {                     \
          r2 = __hfma2(wsum[q], bb[q], r2);                                 \
        }                                                                   \
        const float d_ = __half2float(r2.x) + __half2float(r2.y);           \
        if (u & 1) {                                                        \
          resB[j_] += d_;                                                   \
        } else {                                                            \
          resA[j_] += d_;                                                   \
        }                                                                   \
      }                                                                     \
    }                                                                       \
  }

  AQLM_V4_ISSUE(0, gbuf[0]);
  if (GDEPTH > 2 && NSB > 1) AQLM_V4_ISSUE(1, gbuf[1 % GDEPTH]);
  cp_async_wait_all_sync();  // activations resident from here on
  if (BF16IN) {
    // Staged bytes are bf16: convert to fp16 in place (padded layout: int4
    // j lives at j + j/8) before any lane touches them. The weight/code
    // loads above are already in flight, so the copy/convert still overlap.
    for (int i = threadIdx.x; i < prob_k / 8; i += blockDim.x) {
      const int p = i + (i >> 3);
      sh_b[p] = bf16x8_to_fp16x8(sh_b[p]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int sb = 0; sb < NSB; sb++) {
    if (sb + GDEPTH - 1 < NSB)
      AQLM_V4_ISSUE(sb + GDEPTH - 1, gbuf[(sb + GDEPTH - 1) % GDEPTH]);
    AQLM_V4_COMPUTE(sb, gbuf[sb % GDEPTH]);
  }
#undef AQLM_V4_ISSUE
#undef AQLM_V4_COMPUTE

#pragma unroll
  for (int j = 0; j < RW; j++) {
    float r = resA[j] + resB[j];
    for (int off = G / 2; off > 0; off /= 2) {
      r += __shfl_down_sync(0xffffffff, r, off);
    }
    const int row = base_row + j * R;
    if (row < prob_m && tt == 0) {
      const float s = __half2float(scales[(int64_t)expert * prob_m + row]);
      C_slot[row] = __float2half(r * s);
    }
  }
}

// NVFP4 V4 body. sh_b holds the FULL activation row, flat (int4 j at
// sh_b[j]); same staging contract as aqlm_v4_gemv.
template <int RW, int CPL, bool BF16IN = false>
__device__ __forceinline__ void nvfp4_v4_gemv(
    const int4* __restrict__ packed, const uchar2* __restrict__ bscale,
    const float* __restrict__ scale2, const int s2n, const int expert,
    half* __restrict__ C_slot, const int prob_m, const int prob_k,
    int4* __restrict__ sh_b, const int base_row, const int G,
    const int tt
#if NVFP4_LUT256
    , const half2* __restrict__ lut
#endif
) {
  const int R = 32 / G;
  const int stride = prob_k / 32;  // uint4s (32 fp4) per row

  // Preload ALL weight chunks + block scales for this warp's rows.
  uint4 w[RW][CPL];
  uchar2 bs[RW][CPL];
  bool val[RW][CPL];
#pragma unroll
  for (int j = 0; j < RW; j++) {
    const int row = base_row + j * R;
    const bool rok = row < prob_m;
    const uint4* a_row = reinterpret_cast<const uint4*>(
        packed + ((int64_t)expert * prob_m + row) * stride);
    const uchar2* s_row =
        bscale + ((int64_t)expert * prob_m + row) * stride;
#pragma unroll
    for (int c = 0; c < CPL; c++) {
      const int p = tt + c * G;
      val[j][c] = rok && p < stride;
      if (val[j][c]) {
        // NOTE: __ldcs (evict-first streaming) was measured HERE and LOST
        // (prod-mix w13 112->123us): NVFP4 weight reuse via L2 outweighs
        // protecting the AQLM codebook's residency.
        w[j][c] = a_row[p];
        bs[j][c] = s_row[p];
      }
    }
  }

  cp_async_wait_all_sync();  // activations resident from here on
  if (BF16IN) {
    // Staged bytes are bf16: convert to fp16 in place (flat layout; the
    // LUT region beyond prob_k/8 is untouched).
    for (int i = threadIdx.x; i < prob_k / 8; i += blockDim.x) {
      sh_b[i] = bf16x8_to_fp16x8(sh_b[i]);
    }
    __syncthreads();
  }

  float resA[RW], resB[RW];
#pragma unroll
  for (int j = 0; j < RW; j++) resA[j] = resB[j] = 0.f;
#pragma unroll
  for (int j = 0; j < RW; j++) {
#pragma unroll
    for (int c = 0; c < CPL; c++) {
      if (val[j][c]) {
        const int p = tt + c * G;
        const half2* bb = reinterpret_cast<const half2*>(&sh_b[p * 4]);
        const float d = nvfp4_chunk_dot_v4(w[j][c], bs[j][c], bb
#if NVFP4_LUT256
                                           , lut
#endif
        );
        if (c & 1) {
          resB[j] += d;
        } else {
          resA[j] += d;
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < RW; j++) {
    float r = resA[j] + resB[j];
    for (int off = G / 2; off > 0; off /= 2) {
      r += __shfl_down_sync(0xffffffff, r, off);
    }
    const int row = base_row + j * R;
    if (row < prob_m && tt == 0) {
      const float g =
          scale2[expert * s2n + (int)(((int64_t)row * s2n) / prob_m)];
      C_slot[row] = __float2half(r * g);
    }
  }
}

template <int BOOKS, int RW, int CPL, bool BF16IN = false>
__global__ void __launch_bounds__(256) HybridMatVecMoEV4(
    const int4* __restrict__ codes, const int4* __restrict__ codebooks,
    const half* __restrict__ scales, const int* __restrict__ aqlm_ids,
    const int4* __restrict__ packed, const uchar2* __restrict__ bscale,
    const float* __restrict__ scale2, const int s2n,
    const int* __restrict__ nv_ids, const int4* __restrict__ B_all,
    half* __restrict__ C, const int prob_m, const int prob_k, const int G,
    const int row_div) {
  // TRANSPOSED grid vs V2/V3: slot on blockIdx.x (fastest dispatch dim), row
  // chunk on blockIdx.y. Blocks of all slots interleave in issue order, so
  // L2-bound AQLM blocks and DRAM-bound NVFP4 blocks overlap instead of
  // running as per-slot bursts that serialize on one resource at a time.
  const int slot = blockIdx.x;
  const int a_id = __ldg(&aqlm_ids[slot]);
  const int n_id = __ldg(&nv_ids[slot]);
  const int lane = threadIdx.x % 32;
  const int g = lane / G;
  const int tt = lane - g * G;
  const int R = 32 / G;
  const int warp = (blockDim.x / 32) * blockIdx.y + threadIdx.x / 32;
  const int base_row = warp * (R * RW) + g;
  // Dynamic smem, sized by the host to the ACTUAL activation length (plus
  // LUT) instead of the V4_MAX_ACT_INT4 worst case: 18.4KB static would cap
  // residency at 5 blocks/SM; prod w13 needs only 13.9KB.
  extern __shared__ int4 sh_b[];

  const int n_b = prob_k / 8;
  const int4* B = B_all + (int64_t)(slot / row_div) * n_b;
  half* C_slot = C + (int64_t)slot * prob_m;
  // ROWMAP staging cache policy: with row_div > 1 every slot of a token
  // reads the SAME activation row; blocks of different slots co-resident on
  // one SM re-read it, so allocate it in L1 (.ca) to absorb the duplicate
  // L2 fan-in (measured +4-7us on all-AQLM mixes at T=1 with .cg). The
  // staged bytes are identical either way; row_div == 1 keeps the .cg path
  // (byte-identical instruction stream to the shipped staging loop).
  const bool stage_ca = row_div > 1;

  if (a_id >= 0) {
    // Padded layout (int4 j at j + j/8): lane-strided reads during compute
    // hit rotating banks, same as the V2/V3 32*9 tile.
    if (stage_ca) {
      for (int i = threadIdx.x; i < n_b; i += blockDim.x) {
        cp_async16_ca(&sh_b[i + (i >> 3)], &B[i]);
      }
    } else {
      for (int i = threadIdx.x; i < n_b; i += blockDim.x) {
        cp_async16(&sh_b[i + (i >> 3)], &B[i]);
      }
    }
    cp_async_commit();
    // An AQLM chunk (code int4) covers 64 values = 2 NVFP4 chunks, so the
    // AQLM path needs only ceil(CPL/2) batches per lane.
    aqlm_v4_gemv<BOOKS, RW, (CPL + 1) / 2, BF16IN>(
        codes, codebooks, scales, a_id, C_slot, prob_m, prob_k, sh_b,
        base_row, G, tt);
  } else if (n_id >= 0) {
#if NVFP4_LUT256
    half2* lut = reinterpret_cast<half2*>(sh_b + n_b);  // after flat acts
    nvfp4_fill_lut(lut);  // synced by cp_async_wait_all_sync in the body
#endif
    if (stage_ca) {
      for (int i = threadIdx.x; i < n_b; i += blockDim.x) {
        cp_async16_ca(&sh_b[i], &B[i]);
      }
    } else {
      for (int i = threadIdx.x; i < n_b; i += blockDim.x) {
        cp_async16(&sh_b[i], &B[i]);
      }
    }
    cp_async_commit();
    nvfp4_v4_gemv<RW, CPL, BF16IN>(packed, bscale, scale2, s2n, n_id, C_slot,
                                   prob_m, prob_k, sh_b, base_row, G, tt
#if NVFP4_LUT256
                           , lut
#endif
    );
  } else {  // masked slot: zero-fill this warp's rows
#pragma unroll
    for (int j = 0; j < RW; j++) {
      const int row = base_row + j * R;
      if (row < prob_m && tt == 0) C_slot[row] = __float2half(0.f);
    }
  }
}

// ---------------------------------------------------------------------------
// Prefill dequant kernels: copied unchanged from baseline (not decode-hot).
// ---------------------------------------------------------------------------
template <int BOOKS>
__global__ void CodeKx16DequantMoE(const int4* __restrict__ codes,
                                   half* __restrict__ out,
                                   const int4* __restrict__ codebooks,
                                   const half* __restrict__ scales,
                                   const int* __restrict__ expert_list,
                                   const int prob_m, const int prob_k) {
  const int g = blockIdx.y;
  const int expert = expert_list[g];

  const int a_gl_stride = prob_k / 8 / 8;
  const int row = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  const bool pred = row < prob_m;

  const int4* a_base[BOOKS];
#pragma unroll
  for (int b = 0; b < BOOKS; b++) {
    a_base[b] = codes + ((int64_t)expert * BOOKS + b) * prob_m * a_gl_stride;
  }

  int a_rd = a_gl_stride * row + threadIdx.x % 32;
  const int a_end = a_gl_stride * row + a_gl_stride;

  int4* C = reinterpret_cast<int4*>(out) + (int64_t)g * prob_m * (prob_k / 8);

  const float s =
      pred ? __half2float(scales[(int64_t)expert * prob_m + row]) : 0.f;
  const half2 s2 = __float2half2_rn(s);

  int iters = (prob_k / 8 - 1) / (8 * 32) + 1;
  while (iters--) {
    if (pred && a_rd < a_end) {
      uint32_t dec[4];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        half2 wsum[4] = {};
#pragma unroll
        for (int b = 0; b < BOOKS; b++) {
          const uint16_t* enc =
              reinterpret_cast<const uint16_t*>(&a_base[b][a_rd]);
          asm volatile("ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
                       : "l"((void*)&codebooks[(int64_t)b * 65536 + enc[i]]));
          half2* a = reinterpret_cast<half2*>(&dec);
#pragma unroll
          for (int j = 0; j < 4; j++) wsum[j] = __hadd2(wsum[j], a[j]);
        }
        int4 chunk;
        half2* c2 = reinterpret_cast<half2*>(&chunk);
#pragma unroll
        for (int j = 0; j < 4; j++) c2[j] = __hmul2(wsum[j], s2);
        C[(int64_t)a_rd * 8 + i] = chunk;
      }
    }
    a_rd += 32;
  }
}

__device__ __constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__global__ void NvFp4DequantMoE(const int4* __restrict__ packed,
                                const uchar2* __restrict__ bscale,
                                const float* __restrict__ scale2, const int s2n,
                                half* __restrict__ out,
                                const int* __restrict__ expert_list,
                                const int prob_m, const int prob_k) {
  const int g = blockIdx.y;
  const int expert = expert_list[g];
  const int row = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  const bool pred = row < prob_m;
  if (!pred) return;

  const int a_gl_stride = prob_k / 32;
  const int4* a_row = packed + ((int64_t)expert * prob_m + row) * a_gl_stride;
  const uchar2* s_row = bscale + ((int64_t)expert * prob_m + row) * a_gl_stride;
  half* o_row = out + ((int64_t)g * prob_m + row) * prob_k;
  const float gscale =
      scale2[expert * s2n + (int)(((int64_t)row * s2n) / prob_m)];

  for (int a_rd = threadIdx.x % 32; a_rd < a_gl_stride; a_rd += 32) {
    const uint4 w = *reinterpret_cast<const uint4*>(&a_row[a_rd]);
    const uchar2 bs = s_row[a_rd];
    const float s[2] = {fp8_e4m3_to_float(bs.x) * gscale,
                        fp8_e4m3_to_float(bs.y) * gscale};
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w);
    half2 vals[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
      const float scale = s[i >> 3];
      vals[i] = __floats2half2_rn(kFp4Lut[bytes[i] & 0xF] * scale,
                                  kFp4Lut[bytes[i] >> 4] * scale);
    }
    int4* dst = reinterpret_cast<int4*>(o_row + a_rd * 32);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      dst[j] = reinterpret_cast<const int4*>(vals)[j];
    }
  }
}

// ---------------------------------------------------------------------------
// Fused expert-combine epilogue (AQLM_FUSED_COMBINE=1 in the Python glue,
// default OFF): out[t, j] = sum_k float(y[t*top_k + k, j]) * w[t*top_k + k],
// replacing the per-layer eager tail  y.float() -> * w -> .sum(dim=1) ->
// .to(dtype)  (4 kernel launches). Numerics: each product is a rounded fp32
// multiply (__fmul_rn, never FMA-contracted) and the reduction replicates
// torch's TensorIterator vec4 order for this [T, K, M] shape exactly: four
// zero-initialized fp32 accumulators, acc[k % 4] += p_k in ascending k, then
// the sequential merge ((a0 + a1) + a2) + a3 — probed bit-exact against
// (y.float() * w).sum(dim=1) for T = 1..512 at K = 8, M = 6144, and gated in
// kbench/bench_gemv.py --combine (plain ascending order differs by 1 fp32
// ulp; the zero init also reproduces torch's -0.0 -> +0.0 sums). Output
// converts fp32 -> half/bf16 with the same __float2half / __float2bfloat16
// intrinsics c10 uses on CUDA.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void combine_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void combine_store(half* p, float v) {
  *p = __float2half(v);
}
__device__ __forceinline__ void combine_store(__nv_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}

template <typename OutT>
__global__ void MoECombine(const half* __restrict__ y,
                           const float* __restrict__ w,
                           OutT* __restrict__ out, const int top_k,
                           const int m, const int n_out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_out) return;
  const int t = idx / m;
  const int j = idx - t * m;
  const half* yp = y + (int64_t)t * top_k * m + j;
  const float* wp = w + (int64_t)t * top_k;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < top_k; k++) {
    acc[k & 3] = __fadd_rn(
        acc[k & 3], __fmul_rn(__half2float(yp[(int64_t)k * m]), wp[k]));
  }
  combine_store(out + idx,
                __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]),
                          acc[3]));
}

// ---------------------------------------------------------------------------
// Fused SwiGLU mid-tail (AQLM_FUSED_SILU=1 in the Python glue, default OFF):
// out[t, j] = silu(x[t, j]) * x[t, m + j] over fp16 rows of width 2m,
// replacing the per-layer eager  F.silu(x[..., :m]) -> * x[..., m:]  pair
// (2 kernel launches). Numerics replicate torch's two kernels EXACTLY:
//   silu: ATen silu_kernel fp16 opmath —
//         __float2half(__fdiv_rn(xf, __fadd_rn(1.f, expf(-xf)))); probed
//         bit-exact vs torch.nn.functional.silu over ALL 65536 fp16 bit
//         patterns in the prod image (nvcc 12.9 JIT vs torch cu130 build:
//         libdevice expf agrees on every input; the recip-sigmoid and
//         exp2f-based forms are each 1 fp16 ulp off on isolated inputs).
//   mul:  torch's fp16 binary mul is float-opmath — the silu result is
//         rounded to fp16 FIRST (torch materializes the silu tensor), then
//         __float2half(__fmul_rn(half2float(s), half2float(u))). fp16
//         products are exact in fp32, so this equals torch's mul for every
//         input pair; keeping the intermediate fp16 rounding is what makes
//         the fusion bit-exact rather than "more accurate".
// Gated in kbench/bench_gemv.py --silu (all-bit-pattern gate sweep incl
// -0.0 / denormals / inf / nan, all-bit-pattern up sweep, prod shapes).
// ---------------------------------------------------------------------------
__device__ __forceinline__ half silu_mul_one(const half g, const half u) {
  const float xf = __half2float(g);
  const half s = __float2half(__fdiv_rn(xf, __fadd_rn(1.0f, expf(-xf))));
  return __float2half(__fmul_rn(__half2float(s), __half2float(u)));
}

// Vectorized: one int4 (8 fp16) of gate + up per thread; m % 8 == 0.
__global__ void SiluMulV8(const int4* __restrict__ x, int4* __restrict__ out,
                          const int m8, const int n8) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n8) return;
  const int t = idx / m8;
  const int j = idx - t * m8;
  const int4* row = x + (int64_t)t * 2 * m8;
  const int4 gv = row[j];
  const int4 uv = row[j + m8];
  const half* g = reinterpret_cast<const half*>(&gv);
  const half* u = reinterpret_cast<const half*>(&uv);
  int4 ov;
  half* o = reinterpret_cast<half*>(&ov);
#pragma unroll
  for (int i = 0; i < 8; i++) o[i] = silu_mul_one(g[i], u[i]);
  out[idx] = ov;
}

// Scalar fallback for m % 8 != 0.
__global__ void SiluMul(const half* __restrict__ x, half* __restrict__ out,
                        const int m, const int n_out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_out) return;
  const int t = idx / m;
  const int j = idx - t * m;
  const half* row = x + (int64_t)t * 2 * m;
  out[idx] = silu_mul_one(row[j], row[j + m]);
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------
static void pick_grid_cap(int prob_m, int n_rows_out, dim3& blocks,
                          int& threads, int max_tm) {
  int dev, sms;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > max_tm);
  blocks = dim3(ceildiv(prob_m, thread_m), n_rows_out);
  threads = 32 * thread_m;
}

static void pick_grid(int prob_m, int n_rows_out, dim3& blocks, int& threads) {
  pick_grid_cap(prob_m, n_rows_out, blocks, threads, THREAD_M);
}

template <int BOOKS>
void launch_matvec(const int4* codes, const int4* B, half* C,
                   const int4* codebooks, const half* scales,
                   const int* expert_ids, int n_rows_out, int prob_m,
                   int prob_k, cudaStream_t stream) {
  dim3 blocks;
  int threads;
  pick_grid(prob_m, n_rows_out, blocks, threads);
  CodeKx16MatVecMoE<BOOKS><<<blocks, threads, 0, stream>>>(
      codes, B, C, codebooks, scales, expert_ids, prob_m, prob_k);
}

static bool env_flag(const char* name) {
  const char* e = getenv(name);
  return e && e[0] && e[0] != '0';
}

template <int BOOKS, bool BF16IN = false>
void launch_hybrid(const int4* codes, const int4* codebooks,
                   const half* scales, const int* aqlm_ids, const int4* packed,
                   const uchar2* bscale, const float* scale2, int s2n,
                   const int* nv_ids, const int4* B, half* C, int n_rows_out,
                   int prob_m, int prob_k, int row_div, cudaStream_t stream) {
  // V4 pipelined path (AQLM_GEMV_PIPELINE, default OFF; takes precedence
  // over GLM_MOE_DEDUP / GLM_MOE_LANE_ROWS, and subsumes lane-rows). Falls
  // back to the V2/V3 paths below when the shape doesn't fit the pipelined
  // kernel (activations > smem budget, or > 6 weight chunks per lane).
  // Values: "1"/"both" = V4 for w13+w2; "w2" = V4 only for the small-K
  // (cpl==1, i.e. w2/down) shape; "w13" = only the large-K shape. Live
  // slot-hot-fraction can sit in the regime where V4 w13 regresses vs V3
  // while w2 always wins, so "w2" allows a per-projection split.
  const char* pipe_env = getenv("AQLM_GEMV_PIPELINE");
  const bool pipe_on = pipe_env && pipe_env[0] && pipe_env[0] != '0';
  if (pipe_on && prob_k / 8 <= V4_MAX_ACT_INT4) {
    const int s_n = prob_k / 32;  // NVFP4 uint4s per row
    int G = 32;                   // lane-rows grouping, same rule as V3
    if (s_n >= 2 && s_n <= 16 && (s_n & (s_n - 1)) == 0) G = s_n;
    const int cpl = ceildiv(s_n, G);  // NVFP4 chunks per lane
    // large K: all chunks in flight, 1 row per warp
    auto kern = HybridMatVecMoEV4<BOOKS, 1, 6, BF16IN>;
    int RW = 1;
    bool ok = true;
    if (cpl == 1) {
      // small K: 4 rows per warp
      kern = HybridMatVecMoEV4<BOOKS, 4, 1, BF16IN>;
      RW = 4;
      if (strcmp(pipe_env, "w13") == 0) ok = false;
    } else if (cpl > 6) {
      ok = false;
    } else {
      if (strcmp(pipe_env, "w2") == 0) ok = false;
    }
    if (ok) {
      static bool logged4 = false;
      if (!logged4) {
        logged4 = true;
        fprintf(stderr,
                "[aqlm_moe_v2] V4 pipelined gemv active (G=%d RW=%d cpl=%d)\n",
                G, RW, cpl);
      }
      const int R = 32 / G;
      dim3 blocks;
      int threads;
      pick_grid_cap(ceildiv(prob_m, R * RW), n_rows_out, blocks, threads, 8);
      // Transposed grid: slots on x (see kernel comment).
      dim3 blocks_t(blocks.y, blocks.x);
      const int n_b = prob_k / 8;
      const int lut_int4 = (NVFP4_SMEM_EXTRA + 3) / 4;
      const int smem_int4 =
          n_b + (n_b / 8 > lut_int4 ? n_b / 8 : lut_int4);
      kern<<<blocks_t, threads, smem_int4 * (int)sizeof(int4), stream>>>(
          codes, codebooks, scales, aqlm_ids, packed, bscale, scale2, s2n,
          nv_ids, B, C, prob_m, prob_k, G, row_div);
      return;
    }
  }
  // DECODE-K features, env-gated per launch (cheap; also lets the tier-1
  // variant harness A/B within one process), default OFF -> V2 kernel.
  // GLM_MOE_DEDUP: 0/off | 2 = dedup pairs (UMAX=2, half the smem/regs) |
  // any other nonzero = UMAX=4.
  const char* de = getenv("GLM_MOE_DEDUP");
  int kDedup = 0;
  if (de && de[0] && de[0] != '0') {
    kDedup = atoi(de);
    if (kDedup <= 0) kDedup = 4;  // non-numeric truthy value -> default width
  }
  const bool kLaneRows = env_flag("GLM_MOE_LANE_ROWS");
  dim3 blocks;
  int threads;
  if (kDedup || kLaneRows) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      fprintf(stderr, "[aqlm_moe_v2] DECODE-K V3 kernel active: dedup=%d "
              "lane_rows=%d\n", kDedup, (int)kLaneRows);
    }
    int G = 32;
    if (kLaneRows) {
      const int s_n = prob_k / 32;  // NVFP4 uint4s per row (2*AQLM int4s)
      if (s_n >= 2 && s_n <= 16 && (s_n & (s_n - 1)) == 0) G = s_n;
    }
    const int R = 32 / G;
    pick_grid(ceildiv(prob_m, R), n_rows_out, blocks, threads);
    auto kern = HybridMatVecMoEV3<BOOKS, 1, BF16IN>;
    if (kDedup == 2) {
      kern = HybridMatVecMoEV3<BOOKS, 2, BF16IN>;
    } else if (kDedup) {
      kern = HybridMatVecMoEV3<BOOKS, 4, BF16IN>;
    }
    kern<<<blocks, threads, 0, stream>>>(
        codes, codebooks, scales, aqlm_ids, packed, bscale, scale2, s2n,
        nv_ids, B, C, prob_m, prob_k, n_rows_out, G, row_div);
    return;
  }
  pick_grid(prob_m, n_rows_out, blocks, threads);
  HybridMatVecMoE<BOOKS, BF16IN><<<blocks, threads, 0, stream>>>(
      codes, codebooks, scales, aqlm_ids, packed, bscale, scale2, s2n, nv_ids,
      B, C, prob_m, prob_k, row_div);
}

template <int BOOKS>
void launch_dequant(const int4* codes, half* out, const int4* codebooks,
                    const half* scales, const int* expert_list, int n_experts,
                    int prob_m, int prob_k, cudaStream_t stream) {
  dim3 blocks(ceildiv(prob_m, THREAD_M), n_experts);
  int threads = 32 * THREAD_M;
  CodeKx16DequantMoE<BOOKS><<<blocks, threads, 0, stream>>>(
      codes, out, codebooks, scales, expert_list, prob_m, prob_k);
}

}  // namespace aqlm_moe_v2

// ---------------------------------------------------------------------------
// Host entry points (signatures identical to baseline aqlm_moe.cu)
// ---------------------------------------------------------------------------
torch::Tensor aqlm_moe_gemv(const torch::Tensor& x, const torch::Tensor& codes,
                            const torch::Tensor& codebooks,
                            const torch::Tensor& scales,
                            const torch::Tensor& expert_ids) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16 && x.is_contiguous());
  TORCH_CHECK(codes.dim() == 4 && codes.dtype() == torch::kInt16);
  TORCH_CHECK(codebooks.size(1) == 65536 && codebooks.size(2) == 8);
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32 && expert_ids.is_contiguous());
  const int64_t n = x.size(0);
  const int64_t k = x.size(1);
  const int64_t m = codes.size(2);
  const int64_t books = codes.size(1);
  TORCH_CHECK(codes.size(3) * 8 == k, "codes K mismatch");
  TORCH_CHECK(expert_ids.size(0) == n);
  TORCH_CHECK(k % 64 == 0, "K must be a multiple of 64");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto out = torch::empty({n, m}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (n == 0) return out;
  auto run =
      books == 1 ? aqlm_moe_v2::launch_matvec<1> : aqlm_moe_v2::launch_matvec<2>;
  TORCH_CHECK(books == 1 || books == 2, "books must be 1 or 2");
  run((const int4*)codes.data_ptr(), (const int4*)x.data_ptr(),
      (half*)out.data_ptr(), (const int4*)codebooks.data_ptr(),
      (const half*)scales.data_ptr(), expert_ids.data_ptr<int>(), (int)n,
      (int)m, (int)k, stream);
  return out;
}

torch::Tensor nvfp4_moe_gemv(const torch::Tensor& x,
                             const torch::Tensor& packed,
                             const torch::Tensor& bscale,
                             const torch::Tensor& scale2,
                             const torch::Tensor& expert_ids) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16 && x.is_contiguous());
  TORCH_CHECK(packed.dtype() == torch::kUInt8 && packed.is_contiguous());
  TORCH_CHECK(bscale.dtype() == torch::kUInt8 && bscale.is_contiguous());
  TORCH_CHECK(scale2.dtype() == torch::kFloat32 && scale2.is_contiguous());
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32);
  const int64_t n = x.size(0);
  const int64_t k = x.size(1);
  const int64_t m = packed.size(1);
  TORCH_CHECK(packed.size(2) * 2 == k, "packed K mismatch");
  TORCH_CHECK(bscale.size(2) * 16 == k, "bscale K mismatch");
  TORCH_CHECK(k % 64 == 0);

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto out = torch::empty({n, m}, x.options());
  if (n == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 blocks;
  int threads;
  aqlm_moe_v2::pick_grid((int)m, (int)n, blocks, threads);
  aqlm_moe_v2::NvFp4MatVecMoE<<<blocks, threads, 0, stream>>>(
      (const int4*)packed.data_ptr(), (const uchar2*)bscale.data_ptr(),
      scale2.data_ptr<float>(), (int)scale2.size(1),
      (const int4*)x.data_ptr(), (half*)out.data_ptr(),
      expert_ids.data_ptr<int>(), (int)m, (int)k);
  return out;
}

// Fused per-projection gemv over both storage formats.
//   x:          [N, K] fp16 — or bf16 (AQLM_GEMV_BF16IN glue): converted to
//               fp16 in-kernel during smem staging, bit-identical to feeding
//               x.to(torch.float16)
//   codes/codebooks/scales/aqlm_ids: AQLM set (aqlm_ids[slot] < 0 => not AQLM)
//   packed/bscale/scale2/nv_ids:     NVFP4 set (may be empty when n_nvfp4=0)
//   row_div:    slot s reads x row s / row_div (AQLM_GEMV_ROWMAP glue;
//               default 1 = previous behavior). With row_div = top_k and
//               compact [T, K] activations the output is bit-identical to
//               feeding x.repeat_interleave(top_k, dim=0) at row_div = 1,
//               without materializing the expansion.
// returns [n_slots, M] fp16 (n_slots = N * row_div = aqlm_ids/nv_ids length);
// each slot row computed by exactly one path.
torch::Tensor hybrid_moe_gemv(const torch::Tensor& x,
                              const torch::Tensor& codes,
                              const torch::Tensor& codebooks,
                              const torch::Tensor& scales,
                              const torch::Tensor& aqlm_ids,
                              const torch::Tensor& packed,
                              const torch::Tensor& bscale,
                              const torch::Tensor& scale2,
                              const torch::Tensor& nv_ids,
                              int64_t row_div = 1) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
              (x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16));
  TORCH_CHECK(codes.dim() == 4 && codes.dtype() == torch::kInt16);
  TORCH_CHECK(codebooks.size(1) == 65536 && codebooks.size(2) == 8);
  TORCH_CHECK(aqlm_ids.dtype() == torch::kInt32 && aqlm_ids.is_contiguous());
  TORCH_CHECK(nv_ids.dtype() == torch::kInt32 && nv_ids.is_contiguous());
  TORCH_CHECK(row_div >= 1, "row_div must be >= 1");
  const int64_t n = x.size(0) * row_div;  // slots
  const int64_t k = x.size(1);
  const int64_t m = codes.size(2);
  const int64_t books = codes.size(1);
  TORCH_CHECK(codes.size(3) * 8 == k, "codes K mismatch");
  TORCH_CHECK(aqlm_ids.size(0) == n && nv_ids.size(0) == n,
              "slot ids must have x.size(0) * row_div rows");
  TORCH_CHECK(k % 64 == 0, "K must be a multiple of 64");
  const bool has_nv = packed.numel() > 0;
  int s2n = 1;
  if (has_nv) {
    TORCH_CHECK(packed.dtype() == torch::kUInt8 && packed.is_contiguous());
    TORCH_CHECK(bscale.dtype() == torch::kUInt8 && bscale.is_contiguous());
    TORCH_CHECK(scale2.dtype() == torch::kFloat32 && scale2.is_contiguous());
    TORCH_CHECK(packed.size(1) == m && packed.size(2) * 2 == k);
    TORCH_CHECK(bscale.size(2) * 16 == k);
    s2n = (int)scale2.size(1);
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto out = torch::empty({n, m}, x.options().dtype(torch::kFloat16));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (n == 0) return out;

  const bool bf16in = x.dtype() == torch::kBFloat16;
  auto run = books == 1
                 ? (bf16in ? aqlm_moe_v2::launch_hybrid<1, true>
                           : aqlm_moe_v2::launch_hybrid<1, false>)
                 : (bf16in ? aqlm_moe_v2::launch_hybrid<2, true>
                           : aqlm_moe_v2::launch_hybrid<2, false>);
  TORCH_CHECK(books == 1 || books == 2, "books must be 1 or 2");
  run((const int4*)codes.data_ptr(), (const int4*)codebooks.data_ptr(),
      (const half*)scales.data_ptr(), aqlm_ids.data_ptr<int>(),
      has_nv ? (const int4*)packed.data_ptr() : nullptr,
      has_nv ? (const uchar2*)bscale.data_ptr() : nullptr,
      has_nv ? scale2.data_ptr<float>() : nullptr, s2n,
      nv_ids.data_ptr<int>(), (const int4*)x.data_ptr(),
      (half*)out.data_ptr(), (int)n, (int)m, (int)k, (int)row_div, stream);
  return out;
}

// Fused SwiGLU mid-tail: x [S, 2m] fp16 -> [S, m] fp16,
// out[s, j] = silu(x[s, j]) * x[s, m + j]. Bit-exact vs torch's
// F.silu(x[..., :m]) * x[..., m:] eager pair (see SiluMul kernel note).
torch::Tensor silu_mul(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16 &&
              x.is_contiguous() && x.dim() == 2);
  const int64_t s = x.size(0);
  const int64_t m2 = x.size(1);
  TORCH_CHECK(m2 % 2 == 0, "last dim must be even (gate|up)");
  const int64_t m = m2 / 2;

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto out = torch::empty({s, m}, x.options());
  const int64_t n_out = s * m;
  if (n_out == 0) return out;
  TORCH_CHECK(n_out <= INT32_MAX, "silu_mul output too large");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int threads = 256;
  if (m % 8 == 0) {
    const int n8 = (int)(n_out / 8);
    aqlm_moe_v2::SiluMulV8<<<aqlm_moe_v2::ceildiv(n8, threads), threads, 0,
                             stream>>>(
        (const int4*)x.data_ptr(), (int4*)out.data_ptr(), (int)(m / 8), n8);
  } else {
    aqlm_moe_v2::SiluMul<<<aqlm_moe_v2::ceildiv((int)n_out, threads), threads,
                           0, stream>>>(
        (const half*)x.data_ptr(), (half*)out.data_ptr(), (int)m, (int)n_out);
  }
  return out;
}

torch::Tensor aqlm_moe_dequant(const torch::Tensor& codes,
                               const torch::Tensor& codebooks,
                               const torch::Tensor& scales,
                               const torch::Tensor& expert_list) {
  TORCH_CHECK(codes.is_cuda() && codes.dim() == 4 &&
              codes.dtype() == torch::kInt16);
  TORCH_CHECK(expert_list.dtype() == torch::kInt32 &&
              expert_list.is_contiguous());
  const int64_t g = expert_list.size(0);
  const int64_t books = codes.size(1);
  const int64_t m = codes.size(2);
  const int64_t k = codes.size(3) * 8;

  const at::cuda::OptionalCUDAGuard guard(device_of(codes));
  auto out = torch::empty({g, m, k},
                          codebooks.options().dtype(torch::kFloat16));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (g == 0) return out;
  auto run = books == 1 ? aqlm_moe_v2::launch_dequant<1>
                        : aqlm_moe_v2::launch_dequant<2>;
  TORCH_CHECK(books == 1 || books == 2, "books must be 1 or 2");
  run((const int4*)codes.data_ptr(), (half*)out.data_ptr(),
      (const int4*)codebooks.data_ptr(), (const half*)scales.data_ptr(),
      expert_list.data_ptr<int>(), (int)g, (int)m, (int)k, stream);
  return out;
}

torch::Tensor nvfp4_moe_dequant(const torch::Tensor& packed,
                                const torch::Tensor& bscale,
                                const torch::Tensor& scale2,
                                const torch::Tensor& expert_list) {
  TORCH_CHECK(packed.is_cuda() && packed.dtype() == torch::kUInt8);
  TORCH_CHECK(expert_list.dtype() == torch::kInt32 && expert_list.is_contiguous());
  const int64_t g = expert_list.size(0);
  const int64_t m = packed.size(1);
  const int64_t k = packed.size(2) * 2;

  const at::cuda::OptionalCUDAGuard guard(device_of(packed));
  auto out = torch::empty({g, m, k},
                          torch::TensorOptions()
                              .dtype(torch::kFloat16)
                              .device(packed.device()));
  if (g == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 blocks(aqlm_moe_v2::ceildiv((int)m, aqlm_moe_v2::THREAD_M), g);
  aqlm_moe_v2::NvFp4DequantMoE<<<blocks, 32 * aqlm_moe_v2::THREAD_M, 0,
                                 stream>>>(
      (const int4*)packed.data_ptr(), (const uchar2*)bscale.data_ptr(),
      scale2.data_ptr<float>(), (int)scale2.size(1), (half*)out.data_ptr(),
      expert_list.data_ptr<int>(), (int)m, (int)k);
  return out;
}

// Fused weighted top-k combine (decode epilogue).
//   y:       [S, M] fp16, S = tokens * top_k, slot-major (token t's slots at
//            rows t*top_k .. t*top_k+top_k-1)
//   weights: [S] fp32, same slot order
//   out_mode: 0 = fp32, 1 = fp16, 2 = bf16 output
// returns [S / top_k, M]; ascending-slot fp32 accumulation (see kernel note).
torch::Tensor moe_combine(const torch::Tensor& y,
                          const torch::Tensor& weights, int64_t top_k,
                          int64_t out_mode) {
  TORCH_CHECK(y.is_cuda() && y.dtype() == torch::kFloat16 &&
              y.is_contiguous() && y.dim() == 2);
  TORCH_CHECK(weights.is_cuda() && weights.dtype() == torch::kFloat32 &&
              weights.is_contiguous());
  const int64_t s = y.size(0);
  const int64_t m = y.size(1);
  TORCH_CHECK(top_k >= 1 && s % top_k == 0, "S must be divisible by top_k");
  TORCH_CHECK(weights.numel() == s, "weights/slots mismatch");
  TORCH_CHECK(out_mode >= 0 && out_mode <= 2, "out_mode must be 0/1/2");
  const int64_t t = s / top_k;
  const auto dt = out_mode == 0   ? torch::kFloat32
                  : out_mode == 1 ? torch::kFloat16
                                  : torch::kBFloat16;

  const at::cuda::OptionalCUDAGuard guard(device_of(y));
  auto out = torch::empty({t, m}, y.options().dtype(dt));
  const int64_t n_out = t * m;
  if (n_out == 0) return out;
  TORCH_CHECK(n_out <= INT32_MAX, "combine output too large");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int threads = 256;
  const int blocks = aqlm_moe_v2::ceildiv((int)n_out, threads);
  const half* yp = (const half*)y.data_ptr();
  const float* wp = weights.data_ptr<float>();
  if (out_mode == 0) {
    aqlm_moe_v2::MoECombine<float><<<blocks, threads, 0, stream>>>(
        yp, wp, out.data_ptr<float>(), (int)top_k, (int)m, (int)n_out);
  } else if (out_mode == 1) {
    aqlm_moe_v2::MoECombine<half><<<blocks, threads, 0, stream>>>(
        yp, wp, (half*)out.data_ptr(), (int)top_k, (int)m, (int)n_out);
  } else {
    aqlm_moe_v2::MoECombine<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        yp, wp, (__nv_bfloat16*)out.data_ptr(), (int)top_k, (int)m,
        (int)n_out);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aqlm_moe_gemv", &aqlm_moe_gemv, "AQLM MoE gemv v2 (decode path)");
  m.def("aqlm_moe_dequant", &aqlm_moe_dequant,
        "AQLM MoE batched expert dequant (prefill path)");
  m.def("nvfp4_moe_gemv", &nvfp4_moe_gemv, "NVFP4 MoE gemv v2 (decode path)");
  m.def("nvfp4_moe_dequant", &nvfp4_moe_dequant,
        "NVFP4 MoE batched expert dequant (prefill path)");
  m.def("hybrid_moe_gemv", &hybrid_moe_gemv,
        "Fused AQLM+NVFP4 MoE gemv: one launch per projection",
        py::arg("x"), py::arg("codes"), py::arg("codebooks"),
        py::arg("scales"), py::arg("aqlm_ids"), py::arg("packed"),
        py::arg("bscale"), py::arg("scale2"), py::arg("nv_ids"),
        py::arg("row_div") = 1);
  m.def("moe_combine", &moe_combine,
        "Fused weighted top-k slot combine (decode epilogue)");
  m.def("silu_mul", &silu_mul,
        "Fused SwiGLU mid-tail: silu(x[:, :m]) * x[:, m:], bit-exact vs "
        "torch's eager silu+mul pair");
}
