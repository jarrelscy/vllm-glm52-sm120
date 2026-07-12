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
 *
 * Numerics are kept bit-identical in accumulation order to the baseline for
 * the AQLM path; the NVFP4 path accumulates each 16-value scale block in
 * half2 (8 hfma2) before the fp32 block reduction, matching the baseline to
 * ~1e-3 relative (validated in bench.py).
 */

#include <cuda.h>
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
// AQLM: per-(slot,row) warp gemv body. Call with all threads of the block
// (contains __syncthreads); `expert` must be uniform within the block.
// sh_b must hold >= 32*9 int4.
// ---------------------------------------------------------------------------
template <int BOOKS>
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
      if (b_gl_rd + i < prob_k / 8) sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
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
      if (b_gl_rd + i < prob_k / 8) sh_b[i] = B[b_gl_rd + i];
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
// ---------------------------------------------------------------------------
template <int BOOKS>
__global__ void HybridMatVecMoE(
    const int4* __restrict__ codes, const int4* __restrict__ codebooks,
    const half* __restrict__ scales, const int* __restrict__ aqlm_ids,
    const int4* __restrict__ packed, const uchar2* __restrict__ bscale,
    const float* __restrict__ scale2, const int s2n,
    const int* __restrict__ nv_ids, const int4* __restrict__ B_all,
    half* __restrict__ C, const int prob_m, const int prob_k) {
  const int slot = blockIdx.y;
  const int a_id = aqlm_ids[slot];
  const int n_id = nv_ids[slot];
  const int row = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  const bool pred = row < prob_m;
  __shared__ int4 sh_b[32 * 9 + (NVFP4_SMEM_EXTRA + 3) / 4];

  const int4* B = B_all + (int64_t)slot * (prob_k / 8);
  half* C_slot = C + (int64_t)slot * prob_m;

  if (a_id >= 0) {
    aqlm_slot_gemv<BOOKS>(codes, codebooks, scales, a_id, B, C_slot, prob_m,
                          prob_k, sh_b, row, pred);
  } else if (n_id >= 0) {
    nvfp4_slot_gemv(packed, bscale, scale2, s2n, n_id, B, C_slot, prob_m,
                    prob_k, sh_b, row, pred);
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
template <int BOOKS, int UMAX>
__device__ __forceinline__ void aqlm_multi_gemv(
    const int4* __restrict__ codes, const int4* __restrict__ codebooks,
    const half* __restrict__ scales, const int expert,
    const int4* __restrict__ B_all, half* __restrict__ C,
    const int* __restrict__ slots_u, const int U, const int prob_m,
    const int prob_k, int4* sh_b, const int row0, const int G) {
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
        sh_b[su * 288 + 9 * (ii / 8) + ii % 8] =
            B_all[(int64_t)slots_u[su] * (prob_k / 8) + b_gl_rd + ii];
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

template <int UMAX>
__device__ __forceinline__ void nvfp4_multi_gemv(
    const int4* __restrict__ packed, const uchar2* __restrict__ bscale,
    const float* __restrict__ scale2, const int s2n, const int expert,
    const int4* __restrict__ B_all, half* __restrict__ C,
    const int* __restrict__ slots_u, const int U, const int prob_m,
    const int prob_k, int4* sh_b, const int row0, const int G) {
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
        sh_b[su * 288 + ii] =
            B_all[(int64_t)slots_u[su] * (prob_k / 8) + b_gl_rd + ii];
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

template <int BOOKS, int UMAX>
__global__ void HybridMatVecMoEV3(
    const int4* __restrict__ codes, const int4* __restrict__ codebooks,
    const half* __restrict__ scales, const int* __restrict__ aqlm_ids,
    const int4* __restrict__ packed, const uchar2* __restrict__ bscale,
    const float* __restrict__ scale2, const int s2n,
    const int* __restrict__ nv_ids, const int4* __restrict__ B_all,
    half* __restrict__ C, const int prob_m, const int prob_k,
    const int n_slots, const int G) {
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
    aqlm_multi_gemv<BOOKS, UMAX>(codes, codebooks, scales, my_a, B_all, C,
                                 slots_u, U, prob_m, prob_k, sh_b, row0, G);
  } else if (fmt == 1) {
    nvfp4_multi_gemv<UMAX>(packed, bscale, scale2, s2n, my_n, B_all, C,
                           slots_u, U, prob_m, prob_k, sh_b, row0, G);
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
// Launch helpers
// ---------------------------------------------------------------------------
static void pick_grid(int prob_m, int n_rows_out, dim3& blocks, int& threads) {
  int dev, sms;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);
  blocks = dim3(ceildiv(prob_m, thread_m), n_rows_out);
  threads = 32 * thread_m;
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

template <int BOOKS>
void launch_hybrid(const int4* codes, const int4* codebooks,
                   const half* scales, const int* aqlm_ids, const int4* packed,
                   const uchar2* bscale, const float* scale2, int s2n,
                   const int* nv_ids, const int4* B, half* C, int n_rows_out,
                   int prob_m, int prob_k, cudaStream_t stream) {
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
    auto kern = HybridMatVecMoEV3<BOOKS, 1>;
    if (kDedup == 2) {
      kern = HybridMatVecMoEV3<BOOKS, 2>;
    } else if (kDedup) {
      kern = HybridMatVecMoEV3<BOOKS, 4>;
    }
    kern<<<blocks, threads, 0, stream>>>(
        codes, codebooks, scales, aqlm_ids, packed, bscale, scale2, s2n,
        nv_ids, B, C, prob_m, prob_k, n_rows_out, G);
    return;
  }
  pick_grid(prob_m, n_rows_out, blocks, threads);
  HybridMatVecMoE<BOOKS><<<blocks, threads, 0, stream>>>(
      codes, codebooks, scales, aqlm_ids, packed, bscale, scale2, s2n, nv_ids,
      B, C, prob_m, prob_k);
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
//   x:          [N, K] fp16
//   codes/codebooks/scales/aqlm_ids: AQLM set (aqlm_ids[slot] < 0 => not AQLM)
//   packed/bscale/scale2/nv_ids:     NVFP4 set (may be empty when n_nvfp4=0)
// returns [N, M] fp16; each slot row computed by exactly one path.
torch::Tensor hybrid_moe_gemv(const torch::Tensor& x,
                              const torch::Tensor& codes,
                              const torch::Tensor& codebooks,
                              const torch::Tensor& scales,
                              const torch::Tensor& aqlm_ids,
                              const torch::Tensor& packed,
                              const torch::Tensor& bscale,
                              const torch::Tensor& scale2,
                              const torch::Tensor& nv_ids) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16 && x.is_contiguous());
  TORCH_CHECK(codes.dim() == 4 && codes.dtype() == torch::kInt16);
  TORCH_CHECK(codebooks.size(1) == 65536 && codebooks.size(2) == 8);
  TORCH_CHECK(aqlm_ids.dtype() == torch::kInt32 && aqlm_ids.is_contiguous());
  TORCH_CHECK(nv_ids.dtype() == torch::kInt32 && nv_ids.is_contiguous());
  const int64_t n = x.size(0);
  const int64_t k = x.size(1);
  const int64_t m = codes.size(2);
  const int64_t books = codes.size(1);
  TORCH_CHECK(codes.size(3) * 8 == k, "codes K mismatch");
  TORCH_CHECK(aqlm_ids.size(0) == n && nv_ids.size(0) == n);
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
  auto out = torch::empty({n, m}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (n == 0) return out;

  auto run = books == 1 ? aqlm_moe_v2::launch_hybrid<1>
                        : aqlm_moe_v2::launch_hybrid<2>;
  TORCH_CHECK(books == 1 || books == 2, "books must be 1 or 2");
  run((const int4*)codes.data_ptr(), (const int4*)codebooks.data_ptr(),
      (const half*)scales.data_ptr(), aqlm_ids.data_ptr<int>(),
      has_nv ? (const int4*)packed.data_ptr() : nullptr,
      has_nv ? (const uchar2*)bscale.data_ptr() : nullptr,
      has_nv ? scale2.data_ptr<float>() : nullptr, s2n,
      nv_ids.data_ptr<int>(), (const int4*)x.data_ptr(),
      (half*)out.data_ptr(), (int)n, (int)m, (int)k, stream);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aqlm_moe_gemv", &aqlm_moe_gemv, "AQLM MoE gemv v2 (decode path)");
  m.def("aqlm_moe_dequant", &aqlm_moe_dequant,
        "AQLM MoE batched expert dequant (prefill path)");
  m.def("nvfp4_moe_gemv", &nvfp4_moe_gemv, "NVFP4 MoE gemv v2 (decode path)");
  m.def("nvfp4_moe_dequant", &nvfp4_moe_dequant,
        "NVFP4 MoE batched expert dequant (prefill path)");
  m.def("hybrid_moe_gemv", &hybrid_moe_gemv,
        "Fused AQLM+NVFP4 MoE gemv: one launch per projection");
}
