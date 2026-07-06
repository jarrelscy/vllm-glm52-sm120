/*
 * AQLM fused-MoE kernels for the GLM-5.2 hybrid NVFP4+AQLM build.
 *
 * Adapted from csrc/quantization/aqlm/gemm_kernels.cu (removed in v0.11,
 * originally from https://github.com/Vahe1994/AQLM, Apache-2.0), extended
 * with:
 *   - an expert dimension: codes are [E, BOOKS, M, K/8] and each output row
 *     block resolves its expert id through topk_ids (decode gemv) or an
 *     expert list (prefill dequant),
 *   - K in {1, 2} additive codebooks of 2^16 entries x 8 fp16 values,
 *   - per-expert per-output-channel fp16 scales fused into the epilogue.
 *
 * Built as a JIT torch extension (see aqlm_moe_ext.py); plain CUDA C++,
 * no architecture-specific instructions => runs on SM80..SM120.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace aqlm_moe {

constexpr int THREAD_M = 16;

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

// Decode-path fused MoE gemv.
//   codes:      [E, BOOKS, M, K/8] uint16 (int4-aligned rows)
//   B_all:      [N, K] fp16 activations
//   C:          [N, M] fp16 output (one row per (token, slot))
//   codebooks:  [BOOKS, 65536] int4 entries (8 fp16 each)
//   scales:     [E, M] fp16 per-output-channel
//   expert_ids: [N] int32, expert for each output row
// Grid: (ceildiv(M, thread_m), N). Threads: 32 * thread_m.
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

  const int a_gl_stride = prob_k / 8 / 8;  // int4s per code row
  const int row = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  const bool pred = row < prob_m;

  if (expert < 0) {  // slot handled by another format: contribute zeros
    if (pred && threadIdx.x % 32 == 0) {
      C[(int64_t)slot * prob_m + row] = __float2half(0.f);
    }
    return;
  }

  const int4* B = B_all + (int64_t)slot * (prob_k / 8);

  // Per-book base pointers for this expert's code plane.
  const int4* a_base[BOOKS];
#pragma unroll
  for (int b = 0; b < BOOKS; b++) {
    a_base[b] = codes + ((int64_t)expert * BOOKS + b) * prob_m * a_gl_stride;
  }

  int b_gl_rd = 0;
  int a_rd = a_gl_stride * row + threadIdx.x % 32;
  const int a_end = a_gl_stride * row + a_gl_stride;

  __shared__ int4 sh_b[32 * 9];
  float res = 0;

  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += 32 * 8;

    int b_sh_rd = 9 * (threadIdx.x % 32);
    if (pred && a_rd < a_end) {
      uint32_t dec[4];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        half2 wsum[4] = {};
#pragma unroll
        for (int b = 0; b < BOOKS; b++) {
          const uint16_t* enc = reinterpret_cast<const uint16_t*>(&a_base[b][a_rd]);
          // Bypass L1: codebook rows are effectively random-access.
          asm volatile("ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
                       : "l"((void*)&codebooks[(int64_t)b * 65536 + enc[i]]));
          half2* a = reinterpret_cast<half2*>(&dec);
#pragma unroll
          for (int j = 0; j < 4; j++) wsum[j] = __hadd2(wsum[j], a[j]);
        }
        half2* bb = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
        half2 res2 = {};
#pragma unroll
        for (int j = 0; j < 4; j++) res2 = __hfma2(wsum[j], bb[j], res2);
        res += __half2float(res2.x) + __half2float(res2.y);
        b_sh_rd++;
      }
      a_rd += 32;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = 16; i > 0; i /= 2) res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0) {
      const float s = __half2float(scales[(int64_t)expert * prob_m + row]);
      C[(int64_t)slot * prob_m + row] = __float2half(res * s);
    }
  }
}

// Prefill-path batched expert dequant.
//   out: [G, M, K] fp16, scales fused.
// Grid: (ceildiv(M, thread_m), G). Threads: 32 * thread_m.
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
          const uint16_t* enc = reinterpret_cast<const uint16_t*>(&a_base[b][a_rd]);
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

// ---------------------------------------------------------------------------
// NVFP4 (W4A16) per-expert kernels for the per-expert hybrid MoE.
//   packed: u8 [E, M, K/2]  two fp4 per byte, low nibble = even element
//   bscale: u8 [E, M, K/16] fp8 e4m3 block scales (16 elements per block)
//   scale2: f32 [E, S]      per-projection global scale; a row r uses
//                           scale2[e][r * S / M] (S=2 for w13 gate/up, 1 for w2)
// ---------------------------------------------------------------------------

__device__ __constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t v) {
  __nv_fp8_e4m3 f;
  f.__x = v;
  return float(f);
}

// One warp-row gemv step over a 32-value chunk held in an int4 of packed
// data, with two fp8 block scales.
__device__ __forceinline__ float nvfp4_chunk_dot(const uint4 w,
                                                 const uchar2 bs,
                                                 const half2* __restrict__ bb) {
  const float s[2] = {fp8_e4m3_to_float(bs.x), fp8_e4m3_to_float(bs.y)};
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w);
  float res = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const float scale = s[i >> 3];
    const half2 b2 = bb[i];
    res += scale * (kFp4Lut[bytes[i] & 0xF] * __half2float(b2.x) +
                    kFp4Lut[bytes[i] >> 4] * __half2float(b2.y));
  }
  return res;
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

  if (expert < 0) {
    if (pred && threadIdx.x % 32 == 0) {
      C[(int64_t)slot * prob_m + row] = __float2half(0.f);
    }
    return;
  }

  const int a_gl_stride = prob_k / 32;  // int4s (32 fp4) per row
  const int4* a_row = packed + ((int64_t)expert * prob_m + row) * a_gl_stride;
  const uchar2* s_row =
      bscale + ((int64_t)expert * prob_m + row) * (prob_k / 32);
  const int4* B = B_all + (int64_t)slot * (prob_k / 8);

  __shared__ int4 sh_b[32 * 4];
  float res = 0;

  const int lane = threadIdx.x % 32;
  int iters = (prob_k / 8 + 4 * 32 - 1) / (4 * 32);
  int b_gl_rd = 0;
  int a_rd = lane;
  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 4; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) sh_b[i] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += 32 * 4;

    if (pred && a_rd < a_gl_stride) {
      const uint4 w = *reinterpret_cast<const uint4*>(&a_row[a_rd]);
      const uchar2 bs = s_row[a_rd];
      const half2* bb = reinterpret_cast<const half2*>(&sh_b[lane * 4]);
      res += nvfp4_chunk_dot(w, bs, bb);
      a_rd += 32;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = 16; i > 0; i /= 2) res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0) {
      const float g = scale2[expert * s2n + (int)(((int64_t)row * s2n) / prob_m)];
      C[(int64_t)slot * prob_m + row] = __float2half(res * g);
    }
  }
}

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

template <int BOOKS>
void launch_matvec(const int4* codes, const int4* B, half* C,
                   const int4* codebooks, const half* scales,
                   const int* expert_ids, int n_rows_out, int prob_m,
                   int prob_k, cudaStream_t stream) {
  int dev, sms;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);
  dim3 blocks(ceildiv(prob_m, thread_m), n_rows_out);
  int threads = 32 * thread_m;
  CodeKx16MatVecMoE<BOOKS><<<blocks, threads, 0, stream>>>(
      codes, B, C, codebooks, scales, expert_ids, prob_m, prob_k);
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

}  // namespace aqlm_moe

// x:          [N, K] fp16
// codes:      [E, BOOKS, M, K/8] int16
// codebooks:  [BOOKS, 65536, 8] fp16
// scales:     [E, M] fp16
// expert_ids: [N] int32
// returns     [N, M] fp16
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
  auto run = books == 1 ? aqlm_moe::launch_matvec<1> : aqlm_moe::launch_matvec<2>;
  TORCH_CHECK(books == 1 || books == 2, "books must be 1 or 2");
  run((const int4*)codes.data_ptr(), (const int4*)x.data_ptr(),
      (half*)out.data_ptr(), (const int4*)codebooks.data_ptr(),
      (const half*)scales.data_ptr(), expert_ids.data_ptr<int>(), (int)n,
      (int)m, (int)k, stream);
  return out;
}

// returns [G, M, K] fp16 dequantized (scales applied)
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
  auto run =
      books == 1 ? aqlm_moe::launch_dequant<1> : aqlm_moe::launch_dequant<2>;
  TORCH_CHECK(books == 1 || books == 2, "books must be 1 or 2");
  run((const int4*)codes.data_ptr(), (half*)out.data_ptr(),
      (const int4*)codebooks.data_ptr(), (const half*)scales.data_ptr(),
      expert_list.data_ptr<int>(), (int)g, (int)m, (int)k, stream);
  return out;
}

// x:          [N, K] fp16
// packed:     [E, M, K/2] u8
// bscale:     [E, M, K/16] u8 (fp8 e4m3 bits)
// scale2:     [E, S] f32
// expert_ids: [N] int32 (-1 -> zero row)
// returns     [N, M] fp16
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

  int dev, sms;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int waves = 0, thread_m;
  do {
    waves++;
    thread_m = aqlm_moe::ceildiv((int)m, waves * sms);
  } while (thread_m > aqlm_moe::THREAD_M);
  dim3 blocks(aqlm_moe::ceildiv((int)m, thread_m), n);
  aqlm_moe::NvFp4MatVecMoE<<<blocks, 32 * thread_m, 0, stream>>>(
      (const int4*)packed.data_ptr(), (const uchar2*)bscale.data_ptr(),
      scale2.data_ptr<float>(), (int)scale2.size(1),
      (const int4*)x.data_ptr(), (half*)out.data_ptr(),
      expert_ids.data_ptr<int>(), (int)m, (int)k);
  return out;
}

// returns [G, M, K] fp16 (scale2 applied)
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
  dim3 blocks(aqlm_moe::ceildiv((int)m, aqlm_moe::THREAD_M), g);
  aqlm_moe::NvFp4DequantMoE<<<blocks, 32 * aqlm_moe::THREAD_M, 0, stream>>>(
      (const int4*)packed.data_ptr(), (const uchar2*)bscale.data_ptr(),
      scale2.data_ptr<float>(), (int)scale2.size(1), (half*)out.data_ptr(),
      expert_list.data_ptr<int>(), (int)m, (int)k);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aqlm_moe_gemv", &aqlm_moe_gemv,
        "AQLM MoE gemv (decode path): per-row expert gather");
  m.def("aqlm_moe_dequant", &aqlm_moe_dequant,
        "AQLM MoE batched expert dequant (prefill path)");
  m.def("nvfp4_moe_gemv", &nvfp4_moe_gemv,
        "NVFP4 W4A16 MoE gemv (decode path): per-row expert gather");
  m.def("nvfp4_moe_dequant", &nvfp4_moe_dequant,
        "NVFP4 MoE batched expert dequant (prefill path)");
}
