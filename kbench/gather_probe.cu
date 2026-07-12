// Raw 16B-gather throughput probe: L2 vs local SMEM vs DSMEM (cluster).
// Decides DECODE-K idea 1 (cluster codebook staging).
// Build: nvcc -O3 -arch=sm_120 -o gather_probe gather_probe.cu
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e = (x); if (e) { \
  printf("ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); } } while (0)

__device__ __forceinline__ uint32_t lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s >> 16;  // 16-bit index
}

// A: random 16B gathers from a 1MiB table via L2 (__ldcg), like the kernel.
__global__ void g_l2(const uint4* __restrict__ tab, uint4* out, int iters) {
  uint32_t s = threadIdx.x * 2654435761u + blockIdx.x * 40503u + 1;
  uint4 acc = {};
  for (int i = 0; i < iters; i += 8) {
    uint4 w[8];
#pragma unroll
    for (int u = 0; u < 8; u++) w[u] = __ldcg(tab + lcg(s));
#pragma unroll
    for (int u = 0; u < 8; u++) {
      acc.x ^= w[u].x; acc.y ^= w[u].y; acc.z ^= w[u].z; acc.w ^= w[u].w;
    }
  }
  if (acc.x == 0xdeadbeef) *out = acc;
}

// B: random 16B gathers from local SMEM (48KB slice = 3072 entries).
__global__ void g_smem(const uint4* __restrict__ tab, uint4* out, int iters) {
  __shared__ uint4 sh[3072];
  for (int i = threadIdx.x; i < 3072; i += blockDim.x) sh[i] = tab[i];
  __syncthreads();
  uint32_t s = threadIdx.x * 2654435761u + blockIdx.x * 40503u + 1;
  uint4 acc = {};
  for (int i = 0; i < iters; i += 8) {
    uint4 w[8];
#pragma unroll
    for (int u = 0; u < 8; u++) w[u] = sh[lcg(s) % 3072u];
#pragma unroll
    for (int u = 0; u < 8; u++) {
      acc.x ^= w[u].x; acc.y ^= w[u].y; acc.z ^= w[u].z; acc.w ^= w[u].w;
    }
  }
  if (acc.x == 0xdeadbeef) *out = acc;
}

// C: cluster of CS blocks, each stages 65536/CS entries; random gathers over
// the full table via DSMEM map_shared_rank.
template <int CS>
__global__ void __cluster_dims__(CS, 1, 1) g_dsmem(const uint4* __restrict__ tab,
                                                   uint4* out, int iters) {
  extern __shared__ uint4 sh[];
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int per = 65536 / CS;
  for (int i = threadIdx.x; i < per; i += blockDim.x)
    sh[i] = tab[rank * per + i];
  cl.sync();
  uint32_t s = threadIdx.x * 2654435761u + blockIdx.x * 40503u + 1;
  uint4 acc = {};
  for (int i = 0; i < iters; i += 8) {
    uint4* w[8];
    uint32_t idx[8];
#pragma unroll
    for (int u = 0; u < 8; u++) idx[u] = lcg(s);
#pragma unroll
    for (int u = 0; u < 8; u++) {
      uint4* remote = cl.map_shared_rank(sh, idx[u] / per);
      w[u] = remote + idx[u] % per;
    }
#pragma unroll
    for (int u = 0; u < 8; u++) {
      uint4 v = *w[u];
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  }
  cl.sync();
  if (acc.x == 0xdeadbeef) *out = acc;
}

float run(void (*k)(const uint4*, uint4*, int), dim3 grid, int block, int smem,
          int cs, const uint4* tab, uint4* out, int iters) {
  if (smem)
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[2]; int nat = 0;
  if (cs) {
    at[nat].id = cudaLaunchAttributeClusterDimension;
    at[nat].val.clusterDim.x = cs; at[nat].val.clusterDim.y = 1;
    at[nat].val.clusterDim.z = 1; nat++;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  }
  cfg.attrs = at; cfg.numAttrs = nat;
  // warmup
  CK(cudaLaunchKernelEx(&cfg, k, tab, out, iters));
  CK(cudaDeviceSynchronize());
  cudaEvent_t ev0, ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
  cudaEventRecord(ev0);
  for (int r = 0; r < 5; r++) CK(cudaLaunchKernelEx(&cfg, k, tab, out, iters));
  cudaEventRecord(ev1);
  CK(cudaEventSynchronize(ev1));
  float ms; cudaEventElapsedTime(&ms, ev0, ev1);
  return ms / 5;
}

int main() {
  int dev = 0; cudaSetDevice(dev);
  int cl_launch = 0;
  cudaDeviceGetAttribute(&cl_launch, cudaDevAttrClusterLaunch, dev);
  printf("clusterLaunch=%d\n", cl_launch);

  uint4 *tab, *out;
  CK(cudaMalloc(&tab, 65536 * 16));
  CK(cudaMalloc(&out, 16));
  CK(cudaMemset(tab, 1, 65536 * 16));

  const int ITERS = 4096;
  const int BLK = 256;
  // occupancy-max cluster size probe
  {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1024); cfg.blockDim = dim3(BLK);
    cfg.dynamicSmemBytes = 65536;
    int cs = 0;
    cudaError_t e = cudaOccupancyMaxPotentialClusterSize(&cs, (void*)g_dsmem<16>, &cfg);
    printf("maxPotentialClusterSize(smem64K,blk256): %d (rc=%s)\n", cs,
           cudaGetErrorString(e));
  }

  dim3 grid(188 * 6);
  double n = (double)grid.x * BLK * ITERS * 5 / 5;  // gathers per launch avg
  n = (double)grid.x * BLK * ITERS;

  float ms = run(g_l2, grid, BLK, 0, 0, tab, out, ITERS);
  printf("L2    gathers: %.2f ms  %.1f G/s  (%.2f TB/s sectors32B)\n", ms,
         n / ms / 1e6, n * 32 / ms / 1e9);

  ms = run(g_smem, grid, BLK, 0, 0, tab, out, ITERS);
  printf("SMEM  gathers: %.2f ms  %.1f G/s\n", ms, n / ms / 1e6);

  // DSMEM cluster 8 (portable) and 16 (nonportable): grid must be multiple.
  {
    dim3 g8(188 * 8 / 8 * 8);
    float m8 = run((void (*)(const uint4*, uint4*, int))g_dsmem<8>, dim3(1128),
                   BLK, 65536 / 8 * 16, 8, tab, out, ITERS);
    double n8 = 1128.0 * BLK * ITERS;
    printf("DSMEM cl8  (128KB/blk? no: %dKB/blk): %.2f ms  %.1f G/s\n",
           65536 / 8 * 16 / 1024, m8, n8 / m8 / 1e6);
  }
  {
    float m16 = run((void (*)(const uint4*, uint4*, int))g_dsmem<16>, dim3(1120),
                    BLK, 65536 / 16 * 16, 16, tab, out, ITERS);
    double n16 = 1120.0 * BLK * ITERS;
    printf("DSMEM cl16 (%dKB/blk): %.2f ms  %.1f G/s\n", 65536 / 16 * 16 / 1024,
           m16, n16 / m16 / 1e6);
  }
  CK(cudaGetLastError());
  return 0;
}
