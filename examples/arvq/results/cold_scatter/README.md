# Exact grouped cold output scatter

`VLLM_ARVQ_FUSED_COLD_SCATTER=1` opts into a float4 CUDA copy for qualified
T2048/T4096 grouped prefill. Eligibility and invariant launch arguments are
prepared once per batch. Each expert keeps its original FP32 GEMM output and
unique route destinations; the final route sum is unchanged.

Eight alternating full-layer pairs gave median paired speedups of 1.0599x and
1.03175x for mixed T2048/T4096, and 1.0561x/1.06332x for all-cold. Every affected
pair won; unchanged all-hot controls were neutral. Output bits and peak memory
matched. Raw-bit copy and CUDA-graph tests also passed, including nonfinite bit
patterns. Kernel SASS has vector loads/stores with no floating-point arithmetic.

The earlier Triton implementation and per-expert eligibility checks regressed
full-layer performance and were rejected. The published implementation performs
setup once before the expert loop. These are isolated layer results, not a
whole-model throughput claim. The feature remains disabled by default.
