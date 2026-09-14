# Exact cold GEMM output placement

This opt-in change writes grouped cold down-projection GEMMs directly into
contiguous expert-major FP32 route storage. Native routes use the corresponding
inverse permutation. Final top-8 weighting retains the original route order,
four accumulators, rounded multiplies and output conversion. It changes no
checkpoint weights, P4 activation planes or resident KV limits.

The production gate requires T4096, TP-shard gate/up width1024, hidden6144,
top-k8, chunk128, compact prefill, fused route sum, supported contiguous CUDA
layouts, and enough existing cold-ID storage for the inverse map. The map borrows
dead int64 storage; the route buffer remains the same size. T2048, other shapes,
all-hot and insufficient cold coverage retain their existing paths.

## Numerical and dispatch evidence

Initial real layer3 experiments and repeated T4096 controls preserved final
BF16 bits and measured peak memory. `initial_full.json` and
`initial_repeat4096.json` retain the raw measurements. The latter measured mixed
14.851 to14.004 ms; its unchanged hot control differed by0.38%. These are layer
microbenchmarks, not full-model throughput.

`gemm_dispatch.json` records eighteen raw FP32 output and dispatch comparisons:
gate/up and down at M32/33/64/65/127/128/129/192/257. Output bits, kernel names,
launch geometry and shared memory matched between ordinary and destination-view
GEMMs. Separate M96 descriptor inspection also matched; notably its gate/up
algorithm used eighteen K splits. No universal sixteen-split assumption is made.
See `gemm_m96_algorithms.json`. This is empirical coverage of those shapes, not
an assertion about every possible cuBLAS dispatch.

## Initial live regression and its cause

The first placement image performed poorly: approximately450 and578 tokens/s
on two cold8K requests and771 tokens/s on cold128K, while warm C1 decode measured
about142.7 tokens/s. These are failed candidate results, retained under
`initial_live/`, not improved serving claims.

Routing-dependent cold count C and native-scatter size N were incorrectly
`tl.constexpr` parameters. Every new value compiled another Triton variant.
`compile_storm.json` records1224 inverse and928 scatter TTIR files written within
a600-second observation window. Repeated fixed-routing microbenchmarks missed
this full-model compilation cost. GPU utilization was low despite high clocks.

The correction makes C/N runtime integers and explicitly excludes them from
specialization, while preserving S/H/block constants and the kernel bodies.
The CPU regression test guards this compilation contract.

## Runtime-count correction qualification

`runtime_count_gpu.json` tests C=1/16384/20001/24576/30003/32768 and native rows
1/1024/731/512/117/32 with fixed S32768/H6144. Independent inverse-permutation
checks, raw scatter bits including arbitrary NaN payloads, and graph replay all
pass. A fresh isolated Triton cache contains exactly one inverse and one scatter
kernel after every case, demonstrating bounded compilation cardinality.

`runtime_count_full.json` compares the final repository helper with direct
placement off/on while gate packing and cold scatter remain enabled on both
arms. All six complete real-layer cases preserve output bits and allocated
peaks. Mixed T4096 has a paired median ratio1.0274; all-cold T4096 and fallback
controls are effectively neutral. Full raw timing samples are retained, rather
than selecting fastest runs.

The combined corrected image measured 2098.8 tokens/s at cold8K after code
warmup and 1962.7 at cold128K, versus preceding DMA-only measurements2054.1
and1913.1. Warm C1 decode measured141.24 versus142.56 tokens/s. The first8K
request measured1278.8 and is retained. See `corrected_live/`; these are single
full-model observations with fresh actual UUID prefixes, not isolated direct
placement gains or a statistical nonregression claim. The image also enables
qualified gate packing and cold scatter. C4 decode measured321.52 with all reference IDs equal. C8 measured407.11,
but one of eight streams omitted a token at position494; the remaining seven
matched, including the same prompt in another stream. A repeat measured413.74
with all eight matching. The first discrepancy remains unresolved and is
preserved under `corrected_live/`; the repeat does not establish correctness.
Single-stream long-context stress completed at 256K and 512K: cold prefill
measured 1801.24 and 1664.80 tokens/s, with warm decode 141.04 and 142.41 tokens/s,
respectively. Health and metric checks passed. These long outputs have no
numerical reference comparison; this does not resolve the C8 discrepancy.
See `corrected_live/long_stress/`. The 3000-token/s prefill target is not achieved.
