# Specialized top-8 route pack follow-up

The generic routed pack contained runtime signed 64-bit division. Replacing it
with a right shift is exact for these nonnegative route IDs and fixed top-k 8.
Static pack instruction count fell 544 to 440 (baseline 424), registers 29 to27,
stack 32 to0 bytes, and reciprocal instructions5 to3. Input and route loads
already occur once before the P4 loop; no residual-loop load was hoisted.

Six whole-chain CUDA-graph comparisons passed all pack/scales/descriptor bytes.
Ten alternating rounds included baseline route division/gather, FP16 cast and
packing versus each fused variant. At1024 slots, baseline112–118us, generic
fused213–215us, specialized99–104us. At64 slots baseline14.35us versus8.30us.
Raw samples: pack_results.json. An initial graph harness used a pre-capture
stream pointer and produced empty fused graphs; those invalid results were
replaced after querying the current stream within every launch.

The subsequent fifteen full real-layer cases all passed output int16 equality
and every eligible pack/scales/live-descriptor oracle. Five alternating rounds:

| Tokens | Case | Baseline ms | Specialized ms | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 2048 | Mixed | 10.261 | 9.787 | 1.0484 |
| 4096 | Mixed | 12.884 | 12.702 | 1.0143 |
| 2048 | All hot | 8.856 | 8.707 | 1.0172 |
| 4096 | All hot | 17.299 | 17.487 | 0.9892 |

Eligible peak allocated memory fell exactly12MiB; fallback peaks unchanged.
The prior large regression is removed. Strict full-path nonregression is not
established: all-hot4096 lost1.08%, although unchanged all-cold2048 also lost
1.17%, illustrating timing noise. Do not promote solely from these medians.
All raw samples/peaks are in results_top8.json. GPU0 process exited and lease
was released. No production edits or deployment occurred.
