# Exact routed gate activation packing

`VLLM_ARVQ_FUSED_GATE_PACK=1` enables the opt-in path with paired/wide hot
prefill. It gathers routed BF16 activations, rounds to FP16, and packs the same
four residual FP4 planes in one kernel. Weights and MMA accumulation are unchanged.
Top-k 8 is specialized to eliminate runtime integer division. Unsupported shapes
retain the existing gather and pack implementation.

The twelve-pair repeat uses the original seeded route schedule. Mixed T4096
has a paired median 1.00933x speedup; all-hot T4096 is 1.02825x. Each wins
11 of 12 pairs; the unchanged all-cold control is 0.99990x. Outputs and packed
bytes match exactly, and eligible peak allocated memory falls by 12 MiB.
See `repeat_exact_top8.json` and `repeat_analysis.json` for raw samples and
paired differences. Earlier five-round results, including the unreproduced
all-hot regression, are retained in `TOP8_RESULTS.md` and `results_top8.json`.

The packaged export additionally validates pointers, shape bounds and top-k.
Fifteen focused tests passed, including four GPU packing and graph tests.
Three rebuilt-export full-layer checks also passed bitwise equality and memory
checks. These are layer measurements; a serving throughput gain is not yet
established. The feature remains off by default.
