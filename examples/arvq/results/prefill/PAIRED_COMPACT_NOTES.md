# Paired dense and compact prefill validation

The final image enables paired dense P4 and native dense M16, while grouped prefill remains ON. Compaction was toggled OFF/ON within one boot. The boot marker was enabled before memory profiling. The checkpoint remains unrotated and unchanged.

| Input tokens | Compact OFF seconds | Compact ON seconds | Speedup |
| --- | ---: | ---: | ---: |
| 1024 | 1.9076 | 1.9054 | 1.0011x |
| 4096 | 4.6884 | 4.1647 | 1.1257x |
| 8192 | 9.7238 | 8.5706 | 1.1345x |

Each result averages three alternating measured pairs after one warmup per mode. Every request uses identical input token IDs, a unique cache salt, and exactly one output token. The 8192-token case repeats the saved 4096-token input twice; the 1024-token case truncates it. Legacy raw JSON field `grouped` denotes the selected **compaction** gate in this run: `toggle_env=VLLM_ARVQ_COMPACT_PREFILL` identifies the control. Grouped prefill itself was always enabled.

These same-boot measurements isolate compaction: +12.57% input throughput at 4096 and +13.45% at 8192. Compared across rounds with the first ungrouped baseline, latency moved from 7.192 to 4.165 seconds and from 15.595 to 8.571 seconds, respectively (1.727x and 1.820x throughput). The across-round totals include several changes and are not an isolated compaction claim.

| Streams | Previous total output tokens/s | Final total output tokens/s | Final mean per-request decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 133.403 | 136.055 | 144.803 |
| 2 | 192.740 | 200.529 | 108.297 |
| 4 | 264.203 | 274.149 | 76.090 |

Decode uses 136 actual prompt tokens, 512 output tokens per request, one warmup and two measured batches at each concurrency. Aggregate throughput divides actual completed output tokens by batch wall span; it does not sum per-request rates. All requests had matching final usage and streamed token IDs, and all proposals were accepted (4.0 emitted tokens per draft request step). The prior C2 reference is the M8 run; prior C1/C4 references are the grouped-image regression check.

Native M16 replaces the previous BF16-temporary dense fallback with native P4 activation arithmetic, so this is a performance experiment rather than a claim of identical full-model numerics. Comparing saved output token sequences against the earlier M8 run: 7 of 14 measured requests were identical; the others diverged at their first output token despite identical prompt hashes. Cross-boot greedy output differences were already present in earlier runs. These short synthetic prompts do not measure general model quality.

A fresh 64-output-token decode profile followed all throughput tests, with prompt-cache warmup before profiling. The profiler was stopped and all four rank traces flushed. `paired_compact_decode_profile.json` contains rank-0 attribution over 17 target decode invocations. GPU duration sum is 428.717 ms; its activity window is 465.472 ms and busy union 395.596 ms. Runtime and driver correlation IDs select target decode launches and exclude draft/helper execution ranges. Durations overlap and cannot be added as potential latency savings.

Current decode bottlenecks are remaining BF16 dense GEMM (104.163 ms, 24.30% of summed GPU duration), NCCL collectives (86.027 ms, 20.07%), and native hybrid expert math (84.372 ms, 19.68%). Paired dense P4 math plus reduction is only 16.946 ms (3.95%); the trace directly contains `nvfp4_dense_paired_kernel`. Further work should target remaining dense projections and communication, alongside expert math, rather than assuming the optimized attention output projection still dominates. Exact dense projection mapping requires additional shape/callsite analysis; kernel-name categories alone do not identify every layer.

The server remains healthy with paired P4, native M16, grouped prefill and compaction enabled. Stable-profile persistence is managed separately by the parent task. Raw results are `compact_prefill_ab.json` and `paired_compact_decode.json`; `paired_compact_summary.json` includes token-sequence comparisons. No further GPU experiments were run after the final profile.
