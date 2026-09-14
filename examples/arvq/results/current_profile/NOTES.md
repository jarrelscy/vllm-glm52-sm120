# Current full-context serving profile

Configuration remains max context 1,048,576, max sequences 8, LMCache ON, utilization 0.94, paired native dense maximum 16, grouped/compact prefill ON. No flags, weights or precision changed. Both profiler sessions were stopped and traces flushed before releasing GPUs for experiments.

Primary decode baselines exclude TTFT:

| Streams | Aggregate steady decode tokens/s |
| --- | ---: |
| 1 | 144.885 |
| 4 | 319.512 |
| 8 | 412.797 |

These values reuse the two measured batches in the saved current seq8 configuration. The common window starts immediately after the latest first token emission and ends at the earliest final token emission among streams. Actual streamed token IDs emitted in that interval are counted, then divided by interval duration. Initial emissions and TTFT are excluded. This is an SSE-receipt-timestamp metric, not GPU step timing; it does not sum whole-request rates. `baseline_steady_decode.json` retains window boundaries and counts.

A fresh warmed 136-input/64-output decode trace selects 17 target decode CPU ranges by CUDA runtime and driver launch correlations. Its GPU duration sum is 426.721 ms over a 463.945 ms GPU window. Categories: remaining BF16 dense GEMM 24.33%, NCCL 20.29%, hybrid expert math 19.34%, elementwise 11.20%, sparse attention/indexer 8.70%, paired dense P4 3.97%. Overlapping activity durations are not additive latency savings.

A separate fresh UUID-prefixed 4,096-input/1-output request also used a unique cache salt. Profiled wall time was 4.152 s, or 986.406 input tokens/s. This one instrumented request is diagnostic, not a replacement for repeated unprofiled prefill measurements. Prefill GPU categories: native hybrid 34.69%, NCCL 22.79%, all GEMM 9.19%, elementwise 8.43%, sparse attention 5.71%; native pack/reduce add 3.94%. Compaction reduces native math/pack/reduce launches to 1,760 each, versus 4,800 before compaction.

Source module mapping was independently confirmed from GLM config and model code: qkv_a combines 2048 Q-rank + 512 KV-rank + 64 rotary dimensions; q_b has 16 local heads × 256, while indexer Q has 32 replicated heads × 128. Shared gate/up and down use TP4 intermediate size 512.

Dense mapping combines current compiled graph matrix shapes with trace kernel tile/grid and call counts. Profiler record_shapes was disabled in the existing server, so no direct per-call input-shape tracing is claimed. Extracted compiled calls and kernel grids are retained separately for audit.

| Candidate callsite | Per-rank K → N | Calls per target step | Kernel sum over 17 steps (ms) |
| --- | --- | ---: | ---: |
| Fused Q/K/V input projection | 6144 → 2624 | 78 | 30.429 |
| Q expansion plus active indexer Q projection (combined bucket) | 2048 → 4096 | 99 | 25.195 |
| Shared expert gate/up | 6144 → 1024 | 75 | 14.198 |
| Dense/shared down projections (combined bucket) | 3072 or 512 → 6144 | 78 | 10.800 |

The first dense target alone is 1.790 ms per target decode step. Precision-preserving fusion or better execution of these existing BF16 projections deserves a bounded prototype. The combined buckets cannot be split into separate module timings from kernel names alone. Any candidate must preserve quantization, precision, full context, sequence capacity, LMCache and memory limits, and demonstrate non-regression for prefill, single-stream decode and concurrent aggregate decode before deployment.
