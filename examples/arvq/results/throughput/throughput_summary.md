# Concurrent serving throughput

All rates below are **total completed output tokens divided by the concurrent batch wall span**, including prefill and final completion. They are not sums of independent per-stream decode rates. The current dense NVFP4 ARVQ build was tested unchanged in each mode.

## Short controlled workload

Each stream received the same 136-token repeated-text prompt and generated 512 tokens. Each concurrency used one warmup batch and two measured batches. Temperature 0, seed 173, ignoreEOS and prefix caching were enabled. The table reports mean aggregate rate; latency is median full request duration.

| MTP | Streams | Total output tokens/s | Per-stream decode tokens/s, mean | Request latency, median seconds | TTFT, median ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| on | 1 | 133.71 | 142.05 | 3.829 | 231.8 |
| on | 2 | 72.73 | 37.46 | 14.014 | 356.2 |
| on | 4 | 139.91 | 37.13 | 14.597 | 967.3 |
| off | 1 | 52.21 | 53.40 | 9.806 | 236.6 |
| off | 2 | 79.63 | 40.94 | 12.850 | 368.1 |
| off | 4 | 140.24 | 36.92 | 14.588 | 861.4 |

MTP wins strongly at one stream, loses to no-MTP at two streams, and provides essentially the same total throughput at four streams under this configuration. Four-stream request latency is much higher than one-stream MTP latency. These are simultaneous finite batches, not an open-loop arrival-rate or saturation benchmark.

## Longer document workload

Each stream received a 4115-token synthetic operations report and generated 512 tokens. One measured batch per case, no warmup. A distinct early prefix prevents the C4 case from reusing the C1 case's cached prompt; streams within a case share a prompt. These results include cold-prefill cost and should not be compared directly to the repeated-text rates.

| MTP | Streams | Total output tokens/s | Per-stream decode tokens/s, mean | Request latency, median seconds | TTFT range, seconds | Draft acceptance | Emitted/draft request-step |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| on | 1 | 37.73 | 89.90 | 13.569 | 7.885–7.885 | 51.2% | 2.535 |
| on | 4 | 73.76 | 29.32 | 26.707 | 7.967–9.615 | 70.1% | 3.102 |
| off | 1 | 29.11 | 53.87 | 17.585 | 8.100–8.100 | — | — |
| off | 4 | 92.80 | 36.67 | 22.067 | 8.004–8.164 | — | — |

## Configuration and next opportunities

Native dense P4 is capped at four tokens and CUDA graph capture sizes are `[1,2,4]`. With three speculative drafts, two and four MTP streams can require verification batches of eight and sixteen tokens. These exceed both limits and can use the larger-batch dense dequantization fallback and uncovered graph shapes. Extending graph coverage and testing native M8 execution are concrete next experiments; the present measurements do not isolate their individual contributions. No such variant was enabled during this sweep.

No-MTP uses V1 and MTP uses V2, matching the natural profile defaults. LMCache is off and maximum concurrent sequences is four. Every measured batch passed final token-ID/usage matching and isolated completed-request/generation-token counters. Raw reports include per-stream TTFT, token events, metrics and batch dispatch skew. Compact JSON includes 200ms clock/power summaries per GPU; CSV samples remain available with the experiment.

The server was left running with dense P4 enabled and MTP disabled (`PARALLEL=tp4-1m`). The speed probe makes no new claim about model accuracy.
