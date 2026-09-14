# Fixed MTP concurrency: serving results

**Expanded CUDA-graph coverage fixes the major concurrency regression.** With native M8 enabled, total throughput is **133.20 / 192.74 / 264.21 tokens/s** for one/two/four streams. The previous restricted MTP configuration delivered 133.71 / 72.73 / 139.91. The final server remains running with MTP enabled.

Every total rate below is actual completed output tokens divided by the concurrent batch wall span, including prefill and final completion. It is not a sum of per-stream decode rates. Short cases use the same 136-token repeated-text prompt and 512 output tokens per stream, one warmup and two measured batches. Temperature 0, seed 173, ignoreEOS and prefix caching are enabled.

| Configuration | Streams | Total tokens/s | Per-stream decode tokens/s, mean | Request latency, median seconds | TTFT, median ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP off, caps4/native4 | 1 | 52.21 | 53.40 | 9.806 | 236.6 |
| MTP off, caps4/native4 | 2 | 79.63 | 40.94 | 12.850 | 368.1 |
| MTP off, caps4/native4 | 4 | 140.24 | 36.92 | 14.588 | 861.4 |
| MTP on, caps4/native4 | 1 | 133.71 | 142.05 | 3.829 | 231.8 |
| MTP on, caps4/native4 | 2 | 72.73 | 37.46 | 14.014 | 356.2 |
| MTP on, caps4/native4 | 4 | 139.91 | 37.13 | 14.597 | 967.3 |
| MTP on, caps16/native4 | 1 | 133.34 | 141.78 | 3.840 | 235.6 |
| MTP on, caps16/native4 | 2 | 184.81 | 99.41 | 5.521 | 378.0 |
| MTP on, caps16/native4 | 4 | 265.35 | 73.34 | 7.691 | 763.1 |
| MTP on, caps16/native8 | 1 | 133.20 | 141.69 | 3.844 | 237.3 |
| MTP on, caps16/native8 | 2 | 192.74 | 104.08 | 5.294 | 380.9 |
| MTP on, caps16/native8 | 4 | 264.21 | 73.01 | 7.720 | 745.0 |

`caps4` means `[1,2,4]`; `caps16` means `[1,2,4,8,16]`. The graph-only test kept the original image and native dense maximum 4. It raised two-stream throughput to 184.81 and four-stream throughput to 265.35, establishing graph coverage as the main cause of the regression. Native M8, tested afterwards with identical expanded graph coverage, adds 4.29% at two streams. Four-stream throughput changes by −0.43% across two batches; there is no demonstrated additional four-stream gain from M8. M16 dense execution still uses the fallback.

## Longer document workload

Each stream received the same 4115-token synthetic operations report within its case and generated 512 tokens. One measured batch per case, no warmup. An early per-concurrency discriminator prevents cross-case prefix-cache reuse. Prompt hashes match across old and final builds, but generated text and draft acceptance differ, so these are workload measurements rather than isolated kernel comparisons.

| Configuration | Streams | Total tokens/s | Per-stream decode tokens/s | Median request seconds | TTFT range, seconds | Draft acceptance | Emitted/draft request-step |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| MTP on, caps4/native4 | 1 | 37.73 | 89.90 | 13.569 | 7.885–7.885 | 51.2% | 2.535 |
| MTP on, caps4/native4 | 4 | 73.76 | 29.32 | 26.707 | 7.967–9.615 | 70.1% | 3.102 |
| MTP off, caps4/native4 | 1 | 29.11 | 53.87 | 17.585 | 8.100–8.100 | — | — |
| MTP off, caps4/native4 | 4 | 92.80 | 36.67 | 22.067 | 8.004–8.164 | — | — |
| MTP on, caps16/native8 | 1 | 42.85 | 132.40 | 11.949 | 8.089–8.089 | 93.1% | 3.793 |
| MTP on, caps16/native8 | 4 | 105.90 | 58.45 | 18.356 | 8.045–9.689 | 71.3% | 3.138 |

The long one-stream acceptance change from 51.2% to 93.1% prevents assigning its speed gain solely to execution improvements. No graph-only long run was collected. Long results include roughly eight to ten seconds of cold-prefill latency and are not directly comparable to the short repeated-text workload.

## Evidence and limits

Every measured batch passed returned-token-ID versus final usage checks and isolated completed-request/generation counters. Raw reports include per-stream timing, token events, speculative counters, dispatch skew and safe runtime configuration. The JSON summary includes per-GPU 200ms clock/power samples summarized over each batch; these include prefill. Draft counters count request-level steps, not concurrent scheduler iterations.

Graph-only boot logs report 0.88GiB of capture memory per rank and 21 seconds of capture. An earlier matched MTP capture-allocation log was not retained, so no numeric memory delta is claimed. Expanded graph configuration and successful capture logs are preserved with the graph-only result.

These are finite simultaneous batches, not open-loop arrival-rate or saturation tests. Streams within a batch share the same prompt. No weight or model-quality change was introduced by graph coverage; native M8 changes execution of the same quantized weights. This speed experiment makes no new accuracy claim. The final live profile uses `glm53-arvq-sm120:dense-p4-m8`, native maximum 8, captures `[1,2,4,8,16]` and `PARALLEL=tp4-1m-mtp`.
