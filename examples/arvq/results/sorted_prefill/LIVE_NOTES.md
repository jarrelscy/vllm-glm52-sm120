# Sorted prefill live qualification — NOT QUALIFIED

**Production rollout rejected.** Sorting improved measured prefill throughput, but the pooled eight-stream decode result was 1.80% lower with sorting ON. Decode takes an identical code path for this workload, so causation is unproven; that is insufficient evidence for the user’s strict no-regression requirement. All original and repeat samples are retained. No further rounds were run to seek a favorable result. The stable paired-compact image was restored with sorting disabled. Health returned 200, restart count was zero, and a 16-token completion passed. `final_restore.json` records the final configuration and smoke check.

| Input tokens | OFF prefill tokens/s | ON prefill tokens/s | Change |
| --- | ---: | ---: | ---: |
| 1024 | 594.134 | 593.460 | -0.11% |
| 2048 | 874.637 | 928.452 | +6.15% |
| 4096 | 952.756 | 1023.655 | +7.44% |
| 8192 | 948.842 | 1014.905 | +6.96% |

Prefill used one warmup per arm and four measured pairs per length, with pair order alternating OFF/ON and ON/OFF. Each request had a fresh actual UUID-token prefix and cache salt; the remaining body and exact token count were controlled. This changes some routing, so these are comparable workloads rather than identical token sequences. Every one of 40 requests had histogram count delta 1, GPU cache hits 0 and external cache hits 0. Exactly one output token was requested.

Primary prefill rate is actual prompt tokens divided by the isolated delta of `vllm:request_prefill_time_seconds_sum`. The server interval runs from scheduling to first token, excluding queue time; it includes work to produce that first token and is not a pure kernel timer. HTTP latency remains secondary in raw data. Legacy field `grouped` in the raw prefill JSON means the **sorting toggle** here (`toggle_env=VLLM_ARVQ_SORT_NATIVE_PREFILL`); grouped and compact prefill remained ON in both arms.

| Streams | Initial OFF steady decode tokens/s | Initial ON steady decode tokens/s |
| --- | ---: | ---: |
| 1 | 144.844 | 144.875 |
| 4 | 316.113 | 323.834 |
| 8 | 410.696 | 408.183 |

Initial decode used 136 input tokens and 512 outputs per stream, one warmup and two measured batches in each arm. The common decode interval is `(latest first emission, earliest final emission]`; count actual streamed token IDs inside it and divide by its duration. Thus TTFT and every initial emission are excluded, and rates are not sums of whole-request throughput. All token IDs matched final usage; all proposals were accepted, with 4.0 emitted tokens per draft request step.

The initial C8 result was 0.61% lower ON, triggering one bounded OFF/ON/ON/OFF repeat with two measured batches per block and no additional warmup. Combined original and repeat distributions:

| Sorting | Samples | Mean decode tokens/s | Range | Common-window seconds |
| --- | ---: | ---: | --- | --- |
| OFF | 6 | 419.098 | 409.822–431.330 | 9.336–9.784 |
| ON | 6 | 411.562 | 407.697–418.040 | 9.623–9.848 |

The sorting flag is only inspected inside grouped prefill for `num_tokens>=2048` and eligible native segments of at least 512 routes. Eight short prompts total at most 1088 input tokens; verification has at most 32 token positions. Therefore these decode tests do not execute the sorting path. The variation could reflect scheduling, routing/content variation or measurement noise, but the observed mean does not satisfy the strict rollout gate. Decode gains are not claimed.

Only 16 of 26 matched initial decode request outputs were bit-identical; the others diverged at the first output token despite the unchanged code path and identical prompt hashes. This is not full-model bitwise equivalence evidence. The separate eight-case microbench numerical checks cover only those cases and do not establish general model quality.

Candidate boot preserved all limits: context 1,048,576, sequences 8, LMCache ON, utilization 0.94, CPU 24 GiB/disk 100 GiB per rank, native dense maximum 16 and existing precision. Shared KV capacity was 1,074,176 versus 1,073,920 baseline; available KV remained 13.69 GiB and capture memory was 1.02–1.03 GiB per rank. No permanent compose changes were made by this experiment.

Raw files begin with `live_`; `live_c8_all_samples.json` contains all 12 C8 observations. The reusable harness is `examples/arvq/prefill_experiment/bench_prefill_ab.py`; use `--gate /dev/shm/vllm_arvq_sort_native_prefill_on --toggle-env VLLM_ARVQ_SORT_NATIVE_PREFILL --lengths 1024 2048 4096 8192 --runs 4` with the diagnostic image. The checkpoint and kernels were not tuned or changed during live A/B.

Large raw decode JSON files are stored as lossless `.json.gz` files; use `gzip -dc FILE.json.gz` to read them. `raw_manifest.json` records their uncompressed SHA-256 hashes and sizes. Raw logs and clock samples are in `live_logs_clocks.tar.gz`.
