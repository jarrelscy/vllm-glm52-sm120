# dense_p4_mtp_off_concurrency

Aggregate throughput is total completed output tokens divided by the concurrent batch wall span, including prefill and final stream completion. It is not the sum of per-request decode rates. Each stream uses the same prompt; prefix reuse and identical text are part of this controlled workload.

| Prompt target | Concurrency | Run | Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream | Emitted/draft step |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 128 | 1 | 0 | 512 | 9.802 | 52.232 | 236.1 | — |
| 128 | 1 | 1 | 512 | 9.810 | 52.190 | 237.1 | — |
| 128 | 2 | 0 | 1024 | 12.856 | 79.650 | 493.3, 241.6 | — |
| 128 | 2 | 1 | 1024 | 12.864 | 79.602 | 243.0, 495.1 | — |
| 128 | 4 | 0 | 2048 | 14.589 | 140.378 | 979.3, 979.4, 979.2, 244.2 | — |
| 128 | 4 | 1 | 2048 | 14.619 | 140.095 | 246.6, 743.7, 1009.8, 743.6 | — |

Speculative counters count request-level draft steps, not scheduler iterations. Dividing summed request decode time by these counters does not measure a concurrent GPU batch-step latency. Clock CSVs cover each request batch and include prefill.
