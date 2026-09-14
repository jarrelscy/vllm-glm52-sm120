# dense_p4_mtp_on_m8

Aggregate throughput is total completed output tokens divided by the concurrent batch wall span, including prefill and final stream completion. It is not the sum of per-request decode rates. Each stream uses the same prompt; prefix reuse and identical text are part of this controlled workload.

| Prompt target | Concurrency | Run | Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream | Emitted/draft step |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 128 | 1 | 0 | 512 | 3.843 | 133.232 | 236.9 | 4.0 |
| 128 | 1 | 1 | 512 | 3.845 | 133.159 | 237.7 | 4.0 |
| 128 | 2 | 0 | 1024 | 5.319 | 192.526 | 257.6, 522.6 | 4.0 |
| 128 | 2 | 1 | 1024 | 5.307 | 192.953 | 504.2, 241.7 | 4.0 |
| 128 | 4 | 0 | 2048 | 7.774 | 263.457 | 1027.6, 745.1, 744.9, 243.2 | 4.0 |
| 128 | 4 | 1 | 2048 | 7.729 | 264.971 | 507.7, 1022.6, 1022.6, 243.7 | 4.0 |

Speculative counters count request-level draft steps, not scheduler iterations. Dividing summed request decode time by these counters does not measure a concurrent GPU batch-step latency. Clock CSVs cover each request batch and include prefill.
