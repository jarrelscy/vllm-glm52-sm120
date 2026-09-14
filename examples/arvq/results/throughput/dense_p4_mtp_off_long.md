# dense_p4_mtp_off_long

Aggregate throughput is total completed output tokens divided by the concurrent batch wall span, including prefill and final stream completion. It is not the sum of per-request decode rates. Each stream uses the same prompt; prefix reuse and identical text are part of this controlled workload.

| Prompt target | Concurrency | Run | Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream | Emitted/draft step |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 4096 | 1 | 0 | 512 | 17.586 | 29.113 | 8099.7 | — |
| 4096 | 4 | 0 | 2048 | 22.069 | 92.801 | 8163.1, 8163.7, 8163.1, 8003.8 | — |

Speculative counters count request-level draft steps, not scheduler iterations. Dividing summed request decode time by these counters does not measure a concurrent GPU batch-step latency. Clock CSVs cover each request batch and include prefill.
