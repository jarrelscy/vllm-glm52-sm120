# dense_p4_mtp_on_concurrency

Aggregate throughput is total completed output tokens divided by the concurrent batch wall span, including prefill and final stream completion. It is not the sum of per-request decode rates. Each stream uses the same prompt; prefix reuse and identical text are part of this controlled workload.

| Prompt target | Concurrency | Run | Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream | Emitted/draft step |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 128 | 1 | 0 | 512 | 3.830 | 133.686 | 231.4 | 4.0 |
| 128 | 1 | 1 | 512 | 3.829 | 133.730 | 232.2 | 4.0 |
| 128 | 2 | 0 | 1024 | 14.186 | 72.182 | 539.1, 233.1 | 3.9883268482490273 |
| 128 | 2 | 1 | 1024 | 13.976 | 73.270 | 479.3, 233.0 | 3.9883268482490273 |
| 128 | 4 | 0 | 2048 | 14.598 | 140.292 | 967.4, 967.4, 967.4, 967.2 | 4.0 |
| 128 | 4 | 1 | 2048 | 14.678 | 139.524 | 990.7, 237.1, 719.5, 719.5 | 4.0 |

Speculative counters count request-level draft steps, not scheduler iterations. Dividing summed request decode time by these counters does not measure a concurrent GPU batch-step latency. Clock CSVs cover each request batch and include prefill.
