# dense_p4_mtp_on_m8_long

Aggregate throughput is total completed output tokens divided by the concurrent batch wall span, including prefill and final stream completion. It is not the sum of per-request decode rates. Each stream uses the same prompt; prefix reuse and identical text are part of this controlled workload.

| Prompt target | Concurrency | Run | Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream | Emitted/draft step |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 4096 | 1 | 0 | 512 | 11.949 | 42.847 | 8089.5 | 3.7925925925925927 |
| 4096 | 4 | 0 | 2048 | 19.340 | 105.897 | 8045.2, 9688.9, 9689.0, 9689.3 | 3.1378254211332313 |

Speculative counters count request-level draft steps, not scheduler iterations. Dividing summed request decode time by these counters does not measure a concurrent GPU batch-step latency. Clock CSVs cover each request batch and include prefill.
