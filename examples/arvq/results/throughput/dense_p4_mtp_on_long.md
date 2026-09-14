# dense_p4_mtp_on_long

Aggregate throughput is total completed output tokens divided by the concurrent batch wall span, including prefill and final stream completion. It is not the sum of per-request decode rates. Each stream uses the same prompt; prefix reuse and identical text are part of this controlled workload.

| Prompt target | Concurrency | Run | Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream | Emitted/draft step |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 4096 | 1 | 0 | 512 | 13.569 | 37.732 | 7884.9 | 2.5346534653465347 |
| 4096 | 4 | 0 | 2048 | 27.768 | 73.755 | 9614.8, 7966.8, 9614.8, 9615.1 | 3.1015151515151516 |

Speculative counters count request-level draft steps, not scheduler iterations. Dividing summed request decode time by these counters does not measure a concurrent GPU batch-step latency. Clock CSVs cover each request batch and include prefill.
