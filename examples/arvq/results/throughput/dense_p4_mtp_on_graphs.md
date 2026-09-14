# dense_p4_mtp_on_graphs

Aggregate throughput is total completed output tokens divided by the concurrent batch wall span, including prefill and final stream completion. It is not the sum of per-request decode rates. Each stream uses the same prompt; prefix reuse and identical text are part of this controlled workload.

| Prompt target | Concurrency | Run | Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream | Emitted/draft step |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 128 | 1 | 0 | 512 | 3.839 | 133.371 | 235.3 | 4.0 |
| 128 | 1 | 1 | 512 | 3.841 | 133.301 | 235.8 | 4.0 |
| 128 | 2 | 0 | 1024 | 5.543 | 184.729 | 516.8, 252.7 | 4.0 |
| 128 | 2 | 1 | 1024 | 5.539 | 184.883 | 503.2, 240.2 | 4.0 |
| 128 | 4 | 0 | 2048 | 7.710 | 265.622 | 1018.0, 242.3, 505.8, 1018.4 | 4.0 |
| 128 | 4 | 1 | 2048 | 7.726 | 265.073 | 1021.9, 243.2, 1021.9, 508.3 | 4.0 |

Speculative counters count request-level draft steps, not scheduler iterations. Dividing summed request decode time by these counters does not measure a concurrent GPU batch-step latency. Clock CSVs cover each request batch and include prefill.
