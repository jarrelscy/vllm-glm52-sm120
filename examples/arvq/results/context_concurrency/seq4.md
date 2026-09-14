# context1m_seq4

Aggregate throughput is total completed output tokens divided by the concurrent batch wall span, including prefill and final stream completion. It is not the sum of per-request decode rates. Each stream uses the same prompt; prefix reuse and identical text are part of this controlled workload.

| Prompt target | Concurrency | Run | Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream | Emitted/draft step |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 128 | 1 | 0 | 512 | 3.757 | 136.283 | 233.0 | 4.0 |
| 128 | 1 | 1 | 512 | 3.758 | 136.234 | 232.8 | 4.0 |
| 128 | 4 | 0 | 2048 | 7.277 | 281.444 | 709.1, 976.1, 235.2, 709.1 | 4.0 |
| 128 | 4 | 1 | 2048 | 7.340 | 279.003 | 995.2, 995.5, 246.4, 502.9 | 4.0 |

Speculative counters count request-level draft steps, not scheduler iterations. Dividing summed request decode time by these counters does not measure a concurrent GPU batch-step latency. Clock CSVs cover each request batch and include prefill.
