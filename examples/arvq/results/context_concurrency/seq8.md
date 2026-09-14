# context1m_seq8_lmcache

Aggregate throughput is total completed output tokens divided by the concurrent batch wall span, including prefill and final stream completion. It is not the sum of per-request decode rates. Each stream uses the same prompt; prefix reuse and identical text are part of this controlled workload.

| Prompt target | Concurrency | Run | Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream | Emitted/draft step |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 128 | 1 | 0 | 512 | 3.759 | 136.218 | 231.9 | 4.0 |
| 128 | 1 | 1 | 512 | 3.762 | 136.097 | 234.5 | 4.0 |
| 128 | 4 | 0 | 2048 | 7.366 | 278.019 | 988.4, 988.6, 238.3, 491.3 | 4.0 |
| 128 | 4 | 1 | 2048 | 7.402 | 276.667 | 726.3, 726.2, 1000.6, 239.9 | 4.0 |
| 128 | 8 | 0 | 4096 | 11.910 | 343.903 | 1994.1, 995.1, 995.5, 1994.2, 1994.0, 1993.7, 255.1, 995.0 | 4.0 |
| 128 | 8 | 1 | 4096 | 11.842 | 345.887 | 1990.8, 1990.9, 1990.8, 1990.4, 506.5, 1990.7, 243.3, 1990.3 | 4.0 |

Speculative counters count request-level draft steps, not scheduler iterations. Dividing summed request decode time by these counters does not measure a concurrent GPU batch-step latency. Clock CSVs cover each request batch and include prefill.
