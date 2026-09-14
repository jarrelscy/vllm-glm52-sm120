# Full context and concurrent serving validation

Both tested profiles set `max_model_len=1048576`. The four-sequence baseline used LMCache OFF and GPU utilization 0.96; the eight-sequence candidate used LMCache ON and utilization 0.94. Thus this is a comparison of complete serving configurations, not an isolated max-sequence-count experiment. Paired dense P4, native maximum16, grouped and compact prefill, TP4/DCP4 and MTP remain enabled.

| Profile | Streams | Total output tokens/s | Median request seconds | Median TTFT seconds | Mean request decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| seq4 / LMCache OFF | 1 | 136.258 | 3.757 | 0.233 | 144.990 |
| seq4 / LMCache OFF | 4 | 280.223 | 7.258 | 0.709 | 77.624 |
| seq8 / LMCache ON | 1 | 136.158 | 3.760 | 0.233 | 144.885 |
| seq8 / LMCache ON | 4 | 277.343 | 7.351 | 0.726 | 76.796 |
| seq8 / LMCache ON | 8 | 344.895 | 11.841 | 1.991 | 49.587 |

Short requests used 136 actual input tokens and 512 outputs per stream, one warmup and two measured batches. Total throughput divides summed final completion-token usage by the overlapping batch wall span; it does not sum per-request rates. Token IDs matched usage, speculative counters matched batch isolation, and all proposals were accepted (4.0 emitted tokens per draft request step). Eight sequences preserve C1 throughput (-0.07%) and C4 throughput (-1.03%) within this bounded comparison, while C8 reaches 344.895 total tokens/s.

| Profile | Shared KV token capacity | Available KV GiB | Graph GiB per rank | Graph captures |
| --- | ---: | ---: | ---: | --- |
| seq4 / LMCache OFF | 1222912 | 15.59 | 0.86 | 1,2,4,8,16 |
| seq8 / LMCache ON | 1073920 | 13.69 | 1.02 | 1,2,4,8,16,32 |

The full position limit was accepted at startup; a million-token request was not executed. KV capacity is shared across requests and is not eight independent million-token contexts. Native dense maximum remains16; verification at batch8 can use M32 and falls back to dequantized dense execution, covered by capture32.

The bounded long smoke used eight simultaneous requests with an identical 8221-token synthetic document and 64 outputs each. All512 outputs completed in14.554s, median TTFT12.472s. Draft acceptance was54.15%, with2.624 emitted tokens per draft request step. This one-batch prefill-heavy result is not comparable to the short synthetic throughput workload. Identical inputs can share KV prefixes: it does not establish memory stress for eight independent8k contexts.

NVML used memory per GPU was93,262MiB before requests,93,930MiB after short benchmarks and96,584MiB after the long smoke, against97,887MiB total. These measurements include runtime allocation and allocator retention as well as LMCache; they do not isolate a specific staging buffer. No allocation failure occurred in these bounded runs.

LMCache configuration is CPU24GiB and disk100GiB, chunk256, GPU connectorV3, SHA256-CBOR hashes and PYTHONHASHSEED0. Disk restore across a clean server restart is verified separately by the LMCache proof task. The final restart uses only the base compose file plus the saved optimized profile; no temporary experiment override is required.

Artifacts include seq4/seq8 raw request JSON, seq8_long.json, summary.json, boot excerpts, clock CSVs and NVML snapshots. No weights or kernels changed in this experiment.
