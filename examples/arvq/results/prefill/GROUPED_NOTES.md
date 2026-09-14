# Grouped prefill live A/B

The unrotated ARVQ checkpoint served successfully with the opt-in grouped-prefill image. Grouped prefill improved cold-input latency while short decode throughput remained unchanged in this bounded check.

| Input tokens | OFF mean seconds | ON mean seconds | Speedup |
| --- | ---: | ---: | ---: |
| 1024 | 1.8284 | 1.8331 | 0.9974x |
| 4096 | 7.1921 | 4.7238 | 1.5225x |
| 8192 | 15.5947 | 9.7695 | 1.5963x |

Each length used one warmup per mode followed by three alternating OFF/ON pairs. Each request generated exactly one output token, preventing speculative acceptance from confounding prefill throughput. Identical token IDs came from the earlier 4096-token profile; the 1024 case truncates them and the 8192 case repeats them twice. The 2048-token dispatch threshold leaves the 1024 case on the original path. All measured first output texts matched between modes; this is a smoke check, not a model-quality evaluation.

The reset-prefix-cache endpoint returned HTTP 404 because developer endpoints are disabled. The runner instead supplied a unique supported `cache_salt` on every request, forcing distinct cache hashes without altering input tokens. Server logs reported 0.0% prefix-cache hit rate. HTTP wall time includes input processing and the first output; no profiler was active during these A/B measurements. All requests were sequential and there was no competing GPU benchmark.

| Streams | Previous total output tokens/s | Grouped image total output tokens/s | Mean per-request decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 133.196 | 133.403 | 141.832 |
| 4 | 264.214 | 264.203 | 72.945 |

Decode used the established 136-token prompt and 512 output tokens per stream, one warmup and two measured simultaneous batches at each concurrency. Total throughput divides actual completed output tokens by the full batch wall span; it does not sum independent per-request rates. All speculative proposals were accepted (4.0 emitted tokens per draft request step), and final usage matched streamed token IDs. The previous result came from the preceding M8 image, so small differences are not isolated performance claims.

The temporary image is `glm53-arvq-sm120:grouped-prefill`, with MTP on, native dense maximum 8 and graph captures `[1,2,4,8,16]`. The shared-memory ON marker was touched immediately after container startup, before weight loading and memory profiling. Startup reported 69.2 GiB model loading, 16.07 GiB available KV memory, 1,261,290-token cache capacity, and 0.88–0.89 GiB graph capture per rank. The running server remains healthy with the grouped marker ON. The checkpoint itself was not modified.

Raw results: `grouped_prefill_ab.json`, `grouped_prefill_lengths.json`, `grouped_prefill_decode.json`; compact results: `grouped_summary.json`. Request clocks are in the accompanying CSV files. The boot excerpt is `grouped_boot_summary.log`.
