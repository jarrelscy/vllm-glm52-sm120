# LMCache persistent restore: 8K-token proof

A genuine container recreation preserved an 8,192-token prompt through LMCache
and restored it before any other generation. All four DCP ranks retrieved their
shards. vLLM GPU-prefix hits remained zero. The 32-token completion text was
byte-identical to the cold response; post-restore health returned HTTP 200 and
Docker `RestartCount` remained zero.

| Observation | Cold store | Post-restart restore |
| --- | ---: | ---: |
| Prompt tokens | 8,192 | 8,192 |
| Completion tokens requested | 32 | 32 |
| Complete request time | 8.284 s | 0.528 s |
| GPU-prefix hit counter delta | 0 | 0 |
| External-prefix hit counter delta | 0 | 8,191 |

The observed complete-request ratio is 15.68× for this single prompt, not a
throughput benchmark or a full-window latency claim. The adapter reported all
8,192 tokens available and requested 8,191 externally computed tokens because
the final prompt token must be recomputed. Each worker physically retrieved
2,048 shard tokens, reflecting full stored chunks.

## Configuration and evidence

- Model: `jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid`, unchanged unrotated weights.
- TP4, DCP4, native MTP with three draft tokens; maximum context 1,048,576 and
  maximum sequences 8. This functional proof exercises **8K**, not the full 1M.
- Installed LMCache `0.5.2.dev51`, fork `jarrelscy/LMCache` at
  `b339be5a7fd091109479e998c8716084d60fc4f9`, including DCP and multi-group fixes.
- `ENABLE_LMCACHE=1`, V3 GPU connector, 256-token chunks, `sha256_cbor` hashing,
  `PYTHONHASHSEED=0`, and `LMCacheConnectorV1` with `kv_role=kv_both`.
- CPU tier 24 GiB per worker, disk limit 100 GiB per backend, dedicated persistent
  `/data/lmcache/glm5.3-arvq`; GPU utilization 0.94 reserves staging headroom.

The cold request created 128 new disk data/metadata pairs, 32 per rank. Data
files totaled 448,331,776 bytes (112,082,944 per rank), with 22,016 additional
metadata bytes. Sidecar counts stabilized before restart. Worker processes and
container ID changed; CPU/GPU cache contents could not survive the restart.
The first generation then restored from persistent storage, with zero initial
cache counters and no GPU-prefix hits. Retrieval logs report 2,048/2,048 tokens
on each rank and 71.84–72.08 ms of retrieval time per rank. Those worker times
exclude other request work and are not summed across parallel ranks.

[Raw request, response, counters and disk inventory](restore_proof.json),
[store logs](store_rank_logs.txt), and [restore logs](restore_rank_logs.txt)
contain the evidence. Authentication and environment secrets are not recorded.
ANSI color escapes were removed from the readable log copies. The raw result
omits unrelated disk entries that predated this prompt.

## Repeat the check

Reserve an exclusive request window. Send a unique bounded prompt and retain its
exact token IDs and response. Wait for all four ranks' disk writes to complete.
Recreate the same server profile, preserving its dedicated cache mount; confirm
that the container ID changed. After health is ready, replay the exact prompt
before other generation. Check external-hit counter deltas, zero GPU-prefix
hits, all four rank retrieval logs, output comparison and final health. A warm
request within the same process is insufficient to establish disk persistence.
