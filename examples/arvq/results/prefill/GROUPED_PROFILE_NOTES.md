# Remaining prefill bottlenecks with grouped experts enabled

A single cache-cold 4,096-token request with one output token took **4.576 seconds** under the profiler. It used the same saved input token IDs as the earlier profile and a fresh cache salt. Grouped prefill was ON. All four rank traces were flushed before releasing the GPUs; attribution below is rank 0 only.

| Selected GPU activity | Original profile milliseconds | Grouped profile milliseconds |
| --- | ---: | ---: |
| Native hybrid math | 4545.376 | 1645.215 |
| Native activation packing | 283.604 | 277.337 |
| Native split reduction | 185.048 | 171.937 |
| NCCL collectives | 958.906 | 956.726 |
| Sparse attention/indexer | 246.531 | 239.355 |
| All GEMM, including grouped experts | 172.920 | 386.008 |
| ARVQ expert weight dequantization | 0 | 152.455 |

The strongest next target remains the native route path: math, packing and reduction still account for **44.60% of summed GPU duration**. Their launch counts remain 4,800 each despite offloading many expert routes. The source retains a full native route array with selected cold routes masked out, then launches chunked projections across that array. Compacting only the routes that still require native execution is a concrete next experiment; the trace does not establish how much of the remaining math is avoidable.

Communication is the next large fixed cost: NCCL all-gather is 561.719 ms across 177 calls and reduce-scatter is 395.006 ms across 78. Their combined duration barely changed. This gives a separate DCP communication optimization target after route compaction. Collective durations include device-side waiting and overlap, so they are not a direct prediction of attainable latency savings.

Grouped expert processing introduces many small operations: GPU event count increased from **49,349 to 153,386**, including 17,102 ARVQ weight-dequantization calls. GEMM classification now includes both original dense projections and grouped FP16 expert multiplication; kernel names alone do not cleanly separate every GEMM helper. Source-level batching/fusion of the per-expert decode, gather, GEMM and scatter work is a later target, although total GEMM time is still much smaller than native hybrid math.

The selected GPU duration sum is **4,696.050 ms**, the GPU window is **4,566.981 ms**, and the busy union is **4,485.813 ms**. Overlapping streams explain why the sum exceeds wall time. Selection matches runtime and driver launch correlations within the one target prefill CPU range, excluding draft/helper execution ranges. Comparisons are separate profiled requests, not paired kernel-isolation measurements; clock and synchronization effects can change secondary categories.

Artifacts: `grouped_prefill_capture.json` records the raw trace paths and exact request; `grouped_prefill_profile.json` contains category and kernel attribution. No checkpoint or stable host-compose changes were made by this profiling task. The fresh decode profile is deferred until the server is restored after the kernel microbenchmarks.
