# SM120 TP4 speed inventory

The measured single-stream progression reaches **53.276 tokens/s without MTP**
and **141.661 tokens/s with MTP**. These are two distinct serving modes, not
aggregate throughput across simultaneous requests.

## Measured progression

| Build | No MTP, tokens/s | MTP, tokens/s | What changed |
| --- | ---: | ---: | --- |
| Original NVFP4 + AQLM hybrid | 44.349 | 97.163 | Original serving baseline |
| NVFP4 + ARVQ hybrid | 48.664 | 138.197 | Native FP4 cold-expert multiplication and P4 activations |
| ARVQ + PCIe communication policy | 50.439 | 138.401 | Eligible singleton fused all-reduce/RMSNorm; large-payload DMA capability |
| Above + optional dense NVFP4 P4 | 53.276 | 141.661 | Target attention-output projections only; draft weights unchanged |

The workload uses the same nominal 128-token repeated-pangram prompt
(136 tokens after chat formatting), 512 generated tokens, and ignores EOS.
Reported decode throughput counts verified tokens after the first emission
batch over the corresponding decode interval. Original and latest dense
results use three measured requests; initial ARVQ headline results and the
communication MTP result use one. These sample counts limit comparisons of
small differences. No-MTP uses the V1 runner; MTP uses V2, so an on/off comparison
also includes a runner change.

The MTP measurements accept essentially every draft on this repetitive input;
the latest run accepted 1152/1152 proposed draft tokens across 384 steps. This
is not an acceptance estimate for normal production traffic. See the
[original/ARVQ comparison](results/benchmark_summary.md),
[communication no-MTP result](results/new_pcie_no_mtp_summary.md),
[communication MTP result](results/new_pcie_range_headline.md), and
[dense MTP result](results/dense_p4_mtp_on_summary.md).

The dense addition improves the measured no-MTP rate by 5.62% and MTP rate by
2.36%. It is lossy and remains an opt-in experiment; the small paired quality
probe does not establish general accuracy preservation. See
[DENSE_P4.md](DENSE_P4.md) for scope, quality evidence, and the launch override.
The communication row is a complete build result, not attribution of its gain
to DMA: short single-stream decode does not exercise the large-payload DMA
threshold. [PCIe details](PCIE.md) separate microbench evidence from serving
measurements.

## Where time was spent before dense quantization

The available detailed no-MTP profile is the **50.439 tokens/s communication
build, before dense NVFP4 P4**. Do not treat it as the latest build's profile.
A separate request captured 63 decode invocations. Rank-0 summed kernel times
were:

| Category | GPU ms/token | Interpretation |
| --- | ---: | --- |
| Dense multiplication | 7.434 | Includes Inductor reduction-based GEMVs |
| Sparse attention and indexer | 2.584 | Selection, sparse attention, and related work |
| NCCL communication | 1.983 | Includes DCP communication |
| Elementwise, normalization, reduction | 1.864 | Across the model |
| Fused PCIe all-reduce/RMSNorm | 1.803 | Eligible singleton boundaries |
| ARVQ/NVFP4 expert multiplication | 1.776 | Native hybrid projection kernels |
| Expert activation packing | 0.374 | Four-plane operands |
| Expert split reduction/scales | 0.338 | Native projection epilogues |

Streams overlap, so these are attribution totals, not independently removable
latencies. The trace contains 3625 GPU events/token. Its GPU busy-union time
was 18.071 ms/token, within a 20.354 ms/token trace window. Remaining gaps alone
do not prove CPU overhead. Full evidence and caveats are in the
[profile notes](results/new_pcie_no_mtp_profile_NOTES.md).

The attention-output GEMV previously accounted for 2.530 ms/token across
78 calls, shape N6144×K4096; dense P4 now targets it. Other confirmed dense
sources include fused Q/KV-A at N2624×K6144 (1.196 ms/token, 56 calls in its
largest named kernel) and Q-B at N4096×K2048, fused with normalization
(0.683 ms/token, 56 calls). These counts describe those kernel variants, not
the total number of model layers performing each projection.

## Remaining candidates

### Short-context DSA selection bypass

**CPU audit only; no speedup measured or production bypass implemented.** When
every query has at most 2048 valid global causal keys, top-2048 cannot exclude
a valid key. A specialized path could skip index-query projection, scoring,
selection, and candidate communication while emitting all valid logical IDs.
The existing index-key projection, normalization, RoPE, and cache insertion
must continue so a sequence can later grow beyond the threshold.

The guard must use global per-query lengths, including each draft position's
causal length. A DCP rank-local length below 2048 is insufficient. Output must
preserve the selected canonical descending logical-key order, invalid `-1`
padding, request boundaries, and graph padding. Existing layer-to-layer and MTP
index reuse must retain their semantics.

Current CUDA graph descriptors do not distinguish short and long contexts.
A Python condition during capture would freeze the wrong path when a graph
is reused after the sequence crosses 2048. Full projection and collective
elimination needs a short/long graph variant or equivalent safe mechanism.
Device predicates can initially avoid some kernel work while preserving the
launch sequence, but cannot by themselves remove captured GEMM/NCCL calls.

Implementation locations: indexer projections and fused attention in
[`attention.py`](../../vllm/models/deepseek_v32/nvidia/attention.py),
[selection and DCP merge](../../vllm/model_executor/layers/sparse_attn_indexer.py),
and [graph descriptors](../../vllm/forward_context.py). The current model already
reuses index selections across layers via `index_topk_freq=4`; its profile
contains approximately 21 selection merges/token, not 78.

### Other dense projections

**Further shape-matched microbenchmarks needed.** The Q/KV-A and Q-B shapes
above remain substantial. The existing FP8 hook excludes these projections
because MLA code reads or absorbs raw BF16 weights outside a linear method's
`apply()`. A replacement must preserve those consumers; merely broadening the
suffix list is unsafe. The existing special low-latency QKV-A GEMM is gated to
N2112×K7168 and other GPU families, so it does not dispatch for this GLM/SM120
shape.

The 75-call cuBLAS and WMMA variants are plausibly shared gate/up and down:
call counts match the 75 shared-expert layers and measured isolated latencies
are close. This mapping remains an inference rather than direct launch
correlation; do not attribute them to indexer projections from names alone.

### Batching beyond four tokens

**Concurrent throughput measurement is pending.** Single-stream MTP normally
verifies four tokens, which fits the selected native dense-P4 path. Multiple
requests can exceed this and use temporary weight reconstruction plus BF16
GEMM. Isolated M16/M32 native P4 takes about 89/169 µs; reconstruction plus
BF16 takes about 49 µs, so extending the native threshold blindly regresses
these shapes. Larger graph capture also needs memory analysis because graph
pools can retain reconstructed matrices.

The next comparison should report concurrency, aggregate output tokens/s,
per-request latency, MTP acceptance, and actual dispatch. Single-stream rates
cannot be multiplied by request count. Concurrent artifacts will be recorded
under [results/](results/) when available; no aggregate result is asserted here.

## Candidates tested and not adopted

| Candidate | Measured result | Decision |
| --- | --- | --- |
| Shared-expert NVFP4 P4 | Gate/up 12.58 vs BF16 10.12 µs; down 9.61 vs 5.93 µs | Keep shared BF16 |
| Blanket existing FP8 W8A16 | Attention output improves, but shared projections regress; exceeds 4.5 bpw | Use as comparison only |
| Fused DCP LSE reduction/communication | Best tested one-row fused result about 26.67 vs NCCL 18.44 µs; four-row variants also regress | Retain existing staged NCCL path |
| Expert gate/up split 16→12 | Approximately 0.6–2.8 µs per full synthetic expert MLP across tested route mixtures | Minor isolated candidate; no claimed serving gain |

See [dense microbench JSON](results/dense/dense_micro.json),
[native P4 microbench JSON](results/dense/nvfp4_dense.json),
[DCP sweep](results/tuning/dcp_lse_micro_462.json), and
[expert split sweep](results/tuning/split_sweep.json). The DCP experiment is
separate from the successful singleton all-reduce/RMSNorm fusion already in
the communication build. Tested split timings use synthetic rotating expert
pools; their small differences require end-to-end confirmation.
