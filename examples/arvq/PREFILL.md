# Grouped ARVQ prefill

The host profile enables `VLLM_ARVQ_GROUPED_PREFILL=1`. At 4096 input tokens,
grouping cold-expert routes reduced measured cold-request latency from 7.192 to
4.724 seconds; at 8192 tokens, from 15.595 to 9.769 seconds. Each request emitted
one output token. See [the measured A/B](results/prefill/GROUPED_NOTES.md) for
three alternating pairs per length, cache isolation, and raw results.

The previous prefill path repeated expert matrix-vector work for every routed
token. The grouped path gathers tokens assigned to each cold expert, decodes
that expert's serialized ARVQ weights into temporary FP16 matrices, and uses
FP16 matrix multiplication with FP32 outputs. Matrices are released after each
projection; the checkpoint and resident compressed weight format are unchanged.
Hot experts and cold experts with fewer than 32 routes retain native FP4 P4.
Four-plane packing remains on the native path. Rotation is not used.

The dispatcher requires at least 2048 tokens, no active CUDA capture, and a
routed FP32 output buffer of at most 1 GiB (768 MiB at 4096 tokens, top-8,
hidden width 6144). It falls back to the original path otherwise. The deployed
4096-token batch limit keeps long requests within the scratch budget. Set the
environment variable before startup so memory profiling reserves the required
scratch space. The library default is OFF; the measured host profile opts in.
`toggle` and `/dev/shm/vllm_arvq_grouped_prefill_on` are diagnostic A/B controls;
normal production mode performs no marker-file lookup.

Grouped computation preserves FP16 gate/up and activation rounding and FP32
route-sum order, but temporary FP16 weight reconstruction differs numerically
from native FP4 MMA. The actual mixed layer-3, 4096-token microbenchmark measured
0.128% relative output L2 difference from the native path. Decoder oracle,
route-order, capture, and dispatch tests passed. Matching first output tokens
in the serving A/B is a smoke check, not a full quality evaluation.

Short requests remained at 133.4 total output tokens/s for one stream and
264.2 for four simultaneous streams with MTP. These prompts accepted every
draft; performance on other inputs depends on acceptance. Prefill optimization
does not change the decode dispatch.

## Compact the remaining native routes

`VLLM_ARVQ_COMPACT_PREFILL=1` removes grouped-cold slots from native P4 packing
and projection launches. Remaining hot, low-count cold, and defined-zero routes
keep their original destination and reduction ordering. Tiny tail chunks retain
the original split count. This changes scheduling without another quantization
pass. The integrated 4096-token mixed-layer microbenchmark was bitwise identical
to the preceding grouped helper and improved from 38.496 to 29.923 ms.

In a separate same-boot serving A/B with paired dense execution fixed ON,
compaction reduced 4096-token input latency from 4.68836 to 4.16469 seconds and
8192-token latency from 9.72379 to 8.57062 seconds. These are means of three
alternating pairs with one output token and unique cache salts. The 1024-token
control remains below the 2048-token dispatch threshold and was unchanged.
The original pre-grouping measurements were 7.19210 and 15.59470 seconds;
comparing those earlier measurements with the latest gives about 1.73x and
1.82x overall speedup across rounds, not a single same-boot isolation test.

The compaction library default remains OFF. Diagnostic mode `toggle` uses
`/dev/shm/vllm_arvq_compact_prefill_on`, fixed throughout each TP request.
Production mode `1` performs no filesystem checks. Enable it before startup
memory profiling; grouped-prefill eligibility and scratch limits still apply.

[Final serving measurements and decode profile](results/prefill/PAIRED_COMPACT_NOTES.md)
include raw request data, cross-boot output comparisons, and current bottlenecks.
