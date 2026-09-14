# Q before absorption

`VLLM_GLM_Q_BEFORE_ABSORB=1` enables the qualified large-prefill path. It is off
by default. The gate requires 2048 or 4096 query rows, BF16, DCP4,
`FlashInferMLASparseSM120Impl`, query shape `[T,16,256]` and stride
`[4096,256,1]`, and weight shape `[16,192,512]` and stride `[229376,512,1]`.
Graph capture, padded heads, quantized query input and alternative FP4/FP8 BMM
paths retain the original implementation.

The helper gathers the smaller unabsorbed query and the current weights, then
runs four separate BMMs with the original per-rank dimensions and operand
strides. It preserves the untouched RoPE components and rank/head ordering.
Remote weights are temporary; no duplicate weights remain resident. The
communication-overlap path still starts the query collective before indexer
precomputation and waits afterward. The synchronous path still consumes any
pending indexer merge before gathering.

The isolated four-rank stage benchmark measured approximately 6.424 ms before
versus 4.021 ms after at 4096 rows. Every output BF16 bit matched on all ranks.
Incremental Torch allocation peaks were 679,477,248 versus 601,620,480 bytes;
these exclude fixed inputs and NCCL internal allocations. Small-query cases
were slower and are excluded by the gate. Raw stage measurements are in
[results/q_before_absorb](results/q_before_absorb).

Combined live measurements introduced this path together with wide hot
prefill: cold 128K C1 prefill increased from 988.72 to 1367.54 tokens/s, while
8K C1 decode measured 139.99 versus 140.74 tokens/s. These are bounded
cross-boot observations with different cold UUID prefixes, not isolated
attribution to this helper. Full 1M capacity, eight sequence slots and LMCache
remained configured. Independent cache-restore correctness investigation is
separate from this performance result.

CPU tests cover default-off behavior, thirteen unsupported-path guards,
qualified shapes, collective ordering with and without overlap, and exact BMM
operand strides. This production extraction does not introduce new arithmetic
or rerun live serving qualification.
