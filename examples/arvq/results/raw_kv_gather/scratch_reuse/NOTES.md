# Consumed raw-KV scratch reuse

The existing `VLLM_GLM_RAW_KV_GATHER` flag remains OFF by default. This extends
its bounded C1/T4096 eligibility through 524,288 context tokens. Through
262,144 tokens, the original raw-KV path is used. Above that boundary, completed
attention outputs are copied into already-consumed regions of the temporary
rank-major gather. Resident KV cache and unconsumed peer data are never modified.
No context, concurrency, cache settings, arithmetic, or persistent weights change.

**Live 512K validation remains pending.** These are synthetic complete
attention-stage results, not full-model serving throughput or proof of total
serving capacity. The active production PyNccl tree guard still must pass.

| Context | Q-based baseline ms | Raw path ms | Reuse ms | Baseline peak bytes | Raw peak bytes | Reuse peak bytes |
| --- | --- | --- | --- | --- | --- | --- |
| 131,072 | 12.723 | 6.786 | 6.883 | 635191296 | 464535552 | 464535552 |
| 262,144 | 12.675 | 8.525 | 8.680 | 635191296 | 550518784 | 483409920 |
| 524,288 | 12.764 | 11.871 | 12.058 | 635191296 | 722485248 | 521158656 |

Values average six maximum-rank timings. All four ranks passed all 13
intermediate checks and final bitwise comparison at all three contexts.
The faster raw path remains selected through 256K. At 512K, reuse reduces
peak below the original Q-based baseline; the raw path without reuse exceeds
that baseline peak and is not selected there.

The output copy follows the corresponding backend attention call on the current
CUDA stream. A slot fits wholly within the consumed peer prefix before it is
written. Slots do not overlap. Stashed output views retain the entire gathered
allocation until merge completes, which the memory bound explicitly includes.
The caller deletes its separate output reference before attempting reuse.
The BF16 output, FP32 LSE, correction, and reduction kernels are unchanged.

`memory_bound.py` enumerates every rounded block count from 1,025 through
2,048. Its maximum explicit-tensor bound is 555,794,432 bytes, leaving
79,396,864 bytes below the measured 635,191,296-byte baseline stage peak.
The bound includes the whole gather, external outputs, local Q, final output,
two mapping allocations, all LSEs plus their stack, and the identity table.
It assumes fixed preallocated backend workspace and no hidden context-dependent
allocation. Allocator reservations and whole-model peak require live checks.

`rank0.json` through `rank3.json` preserve raw qualification results.
`source_equivalence.json` verifies the integrated peer loop matches the
qualified overlay and the reuse helper differs only by extra input assertions.
CPU tests cover consumed-prefix protection, output bits, repeat calls,
invalid input, the 256K boundary, default-OFF and unsupported-path fallbacks.
