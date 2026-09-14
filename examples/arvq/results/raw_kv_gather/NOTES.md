# Raw-KV gather qualification

The repository implementation is opt-in with `VLLM_GLM_RAW_KV_GATHER=1`;
its default is OFF. This integration does not change the live profile.
The existing Q-before-absorption path remains the fallback.

The candidate gathers unchanged 656-byte MLA cache records and computes each
DCP partition locally. It retains the existing backend attention calls,
base-2 LSE correction, intermediate BF16 rounding, and destination-specific
reduction order. Before use, an actual PyNccl collective probe must verify
that order on every rank; otherwise all ranks retain the original path.
This is a runtime numerical guard, not a diagnostic dependency.

Eligibility is deliberately narrow: SM120, DCP4, one prefill request,
4,096 query tokens, context from 4,096 through 524,288, and the validated
query/weight/cache layouts and collective settings. Decode, longer contexts,
other batch shapes, and unsupported settings retain the original path.
The option does not allocate a persistent global cache. Through 262,144 tokens,
the original raw-gather path is retained. Above that boundary, completed
outputs reuse consumed gather scratch. See [scratch reuse qualification](scratch_reuse/NOTES.md);
live 512K validation remains pending.

## Original integration evidence

`pipeline_rank0.json` through `pipeline_rank3.json` contain the isolated
complete attention-stage checks. All ranks passed intermediate and final
bitwise comparisons. Rank 0 records these means of the maximum-rank timings:

| Context tokens | Original ms | Candidate ms | Original peak bytes | Candidate peak bytes |
| --- | --- | --- | --- | --- |
| 4,096 | 12.442 | 4.751 | 635191296 | 413138944 |
| 8,192 | 12.448 | 4.799 | 636076032 | 413138944 |
| 131,072 | 12.540 | 6.766 | 635191296 | 464535552 |

Peaks are incremental Torch allocations in that isolated stage. These timings
are not full-model throughput or a guarantee of total serving memory savings.

`qualified_layers.log.gz` preserves the live shadow log losslessly. It records
bitwise passes for all 78 target layers on all four ranks, plus layer 78.
Shadow mode returned the original attention result. `shadow_coverage.json`
records the layer coverage and the uncompressed log SHA256.

`ast_equivalence.json` verifies that the integrated function bodies match the
qualified implementation, except for the opt-in condition and entrypoint name.
The existing Q-before branch is unchanged. `source_manifest.json` identifies
the qualified source and repository file hashes. Imports and module names were
cleaned up; diagnostic and shadow hooks are absent from the integration.

Six CPU tests, including eight subtests, verify the default-OFF behavior,
eligibility limits, layout restrictions, and fallback controls. All scoped
repository pre-commit checks passed. No additional GPU run was needed for
this source-only integration of the already qualified arithmetic.
