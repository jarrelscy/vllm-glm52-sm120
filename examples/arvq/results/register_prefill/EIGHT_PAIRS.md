# Eight-pair register reuse

The register implementation now reuses each weight fragment across eight token
pairs per warp, with independent FP32 accumulators and the same activation
planes, K order and split reduction. The existing whole-batch eligibility and
hot-route selection gates are unchanged. Decode and unsupported prefill shapes
retain their existing execution paths.

The register8 descriptor packs up to three groups of sixteen tokens into seven
integers per 32-slot window. The dedicated `hybrid_pack_register_pairs` export
preserves the older `hybrid_pack_pairs` descriptor ABI. The original wide
entrypoint remains available. The Python wrapper reuses descriptors between
gate/up and down projections exactly as before.

`eight_pairs_current_stack.json` records nine full real-layer comparisons with
fused cold gathering and route reduction enabled on both arms. All final bits
passed. Mixed-route speed ratios were 1.007 at 2048 tokens and 1.022 at 4096;
all-hot ratios were 1.065 and 1.066. The 2177-token fallback and all-cold cases
retain unchanged paths; their small timing differences are not attributed to
this change. Peak allocation did not increase. `eight_pairs.json` preserves
the earlier nine-case qualification with the preceding stack.

The repository integration preserves all six qualified CUDA function bodies
after symbol and formatting normalization. The runtime differs from the prior
repository version only in packing symbol and descriptor width. Five CPU
mapping/ABI tests and two existing runtime-dispatch tests passed. Scoped
pre-commit checks passed. The unchanged build script successfully compiled the
integrated binary for SM120a, retaining all five required old/new exports.

No GPU execution was performed during this integration while the server was
booting. The rebuilt production binary has not been independently run; GPU
qualification here refers to the source-equivalent lab implementation.
`eight_pairs_source_equivalence.json`, `eight_pairs_manifest.json` and
`eight_pairs_build.json` record that distinction and the exact hashes.
