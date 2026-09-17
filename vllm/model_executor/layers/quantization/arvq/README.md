# SM120 NVFP4–ARVQ hybrid

This experimental loader supports three serialized cold formats alongside hot
NVFP4 experts. Each cold vector has eight weights and is the sum of two FP4
codebook vectors.

| Checkpoint `arvq.format` | Index bits | Codebook sizes | Index + scale bpw |
| --- | --- | --- | ---: |
| `rvq256_128x8` | 8 + 7 | 256 + 128 | 1.9375 |
| `rvq256_256x8` | 8 + 8 | 256 + 256 shared | 2.0625 |
| `rvq256_256x8_expert` (v3) | 8 + 8 | 256 + 256 per cold expert | 2.0625 |

The bpw figures include one E4M3 scale per 128 weights; shared codebooks and
projection globals add a small overhead. v3 has separate books per cold expert.
8+8 exceeds the original 2 bpw cold budget. Hot weights remain NVFP4.

All native branches consume four residual activation planes through SM120
FP4 block-scaled MMA, weighted 1, 1/16, 1/256, 1/4096. Both cold formats use
two MMAs per K tile. The 8+8 specialization uses aligned 16-bit index pairs;
it does not add another MMA. No original AQLM weights or fallback are retained.
For eligible long prefills, the optional grouped path reconstructs temporary
FP16 cold weights and uses GEMM with FP32 outputs. Its decoder supports both
serialized layouts. Remaining routes retain native P4 execution.

Build a derived image from the local vision-enabled GLM image:

```bash
docker build -f Dockerfile.arvq -t glm53-arvq-sm120:local .
```

The CUDA source requires CUDA 12.9 or newer and SM120a. The base image provides
vLLM, its existing GLM vision/MTP support and compiled dependencies. The ARVQ
library can also be built with `NVCC=/usr/local/cuda/bin/nvcc bash build.sh`.
`VLLM_ARVQ_KERNEL_LIB` optionally overrides its runtime location.

The checkpoint's existing `quant_method: nvfp4_aqlm_hybrid` envelope must contain:

```json
"arvq": {
  "format": "rvq256_128x8",
  "activation_planes": 4,
  "weight_scale_group": 128,
  "version": 1
}
```

Use `"format": "rvq256_256x8"` for an 8+8 checkpoint, retaining the other
fields above. The marker must agree with the serialized tensor layout: changing
metadata alone does not convert weights. Existing 8+7 checkpoints need no edits.

The marker selects the loader; checkpoints without it retain their existing
behavior. Vision checkpoints must put this marker in both the root and nested
`text_config.quantization_config` envelopes, including the configuration used by
the MTP drafter. The inherited `aqlm_layer_books` field describes expert counts and must
have `n_base=0`. Hot expert tensors and `hyb_kind` keep their existing names.
Each cold projection (`w13` or `w2`) is serialized as:

- `arvq_<projection>_packed`: uint32 `[E,N/16,K/64,W]`, where
  `W=60` for 8+7 and `W=64` for 8+8; no guard in the checkpoint.
- `arvq_<projection>_scales`: uint8 `[E,N/16,K/128,16]`.
- `arvq_<projection>_codebooks`: uint32 `[384]` for 8+7 or `[512]` for 8+8.
  The first 256 words form the first codebook; the remainder form the residual
  codebook. Each word packs eight FP4 E2M1 nibbles.
- `arvq_<projection>_global`: float32 `[1]`.

Each 16×64 tile contains 128 eight-weight groups in native fragment order.
For a group beginning at `(row, col)` within that tile:
`j = row // 8 + 2 * (col // 32)`,
`lane = (row % 8) * 4 + (col % 32) // 8`, and `p = j * 32 + lane`.
The 8+7 layout stores `first | (residual << 8)` in 15 consecutive bits at
bit offset `15*p`. The 8+8 layout stores that pair in 16 bits at offset `16*p`:
word `p//2`, shift `16*(p%2)`.

The original CUDA exports remain 8+7: `hybrid_launch`, `arvq_dequant`, and
`arvq_dequant_fp16`. New exports `hybrid_launch_8x8`, `arvq_dequant_8x8`, and
`arvq_dequant_fp16_8x8` have the same argument lists and handle 8+8. Python
selects the matching specialization from the validated tensor layout. Rebuild
both libraries before loading an 8+8 checkpoint. There is no serving flag that
reinterprets 8+7 storage as 8+8.

The loader slices gate/up rows and down K groups for tensor parallelism, adds one
guard word, and repacks hot weights to native MMA fragment order. vLLM's MoE
runner retains ownership of the tensor-parallel output reduction. Expert
parallelism is unsupported. Both activation and intermediate K dimensions must
be multiples of 128. `VLLM_ARVQ_CHUNK_TOKENS` may be 1–256.

The complete routed MLP is a Torch custom operator with a fake implementation
for compilation. Actual CUDA graph capture includes activation packing, both
hybrid projections, SiLU, and weighted route reduction. No stream is cached
outside a launch; all work uses the current Torch stream.

Run the focused tests from the fork root after building the CUDA library:

```bash
.venv/bin/python -m pytest tests/quantization/test_arvq_hybrid.py -q
```

The CPU tests exercise all four TP shards and invoke the actual main-model and
MTP checkpoint loaders to verify serialized name remapping. The GPU test checks
mixed-format execution, bounded-chunk equivalence, CUDA graph replay, and Torch
full-graph tracing. SM120 is required for the GPU test.

## Version 3: per-expert 8+8 books

Use this exact marker in root `quantization_config.arvq` and in
`text_config.quantization_config.arvq` when present:

```json
{
  "format": "rvq256_256x8_expert",
  "version": 3,
  "codebook_scope": "expert",
  "codebook_sizes": [256, 256],
  "activation_planes": 4,
  "weight_scale_group": 128
}
```

Only the codebook tensor shape changes: uint32 `[E,512]`. `packed`, `scales`,
`global`, fragment ordering, FP4 nibble semantics and activation packing stay
unchanged. Global scale remains `[1]`, not `[E]`. Gate/up share one pair per
cold expert; down has a separate pair. All books are replicated on TP ranks;
only packed weights and scales are sliced on the existing N/K dimensions.
The loader rejects shape/dtype mismatch, including broadcasting `[512]` into
`[E,512]`. Shared `[512]` v2 remains a separate, unchanged ABI path.

`hybrid_launch_8x8_expert` has the same signature as `hybrid_launch_8x8`.
Its cold block reads `cb + cold_ids[slot] * 512` into its 2 KiB shared LUT.
No cross-expert LUT cache exists in this path. Fused down projection selects
the same new ABI. Grouped prefill, reference decode and fused decode/gather
select one expert's contiguous `[512]` row before calling the decoder.
Hot-only paired/wide paths do not consume cold books: their route partition
requires a negative cold slot. Existing grouped GEMM arithmetic is unchanged;
it is not newly claimed to be identical to native FP4-plane prefill.

Book row is always the **cold slot**, never the global expert ID. The legacy
convention assigns cold slots in ascending `hyb_kind == 2` global-ID order.
For any other manifest ordering, mirror the ordered list explicitly in
`aqlm_layer_books["L"]["cold_expert_ids"]` in both quantization envelopes, e.g.
`[2,0]` means packed/scales/books row 0 belongs to global expert 2 and row 1
belongs to global expert 0. IDs must uniquely enumerate the cold `hyb_kind`
entries. The runtime does not discover an external manifest file implicitly.
Reordering storage must reorder packed weights, scales, books and this mapping
together. EP remains explicitly unsupported, rather than silently using a
TP mapping under EP.

For synthetic equivalence tests, duplicate shared books in memory with
`shared_books.repeat(E, 1)`. No production checkpoint conversion is needed.
See `docs/arvq_expert_books_v3.md` for measured parity, timing and limitations.
