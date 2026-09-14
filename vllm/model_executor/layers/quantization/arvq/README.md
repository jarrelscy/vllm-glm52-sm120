# SM120 NVFP4–ARVQ hybrid

This experimental loader supports two serialized cold formats alongside hot
NVFP4 experts. Each cold vector has eight weights and is the sum of two FP4
codebook vectors.

| Checkpoint `arvq.format` | Index bits | Codebook sizes | Index + scale bpw |
| --- | --- | --- | ---: |
| `rvq256_128x8` | 8 + 7 | 256 + 128 | 1.9375 |
| `rvq256_256x8` | 8 + 8 | 256 + 256 | 2.0625 |

The bpw figures include one E4M3 scale per 128 weights; shared codebooks and
projection globals add a small overhead. The existing checkpoint uses 8+7.
8+8 exceeds the original 2 bpw cold budget. Hot weights remain NVFP4.

Both native branches consume four residual activation planes through SM120
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
