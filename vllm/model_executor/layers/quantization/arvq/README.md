# SM120 NVFP4–ARVQ hybrid

This experimental loader reads an actual serialized ARVQ checkpoint. Hot experts
retain NVFP4; cold experts use additive FP4 dictionaries of 256 and 128 vectors,
each eight-dimensional. Their 15-bit index pair takes 1.875 bits/weight. One E4M3
scale per 128 weights adds 0.0625 bits/weight. Dictionary and projection scale
metadata add a small overhead below the 2 bpw cold budget.

Both branches consume four residual activation planes through native SM120
FP4 block-scaled MMA. Plane contributions are weighted 1, 1/16, 1/256, 1/4096.
All token counts, including prefill and speculative verification, use the same
serialized ARVQ weights and activation path. No original AQLM weights or fallback
are retained. Prefill is bounded to 128-token chunks by default; this initial
implementation prioritizes consistent arithmetic over grouped-prefill throughput.

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

The marker selects the new loader; checkpoints without it retain their existing
behavior. Vision checkpoints must put this marker in both the root and nested
`text_config.quantization_config` envelopes, including the configuration used by
the MTP drafter. The inherited `aqlm_layer_books` field describes expert counts and must
have `n_base=0`. Hot expert tensors and `hyb_kind` keep their existing names.
Each cold projection (`w13` or `w2`) is serialized as:

- `arvq_<projection>_packed`: uint32 `[E,N/16,K/64,60]`, no guard in checkpoint.
- `arvq_<projection>_scales`: uint8 `[E,N/16,K/128,16]`.
- `arvq_<projection>_codebooks`: uint32 `[384]`.
- `arvq_<projection>_global`: float32 `[1]`.

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
