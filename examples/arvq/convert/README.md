# Build an NVFP4–ARVQ hybrid checkpoint

These tools transcode the **existing AQLM dictionaries and indices**, not the
original BF16 donor weights. All non-cold tensor payloads are copied byte for
byte. The source checkpoint is never edited. The output requires the matching
custom ARVQ vLLM backend.

Dependencies are Python, PyTorch with CUDA support, NumPy, safetensors, and a
CUDA toolkit supporting SM120. The tested environment used PyTorch
2.11.0+cu130 on RTX PRO 6000 Blackwell Max-Q GPUs. Run these commands from the
fork root using its configured virtual environment:

```bash
nvcc -O3 -gencode arch=compute_120a,code=sm_120a --shared -Xcompiler=-fPIC \
  examples/arvq/convert/assign.cu -o examples/arvq/convert/assign.so

export ARVQ_SOURCE_MODEL=/path/to/local/source/snapshot
export ARVQ_FIT_DIR=/path/to/arvq-fits
export ARVQ_OUTPUT_MODEL=/path/to/GLM-5.3-Vision-NVFP4-ARVQ-hybrid

CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/arvq/convert/fit_codebooks.py \
  --seed 91426
CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/arvq/convert/build_checkpoint.py \
  --chunk-experts 16
.venv/bin/python examples/arvq/convert/audit_checkpoint.py \
  --model "$ARVQ_OUTPUT_MODEL"
```

The scripts also accept `--source-model`, `--fit-dir`, and `--output` arguments
as applicable. `ARVQ_ASSIGN_LIBRARY` optionally overrides the compiled
`assign.so` path. The source environment default is the original local snapshot
used in development; set it explicitly on another machine.

The full frozen fit used seed **91426**, one initial `torch.manual_seed`,
ascending layers **3 through 77**, gate/up then down, beta candidates
**0.5, 0.75, 1.0**, and TF32 disabled. The RNG stream continues across all layers
and projections; fitting an isolated layer with the same seed does not reproduce
its full-run initialization. The algorithm uses 12 k-means iterations,
10 alternating refinement iterations, and three final assignment iterations.
New fit artifacts record seed and layer order. GPU reduction nondeterminism can
prevent bitwise-identical refits; to reproduce the published conversion exactly,
point `ARVQ_FIT_DIR` to its included `arvq_frozen_fits` directory. The build report
records SHA256 for every actual frozen fit.

Each 8-weight vector uses an 8-bit index plus a 7-bit residual index. A
128-weight block has an FP8 E4M3 scale; two shared FP4-constrained codebooks and
a float32 global scale complete the representation. Total cold storage remains
below 2 bits per weight including these tensors. Serialization uses full global
dimensions. The runtime loader slices gate/up row tiles and down input groups
for TP4 and appends the guard word.

Conversion is resumable through `build_records` and atomic shard renames. It
validates native packed indices against independent bit decoding, checks finite
scales and the bit budget, and publishes the final HF index only when every
shard is present. `audit_checkpoint.py` independently checks the entire index,
all file sizes, every cold tensor shape/dtype, and bit budgets.
`BUILD_COMPLETE.json` means checkpoint construction is complete; serving and
model-quality validation are separate.

The tested complete conversion used 16-expert chunks and peaked at approximately
195 MB of PyTorch GPU allocations, plus the CUDA context. All original vision
and MTP tensors are retained. Copying the non-cold source tensors dominates the
disk traffic.
