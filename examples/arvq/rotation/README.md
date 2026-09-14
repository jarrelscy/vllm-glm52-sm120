# Retired rotation experiment

Status: **retired**. Rotation is excluded from the production format and active
tuning plan. The full checkpoint and serving profile are unrotated. This folder
is retained only to reproduce historical experiments; H32/H128 are not planned
production variants. The first matched experiment showed no reconstruction benefit.

`fit_rotation.py` and `projection_bench.py` default to `--blocks 0` (identity).
Use `--blocks 0 32 128` explicitly to reproduce the three-way ablation, or
`--blocks 128` for H128 only. These are conversion/experiment options, not a
runtime switch for existing weights. Already rotated weights must be re-encoded
without rotation before disabling their matching activation transform.

The complete published `jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid` checkpoint
is **unrotated**. The H128 files are standalone tuning samples for layer 3,
cold-local expert 2, TP rank 0. They are not a replacement full checkpoint.
The sample weights and fitted dictionaries are distributed separately through
Hugging Face; Git contains code, format metadata, and measured JSON only.

## Format and algebra

For each cold linear projection independently, store `W_rot = W D H`, where D
is a fixed diagonal sign matrix and H is a block-diagonal normalized Hadamard
matrix (block width 32 or 128). At inference, form row activations `x_rot=x D H`.
Then `x_rot W_rot.T = x W.T` before quantization. There is no output rotation,
and no rotation is moved through SwiGLU. Gate/up and down have separate signs
and dictionaries. Down blocks stay within each TP rank's 512 input channels.

The payload retains two FP4 codebooks (256 and 128 vectors of dimension 8),
15 index bits per eight weights, E4M3 scales per 128 weights, and one FP32
global scale per projection. This prototype stores signs explicitly as int8
per input channel, shared across experts; they could be bit-packed later.
Even the single-expert artifacts are below 2 bpw including signs and codebooks
(approximately 1.943 bpw for down and 1.947 bpw for gate/up; JSON is exact).
File-container headers are not included in tensor payload bpw.

`rotation_pack` fuses the activation transform with all four residual FP4
activation planes. A route-dependent branch applies rotation only when
`cold_ids[slot]>=0`; hot slots preserve the original packed representation.
The existing `hybrid_launch` FP4 MMA and weight layout are unchanged.

Serialized prototype artifacts contain packed indices, packed FP4 codebooks,
native block scales, explicit signs, and global scale. Safetensors metadata
declares transform and block width. These are standalone expert artifacts,
not production checkpoint shards or a vLLM loader integration.

## Matched fitting experiment

The matching original donor is unavailable locally. All sources here are
decoded AQLM approximations, not original BF16 donor weights. Rotation cannot
recover information already lost by AQLM.

Layers 3, 40, and 77; gate/up and down. Train on 512 sampled rows from cold
experts 0 and 1; evaluate 256 rows from held-out cold experts 2 and 3. Each
variant uses identical row selection, scale policy, 32,768 sampled training
vectors, beta sweep (0.5, 0.75, 1), fitting seed, and 16 synthetic activation
vectors. There is no activation calibration, blockwise fine-tuning, or seed
sweep. This is a feasibility ablation, not tuned model accuracy.

| Transform | Mean weight relative L2 | Mean output relative L2 |
| --- | ---: | ---: |
| Identity, refitted control | 35.143% | 35.161% |
| Signed Hadamard 32 | 35.591% | 35.707% |
| Signed Hadamard 128 | 35.576% | 35.792% |

These are arithmetic means over six projections; output error uses synthetic
activations. Neither metric is task accuracy or perplexity. The identity
control is newly fitted under the same procedure, not the existing production
ARVQ dictionary fit. Do not infer a universal tuned-format ranking.

## Kernel verification and latency

`rotation_bench.py` checks 24 cases: widths 6144/512, slots 8/32, identity/H32/H128,
and all-cold/mixed six-cold-two-hot routes. CUDA transform relative L2 is below
3.4e-8 against an independent FP64 Hadamard reference. P4 bits and scales match
the FP32 transform reference exactly; hot slots match original packing exactly.

For mixed routes, H128 adds about 1.07 us (8 slots) or 1.45 us (32 slots)
across gate/up and down packing. This excludes MMA, routing, and communication.

`projection_bench.py` additionally serializes a real layer-3 expert, reloads
and checks every tensor, encodes native cold weights, and executes the existing
FP4 MMA. Maximum error against independently dequantized weight and packed
activation operands is below 2.6e-7. It uses matching activations across variants.

| Token positions | Identity | H32 | H128 |
| --- | ---: | ---: | ---: |
| 1 | 28.82 us | 29.79 us | 30.23 us |
| 4 | 66.47 us | 67.56 us | 70.14 us |

Numbers sum separate gate/up and down pack+MMA microbenchmarks. They are **not**
a chained MLP or serving latency. Each projection rotates through >272 MiB of
distinct weight addresses, using eight cold expert addresses per launch, shared
across token positions. The pool duplicates one real expert's payload: this
tests cache eviction, not diverse expert routing or cross-projection effects.
CUDA graph capture obtains the current stream inside every launch; timings
include packing, MMA, and split reduction, excluding allocations and conversion.

## Reproduce

Published weight samples and fitted dictionaries are available in the
[pinned Hugging Face handoff](https://huggingface.co/jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid/tree/3237d6569412a0396e34bc25cc9b8df055ea7035/rotation-prototype-v1).
`TUNING_HANDOFF.json` pins the matching source revision. There are six layer-3
prototype weight files (two projections times three transforms) and 18 fit
files (three layers times two projections times three transforms).

This folder is self-contained. It includes the exact assignment and native
MMA source dependencies used by the experiment. In a CUDA 12.9+ environment
with PyTorch, safetensors, and an SM120 GPU, from the repository root:

```bash
export ARVQ_SOURCE_MODEL=/models/aqlm-source
export CUDA_VISIBLE_DEVICES=1
bash examples/arvq/rotation/build.sh
/opt/vllm/.venv/bin/python examples/arvq/rotation/fit_rotation.py
/opt/vllm/.venv/bin/python examples/arvq/rotation/rotation_bench.py
/opt/vllm/.venv/bin/python examples/arvq/rotation/projection_bench.py
```

`ARVQ_SOURCE_MODEL` must point to the original AQLM hybrid snapshot containing
the AQLM codes and dictionaries. The published unrotated ARVQ checkpoint cannot
replace that input: it no longer contains the source AQLM representation.
H128 mixes multiple original eight-weight groups, so a full rotated conversion
must decode and re-encode weight blocks; it cannot reuse the previous
65536-entry AQLM-to-ARVQ index translation table. Stream expert weights rather
than materializing the full model in BF16.
`build.sh` builds `assign.so`, `hybrid.so`, and `rotation.so` locally; binaries
are ignored by Git. The Python path above is the serving image environment;
an equivalent activated virtual environment can be used elsewhere.

Scripts write fitted `.pt` files, standalone `.safetensors`, and fresh JSON
beside themselves. To benchmark downloaded fits without retraining, place the
`fit_l3_{gateup,down}_b{0,32,128}.pt` files in this folder, build the libraries,
then run `projection_bench.py`. The independent packing benchmark needs no
model weights. Frozen measured JSON lives in `results/` and is not overwritten.

`original_run_manifest.json` records hashes of the original local experiment,
including its locally compiled library. It does not claim to hash these
portable scripts: imports, paths, help handling, and formatting were adapted
without changing numerical operations. No production weights or serving
configuration were changed to enable rotation.

Active tuning should use unrotated ARVQ (`b0` / identity). Do not use the H32/H128
sample weights for the production checkpoint. Their matching activation transform
is required to interpret those historical artifacts correctly.
