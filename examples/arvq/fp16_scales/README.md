# Experimental FP16 ARVQ block scales

This branch contains the tested SM120 FP16 cold-weight-scale trial. It changes the
cold-scale ABI of `hybrid.so`, `prefill.so`, and `decode_gather.so` together with
the serving loader. Rebuild all three libraries together. Do not mix these
libraries with the original loader or pass U8 scales to their cold-weight paths.
This is an experimental branch, not a backwards-compatible production feature.

## Representation and execution

The loader still reads the existing checkpoint format. After TP sharding, it
converts every ARVQ gate/up and down block-scale tensor from unsigned FP8 E4M3
bytes to FP16, preserving its value exactly. It rejects invalid unsigned scale
codes. Indices, FP4 codebooks, FP32 globals, hot NVFP4 experts, and activation
scales remain unchanged.

Cold decode executes the same two FP4 codebook MMAs with unity native weight
scale, then applies the FP16 scale (exactly promoted to FP32) to each K=64 partial
result with FP32 FMA. No conversion back to FP8 or FP4 occurs. Plain prefill and
fused decode/gather also load FP16 scales directly. Fused activation/down uses
the modified hybrid library.

This changes floating-point accumulation order and is not universally bitwise
identical to native FP8-scaled MMA. Converting existing rounded scales does not
recover fitting precision or demonstrate a quality improvement.

## Build and numerical tests

Inside the serving CUDA image, from the repository root:

```bash
cd vllm/model_executor/layers/quantization/arvq
for f in hybrid prefill decode_gather; do
    nvcc -O3 -gencode arch=compute_120a,code=sm_120a --shared \
        -Xcompiler=-fPIC "$f.cu" -o "$f.so"
done
```

The scripts expect a writable `/work` directory inside the serving image.
Compile the parent commit's `hybrid.cu` to `/work/baseline.so` and `prefill.cu` to
`/work/baseline-prefill.so`, using the same flags. Copy this branch's compiled
`hybrid.so` to `/work/fp16.so`. Install this branch's three libraries and loader
in `/opt/vllm/vllm/model_executor/layers/quantization/` (the libraries in its
`arvq/` subdirectory).

```bash
/opt/vllm/.venv/bin/python examples/arvq/fp16_scales/bench.py
/opt/vllm/.venv/bin/python examples/arvq/fp16_scales/check_prefill.py
```

The projection script tests TP1 and TP4-local gate/up/down shapes, two distinct
expert books, reversed routing, index endpoints, four activation planes, graph
replay and split reduction. FP8-representable scale cases matched exactly in
these fixtures. Arbitrary FP16 scales matched an independent CPU oracle with
relative L2 below 4e-7. These projection timings were collected while another
serving workload was active; they are provisional, not isolated throughput.

The prefill script verifies bitwise equality with the original decoder for
FP8-representable scales, both FP16 and BF16 outputs. Non-FP8-representable scales
and gathered activation rows are checked against an independent CPU oracle.

## Full-model measurement

Checkpoint: `jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid`, pinned revision
`2c88fc41911eaa3b675f1ba950095bba4c16c654` (29/75 PV layers completed).
Four RTX PRO 6000 Blackwell GPUs, TP4/DCP4, MTP3 probabilistic drafting and
standard rejection, CUDA graphs enabled, LMCache disabled, max context 940000,
GPU memory utilization 0.94 for both variants. Three 512-token repetitions per
workload, one request at a time; no concurrent GPU workload during model tests.

Pangram uses raw completions at temperature 0, top_p 1, with
`"The quick brown fox jumps over the lazy dog. "` repeated 17 times. The other
workloads use chat, temperature 1, top_p 0.95, seeds 1234 through 1236. TPS includes
reasoning tokens and uses completion-token count divided by streaming decode
time. The first streamed chunk is included in the count for both variants.

| Workload | FP8 scales tok/s | FP16 scales tok/s |
| --- | ---: | ---: |
| Pangram | 145.37 | 143.80 |
| Counting | 116.28 | 122.42 |
| Prose | 88.10 | 86.01 |
| Code | 101.09 | 96.89 |

Pangram emitted 4.0 tokens/step and produced identical text in all repetitions:
1.08% measured throughput cost. Other workloads changed acceptance and generated
text; their TPS differences do not isolate kernel cost. Warm 7182-token prefill
TTFT was 4.828s vs 4.867s (+0.80%). Both arithmetic checks answered 323; full
reasoning/output identity did not hold, including some temperature-zero requests.
No full task suite or full-context quality evaluation was performed.

All 150 ARVQ projection-scale tensors were verified FP16 on each TP rank. Extra
scale storage is 3.694 GiB model-wide, or 0.924 GiB/GPU at TP4. Model load memory
was approximately 74.08 GiB/GPU before and 75.04 GiB/GPU after conversion.

A subsequent boot at max context 950000 and memory utilization 0.945 succeeded:
12.58 GiB KV cache/GPU, reported capacity 987119 tokens, graph capture on all four
GPUs and a clean 128-token pangram generation. No full 950k-token prompt was
submitted. Checkpoint files were not modified; conversion happens during load.

Numeric measurements are in `fullmodel_results.json` and
`projection_results.json`.

## rs4: residual book at quarter weight (`rvq256_256x8_expert_fp16block_rs4`)

Extension of the FP16 block-scale format. Tensor layout, dtypes, packing, and
bit budget are unchanged from `rvq256_256x8_expert_fp16block`; only the
reconstruction contract differs. Per 8-weight group with main index `a` and
residual index `b`:

```text
w_group = global * fp16_block_scale * (c0[a] + c1[b] / 4)
```

Motivation: with both books on the shared scale, the finest nonzero residual
correction is 0.5 (the smallest e2m1 magnitude), and fitted v3 residual books
pile 84% of their mass on {0, 0.5} while never using magnitudes above 1.5.
Dividing the residual contribution by 4 remaps its used range onto the full
e2m1 grid, giving 0.125-step corrections with no size, layout, or speed cost.

Serving implementation:

- `hybrid.so`: the residual MMA takes constant ue4m3 scale 0.25 (`0x28282828`)
  instead of unity; exact in the MMA's FP32 accumulate, zero extra
  instructions. FP16 block scales are applied afterwards, unchanged.
- `prefill.so` / `decode_gather.so`: the residual half2 is multiplied by 0.25
  before the add; exact in FP16 for e2m1 magnitudes.
- Symbols: `hybrid_launch_8x8_expert_rs4`, `arvq_dequant{,_fp16}_8x8_rs4`,
  `arvq_dequant_gather_fp16_8_rs4`. Rebuild all three libraries together.

Fitting requirements:

- Emulate reconstruction as `scale * (main + residual / 4)`; residual atoms
  remain constrained to the e2m1 grid {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4,
  +-6}, now denoting quarter-scale steps {0, +-0.125, ..., +-1.5}.
- No scale clamp is required: the /4 rides the MMA constant, not the block
  scale.
- Per-layer safetensors metadata: `arvq_format = "rvq256_256x8_expert_fp16block_rs4"`.
  Assembled checkpoint config marker: `format` as above, `version` 3 or 4,
  `codebook_scope = "expert"`, `codebook_sizes = [256, 256]`,
  `activation_planes = 4`, `weight_scale_group = 128`.
