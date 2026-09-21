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

## mcbook16: selectable residual books (`rvq256_mb16_256x8_expert_fp16block`, v5)

Extension of the FP16 block-scale format with sixteen residual books per
expert instead of one. Per layer, per projection:

- `codebooks`: `u32[E, 4352]` — rows 0-255 are the base book (unchanged);
  rows `256 + m*256 .. 256 + (m+1)*256` are residual book `m` for `m` in
  0..15, nibble-encoded on the same plain e2m1 grid.
- `book_factors`: `f32[16]` — the effective residual atom is
  `LUT(nibbles) * book_factors[m]`. Loaded from the checkpoint, never
  hardcoded (the current campaign uses `[1.0]*8 + [0.25]*8`).
- `selectors`: `u8[E, N/16, K/64]`, values 0-15 — one residual-book id per
  packed tile, indexed identically to the first three dims of `packed`.
- `packed`, `scales`, `global`: unchanged from the FP16 block-scale format.

Reconstruction per 8-weight group in tile `(i, j)` with
`m = selectors[e, i, j]`:

```text
w_group = global * fp16_block_scale * (c0[main] + book_factors[m] * c_m[res])
```

Serving implementation:

- Per-LAYER format detection: a projection decodes as mcbook16 exactly when
  the checkpoint provides its `arvq_{proj}_selectors` tensor; without it the
  untouched v4 path runs, so mid-campaign mixed checkpoints load. The config
  marker is `format = "rvq256_mb16_256x8_expert_fp16block"`, `version` 5,
  `codebook_scope = "expert"`.
- `hybrid.so`: `hybrid_launch_8x8_expert_mb16` grows the shared-memory LUT
  from 512 to 4352 words (17.4 KiB static smem) plus the 16 factors; the
  residual MMA accumulates separately and is weighted by `book_factors[m]`
  in FP32 (exact for arbitrary factors, 4 extra FMAs per tile-group step).
  The selector costs one uint8 read per tile per K-group.
- `prefill.so`: `arvq_dequant{,_fp16}_8x8_mb16`; `decode_gather.so`:
  `arvq_dequant_gather_fp16_8_mb16`. Residual half2 is weighted by the
  book's factor in FP32 before the add. Selector/factor pointers trail the
  original argument lists. Rebuild all three libraries together.
- The paired/wide diagnostic prefill runtimes (`VLLM_ARVQ_PAIRED_HOT_PREFILL`)
  have no mcbook16 decode; pairing is automatically disabled on layers that
  carry selectors.
