# Standalone ARVQ 8+8 comparison

This reproduces the format microbenchmark; it does not convert a checkpoint or
change serving defaults. See [results](../results/arvq_8x8/RESULTS.md). The
production CUDA library now also exposes both formats independently.

Build the experimental library and the ordinary production baseline:

```bash
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC \
  -gencode arch=compute_120a,code=sm_120a \
  examples/arvq/format_experiment/hybrid_8x8.cu \
  -o examples/arvq/format_experiment/hybrid_8x8.so
bash vllm/model_executor/layers/quantization/arvq/build.sh
CUDA_VISIBLE_DEVICES=1 ARVQ_LAB_ROOT=/lab \
  python examples/arvq/format_experiment/bench_arvq8x8.py
```

Use an exclusive GPU window. `ARVQ_BASE_KERNEL_LIB` overrides the baseline library
path; `ARVQ8X8_OUTPUT_DIR` chooses the output directory. No generated library is
tracked in Git.

The data loader expects the original experiment artifacts under `ARVQ_LAB_ROOT`:
`scale_experiment/inventory.json`, `rvq_layer3_rank3.safetensors`, and
`fit_l3_gateup.pt` / `fit_l3_down.pt`. The inventory points to the original source
checkpoint, needed for an independent natural-layout oracle. These are the same
conversion artifacts used in the original ARVQ experiment. No weights are
included in this source repository.

Three arms are measured: original 8+7, aligned 8+8 carrying identical indices
below 128, and 8+8 using all 256 residual entries. The upper 128 entries in the
last arm are random valid FP4 vectors; it tests speed and arithmetic only.
Indices are repacked before timing. The script checks matched raw projection
outputs by their integer bit patterns and measures eight alternating-order
rounds after equal warmups. Both warm and rotating expert banks are reported.
