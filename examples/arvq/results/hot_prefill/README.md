# Paired and wide hot prefill on SM120

These optional kernels reuse NVFP4 expert weights across token positions while
retaining all four residual activation planes. Two positions occupy the eight
FP4 MMA columns. The wide kernel shares a weight tile across four token pairs
within a CTA. Each warp keeps the original MMA and reduction order.

Build from the repository root with an SM120a-capable CUDA toolkit:

```bash
bash vllm/model_executor/layers/quantization/arvq/build_prefill_pairs.sh
bash vllm/model_executor/layers/quantization/arvq/build_prefill_wide.sh
```

Enable `VLLM_ARVQ_GROUPED_PREFILL=1`, `VLLM_ARVQ_COMPACT_PREFILL=1`,
`VLLM_ARVQ_PAIRED_HOT_PREFILL=1`, and `VLLM_ARVQ_WIDE_HOT_PREFILL=1` before starting
vLLM. The two new flags default to off. The dispatcher selects whole batches of
2048 or 4096 positions with at least 512 eligible hot routes and at least 32 routes
per selected expert. Original cold handling and token-route destinations remain
part of the common helper. Decode and other batch sizes use the existing path.

The attached measurements used actual initial 8+8 GLM-5.3 layer 3 cold weights and
NVFP4 hot weights on RTX PRO 6000 Blackwell. Eight alternating timing rounds
include routing, sorting, both projections, activation, scatter and combination.
Paired mixed prefill improved by 1.241x at 2048 and1.396x at 4096 versus the preceding
unpaired implementation. Against that paired baseline, the selected wide
variants improved mixed prefill by 1.077x at 2048 (128-thread L1) and1.150x at 4096
(512-thread L1). These are MoE microbenchmarks, not whole-model throughput.

All 18 wide full-pipeline byte comparisons passed; allocation peaks were flat or
slightly lower. The checked build's SASS instruction lines matched the qualified
runtime binaries. CPU coverage and dispatch tests are in
`tests/quantization/test_arvq_paired_prefill.py`.

Raw files: `paired.json`, `wide_128thread.json`, and `wide_512thread.json`.
