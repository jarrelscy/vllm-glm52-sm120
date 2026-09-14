# Exact activation fusion

Two independently opt-in paths preserve the existing half rounding after gate/up,
SiLU, and multiplication. Small-route decode also retains all four residual FP4
activation planes and the original down-projection split/reduction order.

- `VLLM_ARVQ_FUSED_ACTIVATION_PACK=1`: at most64 route slots.
- `VLLM_ARVQ_FUSED_COLD_ACTIVATION=1`: grouped cold expert activation only.
- Build with `bash vllm/model_executor/layers/quantization/arvq/build_activation.sh`.

All65,536 half gate bit patterns at two up values, plus random float inputs,
matched activation bits (including NaNs), packed values and scales. Twelve full
small-route real8x8 MLP cases matched final BF16 bits and graph replay. Nine full
cold-prefill cases matched final BF16 bits and allocation/reservation peaks.
Published CUDA source has identical normalized tokens and identical disassembled
machine code to the tested prototype. A rebuilt binary has not yet undergone a
separate GPU execution check. Raw results retain unchanged-control timing drift.

Microbenchmarks are warm fixed layer3 TP3 weights, not full-model throughput.
Live testing is ongoing; these results do not establish a 3,000 TPS prefill rate.
