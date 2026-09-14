# Experimental cold gather and route reduction fusion

Both options default to off:

```yaml
environment:
  VLLM_ARVQ_FUSED_COLD_GATHER: "1"
  VLLM_ARVQ_FUSED_ROUTE_SUM: "1"
```

The existing grouped-prefill gate still applies. These paths additionally require exactly 2048 or 4096 tokens, hidden size 6144, top-8 routing and chunk size 128. Cold gather requires BF16 contiguous input, gate/up width 1024 and ARVQ 8+8 codebooks. Route reduction requires contiguous FP32 routing weights and routed outputs, with BF16 final output. Other layouts and shapes keep the existing implementation, including ARVQ 8+7 cold decoding.

Cold fusion launches weight-decode CTAs and input-gather/cast CTAs together. It preserves decoder arithmetic, decoded FP16 weights, gathered FP16 inputs, and every expert GEMM's shape and strides. It avoids the intermediate BF16 gather allocation and adds no resident weight cache. Build both prefill libraries with:

```sh
bash vllm/model_executor/layers/quantization/arvq/build_prefill.sh
```

Route fusion preserves Torch's four-accumulator top-8 sum and separate FP32 multiplication rounding. Its scalar inline PTX multiplication is intentional: disabling Triton floating-point fusion alone still allowed packed multiply/add to become FFMA on the tested SM120 toolchain. Replacing it requires bitwise requalification.

## Isolated results

The combined candidate was compared against the qualified Wide hot-expert baseline (mode 0 at 2048 tokens, mode 2 at 4096), using real checkpoint layer 3, TP rank 3. Three alternating rounds included routing, CPU count synchronization, all projections, gather/scatter and final reduction. Independent decoded-weight and gathered-input bit checks passed. All six complete-output checks matched BF16 bits.

| Tokens | Routing | Baseline | Combined | Ratio |
| --- | --- | ---: | ---: | ---: |
| 2048 | Mixed | 13.479 ms | 11.977 ms | 1.125x |
| 4096 | Mixed | 17.985 ms | 16.135 ms | 1.115x |
| 2048 | All cold | 13.140 ms | 10.775 ms | 1.219x |
| 4096 | All cold | 13.637 ms | 11.243 ms | 1.213x |

All-hot controls also matched bits, with 1.017x/1.047x timing ratios. Short control runs showed clock/timing drift; these ratios are not independent proof of that magnitude of improvement. Peak allocated memory never increased; the 4096 mixed case saved 17.3 MB of temporary allocations. Other GPUs ran independent bounded experiments, so these are local microbenchmarks, not whole-server throughput measurements.

Raw [cold-only fusion results](results/fused_prefill/cold_gather.json) and [combined results](results/fused_prefill/combined.json) include per-round timings, memory, checkpoint and source hashes. Live accuracy, capacity and performance qualification remain separate. Neither option changes checkpoint tensors, context limits, cache settings or decode arithmetic.
