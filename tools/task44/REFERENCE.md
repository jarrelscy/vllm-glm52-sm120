# Independent eager references

`vllm/model_executor/layers/quantization/arvq_reference.py` implements diagnostic
PyTorch replacements for the hybrid projection and SM120 sparse MLA attention.
Both switches default off:

- `VLLM_ARVQ_REFERENCE_WEIGHTS=1`: unpack serialized ARVQ 8+7 or 8+8 code indices,
  FP4 codebooks, and E4M3 scales independently; unpack NVFP4 weights and scales;
  reconstruct four residual activation planes; multiply with FP32 matmul.
  Applies to callers of the hybrid projection, including dense hot-only callers.
  Bypasses grouped prefill and the fused MLP activation pack.
- `VLLM_ARVQ_REFERENCE_ATTENTION=1`: decode selected physical `fp8_ds_mla`
  cache entries; compute explicit FP32 scores, softmax, and value multiplication.
  Match the optimized kernel's per-128 FP8 query quantization. Return base-2 LSE
  and zero output / negative-infinity LSE for empty rows.

Use eager execution, no MTP, and no LMCache for the initial reference arm.
The reference module disables TF32. It synchronizes for dynamic expert dispatch;
CUDA graph execution is unsupported. Temporary decoded weights are bounded to
1024 output rows, and attention decodes one query's selected cache entries at a
time. It is intentionally slow and retains no persistent decoded-weight cache.

The model checkpoint is unchanged. Weight quantization and the documented
activation quantization are reproduced; this is not an unquantized donor model.
The indexer, cache writer, dense projections outside the hybrid path, routing,
normalization, and TP/PP communication are still production implementations.
With DCP enabled, physical-index conversion and DCP output combine remain
production code too. PP4 avoids that combine but changes parallel arithmetic.

## Checks

Run `tools/task44/check_reference.py` in the serving image with the repository
mounted at `/work`. The optimized functions must be installed in the image and
the reference switches must be unset for comparison.

- 18 optimized/reference projection comparisons: both cold formats, K=128/256/7168,
  1/4/9 routed rows, two experts, hot/cold/empty routes, gate/up global scales.
- Independent logical hot-weight packing roundtrip, and cold decoding compared
  with the installed CUDA decoder.
- Explicit attention compared with FP64 math, including duplicates and empties.
- Production cache writer and SM120 attention kernel compared with the paged
  reference, including duplicates, partial padding, and a fully empty row.

Initial results: all pass; maximum projection absolute error 6.103515625e-5;
maximum nonempty-row paged LSE absolute error 1.430511474609375e-6.
These fixtures do not establish full-model token-distribution equivalence.
Full-model short/long/batched comparisons are a separate pending experiment.

The first PP4 end-to-end attempt exceeded the default 300-second worker RPC
timeout. Diagnostic runs should set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600`.
The row block was increased from 128 to 1024 to reduce Python dispatch overhead;
the checks also cover N=2048 to exercise multiple output blocks.
