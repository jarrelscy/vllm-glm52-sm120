# Independent eager references

`vllm/model_executor/layers/quantization/arvq_reference.py` implements diagnostic
PyTorch replacements for the hybrid projection and SM120 sparse MLA attention.
Both switches default off:

- `VLLM_ARVQ_REFERENCE_WEIGHTS=1`: unpack serialized ARVQ 8+7 or 8+8 code indices,
  FP4 codebooks, and E4M3 scales independently; unpack NVFP4 weights and scales;
  reconstruct four residual activation planes; multiply with FP32 matmul.
  Applies to callers of the hybrid projection, including dense hot-only callers.
  Bypasses grouped prefill and the fused MLP activation pack.
  The dense P4 dispatcher also bypasses paired MMA and the CUDA weight decoder;
  its large-batch reference preserves the BF16 reconstructed-weight rounding,
  but uses explicit FP32 accumulation rather than truncated BF16 reduction.
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

## Distribution bisection notes

The initial optimized PP4 and independent PP4 reference agreed on the first
eight tokens of `torch-tensor-parallelism@1` (chosen-logprob max delta 0.0174).
This is not a full-vocabulary equivalence claim. Plain optimized TP4, without
DCP or MTP, already diverges from PP4 at the third token. Enabling DCP or MTP
introduces later, close-margin flips on this probe; V1 versus V2 without MTP
was bit-identical for all 32 captured tokens.

Two confounders were identified:

1. Dense output projection quantization runs after TP sharding and uses a
   shard-local maximum. Thus the same BF16 checkpoint produces different
   effective NVFP4 weights in PP4 and TP4. On real layer 3, local maxima were
   85% to 100% of the full-matrix maximum; sampled reconstructed weights differed
   by about 8% in Frobenius norm. `VLLM_NVFP4_P4_GLOBAL_SCALE=1` is a diagnostic
   option to all-reduce the maximum at load time. Its reconstructed shards match
   the unsharded quantization bit-for-bit, but its full-model A/B **did not fix**
   the third-token discrepancy. It remains off by default.
2. Default PyTorch BF16 GEMM permits reduced-precision reduction. In sampled
   32x64 output fixtures at K=4096/16384, errors against FP64 reached
   0.0078125/0.03125. Setting
   `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`
   made those fixtures match the BF16-rounded FP64 result exactly. This is a
   measured numerical approximation, not yet proof of the task-level failure.

`check_dense_global_scale.py` tests quantization invariance under TP slicing.
`check_paired_dense.py` tests paired MMA, BF16 weight decoding, and both dense
reference dispatch paths. Its BF16 comparison disables reduced-precision
reduction explicitly. Same-topology full-model reference and accumulation-policy
controls are required before assigning the remaining distribution shift.
