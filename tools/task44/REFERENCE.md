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
reduction explicitly.

The completed same-topology TP4/DCP1 reference, with both reference switches on,
agreed with native execution on the first eight tokens of the same prompt.
The maximum chosen-token logprob difference was 0.2108 (mean 0.03385); replacing
the kernels did not remove the third-token PP4/TP4 flip. This reference request
took 729 seconds. These eight tokens are insufficient to establish general
distribution equivalence or to explain the benchmark's excessive reasoning.

A native TP4/DCP1 control with BF16 reduced-precision reduction disabled agreed
on all 32 greedy tokens with the original native control. Against the eight-token
reference, its maximum chosen-token logprob difference was 0.0759 (mean 0.02043).
This isolates one numerical contribution to the same-topology discrepancy, but
does not establish a corruption bug or justify a production precision change.
Long-context and concurrent fixed-prefix comparisons remain separate checks.

`check_dense_batch_threshold.py` measures another numerical discontinuity: the
same dense projection switches from FP4 MMA to BF16 reconstructed-weight GEMM
above `VLLM_NVFP4_P4_MAX_TOKENS` rows (16 in production). With BF16 truncated
reduction disabled, two 256-output fixtures at K=4096/16384 differed by relative
L2 0.00249/0.00179 across this threshold. Keeping both batch sizes on FP4 gave
bit-identical BF16 outputs in these fixtures. Applying the global scale after
FP32-output GEMM instead of rounding scaled weights first reduced the relative
differences to 0.000302/0.000293. These are diagnostic measurements, not a claim
that this discontinuity causes the observed benchmark behavior.
The full-model control keeping all 32 test rows on FP4 did not remove serial
drift: twenty identical 131071-token requests still alternated greedy tokens.
The post-GEMM scale prototype is therefore not promoted as a fix for this issue.

## Non-DCP selector control

Fixed-prefix runs up to 131071 input tokens, with 1/4/8 submitted requests,
showed probability shifts even between serial warm-cache repeats. Thus a
single-versus-concurrent difference alone does not isolate a concurrency bug.
On the native control, the largest common-top-20 probability change between
two serial passes was 0.3768, including one greedy flip. Prefix caching was on;
LMCache, DCP, MTP, and experimental DCP flags were off.

`check_nodcp_topk_order.py` identifies a confounder in that control: native
non-DCP prefill top-k changes output order on every one of 100 fixed-logit
repeats at 1 and 32 rows. With rounded, tied logits, the selected set also
changed on 99/100 and 100/100 repeats. Different choices among tied scores are
valid top-k results, but unsuitable for a deterministic reference. The existing
`inkernel` canonical-order setting acts in the DCP merge, which returns early
when DCP size is one. These measurements do not establish production DCP
corruption or show that tie-set changes occur on the real model's logits.

`VLLM_DSA_REFERENCE_TOPK=1` replaces prefill and decode selection with an eager
PyTorch stable sort: descending score, low logical index at ties, then descending
logical index for attention accumulation. It preserves relative row bounds and
uses -1 for padding. `check_reference_topk.py` covers ties, nonzero starts, short
rows, and empties. This adds an independent selector to the diagnostic toolbox;
indexer logits are still computed by the native kernel. The flag defaults off.

The full-model TP4/DCP1 control changing only this selector, with otherwise
native precision and kernels, returned bit-identical top-20 logprobs for all
20 identical 8192-token requests. Its 1/4/8-request comparisons were also
bit-identical on that prompt. This isolates the selector as the cause of the
observed non-DCP repeatability failure on the 8K control. It does not establish
that the production TP4/DCP4/MTP3 path has that failure. CUDA selector checks
also match native selected sets for unique logits under nonzero prefill starts,
ragged 2-D decode bounds, short rows, and empty rows.

### TP4 precision isolation (real checkpoint, September 15)

Three additional probes use the local initial-8x8 checkpoint and saved layer
activations. Paths in the scripts identify the exact checkpoint and trace arm;
raw traces are not included in Git.

- `check_real_tp_shards.py`: layer 3, expert 0, hot and cold gate/up and down.
  Actual production TP loading helpers reconstruct matrices identical to the
  unsharded matrices in all four cases. This is a loader fixture, not an
  exhaustive checkpoint audit.
- `check_real_qb_tp.py`: actual layer-0 BF16 Q projection, captured activation
  repeated over 32 rows. Default BF16 reduced-precision reduction gives 4,350
  full-versus-TP output mismatches; TP differs from FP64-rounded-BF16 at 4,346
  coordinates. Disabling reduced-precision reduction reduces the latter to 2
  (full-matrix GEMM differs at 14). This reproduces the first observed prefill
  layer difference without attention or communication.
- `check_real_p4_tp_partials.py`: actual layer-0 O projection, identical globally
  quantized weights and a real decode activation. Rounding each of four native
  FP4 partial outputs to BF16 before summing produces 2,131/6,144 differences
  against the unsharded native result, relative L2 0.002042. Keeping FP32
  partials until after summing reduces this to 3 differences, relative L2
  0.000001130. The fixture uses FP32 summation, so it isolates partial-output
  rounding rather than reproducing every detail of a BF16 NCCL reduction.

These identify local precision losses. Neither is yet established as the
explanation for the full-model PP4/TP4 third-token probability flip. A diagnostic
combining matched dense weight scales, strict BF16 accumulation, and FP32 TP
partial outputs is being compared against a PP4 control with the same policy.
Production precision defaults have not been changed on this evidence alone.
