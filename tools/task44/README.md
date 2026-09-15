# Task 44: GLM-5.3 SM120 accuracy investigation

The large full-model PP4/TP4 probability flip was traced to a **PP4 reference
bug**: shared indexer selections were not transferred between pipeline stages.
The fix and causal partition-control results are in [REFERENCE.md](REFERENCE.md).
This is separate from the latent ragged indexer bug described below; it does
not establish corruption in production TP4 or explain the benchmark behavior.

The native ragged indexer path had two input-preparation bugs: queries were
packed but weights were not, and causal bounds subtracted the batch maximum
query length instead of each request's query length. Fix: `be7c3b62e`.
The last padded context column must retain the request's final context length:
the paged-logits scheduler uses it to determine its work. Padded Q and weights
are zeroed.

Production SM120 with MTP3 logs `use_flattening=True (next_n=4)` and bypasses
this native padded branch. This fix therefore does not establish the cause of
the reported production behavior. The native dispatcher also requests uniform
decode lengths: ordinary nonuniform requests can be routed through prefill.
The direct-helper ragged repro demonstrates the latent bug, not that an actual
production scheduler batch reaches it.

## Completed numerical checks

- Regression suite: 9 failed / 3 passed before; 12 passed after the fix.
- Corrected S1 fixture: the old form has max relative logits error 0.9883,
  minimum top-k Jaccard 0.264. Both corrected inputs reproduce isolated requests.
- Installed production indexer: four uniform/ragged layouts reproduce isolated
  logits and top-k sets exactly, with all compared logits finite.
- Full DCP attention sweep: 68 cells pass with a reference matching the kernel's
  FP8 query quantization. Maximum all-rank LSE error is 3.33786e-6 (base 2).
  Maximum final output row-normalized error is 0.012132; maximum isolated
  combine error is 0.010910. These output metrics still include BF16 rounding.

The initial LSE discrepancy (up to 0.179717) came from comparing BF16 Q against
a kernel that internally quantizes Q to FP8. It was not an LSE-base mismatch
or an arbitrary-FP32 KV-scale misread. The worst original rows span uniform,
duplicate and singleton patterns; they are not confined to short contexts.
The corrected reference uses an LSE tolerance of 0.005 and output tolerance of
0.03, rather than widening the old LSE tolerance to 0.5.

The handed-over S1 fixture also initially used an invalid indexer cache layout
and masked non-finite comparisons. The archived corrected fixture uses the
production cache writer and asserts finite comparisons. The sparse-MLA
attention ground-truth fixture uses its separate, correct cache format.

## Reproduction

Run inside the banked image for the before case, or the local fixed image
`glm53-arvq-sm120:accuracy-task44-20260915` for the after case. Mount this checkout
at `/work`; the installed indexer being tested resides under `/opt/vllm`.

```sh
/opt/vllm/.venv/bin/python /work/tools/task44/s1_indexer_ragged_repro.py
/opt/vllm/.venv/bin/python /work/tools/task44/s1_installed_repro.py
/opt/vllm/.venv/bin/python -m pytest \
  /work/tests/v1/attention/test_indexer_ragged_decode.py \
  --confcutdir=/work/tests/v1/attention -q
```

The attention sweep needs four otherwise-idle GPUs and 16 GiB shared memory:

```sh
export VLLM_DSA_CANONICAL_TOPK=inkernel
export NCCL_ALGO=RING,TREE NCCL_BUFFSIZE=1048576 NCCL_MAX_NCHANNELS=4
export NCCL_P2P_LEVEL=SYS VLLM_GLM_DCP_AG_RAW_TOPK=1
export VLLM_EXPERIMENT_DCP_BYTEPACK=1 VLLM_EXPERIMENT_DCP_BYTEPACK_OWNER=1
/opt/vllm/.venv/bin/torchrun --standalone --nproc-per-node 4 \
  /work/tools/task44/dcp_gt_matched_q.py --out /work/task44-results.json
```

`results_original_q_rows.json` documents the old-reference errors by row;
`results_matched_q.json` contains the corrected full sweep. These synthetic
checks constrain the attention/DCP hypothesis; they do not measure checkpoint
quality or prove full-model batch invariance. Trajectory and task-level controls
are tracked separately in the workspace's `benchmarks/token_dist_ab/task44/`.
