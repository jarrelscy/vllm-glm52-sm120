# Task 44: reference equivalence diagnostics

These are diagnostic scripts, not production arithmetic changes. They distinguish
mathematical reference outputs from emulations of native rounding. Full-model
production equivalence is **not established**.

## Evidence collected

- `attention_rounding_results.json`: ten synthetic native attention cases.
- `attention_real_rounding_results.json`: eighteen captured GLM attention cases.
- `attention_head_partition_results.json`: full-head versus partitioned-head replay.

On the captured real operands, adding native FP8 softmax-weight rounding and BF16
split-output storage reduces worst relative L2 from 0.29109% to 0.01191%.
Adding source QK accumulation grouping reduces it to 0.00877%; 99.9666% of
output values match exactly. This is not a bit-exact emulator and does not prove
that every remaining difference is rounding. Matching native split tactics makes
all eighteen head-partition replays bit-identical.

## Full-model harness

`fixed_prefix_equivalence.py` captures all 154,880 vocabulary log probabilities
for eight generated tokens, using identical raw input token IDs across arms.
Comparisons stop after prefixes diverge. Explicit extended prefixes provide
additional comparisons at fixed disputed positions. `--scope prod` includes
three short prefixes, a reconstructed 26,450-token task trajectory, and seven
concurrent requests. `--scope all` additionally includes a synthetic 131K prompt.
Concurrent HTTP requests alone do not prove every GPU step has seven sequences.

The harness currently uses local task-44 paths under
`/home/jarrelscy/glm52/dcp-groundtruth/task44` and imports `token_dist_ab` from
`/home/jarrelscy/homeassistant/benchmarks`. It requires NumPy and a running server
on localhost:8001 with model alias `glm-5.3` and `--max-logprobs -1`.
Supply `VLLM_API_KEY` privately through the environment. Never publish credentials,
raw trajectory prompts, raw model tensors, or full Docker environment dumps.

`trace_policy` is mounted as a directory on worker PYTHONPATH; `sitecustomize.py`
activates only with explicit task-44 environment variables. It records eager
layer snapshots while an `enabled` marker is present. The reference chunk wrapper
increases reference expert chunks to 1024 for feasible prefill runtime; it does
not change native kernels. Different matrix shapes can still change rounding.
The FP32 partial policy is an isolated precision-control experiment, not a
production default. Its dense reference allocation must remain FP32 until the
TP reduction; otherwise the intended matched-precision control is invalid.

## Current coverage and limitations

The production-topology comparison uses TP4, DCP4 and MTP3 with original precision,
but eager execution, LMCache disabled and 1024-token prefill chunks. Short probes
already show material distribution differences. Long and concurrent probes were
still running when this file was written. CUDA graphs, LMCache, and the native
4096-token grouped-prefill path require separate controls. Native routing,
indexer, norms and other common operators are not independently replaced by
this full-model reference.

Do not classify checkpoint quality as the sole cause on these results alone.
Do not change tolerances or the mathematical reference to hide observed errors.
