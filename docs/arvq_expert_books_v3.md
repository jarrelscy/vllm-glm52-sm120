# ARVQ v3 per-expert books: serving implementation and SM120 checks

Status: implemented in an isolated worktree based on the active 8+8 serving
fork at `3b19c3660`, branch `feat/arvq-expert-books-v3`. No fitter code changed,
no production checkpoint converted, uploaded or deployed. The temporary GPU
benchmark pause is separate from deployment; the original AQLM container is
restored after testing.

## Contract and integration handoff

The exact v3 marker and layouts are documented in
`vllm/model_executor/layers/quantization/arvq/README.md`. Books are uint32
`[cold_experts,512]`; global scale stays float32 `[1]`; packed index and scale
layouts/sharding are unchanged. `w13` shares its pair between gate and up;
`w2` has a separate pair. Version 2 keeps its shared `[512]` ABI.

For nonascending expert manifests, export the exact ordered cold-slot list as
`aqlm_layer_books["L"]["cold_expert_ids"]` in both root/text quantization config.
Without that optional field, legacy ascending `hyb_kind==2` order applies.
This explicit mapping permits permuted global IDs without changing tensor
storage or adding index bits. The loader validates IDs against `hyb_kind`.
All TP ranks retain every book row. EP remains explicitly rejected.

Fitter coordination is not complete: `/home/coder/git/glm52` is absent on this
host and the fitter is not reachable through the session's agent list. The
user was asked for a host/shared handoff location. No response was available
at report time. This contract and patch are the handoff; agreement with the
fitter's `btx53/arvq88/pack.py` and an actual exported v3 checkpoint is pending.
The synthetic packer follows the current v2 MMA-fragment order and is checked
against the unchanged v2 decoder/kernel, not a new natural-order kernel.

## Path audit

- Native direct FP4 MMA and per-block shared LUT: new C symbol
  `hybrid_launch_8x8_expert`, with `cb + cold_slot * 512`. LUT remains 2 KiB
  per CTA, not E times larger. Arithmetic and four activation planes unchanged.
- No persistent layer-shared weight LUT exists in this kernel. Each cold CTA
  loads its own slot's row; the shared v2 specialization remains separate.
- Fused SiLU/activation-pack/down: dispatch uses book rank and selects the v3
  ABI. Duplicated-book parity tested with this path enabled and disabled.
- Grouped prefill and fused decode/gather: select a contiguous expert book row
  before the existing single-expert decoder. Mixed hot/cold grouped routes,
  compact on/off and direct gather are tested. Grouped FP16 GEMM retains its
  existing arithmetic; it is not claimed identical to native FP4-plane prefill.
- Paired/wide/register/shared-activation prefill kernels are hot-only at their
  serving call sites, enforced by the existing route partition. They do not
  consume expert books. The old paired source's dormant 8+7 cold branch is not
  a v3 entry point. No cold v3 route is sent to it by the serving dispatcher.
- CUDA graphs use the static tensor shape to select the ABI; no device sync
  or expert-ID host reads were added to the native execution path.
- Loader rejects broadcasting shared books into expert books, wrong dimensions,
  dtype/format mismatch, incompatible scope/version, and missing v3 ABI.

## Correctness results

Hardware: four RTX PRO 6000 Blackwell Max-Q GPUs, SM120a, CUDA 12.9 compiler.
104 focused pytest cases passed, including existing v1/v2 loader and prefill
regressions. Tests include deliberately different expert books on identical
indices/scales with known signed outputs, endpoints a/b=0 and 255, finite
E4M3 extremes 0/1/7/8/126, both projections, actual GLM TP1/TP4 shapes,
permuted cold/global IDs, reordered expert storage, unequal route weights,
hot/cold routes, token counts 1/4/33, graph on/off, and fused activation packing.

Duplicated `[E,512]` books matched shared `[512]` books **bit-for-bit** in tested
full-MoE paths and graph replays. A separate test compiled original HEAD
`hybrid.cu` to a separate library and matched both new v2 and new v3 to it.
The benchmark also checks bit equality for every TP1/TP4, route, token and
graph configuration, including real four-GPU NCCL reduction.

Independent natural-index CPU decoding is exact against GPU FP16 weight
decode. Below are maximum discrepancies against the FP4-plane reference,
including intended FP16 intermediate/BF16 output boundaries. TP4 entries
span all four shard positions; `summed` is the sequential reference check.

| TP | Stage | Max absolute | Max relative L2 |
| --- | --- | ---: | ---: |
| 1 | gateup | 1.78814e-07 | 1.04421e-07 |
| 1 | swiglu | 6.10352e-05 | 2.62949e-05 |
| 1 | down | 2.86079e-06 | 2.78876e-05 |
| 1 | moe | 6.10352e-05 | 0.000234083 |
| 1 | summed | 6.10352e-05 | 0.000234083 |
| 4 | gateup | 1.78814e-07 | 1.03123e-07 |
| 4 | swiglu | 6.10352e-05 | 5.38224e-05 |
| 4 | down | 2.86102e-06 | 5.58348e-05 |
| 4 | moe | 6.10352e-05 | 0.000244493 |
| 4 | summed | 6.10352e-05 | 0.000135402 |

An additional actual four-process TP4/NCCL check passed: after reduction,
max absolute error versus the plane reference was 6.10352e-5 and relative L2
was 2.12016e-4 on every rank. These are finite tests, not a universal numerical
identity claim against a differently ordered FP32 reference.

Compute Sanitizer memcheck: 0 errors on three signed/different-book cases.
Racecheck: 0 errors, 0 warnings/hazards on the same three cases. Ruff and diff
checks pass.
The benchmark harness initially retained a graph reference during NCCL teardown;
those completed timings were followed by a stuck teardown, then discarded for
TP4 reporting. The final TP4 command exits successfully after releasing graph
references; `bench-tp4-complete.json` is the reported run.

## Timing and memory

These are complete **single-layer routed MoE** CUDA timings, not model token/s.
Synthetic packed indices/weights and synthetic top-8 routing are used with
173 resident cold experts, 83 hot experts, and real v2 layer-10 shared codewords
from checkpoint revision `59d7c4a5dc30`, duplicated into every cold slot.
TP1 uses N/K 4096/6144 and 6144/2048; TP4 uses 1024/6144 and 6144/512.
TP4 timings include NCCL all-reduce. GPU serving was stopped for isolation.

Eight interleaved A/B samples, order alternated; median GPU latency, graphs on,
mixed routes below. Eager and all-cold results are also saved. Small differences
near 1% can include clock/thermal noise; no model-wide speed claim follows.

| TP | Input tokens | Shared v2 µs | Expert v3 µs | Latency change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 71.86 | 74.66 | +3.90% |
| 1 | 4 | 317.59 | 317.33 | -0.08% |
| 1 | 32 | 2562.20 | 2570.96 | +0.34% |
| 1 | 128 | 10257.73 | 10280.75 | +0.22% |
| 1 | 256 | 20425.82 | 20473.36 | +0.23% |
| 4 | 1 | 59.37 | 59.76 | +0.66% |
| 4 | 4 | 92.18 | 92.37 | +0.21% |
| 4 | 32 | 653.31 | 660.45 | +1.09% |
| 4 | 128 | 2509.25 | 2508.51 | -0.03% |
| 4 | 256 | 5083.36 | 5129.11 | +0.90% |

The 256-token all-cold TP4 graph case costs approximately 1.75% more latency.
Book storage for this measured layer is 4,096 bytes shared versus 708,608 bytes
expert, replicated per rank. For the existing 75-layer allocation with 13,450
cold experts: 52.54 MiB total books per GPU versus 0.293 MiB shared, an increase
of about 52.25 MiB. Indices/scales are unchanged. No expanded pairwise LUT is
allocated. Candidate learned books may have different runtime characteristics.

## Reproduction

Use an isolated instance of image
`glm53-arvq-sm120:fixes12-mtp-buffer-rng-20260917`, mounting this checkout at
`/work` and an artifact directory at `/results`. Do not run the overlay helper
inside a production container. Install pytest/ruff into the image venv with uv.
Build in `/work/vllm/model_executor/layers/quantization/arvq`:

```bash
bash build.sh
bash build_prefill.sh
bash build_activation.sh
```

The helper copies only tested runtime files and built libraries into the
isolated image checkout:

```bash
ARVQ_V3_PARITY_REPORT=/results/parity-final.jsonl \
  bash /work/examples/arvq/expert_books/test_in_image.sh \
  /work/tests/quantization/test_arvq_expert_books.py \
  /work/tests/quantization/test_arvq_hybrid.py \
  /work/tests/quantization/test_arvq_prefill.py -q
```

Set `ARVQ_V2_BASELINE_LIB=/results/baseline-hybrid.so` to include the independently
compiled old-kernel comparison; without it that one case explicitly skips.
Build the old library from `git show 3b19c3660:vllm/model_executor/layers/quantization/arvq/hybrid.cu`
using the same build.sh compiler flags.

```bash
/opt/vllm/.venv/bin/python /work/examples/arvq/expert_books/bench.py \
  --books /results/layer10-books.npz --out /results/bench-tp1.json
/opt/vllm/.venv/bin/torchrun --standalone --nproc-per-node=4 \
  /work/examples/arvq/expert_books/bench.py \
  --books /results/layer10-books.npz --out /results/bench-tp4-complete.json
/opt/vllm/.venv/bin/torchrun --standalone --nproc-per-node=4 \
  /work/examples/arvq/expert_books/distributed_parity.py
```

`layer10-books.npz` contains only uint32 arrays `w13` and `w2`, each `[512]`,
read from the existing checkpoint; it is local test data, not committed.
Sanitizer commands use `compute-sanitizer --tool memcheck` / `--tool racecheck`
with `--error-exitcode 86` and pytest selection
`-k 'known_projection or distinct_books_cpu'`.

All local logs, timing samples and reports:
`/home/jarrelscy/glm52/dcp-groundtruth/task44/arvq-expert-books-v3`.

## Not yet established

No new fitted checkpoint is available: exporter interoperability, real failing
model tasks, full-model generation throughput, quality improvements, long-context
behavior and learned-book end-to-end evaluation remain untested. Synthetic
layer success is not evidence of end-to-end quality. No fitting, alternating
index updates, ARVQ REAP scoring, or checkpoint export was duplicated here.
