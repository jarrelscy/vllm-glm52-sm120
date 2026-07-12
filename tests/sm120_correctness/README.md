# SM120 correctness / regression suite (GLM-5.2 serving path)

Exhaustive correctness gates for the GLM-5.2 hybrid NVFP4+AQLM serving
stack on 4× RTX PRO 6000 (SM120), built so that **every planned
optimization can be validated** and the SM100-class bug — *silent*
reasoning corruption: no crash, no error, deep chain-of-thought quietly
truncated to ≤~9.5k tokens with `finish=stop` and wrong answers — **can
never land silently on SM120 again**.

**Idea → test mapping: see [REGISTRY.md](REGISTRY.md).**
**Mandatory promotion gate: `tier4_canary/test_final_gate_64k.py` — every
prefill/decode change must pass it. A FAIL there or in the reasoning
canary is a STOP-THE-LINE event.**

## Opt-in gating (never runs by accident)

* The whole suite is skipped unless `GLM_SM120_TESTS=1`
  (`conftest.py`; self-contained — no top-level pytest/CI config is
  touched, plain `pytest` in the repo never executes it).
* Server-priced tiers (`tier3_server/`, `tier4_canary/`) are additionally
  skipped unless `GLM_SM120_SERVER_TESTS=1`, and golden capture
  (`make_goldens.sh`) requires the same variable — those consume hours of
  4-GPU server time and are scheduled by the main loop only.
* `run_all.sh` (the intended entry point) sets these variables itself and
  takes the box GPU mutex (`/home/jarrelscy/glm52/coord/gpulease.sh
  acquire sm120-tests <minutes>`) around any GPU work.

## Running

```bash
cd tests/sm120_correctness

./run_all.sh --tier cpu      # anywhere; no GPU, no lease (~2-10 min)
./run_all.sh --tier kernel   # 1 GPU, tiny JIT ctx (~1-2 GB), lease 15 min
./run_all.sh --tier unit     # 2-4 GPUs (torchrun parts), lease 30 min
./run_all.sh --tier server   # live :8001 server, lease 240 min
./run_all.sh --tier canary   # FINAL GATE + 1M + reasoning canary (~1-2 h)
./run_all.sh --tier all
# options: --quick   -k EXPR   --image glm52-sm120:latest   --no-lease
```

Per-tier one-line PASS/FAIL banners + a summary are printed; junit XMLs
land in `results/`. GPU/torch tiers run inside the `glm52-sm120:latest`
container with the repo bind-mounted (the host has no python packages);
server/canary tiers are **stdlib-only** and run directly on the host
(`python3 tier3_server/test_needle_depth.py` works standalone too).

Box rules honored by the runner: GPU work only under the lease; the
kernel tier picks the freest GPU and `common/kernels.py:pick_gpu()`
refuses to run with < `GLM_SM120_MIN_FREE_MIB` (default 8000) free, so a
resident server is never pressured. Never kill another agent's
containers.

## Goldens

```bash
./make_goldens.sh --cpu         # reduction-tree golden (committed)
# The following require GLM_SM120_SERVER_TESTS=1 + the KNOWN-GOOD shipped
# config (tp4-1m-mtp) live on :8001 — scheduled by the main loop:
GLM_SM120_SERVER_TESTS=1 ./make_goldens.sh --server      # logits + acceptance envelopes (~20 min)
GLM_SM120_SERVER_TESTS=1 ./make_goldens.sh --final-gate  # 64K final-gate envelope, N=5 (~1-2 h)
GLM_SM120_SERVER_TESTS=1 ./make_goldens.sh --canary      # canary reference depths (~30-45 min)
```

Committed defaults exist for the acceptance envelope (documented
0.76–0.82 band widened) and canary reference depths (documented
known-good depths), so those gates work before the first capture; the
teacher-forced and final-gate NLL gates *skip with instructions* until
their goldens are captured. **Never capture goldens from a candidate
change** — `make_goldens.sh` refuses to run without an explicit
double-gate and the canary capture refuses to record depths from a
config that answers incorrectly.

## The tiers and every test

### Tier 1 — kernel-level, bit-exact, 1 GPU, tiny memory (`tier1_kernel/`)

| test | guards | gate |
|---|---|---|
| `test_hybrid_gemv_bitexact.py` | the decode-path fused MoE gemv (ideas 1, 2, 5, 6) | `hybrid_moe_gemv` == CPU reference (`common/moe_reference.py`, the documented accumulation tree) with **maxdiff==0**, across {w13 1024×6144, w2 6144×512, small + partial-lane-tail variants} × {all-AQLM, all-NVFP4, mixed 30/70, masked slots, single slot} × {S=1/4/8/32} × {BOOKS 1,2} × {s2n 1,2} + seeded fuzz + adversarial values (denormals, ±65504 scales, zero scales, NaN block scales). The one nvcc-contraction ambiguity (NVFP4 fp32 two-block combine) is pinned by `calibrate_fma_mode` — zero or multiple matching candidates is a hard FAIL ("kernel numerics changed"). |
| `test_kernel_variant_equivalence.py` | any env/compile-gated kernel variant (ideas 1, 2, 5, 6) | auto-discovers `kernel_variants.json`; each variant bit-exact vs the shipped build on a mixed-format case set. Ships with 3 real variants (`NVFP4_LUT256`, `AQLM_CB_L1`, `AQLM_MLP=4`) so the machinery is never vacuous. |
| `test_registry_cpu.py` (CPU-only) | the registry contract | schema valid; every `#ifdef` / `VLLM_*` knob in the shipped source has a registry entry. |
| `test_ds_mla_kv_write.py` | fp8_ds_mla KV writes (the SM100 root-cause area) | 656-byte layout offsets (fp8 NoPE [0,512) / 4×fp32 scales [512,528) / bf16 RoPE [528,656)); scales are ARBITRARY fp32 `max_abs/448` — asserted **not pow2-truncated**; write bytes == CPU reference encoder bit-for-bit; dequant roundtrip within e4m3 bounds at 3 magnitudes; `slot=-1` writes nothing; NaN/Inf isolated to their tile. |
| `test_dequant_prefill.py` | prefill dequant rewrites (idea 7) | AQLM + NVFP4 dequant == CPU reference, maxdiff==0 (incl. real w2 shape, duplicate expert lists); cross-check dequant∘matmul ≈ gemv. |
| `test_reduction_order.py` (CPU-only) | "order-preserving" claims (idea 2) | the exact tree (per-lane fp16 fused-FMA groups of 8 → fp32 lane accumulation in K-order → shfl_down 16/8/4/2/1 → fp32 scale) is executable + pinned to a committed golden; non-vacuity tests prove the golden detects lane/K repartitioning and pins every FMA-mode branch. |

### Tier 2 — distributed/comms units, 2–4 GPUs, no server (`tier2_dist/`)

| test | guards | gate |
|---|---|---|
| `test_dcp_combine.py` | idea 8 (a2a vs ag_rs) | a2a pack kernel lossless (fp32 LSE bit-split); a2a unpack+combine and the ag_rs correction kernel each match the algebraic CPU reference within fp-reorder tolerance; degenerate rows (empty KV shard −inf, NaN/Inf LSE, all-empty row) stay finite; torchrun part: `dcp_a2a_lse_reduce == cp_lse_ag_out_rs` with real comms. |
| `test_allreduce_equivalence.py` | idea 4 (P2P allreduce) | vs exact fp64 sum: elementwise `|diff| ≤ (N−1)·eps·Σ|xᵢ|` on the real message sizes ([1/4/8/16, 6144] bf16/fp16); NCCL repeat-determinism documented (not gated); custom impls auto-discovered from the registry, asserted deterministic when declared. |
| `test_qgather_replication.py` | idea 3 (q_b replication) | replicated-weight GEMM == sharded GEMMs + gather within GEMM-reorder bound, per rank (single-GPU math + torchrun all_gather variant); bitwise-equality fraction reported. |

Multi-GPU parts run under torchrun via `tier2_dist/launch_dist_tests.sh`
(extra gate `GLM_SM120_DIST_TESTS=1`).

### Tier 3 — server-level invariants, live 4-GPU server (`tier3_server/`)

| test | guards | gate |
|---|---|---|
| `test_backend_routing.py` | THE SM100 root-cause class (routing) | SM120 MLA priorities are exactly `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`; backend class contract (cap 12 only, fp8_ds_mla supported); source tripwires: the impl's hard `!= fp8_ds_mla ⇒ raise` guard and the `major==12` branch must exist; live server log mentions the SM120 sparse backend + fp8_ds_mla and no other sparse backend. |
| `test_teacher_forced_logits.py` | any forward-numerics change | fixed ~200-token probe, `prompt_logprobs` vs an N=5 golden envelope (the measured NCCL run-to-run noise): ≥99.5 % of positions inside envelope+0.15, no position >1.5 off the mean track, mean NLL within ±max(0.02, 6σ). |
| `test_needle_depth.py` | coherence at depth | verbatim needle retrieval at 32K and 130K; 749K when `GLM_SM120_LONG=1` and max_model_len allows. |
| `test_mtp_acceptance_envelope.py` | verify-path corruption | overall MTP acceptance within the golden envelope (default [0.72, 0.86] from the documented 0.76–0.82 band); p0 gate when per-position metrics are exported; skips on non-MTP configs. |
| `test_graph_capture_parity.py` | idea 9 (cudagraphs) | two-phase: `--probe graphed` then reboot `CUDAGRAPH=0`, `--probe eager`, then the gate: both probes answer correctly, `finish=stop`, depth ratio ∈ [0.6, 1.67]. Byte equality deliberately NOT gated (NCCL nondeterminism). |

### Tier 4 — canary + capability gates (`tier4_canary/`)

| test | guards | gate |
|---|---|---|
| `test_final_gate_64k.py` — **THE FINAL GATE** | EVERY prefill/decode change | deterministic ~64K prompt (own-LCG document, facts at 10 %/50 %/90 % depth, multi-step synthesis question `FINAL: 95`): teacher-forced mean NLL in golden band AND every 256-token chunk mean inside its N=5 envelope (catches localized prefill corruption); generation answers correctly, depth ∈ [0.6·golden_min, 1.5·golden_max], `finish=stop`. All 5 golden runs stored (`goldens/final_gate_64k_runs.json.gz`). |
| `test_1m_capability.py` | the 1M window | `max_model_len ≥ 950000` and boot-log `GPU KV cache size ≥ 950000` tokens — a candidate must not silently forfeit 1M. |
| `test_reasoning_canary.py` — **the SM100-class detector** | silent reasoning collapse | 3 GPQA-diamond deep questions (gpqa-79/36/13; golds committed, question text never committed — CSV read from `GPQA_CSV` or git-ignored `data/`), temp 0, `reasoning_effort=max`, `max_tokens=40000`: every question ≥60 % of reference depth AND correct; none finishes <8000 tokens with `finish=stop`. **FAIL ⇒ STOP THE LINE** (banner printed). ~30–45 min. |

### `perf/` — non-gating harnesses (promoted from scratchpad)

`microbench.py` (kernel-only timing, shipped vs variant build, real verify
shapes) and `occ_scale.py` (slot-count saturation sweep). Use only AFTER
the Tier-1 gates pass.

## Design notes

* **Bit-exact where possible, envelopes where physics forbids.** The
  multi-GPU server is self-nondeterministic (NCCL ring reductions + MTP
  path dependence; documented acceptance spread 0.76–0.82 on the SAME
  build), so server gates are envelopes/needles/depth. Kernels are
  single-GPU deterministic, so Tier 1 gates are `maxdiff==0`, with the
  CPU reference emulating half-precision fused-FMA exactly (fp64
  evaluation is exact for half a·b+c; single rounding via numpy).
* **The shipped kernel is the reference for variants.** The CPU reference
  anchors the shipped kernel to the documented tree once; every future
  variant is then A/B'd against the shipped build on identical inputs.
* **Secrets/data hygiene:** no API keys in the repo (read from env /
  deployment `.env`); GPQA question text is never committed.
