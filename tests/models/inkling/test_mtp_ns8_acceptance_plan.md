# Task #88 — Inkling MTP speculative decode: ceiling + acceptance-gate plan

Status: **planning only, no GPU touched**. This document is the deliverable
for task #88 ("Inkling: MTP speculative decoding — verify real ceiling +
gate acceptance"). It cannot be turned into a runnable pytest file yet
because the draft-model code it would exercise does not exist in this port.
Section 1 is the actual scope of task #88 as discovered; sections 2-4 are the
test design to execute once that scope is built (tracked separately, not
part of #88 itself unless the mission wants #88 to include the build).

## 0. TL;DR

**MTP is not wired in for Inkling in this vLLM fork.** The checkpoint
(`jarrelscy/Inkling-512k-NVFP4-AQLM-hybrid`, snapshot `db86e0aa...`) ships a
complete 8-layer draft block (`model.mtp.layers.0..7`, matching
`mtp_config.num_nextn_predict_layers=8`), but the port's weight loader
explicitly discards it:

```
vllm/model_executor/models/inkling/model.py:668
vllm/models/inkling/nvidia/model.py:698
    loader = AutoWeightsLoader(module, skip_prefixes=["model.mtp."])
```

and there is no `InklingMTP`-style draft model class or registry entry,
unlike every other MTP-capable model in this fork (`deepseek_v4/nvidia/mtp.py`,
`minimax_m3/nvidia/mtp.py`, `gemma4_mtp.py`, `glm4_moe_mtp.py`, etc. all have
one, and all are registered in `vllm/model_executor/models/registry.py`
lines ~601-648; grep for `"Inkling"` in that block returns nothing). This is
greenfield build work, not a flag flip. Sections below spell out exactly what
must exist before task #88's smoke test can run at all.

## 1. What must be true / built before this can run

### 1.1 Confirmed NOT yet implemented (evidence)

| Needed piece | Evidence it's missing |
|---|---|
| Draft model class (`InklingMTP` analog) | No `vllm/models/inkling/nvidia/mtp.py` or `vllm/model_executor/models/inkling_mtp.py`; compare to `vllm/models/deepseek_v4/nvidia/mtp.py` and `vllm/models/minimax_m3/nvidia/mtp.py` which exist as siblings of their base `model.py` |
| Registry entry | `registry.py` MTP dict (`"DeepSeekV4MTPModel"`, `"MiniMaxM3MTP"`, `"Gemma4MTPModel"`, ... lines 601-648) has no `"Inkling...MTP"` key |
| Weight loading path | `inkling/model.py:668` / `inkling/nvidia/model.py:698`: `AutoWeightsLoader(module, skip_prefixes=["model.mtp."])` — the base-model loader is *told to ignore* every `model.mtp.*` tensor. The weights exist on disk (verified: `model.safetensors.index.json` has exactly 20 tensors per layer for `model.mtp.layers.{0..7}`, none for `.8`) but nothing reads them today |
| `--speculative-config` method name | vLLM's speculative-config factory dispatches on `method` (`"deepseek_mtp"`, `"glm4_moe_mtp"`, `"dspark"`, ...); no method string exists yet for Inkling's self-speculative draft |
| configs.py MTP parsing | `grep -i "mtp\|nextn\|speculative\|draft" vllm/models/inkling/configs.py` → **zero hits**. `mtp_config` from `config.json` is not even parsed into `InklingModelConfig` yet |

### 1.2 What the checkpoint actually contains (grounds the build, verified via `model.safetensors.index.json` + README, no GPU needed)

- 8 distinct draft layers (`model.mtp.layers.0` .. `.7`), **not** one layer
  reused autoregressively (that's the GLM-5.2 `deepseek_mtp` design —
  `num_nextn_predict_layers=1`, ns swept 1-7 by rerunning that one layer).
  Inkling trained 8 *separate* stacked draft blocks — this is the closer-to-
  original DeepSeek-V3/GLM-4.6 chained-MTP design, one dedicated layer per
  lookahead position. **ns=8 is therefore very likely the checkpoint's fixed,
  maximum depth, not a free dial** — unlike GLM-5.2 where ns was swept
  independently of `num_nextn_predict_layers`. This needs confirming once the
  draft class is written (does the vLLM proposer loop allow `num_speculative_tokens
  < num_nextn_predict_layers` by just using the first N draft layers? plausible,
  but unverified) — flag as an open question for whoever picks up the build,
  not assumed either way here.
- Each of the 8 layers has 20 tensors: `embed_norm`, `hidden_norm`,
  `input_proj` (combines previous-step hidden state with the newly-revealed
  token's embedding — standard chained-MTP input), plus a **full transformer
  block**: `attn.{q_norm,k_norm,q/k/v_sconv,wq_du,wk_dv,wv_dv,wo_ud,wr_du,
  rel_logits_proj}` (Inkling's own low-rank-compressed + short-conv + relative-
  position attention, reusing the same primitives as the base model's
  `attention.py`/`sconv_swa_attn.py`) and `mlp.{global_scale,w13_dn,w2_md}`
  (**dense** MLP — no `mlp.gate`/`mlp.shared_experts`, confirmed by grep — so
  draft layers are NOT MoE, unlike GLM-5.2's single MTP layer which *is* a
  full 256-expert MoE). No separate `model.mtp.*` lm-head tensor exists →
  logits are almost certainly produced via the base model's shared
  unembed/`lm_head`, mirroring DeepSeek's `shared_head` pattern.
- `mtp_config.local_layer_ids = [0,2,4,5,6,7]` vs 8 total layers → **draft
  layers 1 and 3 are the only global/full-attention layers**; the other 6 are
  local/SWA-type (matches the base model's mostly-local-with-occasional-global
  layer pattern, `text_config.local_layer_ids` has 55/66 local). This is an
  odd, easy-to-get-wrong interleave (global at positions 1 and 3, not e.g. the
  first or last layer) — worth a specific correctness check once built (see
  §3.4).
- **CORRECTED (2026-07-18, direct safetensors dtype inspection of all 160
  `model.mtp.*` tensors, confirmed twice via a real weight-loading smoke
  test): MTP weights are plain BF16/unquantized, NOT NVFP4-dense** — zero
  hits for `model.mtp` in `hf_quant_config.json`'s `exclude_modules` turned
  out to mean "never quantized at all," not "quantized but excluded from
  that list." No `.weight_scale`/`.weight_scale_2`/`.codes` companion
  tensors exist under `model.mtp.*` anywhere. This does not change the
  conclusion below — plain BF16 dense-linear is even simpler than NVFP4
  dense, so the custom fused hybrid-MoE kernel (`hybrid_moe.py`, tasks
  #87/#96) is still *not* a blocker for MTP; the draft block only needs
  plain BF16 dense-linear support (`quant_config=None`), which trivially
  exists.

### 1.3 Build checklist (in the order a build session should tackle them)

1. Parse `mtp_config` in `vllm/models/inkling/configs.py` (currently absent).
2. Write `vllm/models/inkling/nvidia/mtp.py`: an `InklingMTP` module that
   stacks 8 `InklingDecoderLayer`-shaped blocks (reuse from `model.py`, mirror
   how `deepseek_v4/nvidia/mtp.py` imports `DeepseekV4DecoderLayer` from its
   own `model.py` rather than reimplementing), respecting the local/global
   split at draft-layer indices 1 and 3 vs 0,2,4,5,6,7.
3. Add a loader for `model.mtp.*` weights (currently thrown away by
   `skip_prefixes=["model.mtp."]` in both `model.py:668` and
   `nvidia/model.py:698` — these need a sibling path, not a deletion of the
   skip, since the base model class must still ignore MTP weights when MTP is
   off).
4. Register `"InklingMTPModel"` (or similar) in `registry.py`'s MTP dict.
5. Add an accepted `method` string to vLLM's speculative-config validation
   (self-speculative, checkpoint-bundled — same family as `"deepseek_mtp"`/
   `"glm4_moe_mtp"`, not an external-repo speculator like `"dspark"`).
6. Confirm (by reading, not running) whether the vLLM v1 MTP proposer loop
   can run with `num_speculative_tokens` < `num_nextn_predict_layers` (i.e.
   is ns a free 1..8 dial or hard-pinned to 8?) — affects whether task #88
   can do an ns-sweep at all or only has one config to test.

None of steps 1-6 are done. Task #88 as literally scoped ("verify real
ceiling + gate acceptance") presumes a working draft path exists; it does
not yet. Recommend either re-scoping #88 to include the build, or opening a
prerequisite task and leaving #88 blocked on it.

### 1.4 Compounding untested risk: memory headroom (from `inkling-512k-memory-math` memory note)

Independent of MTP, the 512K/TP4 base-model-only config already runs
weights at **~87.8% of 96 GiB/GPU** with only **~3-6 GiB/GPU slack**
(fp8 KV, `gpu_memory_utilization=0.97`), and that note explicitly lists
"MTP ns=8 draft overhead ... not yet exercised" as an open risk against that
slack, alongside CUDA-graph capture memory (also untested — all runs so far
used `enforce_eager=True`) and concurrency>1. An 8-layer draft block, even
dense/cheap per-layer, adds: 8x draft-layer weights (small, dense NVFP4,
probably low-single-digit GiB total), 8x KV-cache allocation for the 2
global + 6 local draft attention layers, and 8x activation/graph memory if
CUDA graphs are ever turned on for the draft chain. **Do not attempt ns=8 at
the full 512K target on first bring-up** — this compounds two
already-flagged, never-measured risks (graphs, MTP) on top of an already-tight
budget. See §2 for a staged context ladder.

## 2. Launch config

Follow the existing bring-up pattern in this repo (`serve_mtp.sh`: "small
max-model-len for fast boot/iteration; raise once coherent") rather than
starting at the 512K target.

```bash
# Illustrative — assumes the build in §1.3 is done and a method name
# (placeholder: "inkling_mtp") has been registered.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0   # match existing repo convention while bringing up a new spec path

vllm serve /data/huggingface/hub/models--jarrelscy--Inkling-512k-NVFP4-AQLM-hybrid/snapshots/db86e0aa27dc29c776894ab68c439edd622b2363 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --speculative-config '{"method": "inkling_mtp", "num_speculative_tokens": 8}' \
  --enforce-eager \
  --served-model-name inkling \
  --port 8001
```

Rationale for each deviation from the mission's target config (TP4 + ns=8 +
512K + CUDA graphs):

- **`--max-model-len 32768` (not 512K) for first coherence bring-up.**
  Ladder up only after each step gates: 32K → 128K → 256K → 512K. This
  isolates "does the draft chain even work" from "does it fit in the tight
  512K memory budget" (§1.4) and from "does it stay coherent at the deep
  context where the base model's own DSA/SWA-equivalent long-range behavior
  matters" — three different failure modes that should not be debugged
  simultaneously, matching how GLM-5.2's MTP bring-up in this same repo did
  it (`MTP_PROGRESS.md`: short-ctx coherence and ns-sweep first, 1M capability
  as a separate later task).
- **`--enforce-eager` (no CUDA graphs) for first bring-up.** Graph capture for
  Inkling is itself untested (§1.4 item 1) and this mission's own precedent
  (`mtp-perf-ceiling-4xrtx-pcie` memory note) shows MTP+graphs interactions
  are a real, nontrivial source of regressions/deadlocks elsewhere in this
  fork (`b12x-mtp-moe-backend-crash` memory note: a different model
  crash-loops under TP4+MTP+specific graph/backend combos). Don't stack a new
  MTP path on top of an unproven graph path for the FIRST measurement.
- **`--gpu-memory-utilization 0.90`** (below the 0.97 used for base-model-only
  512K tests) — deliberately conservative headroom given §1.4's tight budget
  and the new, unmeasured draft-block memory cost. Raise only after a fit is
  confirmed at each context rung.
- **`--kv-cache-dtype fp8`** — the memory note's own recommendation for
  headroom; carries over unchanged.
- **`--max-num-seqs 1`** — single-stream decode measurement, matching the
  GLM-5.2 acceptance-envelope test's own methodology (`test_mtp_acceptance_envelope.py`,
  `--max-num-seqs 1` equivalent single-stream probes) so acceptance isn't
  confounded by batching effects.
- **`num_speculative_tokens: 8`** — per §1.3 item 6, verify this is actually
  legal (not silently clipped or rejected) before assuming an ns-sweep is
  even possible; if the checkpoint's draft is architecturally fixed at depth
  8 (8 distinct trained layers, not 1 reused), a "ns sweep" the way GLM-5.2 did
  (ns=1..7 on ONE reused layer) may not have an Inkling analog at all — the
  only sweep might be "use the first N of the 8 trained layers", which is a
  different, untested idea (would the checkpoint's layer-3 draft accuracy
  hold if layer-3's `input_proj` never sees layers 4-7 revealed? no — chained
  MTP layers are trained assuming they run to completion in order 0..7, so
  truncating to N<8 is NOT free like GLM-5.2's ns dial was; this needs
  verifying against training methodology, not assumed).

## 3. Acceptance metric and healthy/unhealthy envelope

### 3.1 What to measure (mirrors `tests/sm120_correctness/tier3_server/test_mtp_acceptance_envelope.py`, the closest precedent in this repo)

Two levels, both needed (the GLM-5.2 test only tracks p0 explicitly because
its draft is 1 layer; Inkling's 8-distinct-layer draft needs the *full*
per-position curve as a first-class artifact, not an afterthought):

1. **Overall acceptance rate**: `vllm:spec_decode_num_accepted_tokens_total`
   delta / `vllm:spec_decode_num_draft_tokens_total` delta, over a mixed
   probe (warm + a low-entropy "count 1..N" workload + a high-entropy essay
   workload, matching the GLM-5.2 test's `_run_probe()` exactly — reuse it,
   only the model name / port need to change).
2. **Per-position acceptance p0..p7** (`per_pos_accepted` metric, if/when
   exported for an 8-deep draft — confirm the metrics plumbing actually
   supports 8 positions; GLM-5.2 only ever exercised up to ns=7 through a
   *single reused layer*, so whether the Prometheus exporter's per-position
   array is sized/indexed correctly for a true 8-distinct-layer chain is
   itself worth an assertion, not an assumption).

### 3.2 Is acceptance likely to degrade further at depth 8 than GLM-5.2's ns=7 precedent? Yes, but the mechanism differs — argue both directions explicitly

**Reasons to expect WORSE decay than GLM-5.2 ns=7** (structural, apply
regardless of the distinct-vs-reused-layer difference):
- Longer lookahead is inherently harder — more entropy accumulates the
  further ahead you predict, independent of architecture. GLM-5.2's own data
  shows this cleanly: mean accepted length saturates ~2.8-2.9 even at ns=7,
  and per-position acceptance for positions 5-7 fell below 11% on hard
  content (`glm52-mtp-decode-opt` memory note, "HIGHER-ns SWEEP" table).
- `mtp_config.chain_hidden_post_norm=false` plus the `hidden_norm`/`input_proj`
  tensors present on every layer confirm Inkling's draft chain **also**
  recycles hidden state layer-to-layer (each layer's `input_proj` consumes
  the previous layer's hidden output) — so uncertainty compounds
  autoregressively across the 8 draft layers exactly like GLM-5.2's single
  layer compounds across its ns reuses. This is the dominant shared
  mechanism and there's no architectural reason to expect it to be gentler
  at Inkling's depth 8 than at GLM-5.2's depth 7.

**Reasons Inkling's decay COULD be shallower / healthier than a naive
"worse than ns=7" read suggests**:
- Each Inkling draft position has its **own dedicated trained layer**
  (distinct `w13_dn`/`w2_md`/`attn.*` weights per layer index), specifically
  trained to predict *that* lookahead distance — unlike GLM-5.2 where a
  single layer, trained primarily for next-1-token prediction, is being
  *reused* autoregressively past the horizon it was directly optimized for.
  A dedicated deep-position layer may have learned a better representation
  for "predict 7 tokens ahead" than a 1-layer model repeatedly asked to do
  something it wasn't specifically trained for at that depth. This is the
  standard DeepSeek-V3 MTP argument for why chained multi-layer MTP scales
  to deeper lookahead better than naive autoregressive-self-speculation —
  it's *why* the checkpoint was trained this way instead of GLM-5.2's design.
- Draft layers are **dense**, not MoE (§1.2) — GLM-5.2's MTP-decode-opt
  investigation found the *verify* step's cost (not acceptance) dominated by
  MoE weight loads; that's a throughput finding, not an acceptance one, but
  it does mean Inkling's per-step draft compute is architecturally lighter
  and cleaner (no expert-routing noise/variance feeding into the draft's own
  forward), which could make each layer's predictions more stable / less
  noisy than a MoE draft would be.

**Net call: expect a genuinely degrading p0→p7 curve (do not expect flat
acceptance across depth), magnitude uncertain — do not hardcode a specific
falloff shape a priori.** Capture the full 8-point curve as the primary
artifact (§3.1) rather than committing to a predicted number; the two
competing effects above (accumulated-entropy decay vs. per-position dedicated
training) could plausibly land Inkling anywhere from "GLM-5.2 ns=7-like steep
falloff by position 5" to "gentler, more linear falloff since positions 5-7
have layers actually trained for that job." This test's job is to measure
which one actually happened, not assume it.

**Separately — expect the SAME kind of long-context throughput problem
GLM-5.2 hit, for an analogous reason**: GLM-5.2's `MTP_PROGRESS.md` "why it's
fundamental" analysis shows MTP loses to base at deep context because the
draft chain forces extra O(context) DSA-indexer scans that don't amortize.
Inkling's draft chain has **2 of its 8 layers doing full/global attention**
(layer-local indices 1 and 3, per `mtp_config.local_layer_ids`) — those 2
layers are each an O(context) attention scan over the full (up to 512K)
sequence, structurally the same shape of cost as GLM-5.2's indexer scans,
just via real attention instead of a sparse-indexer. **Predict**: at deep
context, Inkling MTP throughput likely also loses ground to base decode for
the same reason (extra unavoidable O(ctx) work per draft round that the
~2-3x amortization from accepted tokens can't offset) — this is a *ceiling*
question (task #88's other half, "verify real ceiling") separate from the
acceptance-envelope gate, and should be measured as its own tok/s-vs-context
curve, not conflated with the accept-rate pass/fail below.

### 3.3 Concrete pass/fail thresholds proposed, with reasoning

No Inkling MTP golden exists yet (unlike GLM-5.2, which had 3 prior rounds of
real-server measurement backing its `[0.675, 0.7971]` / `[0.8234, 0.9646]`
bands in `goldens/acceptance_envelope.json`). Treat everything below as a
**provisional floor for first bring-up**, to be replaced by a captured
golden (reuse `capture_golden()` from `test_mtp_acceptance_envelope.py`
verbatim — min/max of ≥3 real runs, widened ±0.04) the moment the draft path
is coherent enough to run 3 stable back-to-back probes.

| Gate | Proposed threshold | Reasoning |
|---|---|---|
| **p0 floor** (any workload) | `p0 >= 0.55` | Sanity/wiring-bug floor, not a quality bar. GLM-5.2's p0 never dropped below ~0.70-0.74 even on hard content, and the GLM-5.2 audit's own logic (`glm52-mtp-decode-opt` "ACCEPTANCE CORRECTNESS AUDIT") is that p0 collapsing toward 0 on non-trivial prompts is what a broken draft-input/hidden-recycling/off-by-one bug looks like — pos-0 reaching a high value is only possible if the embed/hidden-recycle/head path is basically correct. 0.55 sits below GLM-5.2's observed floor to give margin for Inkling's different (dense, distinct-layer) architecture and its own quant noise, while still catching a genuinely broken wiring path. |
| **Per-position monotonic-ish** | `p[k+1] <= p[k] + 0.03` for k=0..6 (small noise tolerance) | Catches an indexing bug in the local/global layer interleave specifically. Inkling's draft has global attention at *positions 1 and 3* (not first/last) among 6 local layers (§1.2) — an off-by-one in wiring that interleave would plausibly show up as a NON-monotonic spike or crater at exactly position 1 or 3 relative to neighbors, a distinctive, checkable signature GLM-5.2 didn't need (its single MTP layer has no such interleave). |
| **Deep positions (p5, p6, p7)** | **No hard floor** — log only, assert `0 <= p_k <= 1` | GLM-5.2 saw pos5-7 legitimately fall below 11% on hard content in a *working, non-buggy* configuration (`glm52-mtp-decode-opt` ns=7 sweep table). A hard floor here would false-fail on genuinely healthy-but-hard-content runs. Depth-8 is expected to behave at least as steeply, per §3.2, so gating on it would be gating on an expected property, not a defect. |
| **Overall acceptance rate — mechanical break-even floor** | `rate >= 0.25` on the mixed probe | Below this, an 8-position-deep draft chain almost certainly cannot amortize its own sequential cost even before considering the O(ctx) global-attention layers (§3.2) — i.e. below 0.25 the config is not just "gate-worthy" but "not economically worth serving regardless of gate," so treat it as a hard floor distinct from (below) any quality-envelope band. This is a rough floor, not derived from an Inkling-specific measurement (none exists yet) — GLM-5.2's own economics writeup shows spec decode only wins when accepted length clears roughly `ns+2` at deep context; scaled down for likely much-cheaper dense draft layers, 0.25 (→ mean accepted length ~2 of 8) is a conservative floor beneath which nothing plausible makes this worth shipping, not a precise cutover point. |
| **Metrics-plumbing sanity** | `per_pos_accepted` array has exactly 8 entries when `num_speculative_tokens=8`; test **fails loud** (not skips) if missing or wrong-length | Distinguishes "MTP isn't running" (legitimate skip, matches the existing GLM-5.2 test's `pytest.skip` on absent draft-token metrics) from "MTP is running but the exporter wasn't updated for depth>1/7" (a real bug worth failing on, since every prior model in this fork topped out at ns≤7 via a *reused* layer — an 8-distinct-layer chain is new ground for the metrics code too). |
| **Sample-size floor** | `dn` (drafts_total delta) `>= 25`, `dd` (draft tokens delta) `>= 150` | Scaled from the GLM-5.2 test's `dd > 200` sanity check, but re-derived: with ns=8 (vs GLM-5.2's ns≤7 but usually ns=2-3 in practice) each draft round can emit up to 8 tokens, so a token-count floor alone is a weaker signal of *per-position* sample size than a round-count floor; `dn >= 25` targets ≥25 samples backing each of the 8 per-position rates before treating them as anything but noise, which is a statistical-practice minimum, not a number borrowed from any memory file. |
| **Golden capture** | Run `capture_golden(runs=3)` (reused verbatim from the GLM-5.2 test) the first time all of the above pass, then switch subsequent runs to gate against the captured band, not the provisional floors above | Matches how GLM-5.2 arrived at its actual `[0.675, 0.7971]`/`[0.8234, 0.9646]` bands — by measurement, not by guessing a priori. The provisional floors in this table exist only to bootstrap that first successful measurement safely. |

### 3.4 Specific things this test must NOT do

- Must not assume ns is a free 1..8 sweep dial the way GLM-5.2's ns was
  (§1.3 item 6, §2 last bullet) — confirm truncated-depth semantics are even
  valid before designing an ns-sweep sub-test.
- Must not run at 512K / with CUDA graphs on the first pass (§1.4, §2).
- Must not fail solely on low pos-5/6/7 acceptance (§3.3) — that is an
  expected-shape property per §3.2, not automatically a defect.
- Must not skip capturing the full per-position curve in favor of only an
  overall scalar — for an 8-distinct-layer draft, the shape of the curve
  (graceful decline vs a cliff, and whether positions 1/3 look anomalous
  relative to their local-attention neighbors) is the primary diagnostic
  value of this test, more so than for GLM-5.2's single-layer case.

## 4. Summary of open questions to resolve during the build (not this planning pass)

1. Can `num_speculative_tokens` be set < 8, and if so does the proposer just
   run the first N of the 8 trained layers, or does that require its own
   validation/support given each layer was trained assuming the full chain
   runs? (§1.3.6, §2)
2. Does the existing per-position acceptance Prometheus exporter correctly
   support 8 positions from a genuinely-distinct-layer chain (vs. its
   demonstrated correctness for GLM-5.2's ≤7-deep *reused*-layer chain)?
   (§3.3 metrics-plumbing gate)
3. Real memory fit of the 8-layer dense draft block against the already-thin
   ~3-6 GiB/GPU headroom at 512K (§1.4) — needs an actual measurement once
   built, the arithmetic here is not a substitute for it.
4. Whether logits for draft tokens are produced via the base model's shared
   `lm_head`/unembed (my read of the weight-tensor list, §1.2) or something
   else — needed to write the draft model class correctly in step 2 of §1.3.
