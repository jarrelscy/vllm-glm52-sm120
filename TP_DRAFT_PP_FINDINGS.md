# Draft-TP over Target-PP — findings & spike

**Goal:** run the speculative-decode DRAFT model Tensor-Parallel across all N ranks
(draft TP=N) while the TARGET model runs Pipeline-Parallel across those same N
ranks (target PP=N, TP=1). Motivation: GLM-5.2 (MLA target) + a dense-MHA DSpark
draft (`RedHatAI/GLM-5.2-speculator.dspark`, 5 layers, 64 KV heads). Under pure
PP4 the target gets 1M context (KV split by layer across ranks), but the draft
lands **whole, unsharded, on the last PP rank** → its KV is ~40–80 GiB → OOM at
1M. If the draft runs TP4 (KV sharded 64→16 heads/rank) it shrinks to
~1–1.5 GiB/rank and `1M + DSpark` fits.

**Worktree:** `/home/jarrelscy/glm52/vllm-tppp` (branch `tp-draft-pp-target`),
isolated from the main tree. Pure-Python; test with
`PYTHONPATH=/home/jarrelscy/glm52/vllm-tppp:$PYTHONPATH`.

**Status:** config + process-group + load/KV-shard plumbing implemented and
import-tested (no GPU). The decode-time draft phase is specified but NOT wired
(guarded to raise cleanly instead of deadlocking). See "How far this got".

---

## The architecture that makes this hard

vLLM v1 **fused the draft into the target's model runner**: the draft
(`DSparkSpeculator`) is constructed and executed *inside* the target's
`GPUModelRunner` (v2: `vllm/v1/worker/gpu/model_runner.py`), sharing the
target's parallelism, and it lives **only on the last PP rank**:

- `model_runner.py` `__init__`: `if self.is_last_pp_rank: self.speculator = init_speculator(...)`.
- Draft weights load on last rank only (`load_model`), KV allocated on last rank
  only (`set_attn` + `initialize_kv_cache`), `propose()` called on last rank only
  (`sample_tokens`), CUDA graphs captured on last rank only (`capture`).
- Aux hidden states (DSpark reads target layers `[8,23,39,55,70]`, spread across
  PP stages) are propagated DOWN the pipeline to the last rank via the target's
  `IntermediateTensors` schema, and the last rank broadcasts the produced draft
  tokens back to non-last ranks with `PPHandler.broadcast_draft`
  (`vllm/v1/worker/gpu/pp_utils.py`) so they embed real drafts pp_size steps later.

So today the draft is *whole on one rank*. To shard it TP=N we must give it its
own N-rank parallelism, orthogonal to the target's PP.

### The two levers that drive sharding

1. **Global `_TP` group.** vLLM parallel layers (`ColumnParallelLinear`,
   `RowParallelLinear`, `QKVParallelLinear`, `VocabParallelEmbedding`) read the
   GLOBAL tensor-parallel group for both shard sizing (construction) and
   collectives (forward):
   `get_tensor_model_parallel_world_size()` → `get_tp_group().world_size`
   (`vllm/distributed/parallel_state.py:2012`). There is **no per-model TP
   group** — it's a process-global (`_TP`).

2. **The DSpark draft attention reads that same global for its head count.**
   `DFlashQwen3Attention.__init__` (`vllm/model_executor/models/qwen3_dflash.py:164`)
   does `tp_size = get_tensor_model_parallel_world_size()` and
   `self.num_kv_heads = max(1, total_num_kv_heads // tp_size)` (line 173). It does
   **not** read `model_config.get_num_kv_heads(parallel_config)`.

   **Consequence (key, positive):** if the draft is *constructed* while the
   global `_TP` is an N-rank group, then BOTH its weights AND its KV-head count
   (→ KV cache size, the memory win) shard N-way automatically. One lever
   (swap `_TP` during build) buys the whole load-time memory reduction for this
   draft. (Other drafts that call `get_num_kv_heads(parallel_config)` would ALSO
   need the draft `ParallelConfig` (tp=N) — see "gotchas" — but qwen3_dflash does
   not.)

Under a pure-PP4 target, the standard `_TP` groups are the singletons
`[0],[1],[2],[3]` and `_PP` is `[0,1,2,3]`. The draft-TP group we need has rank
set `[0,1,2,3]` — same members as the PP group, but a *tensor*-parallel
coordinator (all-reduce/all-gather/message-queue-broadcaster semantics).

---

## What was implemented (this worktree)

All flag-guarded by `speculative_config.draft_tp_over_pp`; the default path is
behavior-identical.

### 1. Config — relax + detect the mode  (`vllm/config/speculative.py`)
- `_verify_and_get_draft_tp`: allow `draft_tp == target.pipeline_parallel_size`
  when `target.tensor_parallel_size == 1 and pp > 1` (previously only `1` or
  `target_tp`). Logs "Enabling EXPERIMENTAL draft-TP-over-PP".
- New field `draft_tp_over_pp: bool`, set automatically in the model-validator
  when `target_tp==1 and target_pp>1 and draft_tp==target_pp`.
- `create_draft_parallel_config(..., draft_tp_over_pp)`: the draft's own
  `ParallelConfig` becomes **pp=1, tp=draft_tp** (the draft is not pipelined; it
  is TP across the PP rank set).
- **Import-tested:** `draft_tp=4` accepted for target pp4/tp1 → `draft_tp_over_pp`
  cfg tp=4/pp=1; correctly rejected for tp2/pp1.

### 2. Process group + context manager  (`vllm/distributed/parallel_state.py`)
- `_DRAFT_TP` global, `get_draft_tp_group()`, `draft_tp_group_initialized()`.
- `init_draft_tp_group(backend)`: collective; builds a TP-flavored
  `GroupCoordinator` (message-queue broadcaster on) whose rank groups are the
  contiguous PP blocks `range(base, base+pp)` for each replica. Idempotent.
- `patch_tp_group(group)`: context manager that temporarily installs `group` as
  the global `_TP` (this is the lever from §"two levers": build/run the draft
  inside it and its parallel layers shard across the draft group).
- Registered for teardown in `destroy_model_parallel`.
- **Import-tested:** all four symbols import; group builds the right rank blocks.

### 3. Worker hook  (`vllm/v1/worker/gpu_worker.py`)
- After `ensure_model_parallel_initialized`, if `draft_tp_over_pp`, call
  `init_draft_tp_group(backend)` (collective, guarded).

### 4. Runner — build draft on all ranks + shard load/KV  (`vllm/v1/worker/gpu/model_runner.py`)
- `self.draft_tp_over_pp` attribute; `_draft_tp_ctx()` returns
  `patch_tp_group(get_draft_tp_group())` in the mode, else `nullcontext()`.
- `__init__`: build the speculator on **all** ranks (`is_last_pp_rank OR
  draft_tp_over_pp`), inside `_draft_tp_ctx()`.
- `load_model`: `speculator.load_model(self.model)` wrapped in `_draft_tp_ctx()`
  → sharded weight load (each rank loads its 1/N shard).
- `initialize_kv_cache`: `speculator.set_attn(...)` wrapped in `_draft_tp_ctx()`
  → draft attention backend/metadata + KV allocation use the sharded head count.

### 5. Draft weight sharing guard  (`vllm/v1/worker/gpu/spec_decode/dspark/utils.py`)
- Under `draft_tp_over_pp`, do NOT share the target's (unsharded, tp=1)
  `embed_tokens` / `lm_head` into the draft's (tp=N sharded) layers — that would
  splice a full tensor into sharded modules (shape/rank mismatch). The RedHat
  DSpark checkpoint ships its own `embed_tokens.weight` / `lm_head.weight`, so the
  draft uses those (they load sharded under the patched group).

### 6. Runtime guards (avoid deadlock; clean stop)  (`model_runner.py`)
The decode-time draft is a TP collective across all N ranks, but non-last ranks
return early in `sample_tokens` and the sampler-derived inputs live only on the
last rank (see "Runtime restructure"). Rather than ship a guaranteed-deadlock
half-collective, the flag-on path is guarded:
- Warmup **dummy** propose: skipped under the flag (non-last ranks never reach it
  → would deadlock). Profiling still completes with the target.
- Draft **CUDA-graph capture**: skipped under the flag.
- First real **propose**: raises `NotImplementedError` pointing here.

**Net:** a flag-on boot builds the draft **sharded N-way** (weights + KV), runs
memory profiling and KV allocation (the memory win is observable in per-rank load
logs / `nvidia-smi`), starts the server, and then stops cleanly at the first
spec-decode request. That is the intended spike checkpoint.

---

## The remaining hard part — "Runtime restructure" (NOT wired)

To make decode actually generate, `sample_tokens`
(`vllm/v1/worker/gpu/model_runner.py`) must host a synchronized draft phase that
**all N ranks enter together** (a TP collective under `patch_tp_group`). Today:

- Non-last ranks: `if not self.is_last_pp_rank: ... pp_handler.receive(); return`
  (early return, ~line 1455) — they never call `propose`.
- Last rank: samples, then calls `speculator.propose(...)` (~line 1548) with
  inputs that only it has.

The draft's `propose` inputs and where they live:

| propose arg | source | lives on |
|---|---|---|
| `last_hidden_states` (target final hidden) | target PP forward | **last rank only** |
| `aux_hidden_states` (layers 8,23,39,55,70) | propagated via IntermediateTensors | **last rank only** |
| `num_sampled`, `num_rejected` | rejection sampler | **last rank only** (`self.sampler` is last-rank-only) |
| `temperature`, `seeds` | `self.sampler.sampling_states` | **last rank only** |
| `last_sampled`, `next_prefill_tokens` | `self.req_states` | all ranks |
| `input_batch`, `attn_metadata`, `slot_mappings` | scheduler_output | all ranks (same batch under PP) |

**Required restructure (design):**
1. After the last rank samples, **broadcast** over the draft-TP group (src =
   last rank): `last_hidden_states`, each `aux_hidden_states[i]`, `num_sampled`,
   `num_rejected`, `temperature`, `seeds`. Shapes are derivable on non-last ranks
   from `input_batch` (all PP ranks process the same batch): token count is
   known; hidden/aux dims are `target hidden_size` from config; sampler vectors
   are `[num_reqs]`. Use a dedicated sibling communicator (mirror
   `PPHandler.broadcast_group` via `get_draft_tp_group().make_sibling_device_group`)
   so it doesn't serialize with the PP p2p wire.
2. **All** ranks call `speculator.propose(...)` inside `patch_tp_group(draft_tp)`.
   The draft forward's RowParallel all-reduces / VocabParallel gathers now run
   over the N-rank draft group → correct sharded math; every rank ends with the
   **same** replicated `draft_tokens` (post all-reduce/gather).
3. Because all ranks now hold identical draft tokens, `PPHandler.broadcast_draft`
   becomes **redundant** in this mode (drop it under the flag). The FIFO
   scheduling that embeds real drafts `pp_size` steps later still applies; each
   rank writes its own `req_states.draft_tokens`.
4. Do the same symmetric wrap for the **warmup dummy propose** (so DP/EP sync and
   CUDA-graph capture happen on all ranks) and for **`capture`** — the draft
   graph must be captured symmetrically on all N ranks with the collectives
   inside the graph (NCCL-in-graph is fine if symmetric).

**Collective-ordering caution:** the draft broadcasts must be issued in a fixed
order matched on all ranks (like the existing `[sampled, combined, draft]`
ordering in `pp_utils.py`), or NCCL will mismatch/deadlock. Keep them on a side
stream with explicit events, mirroring `PPHandler`.

**Rejection/verification path:** verification of drafted tokens against the
target still happens on the last rank at the *next* step (rejection sampler is
last-rank-only). That is unchanged — only draft *generation* becomes collective.
The scheduler already schedules `num_speculative_tokens` extra query slots on all
ranks, so no scheduler change is expected.

---

## Gotchas / open questions for whoever finishes this (GPU needed)

1. **Block table / slot mappings for the draft KV on non-last ranks.** The draft
   `BlockTables` and per-request slot mappings are currently built on the last
   rank. Under TP the draft KV exists on all ranks with the *same* logical block
   layout (TP shards heads, not blocks), so the block table can be replicated;
   confirm `set_attn`/`_build_draft_attn_metadata` produce identical block tables
   on every rank (they read `self.block_tables`, which is per-runner). Likely OK
   because the batch/positions are identical across PP ranks.
2. **`get_kv_cache_spec` on all ranks.** With the draft built on all ranks, its
   attention layers register in `static_forward_context` on all ranks, so the KV
   cache spec includes the (sharded) draft group everywhere — good. Verify the
   engine's available-memory reconciliation across ranks doesn't assume the draft
   is last-rank-only.
3. **Non-qwen3_dflash drafts** that size heads via
   `model_config.get_num_kv_heads(parallel_config)` (`vllm/config/model.py:1270`)
   need the draft to be *built* with `parallel_config = draft_parallel_config`
   (tp=N). For this GLM-5.2 DSpark draft (qwen3_dflash backbone) it's unnecessary
   because it reads the global `_TP` directly — but if you swap draft models,
   also pass the draft parallel config into `load_dspark_model`'s
   `draft_vllm_config` (it currently keeps the target's parallel_config).
4. **Memory profiling accounting.** Draft KV is now allocated on every rank;
   ensure the KV-cache budget on non-last ranks accounts for it (the target's PP
   layer-KV per rank + draft's sharded KV). This is the thing to actually measure
   at 1M to confirm the OOM is gone.
5. **`_should_share` / tied lm_head.** With sharing disabled under the flag, the
   draft must ship its own lm_head/embed weights (RedHat DSpark does). If a
   checkpoint relies on tying to the target, you'd need a sharded copy instead.

---

## GPU validation (load-only boot, 2026-07-07)

Booted the spike on the 4x RTX PRO 6000 box: `models/500k` weights, PP4/TP1
target + DSpark draft with `draft_tensor_parallel_size: 4` (→ `draft_tp_over_pp`
auto-enabled). Ran the worktree via the main `.venv` compiled deps (see
"Running the worktree" below).

**Core result — draft KV shards 4× across all PP ranks (the memory win):**
A `[TPPP-DIAG]` log added to `load_dspark_model` printed, on ALL FOUR ranks
(PP0–PP3):

```
draft attn model.layers.0.self_attn: tp_world=4 tp_rank={0,1,2,3}
    num_kv_heads(per-rank)=16  num_heads(per-rank)=16  total_kv=64  kv_size=1024
```

i.e. the DSpark draft's 64 KV heads sharded to **16 per rank (÷4)** across all
four PP ranks — exactly the intended sharding. Baseline (draft on last rank only)
would be 64 KV heads on one rank. Draft **weights** also loaded on every rank
(PP0 reported ~80.6 GiB vs ~78.1–78.6 GiB on PP1–PP3; the ~2.3 GiB delta is the
sharded draft's weights + its own embed/lm_head, which are correctly not shared
with the target under the flag).

**KV-cache gate passed with the sharded draft allocated** (at `max_model_len`
=20000 to clear the target-weight memory wall — see below):
```
Available KV cache memory: 5.8 GiB
GPU KV cache size: 176,549 tokens
Maximum concurrency for 20,000 tokens per request: 8.83x
```
So the draft's sharded KV cache group was allocated on every rank alongside the
PP-split target KV — no "draft piled whole on the last rank" blowup.

**1M reality (complementary fix):** at full `max_model_len=1048576` the boot hit
the *generic* KV gate: the 500k target weights are ~80 GiB/rank, leaving only
0.74 GiB, while 1M target KV needs 34.36 GiB → fail. This is the target-weight
wall (independent of the draft; the same wall the concurrent SWA work hit on the
heavy 500k weights), not a draft-sharding failure. This spike removes the
*draft's* contribution to that wall (draft KV ~¼ and spread across all ranks
instead of whole on the last rank); serving 1M additionally needs the
target-weight/partition headroom (lighter `1m` weights) being pursued separately.
The two fixes are complementary.

**Decode:** as designed, the flag-on path guards the decode-time draft phase and
raises `NotImplementedError` at the first spec-decode step (after a clean boot to
"server ready"); it does not run the unimplemented symmetric collective.

### Running the worktree (non-obvious — the `.venv` is an editable *precompiled* install)

`vllm` is installed via an editable **precompiled finder** (`site-packages/
__editable___vllm..._finder.py`) that maps the whole `vllm` package to the MAIN
tree. Two gotchas:
1. `PYTHONPATH=<worktree>` alone does **not** override it for the modules whose
   resolution goes through `sys.path[0]=''` (cwd). If you `cd` into the main tree
   (as the stock serve scripts do), `import vllm` resolves to the MAIN tree and
   your edits are silently ignored.
2. Fix: **run from the worktree cwd** so `sys.path[0]=''` resolves `vllm` to the
   worktree, and **symlink the build artifacts** (compiled `.so`, plus generated
   `.py` and the `third_party/{deep_gemm,flashmla,...}` include trees incl.
   `.cuh` headers used by runtime JIT) from the main tree into the worktree so it
   is self-contained. The launcher `serve_tppp_loadonly.sh` (in the session
   scratchpad) does this; it activates the main `.venv`, `cd`s to the worktree,
   and sets `PYTHONPATH=<worktree>`.

## How far this got

- ✅ Config relaxed + mode auto-detected (import-tested).
- ✅ Draft-TP `GroupCoordinator` + `patch_tp_group` context manager (import-tested).
- ✅ Collective group-init worker hook (guarded).
- ✅ Draft built on all ranks; weight load + KV alloc wrapped in the draft-TP
  context → **weights and KV heads shard N-way at load time** (the memory win),
  because qwen3_dflash attention reads the global `_TP`.
- ✅ Draft weight-sharing disabled under the flag (avoids full↔sharded splice).
- ⚠️ Decode-time draft phase: specified in detail above, **guarded to raise**
  instead of running (would need the symmetric broadcast + collective propose,
  untestable without the 4 GPUs).
- ❌ No boot attempted — GPUs were leased (SWA fit-tests). A flag-on boot is
  expected to reach "server ready" and then `NotImplementedError` on the first
  spec-decode request; that boot would validate the sharded-KV memory win.

All edits are Python-only (no C-extension changes), so testing needs only
`PYTHONPATH` over the existing `.venv`, no recompile.
