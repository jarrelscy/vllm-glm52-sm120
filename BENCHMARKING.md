# GLM-5.2 hybrid — reproducible benchmarking

`bench.py` reproduces every cell of the config matrix below. It measures **decode
tok/s** the correct way (non-streamed, prefix-cache-warmed, true
`usage.completion_tokens`) and **prefill tok/s**.

> ⚠️ **Do not measure speculative decode by counting streamed SSE chunks.** Under
> MTP/DSpark, vLLM bundles multiple accepted tokens into one SSE delta, so chunk
> counting undercounts throughput by ~the acceptance factor (we saw 104 deltas =
> 300 real tokens). `bench.py` uses non-streamed responses to avoid this.

## ★ Preferred config

**`tp4-1m-mtp` — TP4 + DCP4(ag_rs) + native MTP (ns=3, IndexShare) + PIECEWISE
graphs + bit-exact gemv kernels + forced NCCL P2P.** Coherent **~1M context**
(KV 1,006,063 tok, needle @ 749K depth-0.5 = PASS) *with* lossless MTP spec.
This is the config `./switch.sh glm5.2-hybrid-1m-mtp` ships (2026-07-12 stack;
every knob is env-overridable and each change passed the 64K temp-0 golden gate
+ 1M capability + acceptance envelope, individually and combined).

**Measured (this stack, non-streamed):** short 73/64/52 tok/s (count/code/tcp),
@123K 44/41/32 — vs the pre-stack 55/51/41 and ~29–30 @123K. Fresh prefill
~1,200–1,270 tok/s (was ~750): ~950K prompt ≈ 12.5 min.

```
export NCCL_P2P_LEVEL=SYS VLLM_MTP_INDEX_SHARE=1 GLM_MOE_LANE_ROWS=1 GLM_NVFP4_LUT256=1
vllm serve /models/1m \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 4 --dcp-comm-backend ag_rs \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":3}' \
  --compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE"}' \
  --gpu-memory-utilization 0.97 --kv-cache-dtype fp8_ds_mla \
  --max-model-len 950000 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --no-enable-flashinfer-autotune --served-model-name glm-5.2 --port 8001
```

The 2026-07-12 stack, all lossless-gated:
- **`VLLM_MTP_INDEX_SHARE=1`** — MTP draft steps reuse the verify-anchored DSA
  indexer top-k instead of recomputing it per draft forward (which produced bad
  top-k: count acceptance 0.43→0.96). Draft-only ⇒ lossless by rejection
  sampling. +24–47% on structured content, grows with context.
- **`GLM_MOE_LANE_ROWS=1` + `GLM_NVFP4_LUT256=1`** — bit-exact gemv lane-rows
  repack (w2 −32%) + smem LUT beating the SM120a hw fp4 cvt (NV slice −26%).
- **`--dcp-comm-backend ag_rs` + `NCCL_P2P_LEVEL=SYS`** — NCCL default topology
  detection SHM-bounces this box (all-NODE); forcing P2P is 1.3–2.4× on
  ar/ag but NCCL a2a-over-P2P is pathological, so P2P **requires** ag_rs.
  +3–5% decode @123K, +35% fresh prefill. (This supersedes the old "a2a +5.2%"
  result, which was an artifact of the SHM transport.)
- **chunk 4096 + util 0.97** — +13% prefill, still boots the 950K window.

`cudagraph_mode` default remains `PIECEWISE`. The old FULL-graph deadlock was
specific to the a2a-era collective; `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` now
boots under ag_rs and adds +2–9% (opt-in pending a long soak). Set `NUM_SPEC=7`
to trade generality for structured-content speed (see ns-sweep below).

## 1. Launch a server config

All configs serve an OpenAI-compatible API on `:8001`. Pick a weight variant via
`MODEL_DIR` (`…/glm52-models/{1m,500k,250k}`) and a topology via `PARALLEL`
(docker entrypoint) or the flags in `serve_mtp.sh`.

| col | config | flags (in addition to `--kv-cache-dtype fp8_ds_mla`) |
|-----|--------|----------------------------------------------------------------------|
| A | pp4-base       | `--pipeline-parallel-size 4 --enforce-eager` |
| B | pp4-mtp        | `--pipeline-parallel-size 4 --enforce-eager --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'` |
| C | tp2pp2-base    | `--tensor-parallel-size 2 --pipeline-parallel-size 2 --enforce-eager` |
| D | tp2pp2-mtp     | `--tensor-parallel-size 2 --pipeline-parallel-size 2 --enforce-eager --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'` |
| E | tp4-dspark     | `--tensor-parallel-size 4 --speculative-config '{"model":"RedHatAI/GLM-5.2-speculator.dspark","method":"dspark","num_speculative_tokens":3}'` (graphs default: FULL_AND_PIECEWISE) |
| G | tp4-1m         | `--tensor-parallel-size 4 --decode-context-parallel-size 4 --dcp-comm-backend a2a` (graphs default: FULL_AND_PIECEWISE) |
| H | **tp4-1m-mtp** ★ | `--tensor-parallel-size 4 --decode-context-parallel-size 4 --dcp-comm-backend a2a --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":3}' --compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE"}'` |

Common env for all: `VLLM_DISABLE_FP8_W8A16=1` (v2-only, bit-exact; set `=0` for
the optional v4 fp8 path, +6–8% base), `--gpu-memory-utilization 0.95`,
`--max-num-batched-tokens 2048 --no-enable-flashinfer-autotune`. PP configs use
`VLLM_PP_LAYER_PARTITION=21,19,19,19` (or `20,20,20,18` to reach ~1M with MTP).

**CUDA graphs (cudagraphs-v2):** the V2 NVFP4+AQLM gemv/MoE kernels are registered
as `torch.library` custom ops (opaque to inductor), so graphs now work and are a
clear win for TP configs — default-on (`FULL_AND_PIECEWISE`) for E/G, `PIECEWISE`
for the DCP+spec config H. PP configs (A–D) still run `--enforce-eager` (graphs
are net-negative there). *(This supersedes the old "spec must be enforce-eager"
rule, which predated the V2 custom-op registration.)*

**DCP (decode context parallelism):** `--decode-context-parallel-size 4
--dcp-comm-backend a2a` shards the MLA latent KV *by sequence* across the 4 TP
ranks (each holds ~¼ of the tokens) with an exact LSE combine. This is what lets
plain TP4 reach ~1M (KV 994K) instead of the 371K it hits when MLA KV replicates
under TP. MTP's MLA-shaped draft KV shares the DCP-sharded latent correctly;
DSpark's dense-MHA draft does **not** (accept collapses to ~1.5% under DCP — do
not combine DSpark with DCP).

Or via the docker entrypoint modes:
`PARALLEL=pp4-1m|pp4-mtp|tp2pp2|tp2pp2-mtp|tp4-dspark|tp4-1m|tp4-1m-mtp`.
The entrypoint sets per-mode defaults (DCP, cudagraph_mode, the tp4-1m-mtp
promoted-stack envs above, and NSDEF: MTP=2 for
pp4/tp2pp2, **5** for tp4-1m-mtp, 5 for dspark); `NUM_SPEC` env overrides.

## 2. Run the benchmark

```bash
export VLLM_API_KEY=...            # only if you launched with --api-key
python bench.py                    # 3 workloads, short context
python bench.py --workload code
python bench.py --ctx 200000 --docfile book.txt   # long-context (use a public-domain book)
```

### Exact prompts (baked into `bench.py`)
- **count** (EASY, low-entropy → high acceptance): `Write out the integers from 1 to 500, one per line. Output only the numbers.`
- **code** (medium): `Implement in Python, as separate functions each with a full docstring and inline comments: bubble_sort, insertion_sort, selection_sort, merge_sort, and quick_sort. Output only the code.`
- **tcp** (COMPLEX, high-entropy): `Write a detailed 400-word explanation of how TCP congestion control works.`

All runs use `temperature=0`, `enable_thinking=false`. Acceptance length is read
from the server's `Speculative metrics` log lines (not in the OpenAI response).

## 3. Reference matrix (4× RTX PRO 6000 Blackwell, SM120, non-streamed)

Decode tok/s: base = one content-independent number; spec = `[easy/code/complex]`.
Ceiling = KV-pool tokens that fit.

| variant | A pp4-base | B pp4-mtp | D tp2pp2-mtp | E tp4-dspark | G tp4-1m | H **tp4-1m-mtp** ★ |
|---------|-----------|-----------|--------------|--------------|----------|--------------------|
| **1m**   | 1.27M / 18.7 | 999K / 29/24/24 | 580K / 40/36/30 | 247K / 66/53/28 | 994K / 24.9 | 1.006M / **73/64/52 (short), 44/41/32 @123K** |
| **500k** | 659K / 18.7  | 296K / 28/26/22 | 219K / 39/37/31 | 105K / 64/60/31 | — | — |
| **250k** | 327K / 18.7  | ~31K / 29/27/23 | ~35K / 40/37/31 | ~38K / 65/59/33 | — | — |

DCP configs (G/H) are 1M-only (the whole point is the ~1M ceiling on TP; use the
short-context configs on the smaller variants). H (promoted stack) fresh prefill:
~1,200–1,270 tok/s at 115–123K (chunk 4096 + ag_rs/P2P; was ~750 at chunk 2048
over SHM transport) — a ~950K prompt ≈ 12.5 min. Prefill is NCCL-bound (67% of
chunk time), flat vs context (sparse MLA). Prefix caching amortizes it across
requests.
NVFP4/AQLM expert split: 1m 29/71, 500k 48/52, 250k 57/43.

### MTP ns-sweep (config H, PIECEWISE graphs, counting workload)

Higher `num_speculative_tokens` wins on structured/low-entropy content once graphs
amortize the draft passes:

| num_speculative_tokens | 2 | 3 | 5 (default) | 7 |
|------------------------|------|------|-------------|------|
| count tok/s            | 54.8 | 68.8 | 66.7        | 77.7 |

ns=3 is the shipped default (verify batch ns+1=4 hits the exact graph-4 replay; ns>=4 pads verify to graph-8 = 2x work) (robust across workloads, within ~15% of the ns=7
counting peak). Use `NUM_SPEC=7` for structured/code-heavy traffic, `NUM_SPEC=2`
for general/high-entropy. MTP acceptance stays healthy (65–98%; up to 7.5/8 mean
on counting).

**Takeaways:** decode speed is set by the *config*, not the weight variant.
**`tp4-1m-mtp` (H) is the preferred config** — coherent ~1M + lossless MTP spec at
TP speed with graphs. `tp2pp2-mtp` (D) remains the best speed/context balance at
~580K; `tp4-dspark` (E) is fastest but short-context; `pp4-base`/`pp4-mtp` reach
the longest context (1.27M/999K) with no DCP. MTP beats base on every workload.
Full analysis in `MTP_PROGRESS.md` and `TP4MTP_PROGRESS.md`.
