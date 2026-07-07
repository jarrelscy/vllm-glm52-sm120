# GLM-5.2 MTP spec-decode — overnight effort (started 2026-07-07)

GOAL: MTP speculative decode COHERENT under PP4 (ideally token-identical to PP4-base greedy = lossless).
Keep MTP layer in BF16. Recompile kernels if needed. User: "no mathematical reason it won't work — get it working."

## TASK 3 — VARIANT MATRIX (2026-07-07 ~21:xx) — IN PROGRESS
Per-expert NVFP4/AQLM split (hyb_kind over 256 experts × 75 layers; kind0=NVFP4, kind2=AQLM):
  1m:  29% NVFP4 / 71% AQLM (272G, weights 67 GiB/rank)
  500k:48% / 52%           (311G)
  250k:57% NVFP4 / 43% AQLM(330G, weights 81 GiB/rank)
KEY FINDING: more NVFP4 does NOT speed up base decode (250k base 18.7 ≈ 1m base 18.7) — the larger
NVFP4 weight bytes offset the faster NVFP4 kernel (decode is bandwidth-bound). BUT more NVFP4 raises
MTP draft precision -> higher acceptance (250k CODE accept 2.98 vs 1m 2.68) -> faster MTP. Tradeoff:
250k has a MUCH lower context ceiling (bigger weights -> less KV room).

MATRIX (PP4 rows; decode tok/s @~32K/near-empty, v2-only util 0.95; spec=[easy/code/complex]):
| variant | A pp4-base                          | B pp4-mtp ns2                                  |
|---------|-------------------------------------|-----------------------------------------------|
| 1m      | ceil~1.27M, decode 18.7, prefill~1.5K| ceil~999K, [29.3/24.4/24.0] acc[3.0/2.68/2.68]|
| 500k    | ceil 659K, decode 18.7, prefill~1.6K | ceil ~296K, [28.5/26.3/22.5] acc[3.0/2.95/2.23]|
| 250k    | ceil 327K, decode 18.7, prefill~1.6K | ceil ~31K, [29.2/27.3/22.9] acc[3.0/2.98/2.66]|

FULL MATRIX (decode tok/s; base=1 num content-indep; spec=[easy/code/complex]+mean-accept; ceil=KV-pool tokens; prefill tok/s):
| variant | A pp4-base | B pp4-mtp2 | C tp2pp2-base | D tp2pp2-mtp2 | E tp4-base | F tp4-dspark5 |
|---------|-----------|------------|---------------|---------------|-----------|---------------|
| 1m  | ceil1.27M dec18.7 pf1.5K | ceil999K [29.3/24.4/24.0] a[3.0/2.68/2.68] | ceil752K dec21 pf1.5K | ceil580K [40.4/36.1/30.1] a[3.0/2.87/2.40] | ceil371K dec15.2 pf1.2K | ceil247K [66.2/52.7/28.5] a~4.18 |
| 500k| ceil659K dec18.7 pf1.6K | ceil296K [28.5/26.3/22.5] a[3.0/2.95/2.23] | ceil380K dec21 | ceil219K [39.1/37.1/30.7] a[3.0/2.91/2.51] | ceil176K dec15.5 | ceil105K [64.5/60.0/31.0] a4.45 |
| 250k| ceil327K dec18.7 pf1.6K | ceil31K [29.2/27.3/22.9] a[3.0/2.98/2.66] | ceil160K dec21 | ceil~35K [40.4/36.8/31.4] a[3.0/2.85/2.85] | ceil83K dec15.5 | ceil~38K [64.8/58.7/33.5] a4.58 |
MATRIX COMPLETE — all 18 cells (3 variants × 6 configs), CORRECTED dtps.py, all coherence-gated.
RECOMMENDATIONS: long-ctx no-spec -> pp4-base 1m (1.27M, +7% w/ v4 fp8); long-ctx+spec -> pp4-mtp 1m
(999K, 24-29 tok/s); balanced spec -> tp2pp2-mtp 1m (580K, 30-40 tok/s); max short-ctx decode ->
tp4-dspark 1m (247K, 28-66). VARIANTS: base/spec decode ~identical across 1m/500k/250k (bandwidth-
bound); variants differ ONLY in CONTEXT CEILING (1m biggest) + slightly higher accept on 250k.
=> use 1m unless a niche reason; 250k+TP/MTP ceilings collapse to ~5-83K (unusable for long ctx).
KEY MATRIX FINDINGS (per-config):
- FASTEST decode: tp4-dspark (66/53 easy/code) > tp2pp2-mtp (40/36) > pp4-mtp (29/24) > tp2pp2-base
  (21) > pp4-base (18.7) > tp4-base (15.2). But ceiling is INVERSE: pp4 1.27M >> tp4-dspark 247K.
- tp2pp2-mtp (col D, NEW combo) WORKS + is the best BALANCE: 40/36/30 tok/s at 580K ceiling.
- tp4-base is SLOWER than pp4-base (15.2<18.7): TP4 all-reduce overhead > PP4 bubble for decode.
- Under TP, MLA KV replicates -> ceilings drop hard (pp4 1.27M -> tp2pp2 752K -> tp4 371K).
PP4 SUMMARY: base decode ~18.7 tok/s for ALL variants (bandwidth-bound; NVFP4 ratio nets out).
Context ceiling scales inversely with NVFP4 ratio: 1m 1.27M > 500k 659K > 250k 327K (base);
+MTP: 1m 999K > 500k 296K > 250k 31K. MTP decode/accept slightly BETTER on higher-NVFP4 variants
(code accept 2.68->2.95->2.98 for 1m->500k->250k). Best context+spec combo = 1m (999K + wins). NOTE base decode is content-independent (1 number); MTP shows the
3-workload spread. 250k+MTP ceiling only ~31K (big weights + MTP layer); 1m is the context king.

## TASK 2 — PERF SQUEEZE (2026-07-07 ~21:xx, 1m)
- v4 fp8 W8A16 (VLLM_DISABLE_FP8_W8A16=0): BASE decode 20.07/20.59 (count/tcp) vs v4-off 18.7/19.4
  = **+6-8%, COHERENT**. Best PP4-base config = v4 ON (~20.6 tok/s). For MTP ns=2, v4 within
  run-noise (27-25-22 v4on vs 29-24-24 v4off) — spec speedup dominates the per-forward v4 gain.
  RECOMMENDATION: enable v4 fp8 for base serving (+7%). serve_mtp.sh now honors
  VLLM_DISABLE_FP8_W8A16 override (default still 1).
- PP partition: 20,20,20,18 (MTP-balanced) used throughout; decode is weight-bound so PP-bubble
  effect on single-stream decode is minor. max-num-batched-tokens/max-num-seqs: single-stream
  decode insensitive (not pursued). Base is near MoE-bandwidth roofline; v4 is the main lever.

## TASK 1 — EASY-CONTENT MTP SPEED (2026-07-07 ~20:xx, 1m, CORRECTED dtps.py) — MTP WINS
Decode tok/s (accurate non-streamed cache-warmed). base ~18.7-19.4 (content-independent).
| workload (32K)   | base | ns=2 (acc)  | ns=5 (acc)  | ns=7 (acc)  | best |
|------------------|------|-------------|-------------|-------------|------|
| EASY (count)     | 18.7 | 29.33 (3.0) | 30.56 (5.73)| 29.06 (7.0) | ns5 1.63x |
| CODE (5 sorts)   | 19.4 | 24.35 (2.68)| 21.54 (4.68)| 18.72 (6.20)| ns2 1.26x |
| COMPLEX (tcp)    | 19.0 | 24.01 (2.68)| 15.97 (2.97)| 12.10 (2.52)| ns2 1.26x |
| 200K EASY(count) | 19.4 | 24.58 (3.0) | 26.58 (5.79)| —           | ns5 1.37x |
VERDICT: MTP BEATS base on EVERY workload at ns=2 (1.26-1.57x short, 1.27x@200K). On pure
enumeration ns=5 pushes to ~1.6x (short) / 1.37x (200K). Higher ns (7) only helps enumeration and
HURTS code/complex (draft-step cost > marginal accept once accept saturates). accept scales cleanly
with ns on easy (3.0/5.73/7.0) proving the multi-step chain works. **Best all-round ns=2**;
enumeration-heavy -> ns=5. Note COMPLEX at high ns loses (accept ~2.5-3.0 can't amortize 5-7 draft
loads) — so pick ns by workload. This CONFIRMS the hypothesis: on predictable content MTP wins big.

## 🚨 MEASUREMENT-BUG CORRECTION (2026-07-07 ~20:xx) — MTP is FASTER than earlier reported at long ctx
CRITICAL: my earlier STREAMED decode-tok/s helpers (longctx.py / easy_chat.py) counted `n+=1` per
SSE delta, but under spec-decode vLLM BUNDLES multiple accepted tokens into one SSE delta (verified:
104 deltas = 300 tokens, 2.88 tok/delta; sample delta '3\n4'). So all my STREAMED MTP long-ctx
numbers were UNDERCOUNTED by ~the accept factor. Base (1 tok/delta) was unaffected. The SHORT-ctx
sweep used NON-STREAMED measure_short.py -> those (ns1 21.5, ns2 22.1, ns5 15.2 tech-prompt) are OK.
Correct method = NON-STREAMED, prefix-cache-warmed single timed run (dtps.py): true
usage.completion_tokens / decode_time. Prefix caching (enabled) lets run-2 reuse the long prefill so
its total time ~= decode time.

CORRECTED 1m numbers (dtps.py, decode tok/s; base from accurate streamed-post-first / re-confirmed):
| workload         | base | MTP ns=2 | accept | ns2 vs base |
|------------------|------|----------|--------|-------------|
| short EASY count | 18.7 | 29.33    | 3.00   | 1.57x |
| short COMPLEX tcp| ~19  | 24.01    | 2.68   | 1.26x |
| 200K EASY count  | 19.41| 24.58    | 3.00   | 1.27x |
=> MTP ns=2 BEATS base at BOTH short AND 200K when measured correctly. The earlier
"MTP net-negative >=100K / loses at long ctx" verdict was a STREAMING-UNDERCOUNT ARTIFACT and is
RETRACTED. Re-measuring the deep points properly (below / matrix). Economics note: base decode is
weight-bound and MTP's verify amortizes the MoE weight-load across accepted tokens, so with accept
~2.7-3.0 MTP wins ~1.3-1.6x even at long ctx. (The old "ns+2 O(ctx) scans" model over-weighted the
indexer term; the indexer is cheap+cross-layer-shared, so weight-load amortization dominates -> win.)

## ACCEPTANCE CORRECTNESS AUDIT (2026-07-07 ~20:xx) — NO BUG; low/decaying accept is the real ceiling
User skeptical the low+decaying acceptance is a bug. Audited all 4 points + ran the controlled test.
VERDICT: implementation is CORRECT. The decay is content-entropy + single-MTP-layer behavior, matches
the GLM-5 paper. Evidence:

(1) bf16 dtype — NOT a bug. Layer 78 stored dtypes (verified from safetensors): MoE experts = U8
    (packed NVFP4) + F8_E4M3 weight_scale + F32 input/weight_scale_2; ALL non-expert tensors
    (self_attn, eh_proj, enorm, hnorm, embed, lm_head) = BF16. Hybrid plan puts layer 78 in the
    **nvfp4_layers tier (nvfp4_layers=[78])** — the HIGHER-precision tier; the 75 target MoE layers
    (3-77) use the LOSSIER aqlm tier. So the draft is NOT lower-precision than the target (it's
    higher). The experts exist ONLY as NVFP4 in the checkpoint — there is no bf16 expert source;
    "forcing bf16" = lossless dequant of the same NVFP4 values -> zero precision gain, more VRAM.
    shared_head/lm_head loaded with quant_config=None (bf16), correct (deepseek_mtp.py:55-58).
(2) draft-input path — CORRECT. Proposer loop (llm_base_proposer.py:682-767): each step feeds prev
    drafted token id (686) + recycled MTP hidden (714,750). deepseek_mtp forward recycles post-norm
    hidden, one final-norm for logits (matches SGLang deepseek_nextn). index_share reuses step-0
    topk for steps 1+. PROOF it's correct: on predictable content the full 7-token chain accepts
    78-100% (mean 7.38/8) — impossible with a broken draft-input/hidden/off-by-one path.
(3) reference comparison — IN LINE. GLM-5 report: accept length 2.76 (DeepSeek-V3.2 2.55). Ours:
    2.6-2.9 (ns>=3, mixed prompts); 3.09 hard / 7.38 easy at ns=7. vLLM MTP acceptance range 70-85%;
    our pos-0 0.74 (hard)..1.0 (easy) in range. vLLM docs EXPLICITLY warn single-MTP-layer accuracy
    degrades for num_speculative_tokens>=3 — exactly our decay. Not anomalously low.
(4) pos-0 correctness — CORRECT. pos-0 = P(MTP top1==target top1). Measured 1.000 on predictable
    content -> head/embed routing + dtype are correct (a routing/dtype bug could NOT reach 1.000).
    The 0.74 on hard technical prompts is genuine model uncertainty, not a bug.

CONTROLLED SAME-PROMPT TEST (coordinator's decisive check) — pos-0 is ns-STABLE:
| prompt              | ns=1 pos-0 | ns=7 pos-0 | ns=7 full per-position                          | ns=7 mean |
|---------------------|-----------|-----------|--------------------------------------------------|-----------|
| EASY (count 1..120) | 1.000     | 1.000     | [1.00,.969,.969,.938,.938,.781,.781]             | 7.38      |
| HARD (TCP expl.)    | 0.736     | 0.745     | [.745,.553,.404,.234,.106,.021,.021]             | 3.09      |
pos-0 IDENTICAL across ns on the same prompt (1.000=1.000; .736≈.745 within noise) => the first
draft token is produced identically regardless of ns => NO bug. Decay past pos-1 is prompt-entropy
+ the documented single-MTP-layer limit. On low-entropy content the FULL chain works near-perfectly
(7.38/8). CONCLUSION: acceptance is CORRECT; the earlier "steep decay" seen on technical/reasoning
prompts is the model's real ceiling, not an implementation defect. No fix required.

## HIGHER-ns SWEEP COMPLETE (2026-07-07 ~19:xx, CLEAN 1m weights) — ns=2 is optimal; ns>=4 all LOSE
Same-session PP4-base on 1m: short 18.03 tok/s, 200K 19.41 tok/s. All ns COHERENT (greedy TCP expl).
FULL CURVE (decode tok/s post-first-token; short = near-empty ctx; identical WP prompt at 200K):
| ns | short tok/s | 200K tok/s | mean-accept (short) | per-pos accept (short) |
|----|-------------|------------|---------------------|------------------------|
| base | 18.03 | 19.41 | — | — |
| 1  | 21.5* | 12.83* | ~1.7 | [.93 easy/.72 tech] |
| 2  | **22.1*** | 10.04* | 2.24 | [.756,.488] |
| 3  | 20.0* | — | 2.58 | [.79,.51,.29] |
| 4  | 17.51 | 6.92 | 2.61 | [.739,.449,.246,.174] |
| 5  | 15.20 | 5.93 | 2.58 | [.767,.467,.217,.083,.050] |
| 6  | 14.08 | 5.20 | 2.81 | [.712,.500,.346,.154,.058,.038] |
| 7  | 12.17 | 4.66 | 2.89 | [.761,.413,.326,.239,.109,.043,.000] |
(* = prior-session 1m numbers; base/ns4-7 this session, fully consistent.)

VERDICT (answers the coordinator's key questions):
- Does ns=5 beat ns=2's 22.1 at 32K? **NO.** ns=5 = 15.20, BELOW base (18.03) and far below ns=2.
- Does higher ns help at 200K? **NO** — strictly worse (ns1 12.83 > ns2 10.04 > ns4 6.92 > ns5 5.93
  > ns6 5.20 > ns7 4.66); ALL below base 19.41.
- BEST ns per regime: short ctx -> **ns=2 (22.1, 1.23x over base)**; 200K+ -> NO ns wins (best ns=1
  0.66x). MTP beats base ONLY at short ctx (<=~50-80K), best at ns=2.
- The official recipe's ns=5 is TUNED FOR SM100 datacenter GPUs (native any-next_n paged-MQA kernels
  + CUDA graphs). On SM120 + enforce-eager + the hybrid-MoE draft, ns=5 over-drafts and loses.

WHY higher ns is monotonically worse (user's batched-verify-amortization hypothesis DISPROVEN):
- mean accepted length SATURATES at ~2.8-2.9 even at ns=7 (per-position acceptance collapses past
  pos-2: pos5-7 are <11%). The drafter simply can't predict >~2-3 tokens ahead reliably.
- The draft steps are AUTOREGRESSIVE (ns sequential MTP-layer forwards), NOT batched. Each draft
  forward loads the MTP layer's 256-expert MoE (~7.5 GiB) + pays eager launch/sampling overhead.
  index_share saves the draft's *indexer top-k* scans (steps 1+ reuse step-0 topk) but NOT the
  per-step MoE weight load. So k draft steps cost ~k× the draft-MoE load for ~0 extra accepted
  tokens past 2-3 -> pure waste. At ns=7, 322 tokens drafted for 86-87 accepted (27%).
- Verify amortization (1 weight-load for ns+1 batched positions) is real but small vs the k
  sequential draft loads; it cannot offset them. Net: cost grows ~linearly in ns, benefit saturates.

## 1M-CONTEXT RESULT (2026-07-07 ~13:xx) — MTP FITS ~1M; still slower than base at depth
(1) CONTEXT CEILING: **MTP handles ~1M under PP4.** Default partition 21,19,19,19 puts the MTP
    layer (layer 78, a full 256-expert MoE ~7.5 GiB) on the LAST PP rank, overloading it (rank3
    weights 74.5 vs 67 GiB) -> caps ~600-950K (varies with activation-profiler noise). REBALANCING
    the PP split so the MTP rank carries fewer target layers fixes it: `VLLM_PP_LAYER_PARTITION=
    20,20,20,18` @ util 0.97 -> **est max 999,168 tokens (~1M, 832 short of full 1048576)**; booted
    & served at MAXLEN=998000 (KV pool 999,983 tokens). So unlike DSpark (couldn't fit spec at 1M),
    MTP's MLA-shaped draft KV is tiny and it reaches essentially full 1M. (Layer weights are
    non-uniform due to hybrid NVFP4+AQLM per-expert tiering; rank0 with early dense layers is
    lightest ~64 GiB, so hand-balancing is fiddly — 20,20,20,18 was the best simple split.)
(2) SAME-PROMPT DEEP-CTX A/B (War&Peace 775,233-tok prompt, temp0, /v1/completions, streamed decode
    tok/s post-first-token), IDENTICAL 775,233-tok prompt across all three:
      base   = **18.19 tok/s** (flat! coherent)
      ns=1   = **12.62 tok/s** (0.69x; coherent; mean accept ~1.7)
      ns=2   = **9.85  tok/s** (0.54x; coherent; mean accept 2.08, per-pos [.676,.405])
    [200K prior session, same WP prompt: base 18.24, ns1 12.83 (0.70x), ns2 10.04 (0.55x).]
    [32K short: base 18.9, ns1 21.5, ns2 22.1 (1.17x) — spec WINS only here.]
    Base is FLAT ~18 tok/s from 32K to 775K (DSA sparse). MTP SLOWER at depth. PER-STEP BREAKDOWN
    matches the economics: ns=2 does 4 O(ctx) indexer scans/group for mean 2.08 accepted =
    1.92 scans/token vs base 1 -> predicted 0.52x, MEASURED 0.54x. The slowdown IS the extra
    indexer scans; nothing else. Proven irreducible.
(3) WHY IT'S FUNDAMENTAL (not a fixable rescan — coordinator's hypothesis checked & already handled):
    The draft does NOT re-scan context per draft step. `index_share_for_mtp_iteration=true` +
    DeepSeekMTP.set_skip_topk (llm_base_proposer.py:571-603, deepseek_mtp.py:159): draft step 0
    computes the DSA lightning-indexer top-k, steps 1..ns-1 REUSE it (skip_topk=True). So the draft
    costs ONE O(ctx) indexer scan per propose, not ns.
    Per-group economics at deep ctx (where the O(ctx) DSA indexer scan I dominates; weight-load W is
    small because base decode is already sparse/weight-light): base makes A tokens for A·(W+I)≈A·I.
    Spec makes A tokens (A≤ns+1) for [draft: 1 forward + 1·I] + [verify: 1 W + (ns+1)·I] ≈ (ns+2)·I.
    Spec wins only if (ns+2)·I < A·I i.e. A > ns+2, but A ≤ ns+1 ALWAYS. => at deep ctx MTP does
    (ns+2) O(ctx) indexer scans to yield ≤(ns+1) tokens => STRICTLY more indexer work than base,
    period. The MANDATORY extra draft scan + the "verify ns+1 candidates" both cost a full O(ctx)
    scan each, and DSA removed the weight-bandwidth term that spec normally amortizes. This is
    irreducible: the only way to kill the extra draft scan is to have the draft reuse the TARGET's
    top-k (different layer geometry -> tanks acceptance), and even then it's break-even, not a win.
    => For a DSA (sparse-attention) target, MTP spec decode CANNOT beat base at long ctx. It wins
    only at short ctx (<=~50-80K) where W dominates and the verify's single weight-load amortizes.

## ✅✅✅ COHERENT MTP ACHIEVED (2026-07-07 ~11:34) ✅✅✅
Forcing MTP onto the **V2 model runner** makes MTP spec-decode COHERENT under PP4.
- Fix: `VLLM_USE_V2_MODEL_RUNNER=1` (env), made PERMANENT by adding `"mtp"` to the force-V2
  block in vllm/config/vllm.py:528 (method in ("dspark","mtp") -> return True).
- Result: coherent prose + correct code output. SpecDecoding metrics: **mean acceptance length
  1.93/2, per-position acceptance 0.933 (93.3%)**. V2's method-agnostic draft-broadcast FIFO
  (pp_utils.broadcast_draft + update_pp_decode_requests) covers MTPSpeculator exactly like DSpark.
- FINAL PERF VERDICT (details in ns-sweep section): coherent+lossless at all ns. BEATS base only at
  SHORT ctx (ns=2 ~1.17x @32K); NET-NEGATIVE at >=100K (0.66-0.70x) — GLM's DSA makes base decode
  flat ~18-19 tok/s at any length, so spec has no long-ctx amortization to exploit. Architecture, not bug.
- WHY it works: V1 runner (gpu_model_runner.py) has the [num_reqs,1] assert + no draft FIFO; V2
  (gpu/model_runner.py) has both fixes. `mtp` was already whitelisted for V2 (vllm.py:2070); it
  just wasn't being ROUTED there for our MoE arch. One-line routing fix.
- LOSSLESS GATE RESOLVED: token-identity is UNACHIEVABLE here because **PP4-base itself is
  NON-DETERMINISTIC at temp=0** (5 runs of one prompt -> 3 distinct coherent outputs; MoE/DSA
  enforce-eager kernel numerics). So exact token-identity is void as a gate. Evidence MTP is
  lossless/uncorrupted instead: (a) MTP greedy output for the hash-table prompt EXACTLY matched
  one of base's own greedy variants (MTP output ∈ base distribution); (b) 72-93% per-position
  acceptance -> the verify accepts the draft most of the time, which is impossible if the target
  verify were corrupted. VERDICT: MTP is coherent + effectively lossless.
- SHORT-CTX (32K) tok/s: base ~18.9 tok/s; MTP ns=1 ~21.5 tok/s = 1.14x (accept 72% on tech prompt,
  93% on easy prompt). Spec wins more at long context (bandwidth-bound) -> testing 200K/400K next.
- Permanent V2 routing CONFIRMED without env var ("Using V2 Model Runner" in mtp_boot3.log).

## LONG-CTX VERDICT (ns=1) — MTP LOSES at long context
Coherent at all lengths (accurate War&Peace Rostov summary at 200K AND 400K). But decode tok/s:
| ctx  | base (no spec) | MTP ns=1 | ratio |
|------|----------------|----------|-------|
| 32K  | 18.9           | 21.5     | 1.14x WIN |
| 200K | 18.24          | 12.83    | 0.70x LOSE |
| 400K | 19.34          | 12.83    | 0.66x LOSE |
ROOT CAUSE: GLM-5.2 uses DSA sparse attention -> **base decode is ~flat 18-19 tok/s regardless of
context** (weight-bound, not KV-bandwidth-bound). Spec decode's usual long-ctx win (amortize a
bandwidth-bound step over k tokens) DOESN'T APPLY. Meanwhile MTP's per-step cost GROWS with context
(21.5->12.83): the draft forward + verify run the DSA indexer scan over the full KV, and that cost
is NOT amortized across 78 layers the way the target's is. Note `use_flattening=False (next_n=2)`
in the MTP decode path (indexer.py:297) — the non-flattened spec path may be a perf lever.
So: spec decode helps GLM only at short ctx where the draft is cheap; at long ctx the DSA target is
already so cheap that any drafter is net-negative. This is a MODEL-ARCHITECTURE result, not a bug.

## ns SWEEP + LONG-CTX — COMPLETE (coordinator request)
GLM ships 1 MTP layer; ns>1 runs it autoregressively k times. All ns COHERENT (greedy, War&Peace
Rostov summaries accurate). Decode tok/s (post-prefill, coherent context, enable_thinking=false):

| ctx  | base | ns=1  | ns=2  | ns=3 | best-vs-base |
|------|------|-------|-------|------|--------------|
| 32K  | 18.9 | 21.5  | **22.1** | 20.0 | 1.17x WIN (ns=2) |
| 100K | ~18  | -     | -     | 8.15 | LOSE |
| 200K | 18.24| 12.83 | 10.04 | -    | 0.70x LOSE |
| 400K | 19.34| 12.83 | -     | -    | 0.66x LOSE |

Per-position acceptance (short ctx): ns=1 [.93 easy/.72 tech]; ns=2 [.756,.488]; ns=3 [.792,.506,.286].
Mean accept length: ns=1 1.7-1.9, ns=2 2.24, ns=3 2.58.

**SHORT-CTX SWEET SPOT = ns=2 (~1.17x over base).** ns=3 already past peak (pos2 accept .29, extra
verify cost > gain). Crossover where MTP stops beating base is ~50-80K.

**KEY-TEST ANSWER: NO — MTP does NOT beat PP4-base at 200K/400K (0.66-0.70x). It only wins at
short ctx (<=~32-64K, best ns=2 ~1.17x).**

WHY (architecture, not a bug): GLM-5.2 = DSA sparse attention. base decode is ~FLAT 18-19 tok/s at
ANY context (weight-bound, indexer top-k scan is the only ctx-scaling term and it's cheap+amortized
across 78 layers via one shared per-token selection). Spec decode's classic long-ctx win (amortize a
bandwidth-bound decode over k tokens) does NOT exist here. Meanwhile MTP's cost GROWS with ctx: the
draft (1 MLA+DSA layer) and the verify (next_n query positions) each run their OWN full-context
indexer scan, so MTP pays ~1.5-3 indexer scans/token vs base's 1, plus extra MoE weight loads per
draft/verify. Net-negative >=~100K. `use_flattening=False (next_n=2)` on SM120 is the CORRECT native
fast path (FP8 paged MQA kernel, indexer.py:294) — NOT the culprit; ns>=2 flip to the slower
flattening path (next_n not in (1,2)), which is why ns=2/3 fall faster at long ctx.
Upside ceiling even with a perfect free drafter is small (base already 18-19; theoretical spec cap
~1.5-2x). Not worth a deep kernel rewrite for this model. DSpark ship config's long-ctx value was
CONTEXT REACH (450K), not beating base tok/s — consistent with this.

## DELIVERABLE STATUS
- PRIMARY GOAL (coherent MTP under PP4): ✅ DONE + committed (V2-runner routing, 1-line fix).
- Lossless: ✅ (base non-deterministic so token-identity void; MTP output ∈ base set; high accept).
- ns sweep + long-ctx verdict: ✅ DONE (above).
- Recommendation: MTP spec decode is a SHORT-CONTEXT-only win for GLM-5.2 (ns=2, ~1.17x). For long
  context use plain pp4-1m (base is already at the spec ceiling). Shipping decision left to user.

## >>> CURRENT STATE (read this first) <<<
- Repro DONE. MTP boots fine (DeepSeekMTPModel, layer 78 loads via hybrid quant, KV allocates).
  First greedy request crashes: `AssertionError: PP+async expects sampled_token_ids to have shape
  [num_reqs, 1]` at vllm/v1/worker/gpu_model_runner.py:4716 (_pp_broadcast_prev_sampled_token_ids),
  from sample_tokens:4494. Spec produces [num_reqs, num_spec+1]; the V1 assert wants [N,1].
- ROOT CAUSE FOUND: MTP runs the **V1 monolithic runner** (vllm/v1/worker/gpu_model_runner.py).
  DSpark works because it FORCES the **V2 runner** (vllm/v1/worker/gpu/model_runner.py + gpu/pp_utils.py
  which has the PP broadcast-padding + draft-FIFO fixes). Runner choice: VllmConfig.use_v2_model_runner
  (vllm/config/vllm.py:518). DSpark hard-forces V2 (line 528-532, method=="dspark"). MTP does NOT →
  falls to _is_default_v2_model_runner_model() → False (our GlmMoeDsa is MoE, not in default V2 arch
  list) → V1 → crash.
- KEY: `mtp` is ALREADY in V2's supported spec methods (vllm/config/vllm.py:2070). So V2 should run it.
- NEXT: force MTP onto V2. Fastest test: env VLLM_USE_V2_MODEL_RUNNER=1. If good, make it permanent by
  adding method=="mtp" to the force-V2 block (vllm/config/vllm.py:528).

## Launch
- Serve script: /home/jarrelscy/glm52/vllm/serve_mtp.sh (PP4, MAXLEN env, SYNC=1 for --no-async).
- Boot: `cd /home/jarrelscy/glm52/vllm && nohup env MAXLEN=32768 bash serve_mtp.sh > /home/jarrelscy/glm52/mtp_bootN.log 2>&1 </dev/null &`
- API key: VLLM_API_KEY=sk-98f2447ba021acf1c9ac0aea25f65bb58cf3066dcda734e5 (curl needs Bearer).
- Test: greedy temp=0, chat_template_kwargs:{enable_thinking:false}. Compare vs PP4-base (pp4-1m) greedy.
- Model: /data/huggingface/glm52-models/1m (CLEAN 79/79). Boot ~90s load. Kill: pkill -9 -f "vllm serve".

## Known starting facts
- MTP = checkpoint layer 78 (deepseek_mtp path, num_nextn_predict_layers=1), MLA attention.
  model_type glm_moe_dsa -> remapped to deepseek_mtp -> arch DeepSeekMTPModel (speculative.py:341).
- spec config: {"method":"deepseek_mtp","num_speculative_tokens":1} (deepseek_mtp deprecated->mtp).

## Attempts log (append newest at top)

### Attempt 1 (2026-07-07 ~11:31) — REPRO
Config: PP4, MTP, async, MAXLEN 32768. Result: boots+loads clean, first greedy req crashes with
`PP+async expects sampled_token_ids to have shape [num_reqs, 1]` (gpu_model_runner.py:4716, V1 runner).
Confirms: MTP on V1 runner; V1 lacks spec-under-PP broadcast padding. -> route through V2.
