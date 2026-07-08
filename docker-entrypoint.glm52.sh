#!/usr/bin/env bash
# Entrypoint for the GLM-5.2 hybrid SM120 image. Selects a serving topology via
# $PARALLEL and launches vllm serve on :8000.
#
#   PARALLEL=pp4-1m      PP4, NO speculator, 1M window   (default; the full-context config)
#   PARALLEL=pp4-dspark  PP4 + DSpark, ~256K             (spec decode, reduced context)
#   PARALLEL=tp2pp2      TP2xPP2 + DSpark, ~200K         (spec decode, fastest single-stream decode)
#   PARALLEL=pp4-mtp     PP4 + native MTP self-spec, ~32K   (coherent+lossless; short-ctx ~1.17x @ns=2, net-neg >=100K)
#   PARALLEL=pp4-tpdraft PP4 target + DSpark draft sharded TP4, ~450K  (draft-TP-over-PP, longest spec-decode ctx)
#
# WHY 1M and DSpark are separate modes: on 4x96GB, the 754B hybrid weights (~272 GiB)
# plus a 1M-sized KV cache already fill VRAM. The DSpark drafter (its own embed/
# lm_head/layers + a 1M activation reservation) leaves too little KV for a 1M
# sequence, so speculative decode is only available at capped context.
set -euo pipefail
cd /opt/vllm && source .venv/bin/activate

# Python.h (JIT) + CUDA home (nvidia pip cu13/cu12) discovery
PYINC_DIR=$(find /root/.local/share/uv/python -maxdepth 4 -type d -path "*/include/python3.12" 2>/dev/null | head -1)
[ -n "${PYINC_DIR:-}" ] && export CPATH="$PYINC_DIR" C_INCLUDE_PATH="$PYINC_DIR"
# nvidia is a namespace package (nvidia.__file__ is None) -> use __path__.
NV=$(python -c "import nvidia;print(nvidia.__path__[0])" 2>/dev/null || true)
for c in cu13 cu12; do [ -n "${NV:-}" ] && [ -d "$NV/$c" ] && export CUDA_HOME="$NV/$c" && break; done

export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=1048576
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_DISABLE_FP8_W8A16="${VLLM_DISABLE_FP8_W8A16:-1}"   # v2-only (bit-exact) default; set =0 to opt into v4 fp8 W8A16 (+6-8% base decode)

MODEL_DIR="${MODEL_DIR:-/models/1m}"
PARALLEL="${PARALLEL:-pp4-1m}"
NUM_SPEC_ENV="${NUM_SPEC:-}"   # user override only; per-mode default applied AFTER the case

UTIL_DEFAULT=0.95   # per-mode default; env UTIL overrides
DRAFT_TP=""         # non-empty -> draft_tensor_parallel_size in the spec config
DCP=""              # non-empty -> --decode-context-parallel-size (shard MLA KV across TP ranks -> ~1M at TP4)
CGMODE=""           # per-mode cudagraph_mode override (empty -> FULL_AND_PIECEWISE); DCP+spec MUST use PIECEWISE
NSDEF=""            # per-mode spec-token default override (empty -> MTP=2 / DSpark=5); env NUM_SPEC always wins

case "$PARALLEL" in
  pp4-1m)      # full 1M window, NO speculator (verified: KV 1.27M tokens, ~22 tok/s)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=0; DEFLEN=1048576 ;;
  tp2pp2)      # TP2xPP2 + DSpark, ~223K, ~35 tok/s (best spec-decode single-stream)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-39,39}"
    PAR="--tensor-parallel-size 2 --pipeline-parallel-size 2"; SPEC=1; DEFLEN=200000 ;;
  tp4-dspark)  # TP4 + DSpark, ~200K, ~59.6 tok/s (fastest decode, no PP bubble)
    PAR="--tensor-parallel-size 4"; SPEC=1; DEFLEN=200000 ;;
  tp4-1m)      # TP4 + Decode Context Parallelism (DCP4) -> ~1M at TP speed, NO spec.
    # DCP sequence-shards the MLA latent KV across the 4 TP ranks (TP4 alone caps ~371K
    # because MLA KV replicates); reaches the same ~1.13M KV ceiling as PP4 but at TP speed
    # with no pipeline bubble. VALIDATED: KV 1,128,940 tok @ 950K; needle @ 749,035 tok
    # depth-0.5 = PASS (coherent at depth). DECODE with graphs (default-on for TP) = 24.9
    # tok/s @ ~700K — BEATS PP4-base@1M (18.7) by ~33%; eager is only 10.8 (DCP LSE
    # all-gather over PCIe is comm-bound -> graphs are ESSENTIAL here, they capture the
    # ag_rs comms). Needs the SM120 return-LSE + decode-out-size fix (on this branch).
    PAR="--tensor-parallel-size 4"; SPEC=0; DEFLEN=950000; DCP=4 ;;
  tp4-1m-mtp)  # TP4 + DCP4 + native MTP (ns5 default, preferred) -> coherent ~1M WITH lossless spec at TP speed.
    # THE best long-ctx config. DCP shards KV (fits ~1M) + MTP spec verifies k+1 tok/step.
    # MUST use PIECEWISE cudagraphs (FULL/FULL_AND_PIECEWISE DEADLOCK: the in-graph DCP
    # LSE-combine collective + spec drafter/verify NCCL ordering hangs). PIECEWISE splits at
    # attention -> DCP collective runs eager, MoE/linear graphed -> no hang, keeps the win.
    # VALIDATED: needle@749K PASS (lossless); decode PIECEWISE 54.8/50.8/40.7 short, ~28-30
    # @123K -> ~2x eager (30.8/28.7/24.3) and beats base tp4-1m (24.9). Counting: ns=5 wins
    # (graphs amortize drafts + ~100% accept) — raise NUM_SPEC for predictable workloads.
    PAR="--tensor-parallel-size 4"; SPEC=2; DEFLEN=950000; DCP=4; CGMODE=PIECEWISE; NSDEF=5 ;;
  pp4-dspark)  # PP4 + DSpark, ~130K (drafter co-locates on last rank; dominated by tp2pp2)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=1; DEFLEN=131072 ;;
  pp4-mtp)     # PP4 + native MTP self-speculation (checkpoint layer 78, method=mtp), ~1M ctx.
    # Coherent + lossless. Routed through the V2 model runner (vllm/config/vllm.py force-V2 for
    # method in {dspark,mtp}). FITS ~1M (est ceiling 999,168 tok @ util 0.97; default 950K for
    # margin) — MTP's draft KV is MLA-shaped & tiny, unlike DSpark. PP split MUST be 20,20,20,18
    # (NOT 21,19,19,19): the MTP layer is a full 256-expert MoE (~7.5 GiB) that lands on the LAST
    # rank, so that rank carries fewer target layers or context caps. PERF (non-streamed, CORRECT
    # counting): MTP BEATS base on every workload at ns=2 — 1m @32K [easy 29.3/code 24.4/complex
    # 24.0] vs base 18.7 (1.26-1.57x); still wins @200K (easy 24.6 vs 18.4). ns=2 best all-round.
    # (Earlier "MTP loses at long ctx" was a STREAMED-measurement artifact — SSE bundles accepted
    # tokens; see BENCHMARKING.md.) Short-ctx latency: set MAXLEN=32768.
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-20,20,20,18}"
    PAR="--pipeline-parallel-size 4"; SPEC=2; DEFLEN=950000; UTIL_DEFAULT=0.97 ;;
  tp2pp2-mtp)  # TP2xPP2 + native MTP self-speculation, ~580K ctx — the SPEED/CONTEXT BALANCE pick.
    # Non-streamed: 1m [easy 40.4/code 36.1/complex 30.1] tok/s @ ceiling ~580K — ~2x pp4-mtp's
    # decode while still reaching >0.5M context. Best all-round serving config.
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-39,39}"
    PAR="--tensor-parallel-size 2 --pipeline-parallel-size 2"; SPEC=2; DEFLEN=560000 ;;
  pp4-tpdraft) # PP4 target (MLA KV split by layer -> long ctx) + DSpark draft sharded
    # TP4 across the PP ranks (draft-TP-over-PP). Draft's 64 KV heads shard 16/rank,
    # freeing the KV that a co-located draft would eat -> ~2x the spec-decode ceiling.
    # Validated: KV 514,697 tokens @ 450K, mean accepted ~3.0, ~17-19 tok/s.
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=1; DEFLEN=450000; UTIL_DEFAULT=0.97; DRAFT_TP=4 ;;
  *) echo "unknown PARALLEL=$PARALLEL (use pp4-1m | pp4-mtp | tp2pp2 | tp2pp2-mtp | tp4-dspark | tp4-1m | tp4-1m-mtp | pp4-dspark | pp4-tpdraft)"; exit 1 ;;
esac
UTIL="${UTIL:-$UTIL_DEFAULT}"
# per-mode spec-token default: MTP=2 (matrix optimum, best all-round), DSpark=5; env NUM_SPEC overrides
if [ "$SPEC" = 2 ]; then NUM_SPEC="${NUM_SPEC_ENV:-${NSDEF:-2}}"; else NUM_SPEC="${NUM_SPEC_ENV:-${NSDEF:-5}}"; fi
# NOTE: 1M context and the DSpark drafter cannot co-fit on 4x96GB. The drafter
# needs ~92 GiB KV on its rank vs ~8 GiB available; capping the draft window
# (DRAFT_MAXLEN, below) does NOT free it — vLLM sizes the draft KV at target len.
# For 1M use pp4-1m (no spec); for spec decode use tp4-dspark / tp2pp2 (<=~200K).
MAXLEN="${MAXLEN:-$DEFLEN}"

# CUDA graphs (V2 hybrid kernel is graph-safe via torch.library custom ops).
# FULL_AND_PIECEWISE = one full-graph replay per decode forward (plain PIECEWISE is
# net-NEGATIVE — per-segment replay loop — do NOT use it).
# DEFAULT is per-mode: ON for TP configs, OFF for pure-PP. Graphs amortize TP's
# all-reduce/launch overhead spectacularly (measured @32K: tp4-dspark +25% ->74.6,
# tp2pp2 +54%, tp4-base +180%, tp2pp2-mtp +16-22%) but do ~nothing for pure-PP
# (bubble-bound: pp4 +0-2%). Override with CUDAGRAPH=0/1.
case "$PAR" in *tensor-parallel-size*) CG_DEFAULT=1 ;; *) CG_DEFAULT=0 ;; esac
CUDAGRAPH="${CUDAGRAPH:-$CG_DEFAULT}"
GRAPH_FLAGS=(--enforce-eager)
if [ "$CUDAGRAPH" = 1 ]; then
  GRAPH_FLAGS=(--compilation-config \
    "{\"mode\": 3, \"cudagraph_mode\": \"${CUDAGRAPH_MODE:-${CGMODE:-FULL_AND_PIECEWISE}}\"}")
fi

ARGS=(vllm serve "$MODEL_DIR" $PAR
  --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype fp8_ds_mla
  --max-model-len "$MAXLEN"
  --max-num-seqs 2
  --max-num-batched-tokens 2048
  --no-enable-flashinfer-autotune
  "${GRAPH_FLAGS[@]}"
  --served-model-name "${SERVED_NAME:-glm-5.2}"
  --port "${PORT:-8001}")
[ -n "$DCP" ] && ARGS+=(--decode-context-parallel-size "$DCP" --dcp-comm-backend ag_rs)
if [ "$SPEC" = 1 ]; then
  SC="{\"model\": \"RedHatAI/GLM-5.2-speculator.dspark\", \"method\": \"dspark\", \"num_speculative_tokens\": $NUM_SPEC"
  [ -n "${DRAFT_MAXLEN:-}" ] && SC="$SC, \"max_model_len\": $DRAFT_MAXLEN"
  [ -n "$DRAFT_TP" ] && SC="$SC, \"draft_tensor_parallel_size\": $DRAFT_TP"
  SC="$SC}"
  ARGS+=(--speculative-config "$SC")
elif [ "$SPEC" = 2 ]; then
  # Native MTP self-speculation (checkpoint layer 78). method=deepseek_mtp -> mtp;
  # routed to V2 runner automatically (config/vllm.py force-V2 for mtp).
  ARGS+=(--speculative-config "{\"method\": \"deepseek_mtp\", \"num_speculative_tokens\": $NUM_SPEC}")
fi

echo ">> GLM-5.2 SM120  PARALLEL=$PARALLEL  MAXLEN=$MAXLEN  util=$UTIL  graph=$CUDAGRAPH  spec=$SPEC  draft_tp=${DRAFT_TP:-1}  dcp=${DCP:-1}  served=${SERVED_NAME:-glm-5.2}  model=$MODEL_DIR"
exec "${ARGS[@]}" "$@"
