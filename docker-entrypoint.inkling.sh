#!/usr/bin/env bash
# Inkling-512k-NVFP4-AQLM-hybrid on SM120 (4x RTX PRO 6000, TP4).
# Speed campaigns: task #134 (gated 2026-07-19) + task #137 laneB decode
# round cost (gated 2026-07-20: V2 gemv + fused attn combine + verify via
# gemv, all default-on in model code; see hybrid_moe_kernels.py /
# triton_decode_attention.py / hybrid_moe.py. Escape hatches:
# INKLING_GEMV_V2=0, INKLING_ATTN_FUSED_COMBINE=0,
# INKLING_DISABLE_DECODE_GROUPED=0):
#
#   MODE=512k-mtp (default)  524,288 ctx + MTP ns=2 (lossless spec decode)
#                            + adaptive per-request draft suspension
#                            (INKLING_ADAPTIVE_SPEC=1 default, task #138
#                            laneC; set 0 to disable). Task #137 offline
#                            bench (no adaptive): round 38-40ms short
#                            (was 47-50), 48.7ms @512K depth (was ~61);
#                            decode 50-67 prose / 68-77 code / 74-76
#                            counting tok/s short, ~33-35 pure @512K
#                            depth; prefill ~1174 tok/s warm @512K
#   MODE=640k                655,360 ctx, no MTP (longest context)
#                            decode ~50-52 short (task #137; was ~40.8)
#                            / >=34.5 @640K depth (pre-#137 number,
#                            depth not re-measured); prefill ~1075 warm
#
# Both: FULL_DECODE_ONLY CUDA graphs (capture sizes [1,2,4]), bf16 KV
# (lossless -- fp8 KV excluded by directive), util 0.97, mnbt 2048.
# Runtime knobs: MODEL_DIR, MODE, MAXLEN, NUM_SPEC (512k-mtp only; ns=2 is
# the swept optimum -- larger ns is bimodal on draft-hostile prose).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/inkling}"
MODE="${MODE:-512k-mtp}"

source /opt/vllm/.venv/bin/activate
NV="/opt/vllm/.venv/lib/python3.12/site-packages/nvidia"
CUX="$NV/cu13"; [ -d "$CUX" ] || CUX="$NV/cu12"
export CUDA_HOME="$CUX" PATH="$CUX/bin:$PATH"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL SHM-bounces on this box (all-NODE PCIe topology, AMD-Vi); driver P2P is
# ~55 GB/s/pair. Routing NCCL over P2P halves prefill RS/AG time (the >64-token
# sconv fallback path): 4K prefill 1802 -> 1912 tok/s, 16K 1822 -> 1927 (+6%),
# 120K fill A/B + coherence + unit suite gated 2026-07-19 (lane A, task #136).
# Transport-only change: ring reduction order is unchanged (numerics-preserving).
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"

# --- LMCache KV persistence (task #143) -------------------------------------
# ENABLE_LMCACHE=1 turns on lossless KV disk/CPU persistence via the LMCache MP
# connector (jarrelscy/LMCache branch inkling-mp-hybrid). Inkling's 12-group
# heterogeneous hybrid KV (55 SWA + 11 full-attn + 66 sconv + MTP) cannot use
# the classic in-process connector (vLLM refuses non-HMA connectors under the
# hybrid KV manager), so this launches a background MP server holding a pinned
# host-RAM L1 pool + an fs_native disk L2, and points vllm serve at it. The
# SWA whole-chunk skip (commit fbec1c14) bounds the L1 working set to the
# sliding window, so >=512K restore is lossless: gated cold-restart 519,766-tok
# restore 8.5s (55x vs recompute), needle + byte-identical, 0 errors; graceful
# exact-recompute fallback under a starved L1. Escape hatch: ENABLE_LMCACHE=0.
# Knobs: L1_GB (pinned host pool, default 100), LMCACHE_PORT (default 6879),
# LMCACHE_L2_DIR (default /data/lmcache/inkling-512k-mtp), LMCACHE_L2_GB (800).
KV_XFER_FLAGS=()
KV_MEM_FLAGS=()
if [ "${ENABLE_LMCACHE:-0}" = "1" ]; then
  # MP mode is incompatible with expandable_segments (the CUDA-IPC KV buffers
  # need stable virtual addresses); the connector requires it unset.
  unset PYTORCH_CUDA_ALLOC_CONF
  # util 0.975 is the gated value for >=512K + MTP with LMCache on.
  UTIL="${UTIL:-0.975}"
  # The MP server's native store path (execute_object_group_transfer) makes a
  # lazy per-store GPU scratch allocation in its OWN CUDA context, separate from
  # vLLM's. At util 0.975 vLLM leaves only ~9 MiB VRAM free, so that scratch
  # OOMs and the store fails closed (nothing persists -> a same-server "restore"
  # is then only a vLLM-prefix artifact, not a real LMCache restore). Capping
  # the KV pool a hair below the util target frees a deterministic VRAM headroom
  # for the store scratch while keeping KV >= 524,288 tokens (512K).
  # Boot-log floor: vLLM needs 7.42 GiB KV to fit one 524,288-token request
  # (~15.2 KB/token incl. the 4 padding layers). Uncapped util-0.975 KV is
  # ~7.64 GiB (leaves only ~9 MiB VRAM free). 8.0 GiB (8e9 B) ~= 526K KV tokens
  # (>=512K, ~2K margin for MTP draft slots + block rounding) and frees ~0.2 GiB
  # of VRAM for the MP server's store scratch. Override with KV_CACHE_MEMORY.
  KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-8000000000}"
  KV_MEM_FLAGS=(--kv-cache-memory "$KV_CACHE_MEMORY")
  L1_GB="${L1_GB:-100}"
  LMCACHE_PORT="${LMCACHE_PORT:-6879}"
  LMCACHE_L2_DIR="${LMCACHE_L2_DIR:-/data/lmcache/inkling-512k-mtp}"
  LMCACHE_L2_GB="${LMCACHE_L2_GB:-800}"
  mkdir -p "$LMCACHE_L2_DIR"
  echo "[entrypoint] starting LMCache MP server on :$LMCACHE_PORT "\
"(L1 ${L1_GB}GB pinned, L2 fs_native $LMCACHE_L2_DIR cap ${LMCACHE_L2_GB}GB)" >&2
  python -m lmcache.v1.multiprocess.server \
    --host 0.0.0.0 --port "$LMCACHE_PORT" \
    --chunk-size 256 \
    --l1-size-gb "$L1_GB" \
    --eviction-policy LRU \
    --l2-adapter '{"type":"fs_native","base_path":"'"$LMCACHE_L2_DIR"'","max_capacity_gb":'"$LMCACHE_L2_GB"'}' &
  MP_PID=$!
  # Wait for the MP server to bind before vLLM tries to connect.
  for _ in $(seq 1 120); do
    if python - "$LMCACHE_PORT" <<'PYEOF' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PYEOF
    then break; fi
    kill -0 "$MP_PID" 2>/dev/null || { echo "[entrypoint] MP server died during startup" >&2; exit 1; }
    sleep 1
  done
  # Reap the MP server if vLLM exits, so the container tears down cleanly.
  trap 'kill "$MP_PID" 2>/dev/null || true' EXIT
  KV_XFER_FLAGS=(--kv-transfer-config
    '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":'"$LMCACHE_PORT"'}}')
fi

case "$MODE" in
  512k-mtp)
    MAXLEN="${MAXLEN:-524288}"
    SPEC_FLAGS=(--speculative-config
      '{"method": "mtp", "num_speculative_tokens": '"${NUM_SPEC:-2}"'}')
    # Adaptive draft suspension (lossless): suspend MTP drafting for
    # requests whose acceptance EMA does not pay for the draft cost
    # (draft-hostile prose), keeping counting/code wins. Default ON for
    # this mode (gated 2026-07-19); export INKLING_ADAPTIVE_SPEC=0 to
    # get static ns=2 behavior.
    export INKLING_ADAPTIVE_SPEC="${INKLING_ADAPTIVE_SPEC:-1}"
    ;;
  640k)
    MAXLEN="${MAXLEN:-655360}"
    SPEC_FLAGS=()
    ;;
  *)
    echo "unknown MODE=$MODE (expected 512k-mtp or 640k)" >&2
    exit 1
    ;;
esac

exec vllm serve "$MODEL_DIR" \
  --served-model-name inkling \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 4 \
  --max-model-len "$MAXLEN" \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization "${UTIL:-0.97}" \
  "${KV_MEM_FLAGS[@]}" \
  --kv-cache-dtype auto \
  --no-async-scheduling \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser inkling \
  --kernel-config '{"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false}' \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]}' \
  "${SPEC_FLAGS[@]}" \
  "${KV_XFER_FLAGS[@]}"
