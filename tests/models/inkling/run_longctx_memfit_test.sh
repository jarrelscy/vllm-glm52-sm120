#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Inkling-512k-NVFP4-AQLM-hybrid -- long-context memory-fit sweep
# ---------------------------------------------------------------------------
# PURPOSE
#   Every smoke-test attempt so far (attempts 9-22, see
#   /tmp/claude-1000/-home-jarrelscy-homeassistant/ca20f55c-6e14-4440-b072-adef0dae71ed/scratchpad/run_attempt*.sh)
#   used enforce_eager=True and max_model_len in {1024, 8192}. None of them
#   have exercised CUDA-graph capture at large max_model_len, and none have
#   empirically tested the ~3.2-5.9 GiB/GPU slack estimate (weights 83.41 GiB
#   measured + KV cache 5.55 GiB bf16 / 2.78 GiB fp8 @ 512K tokens, against a
#   94.97 GiB card at gpu_memory_utilization=0.97).
#
#   This script launches the model via `vllm serve` (TP4, enforce_eager=OFF
#   i.e. CUDA graphs ON, kv_cache_dtype configurable / default fp8_e4m3) and
#   steps max_model_len through 8192 -> 32768 -> 131072 -> 262144 -> 524288,
#   recording per-GPU memory and pass/fail/OOM at each step, so the estimate
#   above can be diffed against reality.
#
# !!! THIS SCRIPT IS AUTHORED BUT DELIBERATELY NOT RUN BY THE AGENT THAT   !!!
# !!! WROTE IT. Two other engineers are running real GPU smoke tests right !!!
# !!! now -- do not invoke this until you hold the GPU lease and have      !!!
# !!! confirmed the box is otherwise idle.                                 !!!
#
# !!! READ "KNOWN RISK: fp8 kv-cache dtype mismatch" BELOW BEFORE RUNNING. !!!
#
# USAGE
#   bash run_longctx_memfit_test.sh                    # fp8_e4m3 KV cache (default)
#   KV_CACHE_DTYPE=auto bash run_longctx_memfit_test.sh   # bf16 KV cache fallback
#   MAX_MODEL_LENS="8192 32768" bash run_longctx_memfit_test.sh  # subset
#
# All knobs below are overridable via environment variables so a config
# change doesn't require editing the script.
# ---------------------------------------------------------------------------
set -uo pipefail

# ---------------------------------------------------------------------------
# KNOWN RISK: fp8 kv-cache dtype mismatch (found by static read, NOT tested)
# ---------------------------------------------------------------------------
# vllm/models/inkling/nvidia/attention.py:InklingAttention registers k_scale/
# v_scale buffers and derives self.kv_cache_torch_dtype from cache_config.
# cache_dtype (i.e. it WILL request a float8_e4m3fn-typed paged KV cache when
# --kv-cache-dtype fp8_e4m3 is passed, for both the 11 full-attention layers
# and the 55 local/sliding-window (local_extent-capped) layers).
#
# BUT: ops/fa4_rel_attention.py:inkling_fa4_rel_attention() calls
# vllm.third_party.tml_fa4.flash_attn_varlen_func(..., k=key_cache,
# v=value_cache, ...) WITHOUT ever passing sfk/sfv (the tml_fa4 block-scale
# args) or any k_scale/v_scale-based dequant -- k_scale/v_scale are dead
# buffers, never read anywhere in attention.py or fa4_rel_attention.py
# (grepped, zero hits outside their own registration).
#
# vllm/third_party/tml_fa4/interface.py's non-blockscaled path (the one
# Inkling's call falls into, since sfq/sfk/sfv are all None) asserts:
#     assert q.dtype in [torch.float16, torch.bfloat16]
#     assert q.dtype == k.dtype == v.dtype
# q is always bf16 (model compute dtype). If the KV cache is fp8, k/v will be
# torch.float8_e4m3fn -> q.dtype == k.dtype is FALSE -> AssertionError
# "inputs must have the same dtype", raised on the FIRST attention forward
# call for ANY of the 55 local + 11 full attention layers.
#
# PREDICTION: with --kv-cache-dtype fp8_e4m3, every step below will almost
# certainly crash immediately after the server starts serving (first
# forward pass), NOT during startup memory profiling and NOT with an OOM.
# This would make the memory-fit question undecidable via fp8 until that
# assertion is fixed upstream, independent of how much VRAM slack exists.
#
# This script still requests fp8_e4m3 by default because that is what was
# asked for, but:
#   - it classifies failures (see classify_failure()) so an assertion-style
#     crash is never miscategorized as "OOM" in the results table, and
#   - KV_CACHE_DTYPE=auto (bf16, matching model dtype) is provided as an
#     explicit fallback for whoever runs this, to get a *bf16* memory-fit
#     signal (a conservative /overestimate of KV cost, per the ~5.55 GiB
#     bf16 vs ~2.78 GiB fp8 figures) while the fp8 path is unproven.
# If you see FAIL_ASSERT_OR_DTYPE below for every step, this is why: go fix
# attention.py/fa4_rel_attention.py's fp8 wiring before re-running fp8 here.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config (all overridable via env)
# ---------------------------------------------------------------------------
REPO_ROOT="/home/jarrelscy/inkling-sm120"
VENV_BIN="${VENV_BIN:-/home/jarrelscy/inkling-sm120-venv/bin}"
VLLM_BIN="${VLLM_BIN:-$VENV_BIN/vllm}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_BIN/python}"

MODEL_GLOB="${MODEL_GLOB:-/data/huggingface/hub/models--jarrelscy--Inkling-512k-NVFP4-AQLM-hybrid/snapshots/*}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-inkling-longctx-memfit}"

TP_SIZE="${TP_SIZE:-4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"   # see risk note above; set to "auto" for bf16 fallback
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"              # single/few long-ctx requests, not a concurrency stress test
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
PORT="${PORT:-8010}"                            # NOT 8001: avoid colliding with the homeassistant docker stack
HOST="${HOST:-127.0.0.1}"

# ENFORCE_EAGER left OFF on purpose: vLLM's --enforce-eager is a store_true
# flag. Its ABSENCE means enforce_eager=False (CUDA graphs ON), which is
# what this test is specifically trying to exercise (attempts 9-22 never
# did). Do not add --enforce-eager below.

MAX_MODEL_LENS="${MAX_MODEL_LENS:-8192 32768 131072 262144 524288}"

READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"   # 30 min: cold weight load (83.4 GiB/GPU) + CUDA graph capture
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-5}"
DRAIN_TIMEOUT_SEC="${DRAIN_TIMEOUT_SEC:-180}"     # wait for GPU mem to clear between steps
REQUIRED_FREE_MIB="${REQUIRED_FREE_MIB:-95000}"   # matches settle-guard convention in run_attempt14+.sh

SCRATCH="${SCRATCH:-/tmp/claude-1000/-home-jarrelscy-homeassistant/ca20f55c-6e14-4440-b072-adef0dae71ed/scratchpad}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/tests/models/inkling/longctx_memfit_logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$OUT_DIR/$RUN_TAG"
SUMMARY_FILE="$RUN_DIR/summary.tsv"

# GPU lease (see /home/jarrelscy/glm52/coord/gpulease.sh; convention lifted
# verbatim from run_attempt9.sh..run_attempt22.sh in $SCRATCH).
LEASE_SCRIPT="${LEASE_SCRIPT:-/home/jarrelscy/glm52/coord/gpulease.sh}"
LEASE_NAME="${LEASE_NAME:-inkling-longctx-memfit-$RUN_TAG}"
# Generous: this run does up to 5 sequential cold(ish) server launches, each
# of which may itself take many minutes. Bump if you trim MAX_MODEL_LENS.
LEASE_MAX_MINUTES="${LEASE_MAX_MINUTES:-180}"

PROBE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/longctx_probe_request.py"

mkdir -p "$RUN_DIR"
echo "=== run dir: $RUN_DIR ==="
printf 'max_model_len\tresult\tserver_start_s\tper_gpu_mem_used_mib\tper_gpu_mem_free_mib\tkv_cache_log_line\tprobe_result\tnotes\n' > "$SUMMARY_FILE"

SERVER_PID=""
LEASE_HELD=0

# ---------------------------------------------------------------------------
# Cleanup: always release the lease and kill any server we still own, no
# matter how this script exits (normal completion, error, Ctrl-C).
# ---------------------------------------------------------------------------
cleanup() {
  local rc=$?
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "=== cleanup: killing leftover server pid $SERVER_PID ==="
    kill -TERM "$SERVER_PID" 2>/dev/null
    sleep 5
    kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL "$SERVER_PID" 2>/dev/null
  fi
  if [ "$LEASE_HELD" -eq 1 ]; then
    echo "=== cleanup: releasing GPU lease ($LEASE_NAME) ==="
    bash "$LEASE_SCRIPT" release "$LEASE_NAME" || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Resolve model snapshot path (mirrors inkling_generate_smoke_small.py's
# glob.glob(...)[0] pattern from run_attempt9..22).
# ---------------------------------------------------------------------------
resolve_model_path() {
  local -a matches=( $MODEL_GLOB )
  if [ ${#matches[@]} -eq 0 ] || [ ! -e "${matches[0]}" ]; then
    echo "FATAL: model glob '$MODEL_GLOB' matched nothing" >&2
    exit 96
  fi
  echo "${matches[0]}"
}

# ---------------------------------------------------------------------------
# Classify a failed step from its server log, so OOM vs the fp8-dtype-
# assertion risk (see top-of-file note) vs anything else are never conflated.
# ---------------------------------------------------------------------------
classify_failure() {
  local log="$1"
  if grep -qiE "CUDA out of memory|OutOfMemoryError|out of memory|CUDA error: out of memory" "$log"; then
    echo "FAIL_OOM"
  elif grep -qE "inputs must have the same dtype|must have the same dtype|blockscaled.*must be|AssertionError" "$log"; then
    echo "FAIL_ASSERT_OR_DTYPE"
  elif grep -qiE "Traceback \(most recent call last\)" "$log"; then
    echo "FAIL_OTHER_TRACEBACK"
  else
    echo "FAIL_UNKNOWN"
  fi
}

# ---------------------------------------------------------------------------
# Per-GPU memory snapshot as a single space-joined "used(free)" string, e.g.
# "41000(56249) 41000(56249) 41000(56249) 41000(56249)"
# ---------------------------------------------------------------------------
gpu_mem_snapshot() {
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/^ +| +$/,"",$1); gsub(/^ +| +$/,"",$2); printf "%s(%s) ", $1, $2}'
}

gpu_mem_used_only() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '
}
gpu_mem_free_only() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' '
}

wait_for_gpus_clear() {
  local deadline=$(( $(date +%s) + DRAIN_TIMEOUT_SEC ))
  while true; do
    local free_line min_free
    free_line=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    min_free=$(echo "$free_line" | sort -n | head -1)
    if [ "$min_free" -ge "$REQUIRED_FREE_MIB" ]; then
      echo "=== GPUs clear (min free ${min_free} MiB) ==="
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "WARNING: GPUs did not clear above ${REQUIRED_FREE_MIB} MiB free within ${DRAIN_TIMEOUT_SEC}s (min free ${min_free} MiB) -- proceeding anyway, next step's numbers may be biased" >&2
      return 1
    fi
    sleep 3
  done
}

MODEL_PATH="$(resolve_model_path)"
echo "=== resolved model path: $MODEL_PATH ==="
echo "=== KV_CACHE_DTYPE=$KV_CACHE_DTYPE  TP_SIZE=$TP_SIZE  GPU_MEM_UTIL=$GPU_MEM_UTIL ==="
echo "=== max_model_len sweep: $MAX_MODEL_LENS ==="

# ---------------------------------------------------------------------------
# Acquire GPU lease ONCE for the whole sweep (mirrors run_attempt*.sh, which
# wraps one acquire/release pair around the entire GPU-touching payload).
# ---------------------------------------------------------------------------
echo "=== waiting for GPU lease ($LEASE_NAME, max ${LEASE_MAX_MINUTES}m) ==="
if ! bash "$LEASE_SCRIPT" acquire "$LEASE_NAME" "$LEASE_MAX_MINUTES"; then
  echo "FATAL: acquire command failed/was killed -- aborting, NOT touching the GPU unguarded" >&2
  exit 99
fi
LEASE_HELD=1

HOLDER_LINE=$(cat "$(dirname "$LEASE_SCRIPT")/GPU_LEASE/holder" 2>/dev/null || true)
if [[ "$HOLDER_LINE" != "$LEASE_NAME "* ]]; then
  echo "FATAL: lease file does not show our name after acquire (saw: '$HOLDER_LINE') -- aborting" >&2
  exit 98
fi
echo "=== lease verified held by us ($HOLDER_LINE) ==="

# Settle-guard: identical rationale to run_attempt14.sh onward -- don't
# launch into a still-unwinding CUDA context from the previous lease holder.
echo "=== waiting for all GPUs to show >= ${REQUIRED_FREE_MIB} MiB free before touching them ==="
for i in $(seq 1 30); do
  FREE_LINE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  MIN_FREE=$(echo "$FREE_LINE" | sort -n | head -1)
  echo "poll $i: per-GPU free MiB = $(echo "$FREE_LINE" | tr '\n' ' ') (min=$MIN_FREE)"
  if [ "$MIN_FREE" -ge "$REQUIRED_FREE_MIB" ]; then
    echo "=== GPUs settled, proceeding ==="
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "FATAL: GPUs never settled above ${REQUIRED_FREE_MIB} MiB free after 30 polls -- aborting, NOT launching" >&2
    exit 97
  fi
  sleep 2
done

export CUDA_HOME="${CUDA_HOME:-/home/jarrelscy/inkling-sm120-venv/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$VENV_BIN:$PATH"

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
for LEN in $MAX_MODEL_LENS; do
  STEP_DIR="$RUN_DIR/len_${LEN}"
  mkdir -p "$STEP_DIR"
  SERVER_LOG="$STEP_DIR/server.log"
  echo ""
  echo "############################################################"
  echo "=== STEP max_model_len=$LEN ==="
  echo "############################################################"

  PRE_USED="$(gpu_mem_used_only)"
  PRE_FREE="$(gpu_mem_free_only)"
  echo "pre-launch per-GPU mem used/free (MiB): used=[$PRE_USED] free=[$PRE_FREE]"

  START_TS=$(date +%s)
  EXTRA_ARGS=()
  # COMPILATION_CONFIG_JSON: opt-in escape hatch, default empty (no change
  # in behavior). Added after step max_model_len=8192 crashed FULL cudagraph
  # capture with cudaErrorStreamCaptureUnsupported -- root-caused to
  # hybrid_moe.py's flat_ids.unique().tolist() (line 356) and
  # mask.nonzero().flatten() (line 375), both host-syncing/dynamic-shape ops
  # that torch.cuda.graph()-based FULL capture cannot tolerate mid-capture.
  # PIECEWISE-only capture already proven to pass (4/4) in that same run
  # since MoE routing executes as an eager segment between compiled pieces
  # there. Set COMPILATION_CONFIG_JSON='{"cudagraph_mode": "PIECEWISE"}' to
  # avoid the FULL-mode crash while still exercising real graph capture.
  if [ -n "${COMPILATION_CONFIG_JSON:-}" ]; then
    EXTRA_ARGS+=(--compilation-config "$COMPILATION_CONFIG_JSON")
  fi
  "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code \
    --max-model-len "$LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$PORT" \
    --host "$HOST" \
    "${EXTRA_ARGS[@]}" \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "server pid=$SERVER_PID, log=$SERVER_LOG"

  # ---- wait for readiness (or crash, or timeout) ----
  READY=0
  DEADLINE=$(( $(date +%s) + READY_TIMEOUT_SEC ))
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server process exited early (before becoming ready)"
      break
    fi
    if curl -sf --max-time 5 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep "$POLL_INTERVAL_SEC"
  done

  ELAPSED=$(( $(date +%s) - START_TS ))
  RESULT=""
  PROBE_RESULT="SKIPPED"
  KV_LOG_LINE=""
  NOTES=""

  if [ "$READY" -eq 1 ]; then
    echo "=== server ready after ${ELAPSED}s ==="
    # Give the KV-cache-size log lines a moment to land, then grab per-GPU
    # memory (this is the actual empirical number we're after) and the
    # engine's own reported KV cache stats.
    sleep 2
    POST_USED="$(gpu_mem_used_only)"
    POST_FREE="$(gpu_mem_free_only)"
    echo "post-ready per-GPU mem used/free (MiB): used=[$POST_USED] free=[$POST_FREE]"
    KV_LOG_LINE=$(grep -Ei "GPU KV cache size|# GPU blocks|Maximum concurrency|kv cache" "$SERVER_LOG" | tail -5 | tr '\n' ' | ' || true)

    # Exercise the actual context: send one real request whose prompt
    # approaches max_model_len tokens, to probe the "activations + CUDA
    # graph capture + MTP overhead + concurrency" slack empirically instead
    # of relying only on the startup memory-profiling reservation.
    if [ -f "$PROBE_SCRIPT" ]; then
      echo "=== running long-context probe request (target ~90% of $LEN tokens) ==="
      if "$PYTHON_BIN" "$PROBE_SCRIPT" \
          --host "$HOST" --port "$PORT" \
          --served-model-name "$SERVED_MODEL_NAME" \
          --target-context-tokens "$LEN" \
          --fraction 0.9 \
          --max-tokens 8 \
          > "$STEP_DIR/probe.log" 2>&1; then
        PROBE_RESULT="PASS"
      else
        PROBE_RESULT="FAIL (see $STEP_DIR/probe.log)"
      fi
      # Re-snapshot memory after the probe request: this is the number that
      # actually reflects activation/CUDA-graph/decode-time slack at this
      # context length, not just the startup KV-cache reservation.
      POST_PROBE_USED="$(gpu_mem_used_only)"
      POST_PROBE_FREE="$(gpu_mem_free_only)"
      echo "post-probe per-GPU mem used/free (MiB): used=[$POST_PROBE_USED] free=[$POST_PROBE_FREE]"
      POST_USED="$POST_PROBE_USED"
      POST_FREE="$POST_PROBE_FREE"
    else
      echo "WARNING: probe script not found at $PROBE_SCRIPT, skipping request-level test" >&2
      NOTES="probe_script_missing"
    fi

    RESULT="PASS"
  else
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      RESULT="FAIL_TIMEOUT"
      NOTES="server still running but never became ready within ${READY_TIMEOUT_SEC}s"
    else
      RESULT=$(classify_failure "$SERVER_LOG")
      NOTES="server exited before ready (see $SERVER_LOG); tail: $(tail -c 300 "$SERVER_LOG" | tr '\n' ' ')"
    fi
    POST_USED="$(gpu_mem_used_only)"
    POST_FREE="$(gpu_mem_free_only)"
    echo "=== STEP FAILED: $RESULT ($NOTES) ==="
  fi

  # ---- tear down the server before the next step ----
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "=== stopping server (pid $SERVER_PID) ==="
    kill -TERM "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server did not exit on SIGTERM, sending SIGKILL"
      kill -KILL "$SERVER_PID" 2>/dev/null
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""

  wait_for_gpus_clear || NOTES="${NOTES:+$NOTES; }gpus_did_not_fully_clear_before_next_step"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$LEN" "$RESULT" "$ELAPSED" "$POST_USED" "$POST_FREE" "$KV_LOG_LINE" "$PROBE_RESULT" "$NOTES" \
    >> "$SUMMARY_FILE"

  echo "=== STEP DONE: max_model_len=$LEN result=$RESULT ==="
done

echo ""
echo "=== SWEEP COMPLETE. Summary: $SUMMARY_FILE ==="
column -t -s $'\t' "$SUMMARY_FILE" || cat "$SUMMARY_FILE"
