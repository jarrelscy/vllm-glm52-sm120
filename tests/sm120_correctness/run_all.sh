#!/usr/bin/env bash
# =============================================================================
# SM120 correctness suite runner — THE intended entry point.
#
#   run_all.sh --tier cpu      CPU-only checks (reference goldens, routing
#                              pins, registry schema). Anywhere; no GPU lease.
#   run_all.sh --tier kernel   Tier-1 GPU kernel tests (1 GPU, tiny JIT ctx).
#   run_all.sh --tier unit     Tier-2 distributed/comms tests (2-4 GPUs).
#   run_all.sh --tier server   Tier-3 server invariants (live :8001 server).
#   run_all.sh --tier canary   Tier-4 reasoning canary + FINAL GATE 64K + 1M.
#   run_all.sh --tier all      cpu + kernel + unit + server + canary.
#
# Options: --quick (smaller Tier-1 matrix)  -k EXPR (pytest -k filter)
#          --image IMG (default glm52-sm120:latest)  --no-lease
#
# The suite is OPT-IN: this script sets GLM_SM120_TESTS=1 itself (and
# GLM_SM120_SERVER_TESTS=1 only for server/canary). A plain `pytest` run
# in the repo NEVER executes these tests.
#
# GPU COORDINATION (mandatory on this box): GPU tiers acquire the mutex
# /home/jarrelscy/glm52/coord/gpulease.sh (name sm120-tests) before
# touching a GPU and release it after — kernel tier leases 15 min,
# unit 30, server/canary 240. Never kill another agent's containers.
# =============================================================================
set -uo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SUITE_DIR/../.." && pwd)"
IMAGE="${GLM_TEST_IMAGE:-glm52-sm120:latest}"
LEASE_SH="${GPU_LEASE_SH:-/home/jarrelscy/glm52/coord/gpulease.sh}"
LEASE_NAME="sm120-tests"
RESULTS="$SUITE_DIR/results"
mkdir -p "$RESULTS"

TIER="cpu"; KEXPR=""; QUICK=""; USE_LEASE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    -k) KEXPR="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --no-lease) USE_LEASE=0; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

banner() { printf '\n==== %s ====\n' "$*"; }
PASS_TIERS=(); FAIL_TIERS=()

lease_acquire() {  # $1 = numeric minutes
  [ "$USE_LEASE" = 1 ] && [ -x "$LEASE_SH" ] && bash "$LEASE_SH" acquire "$LEASE_NAME" "$1" || true
}
lease_release() {
  [ "$USE_LEASE" = 1 ] && [ -x "$LEASE_SH" ] && bash "$LEASE_SH" release "$LEASE_NAME" || true
}

in_container() { [ -d /opt/vllm/.venv ]; }

# Run pytest for a path, either directly (inside container) or via docker.
# $1 = tier label, $2 = gpu docker args, $3 = pytest target(s), $4 = extra env
run_pytest() {
  local label="$1" gpuargs="$2" target="$3" extra_env="$4"
  local junit="$RESULTS/junit_${label}.xml"
  local kargs=""
  [ -n "$KEXPR" ] && kargs="-k $KEXPR"
  if in_container; then
    ( source /opt/vllm/.venv/bin/activate
      python -m pytest --version >/dev/null 2>&1 || pip install -q pytest
      cd "$SUITE_DIR"
      env GLM_SM120_TESTS=1 $extra_env PYTHONPATH="$REPO_ROOT:$SUITE_DIR" \
        python -m pytest $target -v --junitxml="$junit" $kargs )
  else
    docker run --rm $gpuargs --shm-size 8g --entrypoint /bin/bash \
      -v "$REPO_ROOT:/work" \
      -v "${TORCH_EXT_CACHE:-$HOME/.cache/sm120_test_ext}:/root/.cache/torch_extensions" \
      -e GLM_SM120_TESTS=1 ${QUICK:+-e GLM_SM120_QUICK=1} \
      "$IMAGE" -c "
        source /opt/vllm/.venv/bin/activate
        python -m pytest --version >/dev/null 2>&1 || pip install -q pytest
        cd /work/tests/sm120_correctness
        env $extra_env PYTHONPATH=/work:/work/tests/sm120_correctness \
          python -m pytest $target -v \
          --junitxml=/work/tests/sm120_correctness/results/junit_${label}.xml \
          $kargs"
  fi
}

# stdlib-only host runner for server/canary modules (no pytest needed)
run_host_module() {  # $1 = module file
  ( cd "$SUITE_DIR" && \
    GLM_SM120_TESTS=1 GLM_SM120_SERVER_TESTS=1 python3 "$1" )
}

tier_cpu() {
  banner "TIER cpu — reference goldens / routing pins / registry (no GPU)"
  # CPU tests that need numpy/vllm run in the container without GPUs;
  # source tripwires also run there (repo mounted).
  # NB: the four routing pins live in tier3_server (conftest double-gates
  # that dir), but they are pure CPU checks — no server contact, no GPU
  # cost — so the runner un-gates exactly those node IDs here.
  run_pytest cpu "" \
    "tier1_kernel/test_reduction_order.py tier1_kernel/test_registry_cpu.py tier3_server/test_backend_routing.py::test_sm120_priority_list tier3_server/test_backend_routing.py::test_sm120_backend_class_contract tier3_server/test_backend_routing.py::test_impl_hard_fails_on_non_ds_mla_source_tripwire tier3_server/test_backend_routing.py::test_cuda_routing_block_source_tripwire" \
    "GLM_SM120_SERVER_TESTS=1"
}

tier_kernel() {
  banner "TIER kernel — Tier-1 GPU bit-exact (1 GPU, tiny JIT context)"
  lease_acquire 15
  local rc=0
  run_pytest kernel "--gpus all" "tier1_kernel/" \
    "CUDA_VISIBLE_DEVICES=${GLM_TEST_GPU:-$(pick_gpu_idx)}" || rc=$?
  lease_release
  return $rc
}

pick_gpu_idx() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' '
}

tier_unit() {
  banner "TIER unit — Tier-2 distributed/comms (single-GPU parts + torchrun)"
  lease_acquire 30
  local rc=0
  run_pytest unit_single "--gpus all" "tier2_dist/" \
    "CUDA_VISIBLE_DEVICES=${GLM_TEST_GPU:-$(pick_gpu_idx)}" || rc=$?
  # full-comms torchrun parts (need all GPUs idle)
  bash "$SUITE_DIR/tier2_dist/launch_dist_tests.sh" "${GLM_DIST_NPROC:-4}" || rc=$?
  lease_release
  return $rc
}

tier_server() {
  banner "TIER server — Tier-3 invariants against the live :8001 server"
  lease_acquire 240
  local rc=0
  for m in tier3_server/test_backend_routing.py \
           tier3_server/test_teacher_forced_logits.py \
           tier3_server/test_needle_depth.py \
           tier3_server/test_mtp_acceptance_envelope.py \
           tier3_server/test_graph_capture_parity.py; do
    run_host_module "$m" || rc=$?
  done
  lease_release
  return $rc
}

tier_canary() {
  banner "TIER canary — FINAL GATE 64K + 1M capability + reasoning canary"
  echo "(the mandatory promotion gate; a FAIL here = STOP THE LINE)"
  lease_acquire 240
  local rc=0
  for m in tier4_canary/test_1m_capability.py \
           tier4_canary/test_final_gate_64k.py \
           tier4_canary/test_reasoning_canary.py; do
    run_host_module "$m" || rc=$?
  done
  lease_release
  return $rc
}

run_tier() {
  local t="$1" rc=0
  case "$t" in
    cpu) tier_cpu || rc=$? ;;
    kernel) tier_kernel || rc=$? ;;
    unit) tier_unit || rc=$? ;;
    server) tier_server || rc=$? ;;
    canary) tier_canary || rc=$? ;;
    *) echo "unknown tier: $t"; return 2 ;;
  esac
  if [ $rc -eq 0 ]; then
    PASS_TIERS+=("$t"); printf '\n#### TIER %-6s : PASS ####\n' "$t"
  else
    FAIL_TIERS+=("$t"); printf '\n#### TIER %-6s : FAIL ####\n' "$t"
  fi
  return 0
}

case "$TIER" in
  all) for t in cpu kernel unit server canary; do run_tier "$t"; done ;;
  *)   run_tier "$TIER" ;;
esac

banner "SUMMARY"
[ ${#PASS_TIERS[@]} -gt 0 ] && echo "PASS: ${PASS_TIERS[*]}"
if [ ${#FAIL_TIERS[@]} -gt 0 ]; then
  echo "FAIL: ${FAIL_TIERS[*]}"
  echo "(junit XMLs in $RESULTS)"
  exit 1
fi
echo "all requested tiers passed (junit XMLs in $RESULTS)"
