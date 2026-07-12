#!/usr/bin/env bash
# =============================================================================
# Golden generation for the SM120 correctness suite.
#
#   make_goldens.sh --cpu          reduction-tree golden (runs anywhere the
#                                  container runs; no GPU, no server)
#   make_goldens.sh --server       teacher-forced logits envelope (N=5) +
#                                  MTP acceptance envelope (N=3)
#   make_goldens.sh --final-gate   FINAL GATE 64K: 5x teacher-forced 64K
#                                  passes + 5x temp-0 generations (~1-2 h)
#   make_goldens.sh --canary       re-measure canary reference depths
#                                  (~30-45 min; only on a TRUSTED config)
#   make_goldens.sh --all          all of the above
#
# RULES:
#  * server-side goldens MUST be captured against the KNOWN-GOOD shipped
#    config (tp4-1m-mtp default boot) — never against a candidate change.
#  * capture requires the GPU lease (this script takes it for the server
#    parts: gpulease.sh acquire sm120-tests 240).
#  * commit the refreshed goldens/ files together with a note of the
#    server config they were captured on.
# =============================================================================
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SUITE_DIR/../.." && pwd)"
IMAGE="${GLM_TEST_IMAGE:-glm52-sm120:latest}"
LEASE_SH="${GPU_LEASE_SH:-/home/jarrelscy/glm52/coord/gpulease.sh}"

do_cpu=0; do_server=0; do_final=0; do_canary=0
for a in "$@"; do
  case "$a" in
    --cpu) do_cpu=1 ;;
    --server) do_server=1 ;;
    --final-gate) do_final=1 ;;
    --canary) do_canary=1 ;;
    --all) do_cpu=1; do_server=1; do_final=1; do_canary=1 ;;
    *) echo "usage: make_goldens.sh [--cpu|--server|--final-gate|--canary|--all]"; exit 2 ;;
  esac
done
[ $((do_cpu + do_server + do_final + do_canary)) -gt 0 ] || { echo "pick a target (--cpu/--server/--final-gate/--canary/--all)"; exit 2; }

if [ "$do_cpu" = 1 ]; then
  echo "== CPU golden: reduction_tree_golden.json =="
  gen() {
    cd "$1/tests/sm120_correctness"
    PYTHONPATH=. python - <<'EOF'
import json, pathlib, sys
sys.path.insert(0, ".")
sys.path.insert(0, "tier1_kernel")
from test_reduction_order import GOLDEN_PATH, compute_goldens
g = compute_goldens()
GOLDEN_PATH.parent.mkdir(exist_ok=True)
json.dump(g, open(GOLDEN_PATH, "w"), indent=1)
print(f"wrote {GOLDEN_PATH}: {len(g)} cases")
EOF
  }
  if [ -d /opt/vllm/.venv ]; then
    source /opt/vllm/.venv/bin/activate
    python -c "import pytest" 2>/dev/null || pip install -q pytest
    gen "$REPO_ROOT"
  else
    docker run --rm --entrypoint /bin/bash -v "$REPO_ROOT:/work" "$IMAGE" -c \
      "source /opt/vllm/.venv/bin/activate && \
       (python -c 'import pytest' 2>/dev/null || pip install -q pytest) && \
       cd /work/tests/sm120_correctness && \
       PYTHONPATH=.:tier1_kernel python -c '
import json, sys
sys.path[:0] = [\".\", \"tier1_kernel\"]
from test_reduction_order import GOLDEN_PATH, compute_goldens
g = compute_goldens()
json.dump(g, open(GOLDEN_PATH, \"w\"), indent=1)
print(\"wrote\", GOLDEN_PATH, len(g), \"cases\")'"
  fi
fi

server_part() {
  bash "$LEASE_SH" acquire sm120-tests 240 || true
  trap 'bash "$LEASE_SH" release sm120-tests || true' EXIT
  cd "$SUITE_DIR"
  if [ "$do_server" = 1 ]; then
    echo "== server goldens: teacher-forced envelope (N=5) =="
    GLM_SM120_TESTS=1 python3 tier3_server/test_teacher_forced_logits.py --capture-golden 5
    echo "== server goldens: acceptance envelope (N=3) =="
    GLM_SM120_TESTS=1 python3 tier3_server/test_mtp_acceptance_envelope.py --capture-golden 3
  fi
  if [ "$do_final" = 1 ]; then
    echo "== FINAL GATE 64K golden (N=5; ~1-2 h) =="
    GLM_SM120_TESTS=1 python3 tier4_canary/test_final_gate_64k.py --capture-golden 5
  fi
  if [ "$do_canary" = 1 ]; then
    echo "== canary reference depths (~30-45 min) =="
    GLM_SM120_TESTS=1 python3 tier4_canary/test_reasoning_canary.py --capture-golden
  fi
  bash "$LEASE_SH" release sm120-tests || true
  trap - EXIT
}

if [ $((do_server + do_final + do_canary)) -gt 0 ]; then
  # golden capture consumes hours of 4-GPU server time: double-gated like
  # the server/canary test tiers.
  if [ "${GLM_SM120_SERVER_TESTS:-}" != "1" ]; then
    echo "ERROR: server-side golden capture is double-gated; set" >&2
    echo "GLM_SM120_SERVER_TESTS=1 explicitly (the main loop schedules" >&2
    echo "this — it requires pausing the prefill campaign)." >&2
    exit 1
  fi
  if ! curl -sf -o /dev/null --max-time 5 http://localhost:8001/v1/models \
       -H "Authorization: Bearer ${VLLM_API_KEY:-$(grep '^VLLM_API_KEY=' /home/jarrelscy/homeassistant/.env 2>/dev/null | cut -d= -f2-)}"; then
    echo "ERROR: no live server on :8001 — boot the KNOWN-GOOD shipped"
    echo "config (tp4-1m-mtp) first; goldens must never be captured from"
    echo "a candidate change." >&2
    exit 1
  fi
  server_part
fi
echo "done."
