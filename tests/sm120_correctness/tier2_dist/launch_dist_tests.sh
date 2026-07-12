#!/usr/bin/env bash
# Launch the multi-GPU (torchrun) parts of tier2_dist inside the
# glm52-sm120 container.  Needs ALL GPUs idle: acquire the GPU lease
# first (run_all.sh --tier unit does both).
#
# Usage: launch_dist_tests.sh [nproc (default 4)] [extra pytest args...]
set -euo pipefail
NPROC="${1:-4}"; shift || true
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SUITE_DIR/../.." && pwd)"
IMAGE="${GLM_TEST_IMAGE:-glm52-sm120:latest}"

if [ -d /opt/vllm/.venv ]; then
  # already inside the container
  source /opt/vllm/.venv/bin/activate
  cd "$SUITE_DIR"
  GLM_SM120_TESTS=1 GLM_SM120_DIST_TESTS=1 PYTHONPATH="$REPO_ROOT:$SUITE_DIR" \
    torchrun --standalone --nproc-per-node "$NPROC" \
    -m pytest tier2_dist/ -v "$@"
else
  exec docker run --rm --gpus all --shm-size 16g --entrypoint /bin/bash \
    -v "$REPO_ROOT:/work" \
    -e GLM_SM120_TESTS=1 -e GLM_SM120_DIST_TESTS=1 \
    "$IMAGE" -c "
      source /opt/vllm/.venv/bin/activate &&
      python -m pytest --version >/dev/null 2>&1 || (uv pip install -q pytest 2>/dev/null || python -m pip install -q pytest) &&
      cd /work/tests/sm120_correctness &&
      PYTHONPATH=/work:/work/tests/sm120_correctness \
      torchrun --standalone --nproc-per-node $NPROC -m pytest tier2_dist/ -v $*"
fi
