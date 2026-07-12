#!/usr/bin/env bash
# DECODE-K GPU batch 3: post-election-fix validation.
set -uo pipefail
WT=/home/jarrelscy/glm52/vllm-decodeK
PRISTINE=/home/jarrelscy/glm52/vllm-sm120tests
CACHE=/home/jarrelscy/.cache/glm-dk-ext

run() {
  local envs=()
  while [ "$1" != "--" ]; do envs+=(-e "$1"); shift; done
  shift
  docker run --rm --gpus '"device=0"' --entrypoint /bin/bash --name glm-dk-b \
    -v "$WT:/work" -v /home/jarrelscy/glm52/vllm:/shipped:ro \
    -v "$CACHE:/root/.cache/torch_extensions" \
    "${envs[@]}" glm52-sm120:latest -c \
    "source /opt/vllm/.venv/bin/activate && python /work/kbench/bench_gemv.py $*"
}

echo "=================== A. BIT-EXACT RE-CHECKS (post election fix) ==================="
for envs in "GLM_MOE_DEDUP=1" "GLM_MOE_DEDUP=2" "GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1" "GLM_MOE_DEDUP=2 GLM_MOE_LANE_ROWS=1"; do
  for mix in prod dup realdup; do
    echo "--- mix=$mix env=[$envs]"
    # shellcheck disable=SC2086
    run $envs -- --tokens 4 --mix "$mix" --check --iters 30 2>&1 | grep -E "CHECK|NOT|error" | head -3
  done
done
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 -- --tokens 8 --mix dup --check --iters 30 2>&1 | sed -n "s/^/S64: /p" | grep -E "CHECK|NOT" | head -3

echo "=================== B. PERF (tokens=4) ==================="
for envs in "GLM_MOE_DEDUP=1" "GLM_MOE_DEDUP=2" "GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1" "GLM_MOE_DEDUP=2 GLM_MOE_LANE_ROWS=1"; do
  for mix in prod realdup; do
    echo "--- env=[$envs] mix=$mix"
    # shellcheck disable=SC2086
    run $envs -- --tokens 4 --mix "$mix" 2>&1 | grep -E "w13 |w2 |slots"
  done
done
echo "--- stacked winner + LUT256, prod + realdup + tokens=1"
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 KB_CFLAGS=-DNVFP4_LUT256=1 -- --tokens 4 --mix prod 2>&1 | grep -E "w13 |w2 "
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 KB_CFLAGS=-DNVFP4_LUT256=1 -- --tokens 4 --mix realdup 2>&1 | grep -E "w13 |w2 |slots"
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 KB_CFLAGS=-DNVFP4_LUT256=1 -- --tokens 1 --mix prod 2>&1 | grep -E "w13 |w2 "

echo "=================== C. PRISTINE suite: fma calibration + kv_write (pre-existing?) ==================="
cd "$PRISTINE/tests/sm120_correctness" && ./run_all.sh --tier kernel --no-lease -k "test_hybrid_gemv_bitexact and all_aqlm and S1 and b1 or padded_slot_skipped" 2>&1 | grep -E "PASSED|FAILED|ERROR|passed|failed|error" | tail -6

echo "=================== D. variant equivalence rerun on FIXED kernel ==================="
cd "$WT/tests/sm120_correctness" && ./run_all.sh --tier kernel --no-lease -k "variant" 2>&1 | grep -E "passed|failed|error" | tail -3
