#!/usr/bin/env bash
# DECODE-K GPU batch 2: V3 correctness + microbench perf + tier1 + ncu deltas.
# Run ONLY under the decode-k lease. Uses GPU 0 (tiny JIT ctx).
set -uo pipefail
WT=/home/jarrelscy/glm52/vllm-decodeK
CACHE=/home/jarrelscy/.cache/glm-dk-ext

run() {  # run bench in container: run <envflags...> -- <args...>
  local envs=()
  while [ "$1" != "--" ]; do envs+=(-e "$1"); shift; done
  shift
  docker run --rm --gpus '"device=0"' --entrypoint /bin/bash --name glm-dk-b \
    -v "$WT:/work" -v /home/jarrelscy/glm52/vllm:/shipped:ro -v "$CACHE:/root/.cache/torch_extensions" \
    "${envs[@]}" glm52-sm120:latest -c \
    "source /opt/vllm/.venv/bin/activate && python /work/kbench/bench_gemv.py $*"
}

echo "=================== A. BIT-EXACT CHECKS vs shipped ==================="
for mix in prod aqlm nv dup realdup; do
  for envs in "GLM_MOE_DEDUP=1" "GLM_MOE_LANE_ROWS=1" "GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1"; do
    echo "--- mix=$mix env=[$envs]"
    # shellcheck disable=SC2086
    run $envs -- --tokens 4 --mix "$mix" --check --iters 50 2>&1 | grep -E "CHECK|NOT|Error|error" | head -4
  done
done
echo "--- dup mix, 8 tokens (S=64), both flags"
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 -- --tokens 8 --mix dup --check --iters 50 2>&1 | grep -E "CHECK|NOT|error" | head -4

echo "=================== B. PERF (tokens=4 unless noted) ==================="
echo "--- baseline (V2)"
run NOOP=0 -- --tokens 4 --mix prod 2>&1 | grep -E "w13 |w2 "
run NOOP=0 -- --tokens 4 --mix realdup 2>&1 | grep -E "w13 |w2 |slots"
echo "--- DEDUP only"
run GLM_MOE_DEDUP=1 -- --tokens 4 --mix prod 2>&1 | grep -E "w13 |w2 "
run GLM_MOE_DEDUP=1 -- --tokens 4 --mix realdup 2>&1 | grep -E "w13 |w2 |slots"
echo "--- LANE_ROWS only"
run GLM_MOE_LANE_ROWS=1 -- --tokens 4 --mix prod 2>&1 | grep -E "w13 |w2 "
echo "--- BOTH"
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 -- --tokens 4 --mix prod 2>&1 | grep -E "w13 |w2 "
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 -- --tokens 4 --mix realdup 2>&1 | grep -E "w13 |w2 |slots"
echo "--- BOTH + LUT256 (KB_CFLAGS)"
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 KB_CFLAGS=-DNVFP4_LUT256=1 -- --tokens 4 --mix prod 2>&1 | grep -E "w13 |w2 "
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 KB_CFLAGS=-DNVFP4_LUT256=1 -- --tokens 4 --mix realdup 2>&1 | grep -E "w13 |w2 |slots"
echo "--- BOTH + LUT256, tokens=1 (non-verify decode step shape)"
run GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 KB_CFLAGS=-DNVFP4_LUT256=1 -- --tokens 1 --mix prod 2>&1 | grep -E "w13 |w2 "

echo "=================== C. ncu delta (LANE_ROWS on w2, DEDUP on w13) ==================="
docker run --rm --gpus '"device=0"' --cap-add SYS_ADMIN --entrypoint /bin/bash --name glm-dk-b \
  -v "$WT:/work" -v /home/jarrelscy/glm52/vllm:/shipped:ro -v "$CACHE:/root/.cache/torch_extensions" \
  -e GLM_MOE_DEDUP=1 -e GLM_MOE_LANE_ROWS=1 glm52-sm120:latest -c '
source /opt/vllm/.venv/bin/activate
NCU=/opt/nvidia/nsight-compute/2025.2.1/ncu
echo "### V3 both flags, realdup w13"
$NCU -k regex:HybridMatVecMoEV3 -c 1 --section SpeedOfLight --section Occupancy python /work/kbench/bench_gemv.py --tokens 4 --mix realdup --shape w13 --prof 2>/dev/null | grep -E "Throughput|Duration|Achieved Occ|Registers"
echo "### V3 both flags, realdup w2"
$NCU -k regex:HybridMatVecMoEV3 -c 1 --section SpeedOfLight --section Occupancy python /work/kbench/bench_gemv.py --tokens 4 --mix realdup --shape w2 --prof 2>/dev/null | grep -E "Throughput|Duration|Achieved Occ|Registers"'

echo "=================== D. tier1 kernel suite (full matrix) ==================="
cd "$WT/tests/sm120_correctness" && ./run_all.sh --tier kernel --no-lease 2>&1 | tail -25
