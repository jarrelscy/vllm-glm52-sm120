#!/usr/bin/env bash
# DECODE-K batch 4: SERVER A/B (control = shipped defaults; candidate =
# GLM_MOE_LANE_ROWS=1 + GLM_NVFP4_LUT256=1) + lossless gates.
# Control and candidate both run tp4-1m-mtp PIECEWISE graphs, no IndexShare
# (shipped defaults) for clean attribution. Run ONLY under the decode-k lease.
set -uo pipefail
WT=/home/jarrelscy/glm52/vllm-decodeK
BOOK=/home/jarrelscy/glm52/mtp_cmp/book.txt
SUITE=$WT/tests/sm120_correctness
cd /home/jarrelscy/glm52

bench_all() {  # $1 = tag
  echo "### [$1] WARMUP pass (discarded; ~10% first-requests-after-boot penalty)"
  python3 /home/jarrelscy/glm52/vllm/bench.py --ngen 120 2>&1 | grep -E "DECODE|coherent" | sed 's/^/  warm /'
  for rep in 1 2 3; do
    echo "### [$1] 32K-short rep$rep"
    python3 /home/jarrelscy/glm52/vllm/bench.py --ngen 240 2>&1 | grep -E "DECODE|coherent"
  done
  echo "### [$1] 123K rep1 (includes prefill warm)"
  python3 /home/jarrelscy/glm52/vllm/bench.py --ngen 240 --ctx 123000 --docfile $BOOK 2>&1 | grep -E "DECODE|coherent|long-context"
  echo "### [$1] 123K rep2"
  python3 /home/jarrelscy/glm52/vllm/bench.py --ngen 240 --ctx 123000 --docfile $BOOK 2>&1 | grep -E "DECODE|coherent"
}

echo "=================== 0. microbench missing cell: LR+LUT (no dedup) ==================="
docker run --rm --gpus '"device=0"' --entrypoint /bin/bash --name glm-dk-b \
  -v "$WT:/work" -v /home/jarrelscy/.cache/glm-dk-ext:/root/.cache/torch_extensions \
  -e GLM_MOE_LANE_ROWS=1 -e KB_CFLAGS=-DNVFP4_LUT256=1 glm52-sm120:latest -c \
  'source /opt/vllm/.venv/bin/activate && python /work/kbench/bench_gemv.py --tokens 4 --mix prod && python /work/kbench/bench_gemv.py --tokens 1 --mix prod' 2>&1 | grep -E "w13 |w2 |slots"

echo "=================== 1. CONTROL boot (shipped defaults) ==================="
bash $WT/kbench/serve_dk.sh || exit 1
sleep 10
bench_all CONTROL
docker logs glm-dk 2>&1 | grep -iE "aqlm_moe_v2. DECODE-K|Building aqlm" | tail -3
echo "acceptance (control, from log):"; docker logs glm-dk 2>&1 | grep -i "acceptance" | tail -2

echo "=================== 2. CANDIDATE boot (LANE_ROWS + LUT256) ==================="
bash $WT/kbench/serve_dk.sh -e GLM_MOE_LANE_ROWS=1 -e GLM_NVFP4_LUT256=1 || exit 1
sleep 10
bench_all CANDIDATE
echo "--- proof of activation:"
docker logs glm-dk 2>&1 | grep -iE "aqlm_moe_v2. DECODE-K|Building aqlm|lut256" | tail -4
echo "acceptance (candidate, from log):"; docker logs glm-dk 2>&1 | grep -i "acceptance" | tail -2

echo "=================== 3. GATES on candidate ==================="
export GLM_SM120_TESTS=1 GLM_SM120_SERVER_TESTS=1 PYTHONPATH=$SUITE
echo "--- MTP acceptance envelope"
python3 $SUITE/tier3_server/test_mtp_acceptance_envelope.py; echo "envelope rc=$?"
echo "--- FINAL GATE 64K (~20-30 min)"
python3 $SUITE/tier4_canary/test_final_gate_64k.py; echo "final64k rc=$?"
echo "--- 1M capability"
python3 $SUITE/tier4_canary/test_1m_capability.py; echo "cap1m rc=$?"

echo "=================== 4. teardown ==================="
docker rm -f glm-dk
echo BATCH4-DONE
