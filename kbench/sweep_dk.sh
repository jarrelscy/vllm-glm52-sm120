#!/usr/bin/env bash
# DECODE-K batch-scaling sweep on SM100 (B200). One build; A/B via runtime env.
# For each token count x flag-combo, print w13/w2 us. tokens*8 = MoE slots.
cd /data/vllm-glm52-sm120; source .venv/bin/activate 2>/dev/null
export CUDA_HOME=/data/vllm-glm52-sm120/.venv/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="10.0a"
MIX="${MIX:-prod}"
ITERS="${ITERS:-100}"
run() { # $1 label ; env DEDUP/LANE already set
  local out us13 us2
  out=$(.venv/bin/python kbench/bench_gemv.py --tokens "$T" --shape both --mix "$MIX" --iters "$ITERS" 2>/dev/null)
  us13=$(echo "$out" | awk '/w13 /{print $2}')
  us2=$(echo  "$out" | awk '/w2  /{print $2}')
  printf "  %-14s w13=%8s us   w2=%8s us\n" "$1" "$us13" "$us2"
}
for T in 4 16 64 128 256 384; do
  echo "=== tokens=$T  (slots=$((T*8)))  mix=$MIX ==="
  GLM_MOE_DEDUP=0 GLM_MOE_LANE_ROWS=0 run "off"
  GLM_MOE_DEDUP=0 GLM_MOE_LANE_ROWS=1 run "lane"
  GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=0 run "dedup1"
  GLM_MOE_DEDUP=1 GLM_MOE_LANE_ROWS=1 run "dedup1+lane"
done
echo "DK_SWEEP_DONE"
