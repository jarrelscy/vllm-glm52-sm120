# Serving GLM-5.3 hybrid (NVFP4+AQLM) on 4x SM120 — deployment guide

This fork serves `jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m` (274 GB, MoE with
29.9% hot NVFP4 experts + AQLM cold experts, self-contained vision tower) at a
~950K-token context window with MTP speculative decoding, reaching ~84 tok/s
single-stream code decode and bit-deterministic temp-0 output on 4x NVIDIA RTX
PRO 6000 Blackwell (96 GB, SM120, PCIe Gen5, no NVLink).

Branch: `main` is the integration tip (all rounds merged; `v4-race-fix` is the
same history under its original working name). Historical per-lever branches:
`gemv-pipeline`, `dense-gemm`, `tail-fusion`, `tail-fusion-2`, `comm-overlap`,
`canonical-inkernel`, `round3-comm`, `round4-tail`, `round5-copies`,
`custom-ar-force` (rejected lever, kept for the record).

## Requirements

- 4 GPUs, SM120 (Blackwell), 96 GB each. The window/UTIL numbers below are
  calibrated for exactly this; smaller VRAM needs a smaller MAXLEN.
- Weights: `huggingface-cli download jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m`
  (274 GB). The older 750K-window checkpoint is
  `jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid` (289 GB, needs MAXLEN=750000).

## Build

```bash
git clone -b main https://github.com/jarrelscy/vllm-glm52-sm120
cd vllm-glm52-sm120
docker build -f Dockerfile.glm52-sm120 -t glm52-vision-sm120 .
```

The entrypoint (`docker-entrypoint.glm52.sh`) drives `vllm serve`; select the
parallelism mode with `PARALLEL=tp4-1m-mtp` (TP4 + DCP4 + MTP ns=3).

## Run

```bash
docker run --gpus all --ipc=host -p 8001:8001 \
  -v /data/huggingface:/data/huggingface:ro \
  -v /data/triton_cache:/data/triton_cache \
  -e MODEL_DIR=/data/huggingface/hub/models--jarrelscy--GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m/snapshots/<sha> \
  -e SERVED_NAME=glm-5.3 \
  -e PARALLEL=tp4-1m-mtp \
  -e MAXLEN=950000 -e UTIL=0.96 -e MAX_NUM_SEQS=4 \
  -e CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  -e TRITON_CACHE_DIR=/data/triton_cache \
  glm52-vision-sm120 \
  --trust-remote-code --reasoning-parser glm47 --enable-prefix-caching
```

`VLLM_API_KEY` (if you want auth) comes from the environment; never bake it in.

`MAX_NUM_SEQS=4` is the default (raised from 2 once the indexer off-by-one was
fixed). It leaves single-stream B=1 decode untouched (~42.2 ms/step) while
lifting peak aggregate throughput from ~106 to ~113 tok/s at 4 concurrent
streams — this box is memory-bandwidth-bound, so concurrency scales sub-linearly
(per-stream drops from ~50 to ~28 tok/s) but the aggregate still gains. The
decode CUDA graph for the full batch (4 seqs x next_n 4 = size 16) is in the
capture list, so there is no extra warmup cost.

## Performance/determinism flags — all gated lossless, recommended ON

Every flag defaults OFF in code so the fork stays a clean superset of upstream
behavior. This set is what the numbers above were measured with; each was gated
individually for bitwise losslessness on live serving.

```
# MoE decode gemv + tail (AQLM experts)
AQLM_GEMV_PIPELINE=1        # V4 cp.async-pipelined gemv
AQLM_GEMV_BF16IN=1          # bf16 activations straight into gemv
AQLM_FUSED_COMBINE=1        # single-kernel MoE combine (torch-order bit-exact)
AQLM_GLUE_OPT=1             # stacked-table gather
AQLM_GEMV_ROWMAP=1          # in-kernel row mapping, kills repeat_interleave
AQLM_FUSED_SILU=1           # bit-equal fused SiLU
# Router + comms
VLLM_SM120_ROUTER_GEMM=1    # dsv3_router_gemm on SM120 (vs fp32 SIMT fallback)
VLLM_GLM_COMM_OVERLAP=1     # DCP AG(q)/AG(idx) hidden behind index compute
# Determinism (temp-0 bit-determinism within a boot, ~zero cost)
VLLM_DSA_CANONICAL_TOPK=inkernel
# Step tail (round 4)
VLLM_GLM_IDX_FUSED_LOCALIZE=1  # DCP seq-len localization fused into decode kernel
VLLM_GLM_MM_MASK_REUSE=1       # persistent all-False mm mask on text-only steps
VLLM_GLM_EMBED_GRAPH=1         # CUDA-graphs the decode embed prologue (needs MM_MASK_REUSE)
# DCP attention-epilogue copies (round 5)
VLLM_GLM_DCP_RS_STAGED=1    # correction kernel writes straight into RS staging
VLLM_GLM_DCP_RS_VIEW=1      # consume RS output as a view (bmm stride-invariant)
VLLM_GLM_DCP_AG_RAW_TOPK=1  # top-k merge reads raw AG layout, no repack
```

Known-rejected on this hardware (kept in-tree, leave OFF):
`VLLM_FORCE_CUSTOM_ALLREDUCE` (−15-20% on PCIe), `VLLM_DCP_A2A_EXACT` (−8%
inside FULL cudagraphs despite winning eager microbenches), `LOCAL_ARGMAX`
(draft tokens not exact on prose), `VLLM_GLM_COMM_COALESCE` (neutral).

## Gate methodology (if you change kernels)

- Temp-0 outputs are deterministic only WITHIN a boot. Never compare outputs or
  tok/s across boots; compare ms/step (tok/s ÷ tok-per-step from the spec-decode
  acceptance metrics).
- Determinism probes must interleave varied predecessor requests (alternating
  prompts); straight same-prompt repeats always pass and prove nothing.
- Eager microbench wins do not transfer into FULL cudagraphs — measure in-graph
  (capture-replay harnesses in `tools/round5/`).
- Bit-exactness proofs belong at the kernel level, in-process (see
  `tools/round5/bitexact_*.py` for the pattern).

## Known issues

- First request after a cold boot is slow (~1-3 min) while torch.compile/Triton
  JIT finish; persist `TRITON_CACHE_DIR` to a volume and send a ~15K-token
  warmup request.
- UTIL above 0.96 can OOM in the fp32 MoE dequant transient under long
  prefills on this checkpoint. Keep 0.96. At `MAX_NUM_SEQS=4` the worst case
  (a >200K-token chunked prefill co-batched with 3 concurrent decodes) peaks
  at ~1.3 GiB free/GPU — tight but gated safe; do not raise UTIL past 0.96.
- The off-by-one crash in `mla/indexer.py _prepare_decode_tensors` at a length
  boundary (formerly ~once/10h under concurrent MTP, engine auto-restarted) is
  FIXED as of 91bb50936 — the variable-decode-lengths path now truncates the
  runner's manager-rounded block-table row (one kernel-block wider) to the
  indexer buffer width, which is provably lossless. This is what makes
  `MAX_NUM_SEQS=4` the default; no `MAX_NUM_SEQS=1` mitigation is needed.
