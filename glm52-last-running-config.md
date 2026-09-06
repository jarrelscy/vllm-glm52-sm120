# GLM-5.2 hybrid — last known-good production serve config

Reconstructed from the (now-defunct) "Host GLM-5.2" companion session transcript
(`e2e54646`): the live-engine config report, the EngineCore boot logs, and the
venv/toolchain probes captured in that session.

**Caveats**
- Read from the session transcript/logs — the box is defunct now, so its live
  state may differ.
- The vLLM "commit" `2fafcf0aa` is the **squashed fork** commit, not an upstream
  vLLM SHA. The reproducible source of truth is the fork itself:
  `jarrelscy/vllm-glm52-sm120` @ branch `glm52-sm120`.

## Model

- **Path:** `/data/glm52-1m` (GLM-5.2 NVFP4 + AQLM hybrid)
- **Served name:** `GLM-5.2-NVFP4-AQLM-hybrid`
- **Endpoint:** `0.0.0.0:8000`
- **HF repo (reproducer):** `jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid` (272 GB / 77 shards)

## Version stack

| Component | Version |
|---|---|
| vLLM | `v0.1.dev1+g2fafcf0aa` — fork `jarrelscy/vllm-glm52-sm120`, branch `glm52-sm120` (single squashed commit `2fafcf0aa`), with local rebuild of the sparse-MLA extension |
| nvcc / CUDA toolkit | release 13.0, V13.0.88 (`Build cuda_13.0.r13.0/compiler.36424714_0`), from `.venv/…/nvidia/cu13/bin/nvcc` |
| PyTorch | `2.11.0+cu130` |
| Python | 3.12 |
| GPU arch | `sm_100` / `10.0f` (B200), auto-detected |

## Serve config

| Setting | Value |
|---|---|
| Tensor parallel | TP8 (all 8 B200s) |
| DCP | none (`decode_context_parallel_size=1`) |
| KV cache dtype | `fp8_ds_mla` (packed MLA, 656 B/tok/layer) |
| Max context | 1,048,576 (1M) |
| Speculative decode | none (`speculative_config=None` — no-MTP) |
| CUDA graphs | `PIECEWISE` (compilation mode 3) |
| max-num-seqs | 256 |
| max-num-batched-tokens | 16,384 |
| gpu-memory-utilization | 0.90 |
| custom all-reduce | disabled (`--disable-custom-all-reduce`) |
| tool calling | on (`glm47` parser) |

## Key env / kernel flags

- `GLM_MOE_LANE_ROWS=1` — DECODE-K lane-rows gemv (bit-exact, ~20% w2 win)
- `GLM_MOE_DEDUP` **unset (off)** — pathological on SM100
- `VLLM_DISABLE_FP8_W8A16=1` (v2 bit-exact)
- `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=1048576`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`
- `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1024`
- `FI_AR_MAX_MB=32`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- NCCL unthrottled (full NVLink)
- Includes **Fix C** (sparse-indexer graph-padding reshape) and the FlashMLA
  read-side fix (Fix A/A2)

## Derived capacity

- KV pool: 2,044,672 tokens (~1.95× concurrency at full 1M context)
- VRAM: ~168.9 / 183.4 GB per GPU (92%)

## Launch

- Production: `serve_b200_tp8.sh` via `restart_prod.sh` (watchdog recovery script,
  relaunches this exact config).
- Reproducer: `~/git/glm52/start.sh` — self-contained, idempotent bootstrap
  (clone fork → build venv → download weights → load/gen API key → launch),
  verified an argv-exact match to the live production server.
- This was the **no-MTP** variant (switched from the MTP config for better
  multi-stream Terminal-Bench throughput).
