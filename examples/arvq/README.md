# GLM-5.3 NVFP4–ARVQ on SM120

This branch serves a serialized hybrid checkpoint: hot experts retain NVFP4,
and cold experts use additive FP4 vector codebooks. Both hybrid branches use
four residual activation planes in the native FP4 MMA path. Prefill and decode
use the same quantized weights.

The initial checkpoint is an **untuned transcode of existing AQLM weights**.
It is intended for serving and kernel experiments. Its numerical kernel checks
do not establish model accuracy; conversion adds substantial approximation.

## Build and serve

Use the GLM vision image built by this fork's `Dockerfile.glm52-sm120` as the
base. The derived image overlays this branch's complete Python package and
compiles the ARVQ CUDA library. It requires four SM120 GPUs for the supplied
GLM profile and CUDA 12.9 or newer to compile the kernel.

```bash
docker build -f Dockerfile.arvq -t glm53-arvq-sm120:local .
ARVQ_MODEL_DIR=/data/models/jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid \
  docker compose -f examples/arvq/compose.yaml up -d
```

The API listens on port 8001 and serves
`jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid`. The checkpoint's `arvq` metadata
automatically selects the loader; no activation flag or on-load transcode is
required. The supplied profile enables TP4, DCP4, native MTP with three draft
tokens, and full CUDA graphs. Set `PARALLEL=tp4-1m` to disable drafting.

Conversion tools, frozen-fit reuse, and checkpoint auditing are described in
[convert/README.md](convert/README.md). The format and loader details are in
[the kernel documentation](../../vllm/model_executor/layers/quantization/arvq/README.md).

## Measure

Use the same prompts, output limits, concurrency, and runtime settings for
original/new checkpoints with drafting off/on. This runner counts actual
completion tokens from streamed usage and token IDs, rather than SSE chunks.
It also records speculative counters and keeps the acceptance-normalized
estimate separate from measured throughput.

```bash
.venv/bin/python examples/arvq/bench_serving.py \
  --label new_mtp_on --prompt-lengths 128 4096 \
  --max-tokens 512 --runs 3 --ignore-eos \
  --output new_mtp_on.json
```

For normalization, add `--normalize-reference original_mtp_on.json`.
Authentication, if configured, is read from `OPENAI_API_KEY` and is never
serialized in the report.
