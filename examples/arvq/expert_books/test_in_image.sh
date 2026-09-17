#!/usr/bin/env bash
set -euo pipefail
# Run inside an isolated serving-image container with this checkout at /work.
q=vllm/model_executor/layers/quantization
for file in nvfp4_arvq_hybrid.py nvfp4_arvq_prefill.py arvq_reference.py; do
  cp "/work/$q/$file" "/opt/vllm/$q/$file"
done
cp /work/$q/arvq/{hybrid,prefill,activation_pack,decode_gather}.so /opt/vllm/$q/arvq/
cd /opt/vllm
exec .venv/bin/python -m pytest --import-mode=importlib --confcutdir=/work/tests/quantization "$@"
