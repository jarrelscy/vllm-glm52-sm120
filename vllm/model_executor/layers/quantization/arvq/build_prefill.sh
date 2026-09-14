#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
arvq_prefill_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${NVCC:-nvcc}" -O3 -std=c++17 -shared -Xcompiler=-fPIC \
  -gencode arch=compute_120a,code=sm_120a \
  "$arvq_prefill_dir/prefill.cu" -o "$arvq_prefill_dir/prefill.so"

"${NVCC:-nvcc}" -O3 -std=c++17 -shared -Xcompiler=-fPIC \
  -gencode arch=compute_120a,code=sm_120a \
  "$arvq_prefill_dir/decode_gather.cu" -o "$arvq_prefill_dir/decode_gather.so"
