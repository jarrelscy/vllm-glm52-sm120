#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
arvq_source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${NVCC:-nvcc}" -O3 -std=c++17 -shared -Xcompiler=-fPIC \
  -gencode arch=compute_120a,code=sm_120a \
  "$arvq_source_dir/prefill_wide.cu" \
  "$arvq_source_dir/prefill_register.cu" -o "$arvq_source_dir/prefill_wide.so"
