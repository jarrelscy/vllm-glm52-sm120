#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
arvq_topk_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
arvq_repo_root="$(cd -- "$arvq_topk_dir/../../../../.." && pwd)"
arvq_python="${ARVQ_PYTHON:-$arvq_repo_root/.venv/bin/python}"
arvq_torch_dir="$("$arvq_python" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"
"${NVCC:-nvcc}" -DUSE_CUDA -DTORCH_TARGET_VERSION=0x020B000000000000ULL \
  -O3 -std=c++17 -shared -Xcompiler=-fPIC \
  -gencode arch=compute_120a,code=sm_120a \
  -I"$arvq_repo_root/csrc/libtorch_stable" \
  -I"$arvq_torch_dir/include" -I"$arvq_torch_dir/include/torch/csrc/api/include" \
  -L"$arvq_torch_dir/lib" -ltorch_cpu -lc10 -ltorch_cuda -lc10_cuda \
  "$arvq_topk_dir/topk_ext.cu" -o "$arvq_topk_dir/topk.so"
