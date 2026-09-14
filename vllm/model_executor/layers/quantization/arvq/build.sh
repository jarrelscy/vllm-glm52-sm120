#!/usr/bin/env bash
set -euo pipefail
arvq_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${NVCC:-nvcc}" -O3 -gencode arch=compute_120a,code=sm_120a --shared -Xcompiler=-fPIC "$arvq_dir/hybrid.cu" -o "$arvq_dir/hybrid.so"
