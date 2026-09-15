// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Rebuild the corrected dispatcher when the image retains its base _C binary.
#define persistent_topk arvq_persistent_topk_impl
#include "topk.cu"
#undef persistent_topk

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(arvq_indexer, ops) {
  ops.def(
      "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(arvq_indexer, CUDA, ops) {
  ops.impl("persistent_topk", TORCH_BOX(&arvq_persistent_topk_impl));
}
