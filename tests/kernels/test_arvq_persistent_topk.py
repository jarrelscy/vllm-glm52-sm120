# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the packaged selector after the retained core initializes CUDA caches."""

from pathlib import Path

import pytest
import torch

from vllm import _custom_ops  # noqa: F401
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@torch.inference_mode()
def test_packaged_topk_after_core_initialization():
    from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid
    from vllm.v1.attention.ops.arvq_persistent_topk import persistent_topk

    library = Path(nvfp4_arvq_hybrid.__file__).with_name("arvq") / "topk.so"
    if not library.is_file():
        pytest.skip("Build the SM120 packaged selector with arvq/build_topk.sh")
    if not current_platform.is_device_capability_family(120):
        pytest.skip("Packaged kernel targets SM120")
    logits = torch.randn((1, 1024), device="cuda", dtype=torch.float32)
    lengths = torch.tensor([1024], device="cuda", dtype=torch.int32)
    output = torch.empty((1, 2048), device="cuda", dtype=torch.int32)
    workspace = torch.empty(2**20, device="cuda", dtype=torch.uint8)
    # Reproduces a CUDA 13 core / CUDA 12.9 extension device-property cache
    # collision. Testing the packaged operator alone did not expose the fault.
    torch.ops._C.persistent_topk(logits, lengths, output, workspace, 2048, 1024)
    persistent_topk(logits, lengths, output, workspace, 2048, 1024)
    torch.testing.assert_close(
        output[0, :1024].sort().values,
        torch.arange(1024, device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    assert (output[0, 1024:] == -1).all()
