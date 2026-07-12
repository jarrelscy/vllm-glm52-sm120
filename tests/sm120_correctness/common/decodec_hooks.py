# SPDX-License-Identifier: Apache-2.0
"""Compare hooks for the decode-C comms variants (kernel_variants.json).

pcie_custom_allreduce: exercises the VLLM_PCIE_CUSTOM_AR un-gated custom
(CUDA IPC P2P) allreduce on 4x PCIe SM120 against NCCL.  Only meaningful
under torchrun (tier2_dist/test_allreduce_equivalence.py, which passes
``{"x": tensor, "group": WORLD}`` and checks the result against the exact
fp64 sum with the fp-reorder bound).  In the single-GPU tier1 sweep (which
calls every python_env hook with a MoE gemv case) the hook returns a
trivially-equal pair — a comms variant has no single-GPU content.
"""

import os

_CA = None  # cached CustomAllreduce (one IPC buffer set per process)


def pcie_custom_allreduce(case, device):
    import numpy as np

    try:
        import torch
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        z = np.zeros(1, np.float32)
        return z, z

    if (
        not isinstance(case, dict)
        or "x" not in case
        or not dist.is_initialized()
        or dist.get_world_size() < 2
    ):
        # tier1 single-GPU context: nothing to compare
        z = np.zeros(1, np.float32)
        return z, z

    x = case["x"]
    ref = x.clone()
    dist.all_reduce(ref)  # NCCL reference

    global _CA
    if _CA is None:
        os.environ["VLLM_PCIE_CUSTOM_AR"] = "1"
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        gloo = dist.new_group(backend="gloo")
        _CA = CustomAllreduce(group=gloo, device=torch.device(device))
    assert not _CA.disabled, "VLLM_PCIE_CUSTOM_AR custom allreduce failed to enable"
    out = _CA.custom_all_reduce(x.contiguous())
    assert out is not None, "custom allreduce declined the message"
    torch.cuda.synchronize()
    return ref.float().cpu().numpy(), out.float().cpu().numpy()
