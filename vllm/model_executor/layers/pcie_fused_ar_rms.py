# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-wired fused all-reduce + add-residual + RMSNorm via b12x PCIe one-shot.

Lever #2 of the b12x PCIe graft (2026-09-13). The b12x fork fuses
all_reduce -> fused_add_rms_norm pairs with an inductor IR pass; that pass is
fork-specific, so here the runtime API is called directly from the GLM
(GlmMoeDsa / DeepseekV2) decoder-layer call sites, mirroring the in-tree
DSV4.1 precedent (vllm/models/deepseek_v32/nvidia/fused_ops.py) but wrapped
in a torch custom op because DeepseekV2Model runs under torch.compile.

Numerics: the b12x fused kernel accumulates in fp32 — treat as
lossless-by-gate (ms/step + coherence + needle + acceptance), NOT bit-exact.
The fallback path (comm unavailable, size above the one-shot ceiling, or any
geometry check failing) is the exact stock sequence:
tensor_model_parallel_all_reduce + ir.ops.fused_add_rms_norm.

Everything is inert unless BOTH VLLM_ENABLE_PCIE_ALLREDUCE=1 and
VLLM_GLM_PCIE_FUSED_AR_RMS=1.
"""

import torch

import vllm.envs as envs
from vllm import ir
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.b12x_pcie_all_reduce import _parse_byte_size
from vllm.distributed.utils import is_weak_contiguous
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def pcie_fused_ar_rms_enabled() -> bool:
    return bool(
        envs.VLLM_ENABLE_PCIE_ALLREDUCE
        and envs.VLLM_PCIE_ALLREDUCE_BACKEND == "b12x"
        and envs.VLLM_GLM_PCIE_FUSED_AR_RMS
    )


def _run_b12x_fused(
    comm,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    out: torch.Tensor,
) -> bool:
    """Run the b12x fused AR+add+rmsnorm if every gate accepts the input.

    Mirrors B12xPcieAllReduce.try_fused_add_rms_norm validation, with one
    difference: the normed output goes to a separate ``out`` buffer while the
    new residual is written in place into ``residual`` (declared mutation of
    the wrapping custom op).
    """
    if (
        not comm.supports_fused_add_rms_norm()
        or x.nbytes > comm.fused_max_bytes
        or x.ndim == 0
        or residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
        or not is_weak_contiguous(residual)
        or weight.shape != (x.shape[-1],)
        or weight.dtype != x.dtype
        or weight.device != x.device
        or not weight.is_contiguous()
        or x.shape[-1] * x.element_size() % 16 != 0
        or x.data_ptr() == residual.data_ptr()
        or epsilon < 0
    ):
        return False

    runtime = comm._runtime
    if runtime is None:
        return False
    stream = comm._runtime_stream()
    if not runtime.for_stream(stream).should_allreduce(x):
        return False
    if comm._is_capturing and not torch.cuda.is_current_stream_capturing():
        # Graph warmup phase: pre-plan the graph-safe channel, then run
        # eagerly (same contract as the adapter's try_fused_add_rms_norm).
        prepare = getattr(runtime, "prepare_graph_fused_add_rms_norm", None)
        if prepare is not None:
            prepare(x, stream=stream)
    runtime.all_reduce_fused_add_rms_norm(
        x,
        residual,
        weight,
        epsilon,
        out=out,
        residual_out=residual,
        stream=stream,
    )
    return True


def _glm_pcie_fused_ar_rms(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """AR(x) + residual add + RMSNorm. Mutates ``residual``; returns norm out.

    ``x`` is the per-rank PARTIAL output of a row-parallel linear / deferred
    MoE combine (reduce_results=False). On return ``residual`` holds
    all_reduce(x) + residual and the returned tensor holds the RMSNorm of it.
    """
    from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (
        get_b12x_pcie_allreduce,
    )

    comm = get_b12x_pcie_allreduce()
    if comm is not None:
        out = torch.empty_like(x)
        if _run_b12x_fused(comm, x, residual, weight, epsilon, out):
            return out

    # Fallback = exact stock path (bit-identical kernels to the unfused
    # sequence): TP all-reduce, then vLLM's fused add+rmsnorm.
    reduced = tensor_model_parallel_all_reduce(x)
    out_t, new_residual = ir.ops.fused_add_rms_norm.maybe_inplace(
        reduced, residual, weight, epsilon
    )
    if new_residual is not residual:
        residual.copy_(new_residual)
    return out_t


def _glm_pcie_fused_ar_rms_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="glm_pcie_fused_ar_rms",
    op_func=_glm_pcie_fused_ar_rms,
    mutates_args=["residual"],
    fake_impl=_glm_pcie_fused_ar_rms_fake,
)


@torch.compiler.assume_constant_result
def _fused_max_bytes() -> int:
    return _parse_byte_size(envs.VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE)


def fused_ar_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call-site helper: norm(all_reduce(hidden_states), residual).

    ``hidden_states`` must be the per-rank partial sum. Returns
    ``(normed, residual)`` where ``residual`` is updated in place — the same
    python variable keeps flowing, matching the stock in-place
    fused_add_rms_norm contract.
    """
    # vLLM traces once and specializes ranges afterwards. A Python shape
    # branch during tracing loses the one-row arm permanently. Keep the
    # ordinary operations visible; B12xAllReduceRMSFusionPass replaces the
    # pair only in the independently compiled [1,1] range.
    if torch.compiler.is_compiling():
        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return norm(reduced, residual)
    max_bytes = _fused_max_bytes()
    if hidden_states.numel() * hidden_states.element_size() > max_bytes:
        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return norm(reduced, residual)
    out = torch.ops.vllm.glm_pcie_fused_ar_rms(
        hidden_states, residual, norm.weight, norm.variance_epsilon
    )
    return out, residual


def defer_mlp_all_reduce(mlp: torch.nn.Module) -> bool:
    """Make an MLP/MoE module return per-rank partial sums (defer its TP AR).

    Returns True when the deferral was applied; the caller MUST then fuse the
    all-reduce into the next consumer (next layer's input_layernorm or the
    model's final norm). Handles both DeepseekV2MLP (dense down_proj) and
    DeepseekV2MoE (MoERunner late all-reduce, DSV4.1-precedent
    skip_final_all_reduce).
    """
    down_proj = getattr(mlp, "down_proj", None)
    if down_proj is not None:
        if getattr(down_proj, "reduce_results", False) and (
            getattr(down_proj, "tp_size", 1) > 1
        ):
            down_proj.reduce_results = False
            return True
        return False

    experts = getattr(mlp, "experts", None)
    moe_config = getattr(experts, "moe_config", None)
    if moe_config is None:
        return False
    if moe_config.is_sequence_parallel or moe_config.skip_final_all_reduce:
        return False
    if not (moe_config.tp_size > 1 or moe_config.ep_size > 1):
        return False
    # The deferral is only valid on the "late" all-reduce path where neither
    # the fused nor the shared output has been individually reduced.
    if getattr(experts, "_fused_output_is_reduced", False):
        return False
    # ZeroExpertRouter adds its contribution after the runner's all-reduce;
    # deferring would multiply it by world_size. Not used by GLM-5.3, but
    # guard anyway.
    router = getattr(experts, "router", None)
    if router is not None and type(router).__name__ == "ZeroExpertRouter":
        return False
    moe_config.skip_final_all_reduce = True
    return True
