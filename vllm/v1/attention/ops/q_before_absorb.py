# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve per-rank absorption arithmetic while gathering smaller MLA queries."""

import os
from typing import Any

import torch

from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.v1.attention.ops.dcp_comm_overlap import (
    AsyncAllGather,
    consume_pending_dcp_merge,
    dcp_comm_overlap_enabled,
)

logger = init_logger(__name__)


def q_before_absorb_eligible(
    layer: Any, query: torch.Tensor, fp8_attention: bool
) -> bool:
    # Existing decode/graph and unsupported-layout paths remain unchanged.
    eligible = (
        os.environ.get("VLLM_GLM_Q_BEFORE_ABSORB", "0") == "1"
        and query.shape[0] in (2048, 4096)
        and query.dtype == torch.bfloat16
        and query.is_cuda
        and not torch.cuda.is_current_stream_capturing()
        and layer.impl.__class__.__name__ == "FlashInferMLASparseSM120Impl"
        and layer.impl.dcp_world_size == 4
        and layer.q_pad_num_heads is None
        and not layer.is_aiter_triton_fp4_bmm_enabled
        and not layer.is_aiter_triton_fp8_bmm_enabled
        and not (fp8_attention and layer.impl.supports_quant_query_input)
        and query.shape[1:] == (16, 256)
        and query.stride() == (4096, 256, 1)
        and layer.W_UK_T.shape == (16, 192, 512)
        and layer.W_UK_T.dtype == torch.bfloat16
        and layer.W_UK_T.stride() == (229376, 512, 1)
    )
    if eligible:
        logger.info_once("Using Q-before-absorption for qualified DCP4 prefill")
    return eligible


def q_before_absorb(
    query: torch.Tensor, weight: torch.Tensor, impl: Any, attn_metadata: Any
) -> tuple[torch.Tensor, Any]:
    # Rank-major Q preserves the original per-rank BMM strides and dimensions.
    # No remote weight survives this call. One output chunk is live at a time;
    # measured Torch peak at T4096 is lower than the original Q all-gather.
    group = get_dcp_group()
    tokens = query.shape[0]
    precomputed_indices = None
    if dcp_comm_overlap_enabled() and hasattr(impl, "precompute_mqa_indices"):
        ag_q = AsyncAllGather(group, query, dim=1, flush_coalesce=True)
        precomputed_indices = impl.precompute_mqa_indices(attn_metadata, tokens)
        gathered_q = ag_q.wait_raw()
    else:
        consume_pending_dcp_merge()
        gathered_q = group.all_gather(query, dim=0)
    gathered_q = gathered_q.view(4, tokens, 16, 256)
    packed_weight = weight.contiguous()
    gathered_w = group.all_gather(packed_weight, dim=0).view(4, 16, 192, 512)
    final = query.new_empty((tokens, 64, 576))
    for peer in range(4):
        remote = torch.empty_strided(
            weight.shape, weight.stride(), dtype=weight.dtype, device=weight.device
        )
        remote.copy_(gathered_w[peer])
        qi = gathered_q[peer]
        absorbed = query.new_empty((16, tokens, 512))
        torch.bmm(qi[:, :, :192].transpose(0, 1), remote, out=absorbed)
        local_result = torch.cat((absorbed.transpose(0, 1), qi[:, :, 192:]), dim=-1)
        del absorbed
        final[:, peer * 16 : (peer + 1) * 16].copy_(local_result)
        del local_result, remote
    return final, precomputed_indices
