# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather unchanged MLA cache bytes for bounded DCP4 prefill attention."""

import os

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.ops.common import _dcp_rs_copyfree_supported
from vllm.v1.attention.ops.dcp_comm_overlap import consume_pending_dcp_merge
from vllm.v1.attention.ops.raw_kv_collective_guard import TREES, qualify
from vllm.v1.attention.ops.raw_kv_merge import merge

logger = init_logger(__name__)


@triton.jit
def pack_pages(
    KV,
    BT,
    OUT,
    VALID: tl.constexpr,
    TOTAL: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offset // 656
    col = offset % 656
    block = tl.load(BT + token // 64, (offset < TOTAL) & (token < VALID), other=0)
    data = tl.load(
        KV + block * STRIDE + (token % 64) * 656 + col,
        (offset < TOTAL) & (token < VALID),
        other=0,
    )
    tl.store(OUT + offset, data, offset < TOTAL)


def _eligible(layer, query, cache, meta, fp8_attention, group):
    # These pre-collective predicates must be uniform across TP ranks: shared
    # model/config/metadata and identically laid-out per-rank tensors. The tree
    # probe below separately reduces its numerical verdict across all ranks.
    impl = layer.impl
    return (
        query.shape == (4096, 16, 256)
        and query.stride() == (4096, 256, 1)
        and query.dtype == torch.bfloat16
        and not torch.cuda.is_current_stream_capturing()
        and meta.num_reqs == 1
        and meta.num_actual_tokens == 4096
        and 4096 <= meta.max_seq_len <= 131072
        and meta.topk_tokens == 2048
        and meta.block_size == 64
        and meta.cp_kv_cache_interleave_size == 1
        and impl.__class__.__name__ == "FlashInferMLASparseSM120Impl"
        and impl.dcp_world_size == 4
        and group.world_size == 4
        and not layer.dcp_a2a
        and not layer.dcp_a2a_exact
        and impl.need_to_return_lse_for_decode
        and os.getenv("VLLM_GLM_DCP_RS_STAGED") == "1"
        and os.getenv("VLLM_GLM_DCP_RS_VIEW") == "1"
        and _dcp_rs_copyfree_supported(group)
        and impl.kv_scale_format == "arbitrary_fp32"
        and not impl.lse_base_on_e
        and layer.q_pad_num_heads is None
        and not layer.is_aiter_triton_fp4_bmm_enabled
        and not layer.is_aiter_triton_fp8_bmm_enabled
        and not (fp8_attention and impl.supports_quant_query_input)
        and layer.W_UK_T.shape == (16, 192, 512)
        and layer.W_UK_T.stride() == (229376, 512, 1)
        and layer.W_UK_T.dtype == torch.bfloat16
        and cache.ndim == 3
        and cache.shape[1:] == (64, 656)
        and cache.dtype == torch.uint8
        and cache.stride(2) == 1
        and cache.stride(1) == 656
        and os.getenv("VLLM_DSA_CANONICAL_TOPK") == "inkernel"
        and os.getenv("NCCL_MAX_NCHANNELS") == "4"
        and os.getenv("NCCL_BUFFSIZE") == "1048576"
        and tuple(x.strip().upper() for x in os.getenv("NCCL_ALGO", "").split(","))
        == ("RING", "TREE")
        and os.getenv("NCCL_P2P_LEVEL") == "SYS"
        and torch.cuda.get_device_capability(query.device) == (12, 0)
    )


def try_raw_kv_gather(layer, query, cache, meta, fp8_attention):
    if not envs.VLLM_GLM_RAW_KV_GATHER:
        return None
    group = get_dcp_group()
    if not _eligible(layer, query, cache, meta, fp8_attention, group):
        return None
    impl = layer.impl
    # Finishes exactly the existing global top-k merge before changing QAG.
    # Deferred coalesced gathers self-launch solo in the existing wait path.
    consume_pending_dcp_merge()
    if not qualify(group, 4096):
        logger.warning_once(
            "Raw-KV gather production-pynccl tree probe failed; "
            "using original attention."
        )
        return None
    rank = group.rank_in_group
    length = meta.max_seq_len
    blocks = (length + 255) // 256
    valid = length // 4 + int(rank < length % 4)
    if meta.block_table.shape[1] < blocks:
        return None
    packed = torch.empty((blocks, 64, 656), device=cache.device, dtype=cache.dtype)
    pack_pages[(triton.cdiv(packed.numel(), 1024),)](
        cache, meta.block_table[0], packed, valid, packed.numel(), cache.stride(0), 1024
    )
    allkv = group.all_gather(packed, dim=0).view(4, blocks, 64, 656)
    del packed
    absorbed = query.new_empty((16, 4096, 512))
    torch.bmm(query[:, :, :192].transpose(0, 1), layer.W_UK_T, out=absorbed)
    localq = torch.cat((absorbed.transpose(0, 1), query[:, :, 192:]), dim=-1)
    del absorbed
    table = torch.arange(blocks, device=query.device, dtype=torch.int32).view(1, blocks)
    topk = impl.topk_indices_buffer[:4096]
    parts = []
    lses = []
    # Use backend calls to preserve actual scale, workspace and empty-row policy.
    # Temporarily changing impl.dcp_rank is unnecessary: provide explicit mapping.
    for peer in range(4):
        indices, counts = triton_filter_and_convert_dcp_index(
            meta.req_id_per_token[:4096],
            table,
            topk,
            dcp_size=4,
            dcp_rank=peer,
            cp_kv_cache_interleave_size=1,
            BLOCK_SIZE=64,
            NUM_TOPK_TOKENS=2048,
            return_valid_counts=True,
        )
        empty = None if impl._skip_empty_fill else (indices == -1).all(dim=-1)
        output, lse = impl.forward_mqa(
            localq,
            allkv[peer],
            meta,
            layer,
            precomputed_indices=(indices, counts, empty),
        )
        parts.append(output)
        lses.append(lse)
    del allkv
    result = merge(parts, torch.stack(lses), TREES[rank])
    logger.info_once(
        "Raw-KV gather eligible: C1 T4096 context<=128K, "
        "actual-pynccl tree probe passed."
    )
    return result
