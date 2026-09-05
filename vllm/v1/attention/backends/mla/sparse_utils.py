# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for sparse MLA backends."""

import os

import torch

from vllm.triton_utils import tl, triton

# See sparse_attn_indexer._CANONICAL_TOPK (determinism audit 2026-09-05).
# The COMPACT_TO_FRONT path below reserves each tile's output slice with an
# atomic add, so the compacted prefix ORDER is scheduling-dependent (measured
# 500/500 order changes on fixed inputs, kbench/stress_indexer_order.py) even
# when the upstream merge order is canonical. The trtllm-gen sparse kernel
# accumulates the first valid_count entries in order -> fp reduction-order
# nondeterminism. With the flag on, the compacted rows are re-sorted into a
# canonical (descending physical slot) order; -1 padding stays at the tail.
# Set-preserving => lossless. Default OFF pending live gating.
#
# VLLM_DSA_CANONICAL_TOPK=logical (2026-09-05 follow-up): the descending
# PHYSICAL sort above is deterministic only per KV-block layout — physical
# slot ids are block_table[req, i]*BLOCK_SIZE + off, and the block ids a
# request gets depend on the block-pool free-list state left by PREVIOUS
# requests. Two runs of the same prompt after different predecessors select
# the same logical token set but sort it into a different physical order ->
# different fp accumulation order in the sparse attention -> temp-0 flips
# that track predecessor request shapes (flag-independent; observed on every
# campaign-flag combination). "logical" mode removes the layout dependence:
# the merged LOGICAL top-k is canonicalized (descending global token id) at
# the merge (sparse_attn_indexer), and the conversion below runs in
# order-preserving mode followed by a STABLE compaction of the valid slots,
# so the attention accumulation order is a pure function of request content.
# Same selected set => lossless; deterministic across block layouts.
#
# VLLM_DSA_CANONICAL_TOPK=inkernel (2026-09-05 follow-up 2): same canonical
# order as "logical" (the DCP merge kernel emits descending-global-token-id
# rows itself now, see dcp_indexer_cutedsl), but the compaction keeps that
# order IN-KERNEL: instead of the atomic slot allocator (order unspecified)
# or the "logical" post-hoc stable argsort+gather (~63us/layer), each tile
# derives its output base deterministically as the number of valid entries in
# all PRECEDING columns of its row (DETERMINISTIC_BASE below). Validity is a
# pure function of the token id and compile-time constants (DCP ownership +
# block-bound check need no block_table load), so the prefix re-evaluation
# costs one extra masked row read per tile and zero atomics/sorts.
# Bit-identical output to "logical" mode.
_ct = os.environ.get("VLLM_DSA_CANONICAL_TOPK", "0").strip().lower()
_CANONICAL_TOPK = _ct not in ("", "0")
_CANONICAL_TOPK_LOGICAL = _ct == "logical"
_CANONICAL_TOPK_INKERNEL = _ct == "inkernel"
del _ct


# Kernel with prefill workspace support and valid count tracking
@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    valid_count_ptr,  # int32 [num_tokens] - output valid count per row
    prefill_request_id_ptr,  # int32 [num_tokens], -1 for decode, >=0 for prefill
    workspace_starts_ptr,  # int32 [num_prefill_reqs+1] or nullptr
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
    # When set, scatter valid slots to a contiguous prefix [0, valid_count) using
    # valid_count_ptr as an atomic slot allocator (DCP filtering leaves interior
    # -1 gaps; the trtllm-gen sparse kernel reads the first valid_count entries).
    # Requires COUNT_VALID and an out buffer pre-filled with -1. Order within the
    # prefix is unspecified (only the selected set matters).
    COMPACT_TO_FRONT: tl.constexpr,
    # With COMPACT_TO_FRONT: derive each tile's output base deterministically
    # (count of valid entries in all preceding columns of the row) instead of
    # the atomic slot allocator, so the compacted prefix PRESERVES the input
    # column order — a pure function of the row content, scheduling-independent.
    # valid_count_ptr is then only written when COUNT_VALID is set (and may be
    # null otherwise). Decode-only (asserted host-side: no prefill workspace).
    DETERMINISTIC_BASE: tl.constexpr,
    TOPK_TOTAL: tl.constexpr,  # full row width, for the prefix re-evaluation
    # DCP de-interleave: with DCP_SIZE == 1 these are an exact no-op
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0

    # DCP de-interleave the global token id into this rank's local slot.
    # Tokens are interleaved in groups of DCP_INTERLEAVE across ranks. With
    # DCP_SIZE == 1 (and any interleave) owning_rank == 0 == DCP_RANK (never
    # remote) and local_idx == tok, so this reduces to the non-DCP path; with
    # DCP_INTERLEAVE == 1 it reduces to plain round-robin (tok % / // DCP_SIZE).
    owning_rank = (tok // DCP_INTERLEAVE) % DCP_SIZE
    is_remote = owning_rank != DCP_RANK
    local_idx = (
        tok // (DCP_SIZE * DCP_INTERLEAVE)
    ) * DCP_INTERLEAVE + tok % DCP_INTERLEAVE

    # Compute block id and in-block offset
    block_id = local_idx // BLOCK_SIZE
    inblock_off = local_idx % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block | is_remote
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill & ~is_remote, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    # Override with prefill output if prefill is enabled
    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    if COMPACT_TO_FRONT:
        # Scatter valid slots to a contiguous prefix. A per-tile exclusive prefix
        # sum gives each valid lane a distinct local offset; one atomic add of the
        # tile's valid count reserves a contiguous base across racing tiles. The
        # out buffer is pre-filled with -1, so unwritten tail slots stay -1.
        is_valid = (~is_invalid_tok).to(tl.int32)
        local_offset = tl.cumsum(is_valid) - is_valid
        tile_valid_count = tl.sum(is_valid)
        if DETERMINISTIC_BASE:
            # Deterministic base: number of VALID entries among all columns
            # before this tile. Validity is a pure function of the token id
            # (DCP ownership + block-bound check) and compile-time constants,
            # so one masked read of the row prefix reproduces it exactly —
            # matching the is_invalid_tok computation above term for term
            # (tok < 0, block bound, remoteness; no block_table load needed).
            prev_cols = tl.arange(0, TOPK_TOTAL)
            prev_mask = prev_cols < tile_id * BLOCK_N
            ptok = tl.load(
                token_indices_ptr + token_id * ti_stride0 + prev_cols * ti_stride1,
                mask=prev_mask,
                other=-1,
            )
            p_local = (
                ptok // (DCP_SIZE * DCP_INTERLEAVE)
            ) * DCP_INTERLEAVE + ptok % DCP_INTERLEAVE
            p_block = p_local // BLOCK_SIZE
            p_invalid = (
                (ptok < 0)
                | (((ptok // DCP_INTERLEAVE) % DCP_SIZE) != DCP_RANK)
                | (p_block >= max_num_blocks_per_req)
                | (p_block < 0)
            )
            base = tl.sum((~p_invalid).to(tl.int32))
            if COUNT_VALID:
                tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)
        else:
            base = tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)
        dest = base + local_offset
        out_ptr_dest = out_ptr + token_id * out_stride0 + dest * out_stride1
        tl.store(out_ptr_dest, out_val, mask=is_valid == 1)
    else:
        # Store results in place (input column == output column).
        out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
        tl.store(out_ptr_ij, out_val)

        # Count valid indices in this tile and atomically add to row total
        if COUNT_VALID:
            tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
            tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.

    When HAS_PREFILL_WORKSPACE is True, prefill tokens are mapped to workspace offsets
    instead of global cache slots. prefill_workspace_request_ids and
    prefill_workspace_starts must be provided.

    prefill_workspace_request_ids: int32 [num_tokens], -1 for decode else
        prefill request index (maps to prefill_workspace_starts)
    prefill_workspace_starts: int32 [num_prefills], 0-indexed workspace
        starts for each prefill request

    When return_valid_counts is True, also returns the count of valid (non -1)
    indices per row, computed during the same kernel pass (no extra overhead).
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None
        assert prefill_workspace_starts is not None
        assert prefill_workspace_request_ids.dtype == torch.int32
        assert prefill_workspace_starts.dtype == torch.int32

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Allocate valid count buffer if needed (must be zero-initialized for atomics)
    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Prepare prefill pointers
    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None  # for mypy
        assert prefill_workspace_starts is not None  # for mypy
        assert prefill_workspace_request_ids.is_contiguous()
        assert prefill_workspace_starts.is_contiguous()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        HAS_PREFILL_WORKSPACE,
        return_valid_counts,
        False,  # COMPACT_TO_FRONT: keep input column == output column
        False,  # DETERMINISTIC_BASE (no compaction here)
        NUM_TOPK_TOKENS,
        # DCP disabled (no-op de-interleave)
        1,
        0,
        1,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out


def triton_filter_and_convert_dcp_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int = 1,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    return_valid_counts: bool = False,
    compact_valid_to_front: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Filter global per-request indices to this DCP rank's local slots.

    With ``compact_valid_to_front`` (default), the conversion kernel scatters
    this rank's owned slots to a contiguous prefix ``[0, valid_count)`` and
    leaves the rest ``-1``. DCP filtering marks non-owned slots ``-1`` and so
    creates interior gaps; the trtllm-gen sparse kernel reads the first
    ``valid_count`` entries of each row, so they must be a contiguous prefix.
    Compaction is fused into the kernel (atomic slot allocator) rather than a
    separate sort/gather pass. Prefix order is unspecified (only the set
    matters) — except under VLLM_DSA_CANONICAL_TOPK: "1" re-sorts the prefix
    (descending physical slot), "logical" compacts stably after an
    order-preserving conversion, and "inkernel" keeps the fused compaction but
    derives each tile's output base deterministically so the prefix preserves
    the (already canonical) input order at atomic-path cost.
    """
    assert dcp_size >= 1
    assert 0 <= dcp_rank < dcp_size
    # Interleave groups must align to KV blocks (globally enforced by
    # VllmConfig: block_size % cp_kv_cache_interleave_size == 0); assert the
    # local invariant so local_idx // BLOCK_SIZE never straddles a group.
    assert BLOCK_SIZE % cp_kv_cache_interleave_size == 0, (
        f"BLOCK_SIZE ({BLOCK_SIZE}) must be divisible by "
        f"cp_kv_cache_interleave_size ({cp_kv_cache_interleave_size})."
    )
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0

    if dcp_size == 1:
        return triton_convert_req_index_to_global_index(
            req_id,
            block_table,
            token_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_TOPK_TOKENS=NUM_TOPK_TOKENS,
            BLOCK_N=BLOCK_N,
            return_valid_counts=return_valid_counts,
        )

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    # VLLM_DSA_CANONICAL_TOPK=logical: skip the in-kernel atomic compaction
    # and compact deterministically afterwards (stable, order-preserving), so
    # the prefix order inherits the canonical LOGICAL order of the input row
    # instead of an atomic/physical order. See flag comment at top of file.
    logical_compact = _CANONICAL_TOPK_LOGICAL and compact_valid_to_front
    if logical_compact:
        compact_valid_to_front = False
    # VLLM_DSA_CANONICAL_TOPK=inkernel: keep the fused compaction but derive
    # each tile's output base deterministically inside the kernel (order-
    # preserving, no atomics/sorts). See flag comment at top of file.
    deterministic_base = _CANONICAL_TOPK_INKERNEL and compact_valid_to_front

    # The atomic compaction uses the valid-count buffer as a slot allocator, so
    # it requires counting; the deterministic-base compaction only counts when
    # the caller asked for counts. Pre-fill out with -1 so the unwritten tail
    # stays -1.
    count_valid = return_valid_counts or (
        compact_valid_to_front and not deterministic_base
    )
    if compact_valid_to_front:
        out = torch.full_like(token_indices_c, -1)
    else:
        out = torch.empty_like(token_indices_c)

    valid_counts: torch.Tensor | None = None
    if count_valid:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    _convert_req_index_to_global_index_kernel[(num_tokens, tiles_per_row)](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        # No prefill workspace on the DCP decode path.
        None,
        None,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        False,  # HAS_PREFILL
        count_valid,
        compact_valid_to_front,
        deterministic_base,
        NUM_TOPK_TOKENS,
        dcp_size,
        dcp_rank,
        cp_kv_cache_interleave_size,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )

    if logical_compact:
        # Stable compaction of the order-preserving conversion output: valid
        # slots move to a contiguous prefix KEEPING their relative (logical,
        # already canonicalized at the merge) order; -1 holes move to the
        # tail. Deterministic regardless of KV block layout: the prefix order
        # is a pure function of the selected logical token set.
        order = torch.argsort(
            (out == -1).to(torch.int8), dim=1, stable=True
        )
        out = torch.gather(out, 1, order)
    elif (
        _CANONICAL_TOPK
        and not _CANONICAL_TOPK_INKERNEL
        and compact_valid_to_front
    ):
        # Canonicalize the compacted prefix (atomic slot-allocator order is
        # scheduling-dependent): descending sort keeps valid physical slots
        # in a deterministic order and the -1 padding at the tail. Same set,
        # deterministic downstream attention accumulation order — but only
        # per KV-block layout; see the "logical" mode above for the
        # layout-independent variant. "inkernel" mode needs no post-pass: the
        # DETERMINISTIC_BASE compaction already preserved the canonical order.
        out = torch.sort(out, dim=1, descending=True).values

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out
