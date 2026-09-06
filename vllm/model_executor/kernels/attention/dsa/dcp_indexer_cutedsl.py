# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Uint32, Uint64
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import recast_val
from vllm.triton_utils import tl, triton

# VLLM_DSA_CANONICAL_TOPK=inkernel (determinism follow-up 2026-09-05): the
# selector below emits the selected set via an atomic slot allocator, so its
# output ORDER is warp-scheduling dependent (see sparse_attn_indexer). The
# "logical" mode fixes that with a post-hoc torch.sort (~63us/layer). In
# "inkernel" mode the kernel itself finishes with a block-wide bitonic sort of
# the output row (descending token id, -1 padding at the tail) in shared
# memory -- byte-identical order to the "logical" torch.sort, at in-kernel
# cost. Read once at import (static branch, capture-safe); kbench overrides
# via the explicit `canonical=` argument.
_CANONICAL_TOPK_INKERNEL = (
    os.environ.get("VLLM_DSA_CANONICAL_TOPK", "0").strip().lower() == "inkernel"
)


def stable_topk_from_gathered_candidates_cutedsl(
    gathered: torch.Tensor,
    topk: int,
    out: torch.Tensor | None = None,
    canonical: bool | None = None,
) -> torch.Tensor:
    """Select the global top-k from the gathered per-rank candidates.

    ``gathered`` is either the concatenated view ``[rows, ws*topk, 2]``
    (baseline; requires the movedim+reshape copy of the all-gather output)
    or -- VLLM_GLM_DCP_AG_RAW_TOPK -- the collective's RAW rank-major output
    viewed as ``[ws, rows, topk, 2]`` (4-D, no copy). The raw path reads
    candidate ``(rank, j)`` at the same logical position ``rank*topk + j``
    the 3-D path reads it from, so keys, selection and the canonical
    in-kernel sort are bit-identical between the two layouts.
    """
    raw = gathered.ndim == 4
    if raw and (gathered.shape[2] % StableTopKFromGatheredCandidatesKernel.tb_size):
        # Per-rank candidate count not a tb_size multiple: fall back to the
        # concat layout (materializes the copy the raw path avoids).
        ws, rows, k, _ = gathered.shape
        gathered = gathered.movedim(0, 1).reshape(rows, ws * k, 2)
        raw = False
    num_rows = gathered.shape[1] if raw else gathered.shape[0]
    if out is None:
        out = torch.empty(
            (num_rows, topk),
            dtype=torch.int32,
            device=gathered.device,
        )
    if canonical is None:
        canonical = _CANONICAL_TOPK_INKERNEL
    if raw:
        world_size = gathered.shape[0]
        num_candidates = world_size * gathered.shape[2]
        StableTopKFromGatheredCandidatesKernel.compile(
            topk, num_candidates, canonical, world_size
        )(gathered, out)
    else:
        StableTopKFromGatheredCandidatesKernel.compile(
            topk, gathered.shape[1], canonical
        )(gathered, out)
    return out


def pack_dcp_topk_candidates_cutedsl(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    packed: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None,
) -> None:
    topk = topk_indices.shape[1]
    grid = (topk_indices.shape[0], triton.cdiv(topk, 512))
    row_starts_arg = row_starts if row_starts is not None else topk_indices
    _pack_dcp_topk_candidates_triton_kernel[grid](
        logits,
        topk_indices,
        packed,
        row_starts_arg,
        logits.stride(0),
        logits.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        logits.shape[1],
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
        CP_INTERLEAVE=cp_interleave,
        HAS_ROW_STARTS=row_starts is not None,
        TOPK=topk,
        BLOCK_SIZE=512,
        num_warps=8,
    )


@triton.jit
def _pack_dcp_topk_candidates_triton_kernel(
    logits,
    topk_indices,
    packed,
    row_starts,
    logits_stride0: tl.constexpr,
    logits_stride1: tl.constexpr,
    topk_stride0: tl.constexpr,
    topk_stride1: tl.constexpr,
    packed_stride0: tl.constexpr,
    packed_stride1: tl.constexpr,
    packed_stride2: tl.constexpr,
    num_cols,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < TOPK

    local_idx = tl.load(
        topk_indices + row * topk_stride0 + cols * topk_stride1,
        mask=mask,
        other=-1,
    )
    valid = local_idx >= 0
    safe_local_idx = tl.maximum(local_idx, 0)

    row_start = 0
    if HAS_ROW_STARTS:
        row_start = tl.load(row_starts + row)

    score_col = safe_local_idx + row_start
    score_col = tl.minimum(score_col, tl.maximum(num_cols - 1, 0))
    score = tl.load(
        logits + row * logits_stride0 + score_col * logits_stride1,
        mask=mask & valid,
        other=-float("inf"),
    )

    global_id = (
        (safe_local_idx // CP_INTERLEAVE) * (DCP_WORLD_SIZE * CP_INTERLEAVE)
        + DCP_RANK * CP_INTERLEAVE
        + safe_local_idx % CP_INTERLEAVE
    )
    global_id = tl.where(valid, global_id, -1)

    packed_base = packed + row * packed_stride0 + cols * packed_stride1
    tl.store(packed_base, score, mask=mask)
    tl.store(packed_base + packed_stride2, global_id.to(tl.float32), mask=mask)


@cute.jit
def _warp_scan_inclusive_i32(val: Int32, lane: Int32) -> Int32:
    for i in cutlass.range_constexpr(cute.arch.WARP_SIZE.bit_length() - 1):
        offset = 1 << i
        partial = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial
    return val


@cute.jit
def _block_scan_inclusive_i32(
    val: Int32,
    lane: Int32,
    warp_id: Int32,
    warp_scratch: cute.Tensor,
    warps_per_block: int,
) -> Int32:
    prefix = _warp_scan_inclusive_i32(val, lane)
    if lane == Int32(cute.arch.WARP_SIZE - 1):
        warp_scratch[0, warp_id] = prefix
    cute.arch.sync_threads()

    if warp_id == Int32(0):
        warp_total = Int32(0)
        if lane < Int32(warps_per_block):
            warp_total = warp_scratch[0, lane]
        warp_prefix = _warp_scan_inclusive_i32(warp_total, lane)
        if lane < Int32(warps_per_block):
            warp_scratch[0, lane] = warp_prefix - warp_total
    cute.arch.sync_threads()

    return prefix + warp_scratch[0, warp_id]


class StableTopKFromGatheredCandidatesKernel:
    tb_size = 512
    hist_bins = 2048
    radix_bits = (hist_bins - 1).bit_length()
    assert hist_bins == 1 << radix_bits
    key_bits = Uint64.width
    radix_passes = (key_bits + radix_bits - 1) // radix_bits
    final_radix_bits = key_bits - radix_bits * (radix_passes - 1)
    hist_chunks = (hist_bins + tb_size - 1) // tb_size
    warps_per_block = tb_size // cute.arch.WARP_SIZE

    def __init__(
        self,
        topk: int,
        num_candidates: int,
        canonical: bool = False,
        raw_world_size: int = 0,
    ):
        assert num_candidates % self.tb_size == 0, (
            "StableTopKFromGatheredCandidatesKernel requires candidate count "
            f"to be a multiple of {self.tb_size}, got {num_candidates}"
        )
        self.topk = topk
        self.keys_per_thread = num_candidates // self.tb_size
        self.canonical = canonical
        # VLLM_GLM_DCP_AG_RAW_TOPK: input is the raw rank-major all-gather
        # output [ws, rows, per_rank, 2] instead of the concatenated
        # [rows, ws*per_rank, 2] copy. Candidate (rank, j) is read at logical
        # position rank*per_rank + j -- the identical (thread, key_idx) slot
        # the 3-D path reads it from -- so the resulting keys array, and thus
        # everything downstream, is bit-identical.
        self.raw_world_size = raw_world_size
        if raw_world_size:
            assert num_candidates % raw_world_size == 0, (
                num_candidates,
                raw_world_size,
            )
            per_rank = num_candidates // raw_world_size
            assert per_rank % self.tb_size == 0, (
                "raw-layout stable topk requires per-rank candidate count "
                f"to be a multiple of {self.tb_size}, got {per_rank}"
            )
            self.raw_per_rank = per_rank
            self.raw_chunks_per_rank = per_rank // self.tb_size
        if canonical:
            # The canonical sort runs in the histogram smem after the radix
            # passes finish, so the row must fit in it; topk is also required
            # to be a power of two (bitonic network) and a multiple of tb_size
            # (uniform per-thread element counts between barriers). All hold
            # for the supported topk values (512, 1024, 2048).
            assert topk <= self.hist_bins, (topk, self.hist_bins)
            assert topk & (topk - 1) == 0, topk
            assert topk % self.tb_size == 0, (topk, self.tb_size)

        @cute.struct
        class SharedStorage:
            hist: cute.struct.MemRange[Int32, self.hist_bins]
            committed_count: cute.struct.MemRange[Int32, 1]
            running_count: cute.struct.MemRange[Int32, 1]
            threshold_bin: cute.struct.MemRange[Int32, 1]
            threshold_found: cute.struct.MemRange[Int32, 1]
            include_threshold_bin: cute.struct.MemRange[Int32, 1]
            prefix_s: cute.struct.Align[cute.struct.MemRange[Uint64, 1], 8]
            warp_totals: cute.struct.MemRange[Int32, self.warps_per_block]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(
        self,
        gathered: cute.Tensor,
        out: cute.Tensor,
        stream: CUstream,
    ):
        # One block per output row (out rows == gathered rows in both the
        # 3-D concat layout and the 4-D raw rank-major layout).
        grid = (out.shape[0], 1, 1)
        self.kernel(gathered, out).launch(
            grid=grid,
            block=(self.tb_size, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _stable_key(self, score: Float32, token_id: Int32) -> Uint64:
        bits = recast_val(score, Uint32)
        mask = Uint32(0x80000000)
        if (bits & Uint32(0x80000000)) != Uint32(0):
            mask = Uint32(0xFFFFFFFF)
        score_key = Uint64(bits ^ mask) << Uint64(32)
        id_key = Uint64(~Uint32(token_id))
        key = score_key | id_key
        if token_id < Int32(0):
            key = Uint64(0)
        return key

    @cute.jit
    def _prefix_matches(
        self,
        key: Uint64,
        prefix: Uint64,
        prefix_bits: Int32,
    ):
        matches = prefix_bits == Int32(0)
        if prefix_bits != Int32(0):
            shift = Int32(self.key_bits) - prefix_bits
            matches = (key >> Uint64(shift)) == (prefix >> Uint64(shift))
        return matches

    @cute.jit
    def _radix_pass(
        self,
        keys: cute.Tensor,
        output: cute.Tensor,
        storage,
        tid: Int32,
        step: Int32,
        bits: int,
        is_final_pass: bool,
    ):
        hist_smem = storage.hist.get_tensor(cute.make_layout((self.hist_bins,)))
        committed_count_smem = storage.committed_count.data_ptr()
        running_count_smem = storage.running_count.data_ptr()
        threshold_bin_smem = storage.threshold_bin.data_ptr()
        threshold_found_smem = storage.threshold_found.data_ptr()
        include_threshold_bin_smem = storage.include_threshold_bin.data_ptr()
        prefix_smem = storage.prefix_s.data_ptr()
        warp_totals_smem = storage.warp_totals.get_tensor(
            cute.make_layout((1, self.warps_per_block))
        )

        prefix_bits = step * Int32(self.radix_bits)
        num_bins = 1 << bits
        block_scan_iterations = (num_bins + self.tb_size - 1) // self.tb_size
        shift = Int32(self.key_bits) - prefix_bits - Int32(bits)
        bin_mask = Uint64(num_bins - 1)
        prefix = prefix_smem.load()

        for chunk in cutlass.range_constexpr(self.hist_chunks):
            hist_smem[tid + Int32(chunk * self.tb_size)] = Int32(0)
        if tid == Int32(0):
            running_count_smem.store(committed_count_smem.load())
            include_threshold_bin_smem.store(Int32(0))
            threshold_found_smem.store(Int32(0))
        cute.arch.sync_threads()

        for key_idx in cutlass.range_constexpr(self.keys_per_thread):
            key = keys[key_idx]
            if self._prefix_matches(key, prefix, prefix_bits):
                bin_idx = Int32((key >> Uint64(shift)) & bin_mask)
                cute.arch.atomic_add(
                    hist_smem.iterator + bin_idx,
                    Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
        cute.arch.sync_threads()

        lane = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx()
        # Each iteration scans one tb_size-wide slice of bins, high to low.
        iter = Int32(0)
        threshold_found = threshold_found_smem.load()
        while threshold_found == Int32(0) and iter < Int32(block_scan_iterations):
            bin_idx = Int32(num_bins - 1) - (iter * Int32(self.tb_size) + tid)
            count = hist_smem[bin_idx]
            chunk_inclusive = _block_scan_inclusive_i32(
                count,
                lane,
                warp_id,
                warp_totals_smem,
                self.warps_per_block,
            )
            running_count = running_count_smem.load()
            prior_in_scan_slice = chunk_inclusive - count
            remaining = Int32(self.topk) - running_count - prior_in_scan_slice
            if count > Int32(0) and remaining > Int32(0) and remaining <= count:
                threshold_bin_smem.store(bin_idx)
                if count <= remaining or cutlass.const_expr(is_final_pass):
                    include_threshold_bin_smem.store(Int32(1))
                threshold_found_smem.store(Int32(1))
            # Barrier: every thread must finish reading running_count for this
            # slice before tb_size-1 advances it, else a warp racing ahead to
            # the store makes a lagging thread double-count the slice total
            # (-> remaining too small -> threshold too high -> under-fill).
            cute.arch.sync_threads()
            if tid == Int32(self.tb_size - 1):
                running_count_smem.store(running_count + chunk_inclusive)
            cute.arch.sync_threads()

            threshold_found = threshold_found_smem.load()
            iter += Int32(1)

        threshold = threshold_bin_smem.load()
        should_include_threshold = include_threshold_bin_smem.load() != Int32(0)
        for key_idx in cutlass.range_constexpr(self.keys_per_thread):
            key = keys[key_idx]
            if self._prefix_matches(key, prefix, prefix_bits):
                bin_idx = Int32((key >> Uint64(shift)) & bin_mask)
                selected = bin_idx > threshold
                if should_include_threshold:
                    selected = selected or bin_idx == threshold
                if selected:
                    dst = cute.arch.atomic_add(
                        committed_count_smem,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if dst < Int32(self.topk):
                        output[dst] = recast_val(~Uint32(key), Int32)
        cute.arch.sync_threads()

        pass_finished = include_threshold_bin_smem.load()
        if tid == Int32(0) and pass_finished == Int32(0):
            prefix_smem.store(prefix | (Uint64(threshold) << Uint64(shift)))
        cute.arch.sync_threads()
        return pass_finished

    @cute.jit
    def _canonical_sort_desc(
        self,
        output_row: cute.Tensor,
        storage,
        tid: Int32,
    ):
        """Block-wide bitonic sort of output_row[0:topk], DESCENDING.

        Values are int32 token ids (unique) with -1 padding; descending order
        keeps the padding at the tail. Byte-identical result to the "logical"
        mode's post-hoc ``torch.sort(..., descending=True)``, but runs inside
        the selector launch.

        Element mapping: thread ``tid`` keeps ``elems = topk / tb_size``
        values in registers, element index ``i = e * tb_size + tid``. For a
        compare-exchange stride ``j`` the partner element then lives
          * in the SAME THREAD for ``j >= tb_size`` (register swap, free),
          * in the SAME WARP for ``j < 32`` (butterfly shuffle, no barrier),
          * across warps otherwise (one smem exchange + 2 barriers).
        For topk=2048/tb=512 that is 3 register steps, 45 shuffle steps and
        only 18 smem steps of the 66-step network (vs 66 all-smem barrier
        steps for the naive version). All lanes are active in every step
        (each element keeps max or min of its pair; direction and side derive
        from bits k and j of the element index). The histogram smem is reused
        as the exchange buffer (the radix passes are done with it by now).
        """
        sort_smem = storage.hist.get_tensor(cute.make_layout((self.hist_bins,)))
        elems = self.topk // self.tb_size
        vals = cute.make_rmem_tensor((elems,), Int32)
        for e in cutlass.range_constexpr(elems):
            vals[e] = output_row[tid + Int32(e * self.tb_size)]

        log2n = self.topk.bit_length() - 1
        for kk in cutlass.range_constexpr(1, log2n + 1):
            k = 1 << kk
            for jj in cutlass.range_constexpr(kk):
                j = 1 << (kk - 1 - jj)
                if cutlass.const_expr(j >= self.tb_size):
                    # In-thread register step: pair (e, e ^ j/tb) of this
                    # thread's own elements. Direction depends only on bit k
                    # of i = e*tb + tid, which is an e bit here (k > j >= tb),
                    # so it is compile-time per pair.
                    ej = j // self.tb_size
                    for e in cutlass.range_constexpr(elems):
                        if cutlass.const_expr((e & ej) == 0):
                            pe = e | ej
                            # keep max at the LOW index iff descending block
                            keep_max_low = ((e * self.tb_size) & k) == 0
                            a = vals[e]
                            b = vals[pe]
                            if cutlass.const_expr(keep_max_low):
                                if a < b:
                                    vals[e] = b
                                    vals[pe] = a
                            else:
                                if a > b:
                                    vals[e] = b
                                    vals[pe] = a
                elif cutlass.const_expr(j >= cute.arch.WARP_SIZE):
                    # Cross-warp step: exchange through smem. Two barriers:
                    # publish, then read-partner (the second barrier protects
                    # the next step's writes from overtaking these reads).
                    for e in cutlass.range_constexpr(elems):
                        sort_smem[tid + Int32(e * self.tb_size)] = vals[e]
                    cute.arch.sync_threads()
                    for e in cutlass.range_constexpr(elems):
                        i = tid + Int32(e * self.tb_size)
                        b = sort_smem[i ^ Int32(j)]
                        a = vals[e]
                        mn = a
                        mx = b
                        if a > b:
                            mn = b
                            mx = a
                        # keep max iff bit k and bit j of i agree (both zero:
                        # low side of a descending block; both one: high side
                        # of an ascending block).
                        keep = mn
                        if ((i >> Int32(kk)) & Int32(1)) == (
                            (i >> Int32(kk - 1 - jj)) & Int32(1)
                        ):
                            keep = mx
                        vals[e] = keep
                    cute.arch.sync_threads()
                else:
                    # Intra-warp step: butterfly shuffle, no barrier.
                    for e in cutlass.range_constexpr(elems):
                        i = tid + Int32(e * self.tb_size)
                        a = vals[e]
                        b = Int32(
                            cute.arch.shuffle_sync_bfly(a, offset=j)
                        )
                        mn = a
                        mx = b
                        if a > b:
                            mn = b
                            mx = a
                        keep = mn
                        if ((i >> Int32(kk)) & Int32(1)) == (
                            (i >> Int32(kk - 1 - jj)) & Int32(1)
                        ):
                            keep = mx
                        vals[e] = keep

        for e in cutlass.range_constexpr(elems):
            output_row[tid + Int32(e * self.tb_size)] = vals[e]

    @cute.kernel
    def kernel(
        self,
        input: cute.Tensor,
        out: cute.Tensor,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.raw_world_size > 0):
            # Raw rank-major layout [ws, rows, per_rank, 2]: slice this row
            # across all ranks -> (ws, per_rank, 2).
            input_row = input[None, row, None, None]
        else:
            input_row = input[row, None, None]
        output_row = out[row, None]
        keys = cute.make_rmem_tensor((self.keys_per_thread,), Uint64)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage, 8)
        committed_count_smem = storage.committed_count.data_ptr()
        prefix_smem = storage.prefix_s.data_ptr()
        for i in range(tid, self.topk, self.tb_size):
            output_row[i] = Int32(-1)

        for key_idx in cutlass.range_constexpr(self.keys_per_thread):
            if cutlass.const_expr(self.raw_world_size > 0):
                # Logical position col = tid + key_idx*tb_size = rank*per_rank
                # + col_local: identical (thread, key_idx) -> candidate map
                # as the 3-D concat layout, just addressed through the raw
                # 4-D tensor. rank / chunk are compile-time per key_idx.
                rank_c = key_idx // self.raw_chunks_per_rank
                col_local = tid + Int32(
                    (key_idx % self.raw_chunks_per_rank) * self.tb_size
                )
                score = Float32(input_row[rank_c, col_local, 0])
                token_id = Int32(input_row[rank_c, col_local, 1])
            else:
                col = tid + Int32(key_idx * self.tb_size)
                score = Float32(input_row[col, 0])
                token_id = Int32(input_row[col, 1])
            keys[key_idx] = self._stable_key(score, token_id)

        if tid == Int32(0):
            committed_count_smem.store(Int32(0))
            prefix_smem.store(Uint64(0))
        cute.arch.sync_threads()

        step = Int32(0)
        finished = Int32(0)
        while finished == Int32(0) and step < Int32(self.radix_passes - 1):
            finished = self._radix_pass(
                keys,
                output_row,
                storage,
                tid,
                step,
                self.radix_bits,
                False,
            )
            step += Int32(1)

        if finished == Int32(0):
            self._radix_pass(
                keys,
                output_row,
                storage,
                tid,
                Int32(self.radix_passes - 1),
                self.final_radix_bits,
                True,
            )

        if cutlass.const_expr(self.canonical):
            # Every _radix_pass exits through a sync_threads, so all committed
            # output_row stores are visible block-wide here.
            self._canonical_sort_desc(output_row, storage, tid)

    @cache
    @staticmethod
    def compile(
        topk: int,
        num_candidates: int,
        canonical: bool = False,
        raw_world_size: int = 0,
    ):
        num_rows = cute.sym_int()

        if raw_world_size:
            per_rank = num_candidates // raw_world_size
            # Raw rank-major all-gather output: [ws, rows, per_rank, 2]
            # contiguous. The rank stride (rows*per_rank*2) is symbolic; the
            # row stride is the compile-time per-rank payload size.
            gathered = cute.runtime.make_fake_tensor(
                Float32,
                (raw_world_size, num_rows, per_rank, 2),
                stride=(cute.sym_int64(divisibility=2), per_rank * 2, 2, 1),
                assumed_align=8,
            )
        else:
            gathered = cute.runtime.make_fake_tensor(
                Float32,
                (num_rows, num_candidates, 2),
                stride=(cute.sym_int64(divisibility=2), 2, 1),
                assumed_align=8,
            )
        out = make_fake_tensor(Int32, (num_rows, topk), divisibility=1)

        kernel = StableTopKFromGatheredCandidatesKernel(
            topk, num_candidates, canonical, raw_world_size
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            gathered,
            out,
            stream,
            options="--enable-tvm-ffi",
        )
