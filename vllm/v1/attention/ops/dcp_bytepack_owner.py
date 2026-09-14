# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental single-stream loan of existing B12X BF16 ring scratch.

No owned CUDA allocation, event, stream or counter is created by this adapter.
The caller must serialize EVERY use of the underlying DMA channel, not merely
calls to this object. Construction and calls must be collective across TP4.
"""

import weakref

import torch

READY = 192
DATA = 196
DONE = 200
MAX_GENERATIONS = (1 << 31) - 1
_claimed: weakref.WeakSet[object] = weakref.WeakSet()


def layout(rows, width, rank):
    if not (128 <= rows <= 4096 and width in (6, 7) and 0 <= rank < 4):
        raise ValueError("Unsupported owner shape")
    splits = [rows // 4 + int(peer < rows % 4) for peer in range(4)]
    offsets = [sum(splits[:peer]) for peer in range(4)]
    return splits, offsets, splits[rank], width * 2048


class BorrowedOwner:
    def __init__(self, dma):
        from b12x.comm.pcie import pcie_dma
        from b12x.comm.pcie.pcie_dcp_topk import _tensor_from_cuda_pointer

        if (
            pcie_dma.FLAG_SLOTS != 256
            or pcie_dma.FLAG_STRIDE != 128
            or pcie_dma.MAX_PIECES != 8
            or dma.world_size != 4
            or dma.wire_mode != "bf16"
            or dma.max_bytes != 4096 * 6144 * 4
            or dma.shard_capacity != 24 * 1024 * 1024
            or len(dma._scratch_base) != 4
            or dma._send_counters.numel() != 256
            or dma._wait_counters.numel() != 256
            or len(dma._copied_events) < 4
            or dma._closed
        ):
            raise ValueError("Borrowing requires the reviewed B12X TP4 DMA layout")
        if dma in _claimed:
            raise ValueError("DMA channel already has a borrower")
        _claimed.add(dma)
        self.dma = dma
        self.rank = dma.rank
        self.tensor_view = _tensor_from_cuda_pointer
        self.generations = 0
        self.stream = None
        self.busy = False
        self.poisoned = False

    def _publish(self, phase, peer):
        d = self.dma
        d._kernels.dma_set_flag(
            d._flag_ptr(peer, phase + self.rank),
            d._counter_ptr(d._send_counters, phase + peer),
        )

    def _wait(self, phase, peer):
        d = self.dma
        d._kernels.dma_wait_flag(
            d._flag_ptr(self.rank, phase + peer),
            d._counter_ptr(d._wait_counters, phase + peer),
        )

    def _barrier(self, phase, peers):
        # All publications precede waits, preventing a mutual-wait deadlock.
        for peer in peers:
            self._publish(phase, peer)
        for peer in peers:
            self._wait(phase, peer)

    def run(
        self, local, width, consume, *, producer_delay_cycles=0, consumer_delay_cycles=0
    ):
        """Call consume on borrowed [sender, owner_row, byte] storage.

        consume must enqueue all reads on the current stream and return only
        an independent Tensor or a Python bool diagnostic verdict.
        Diagnostic delays never run in timed measurements.
        A failure after entering the protocol must abort the distributed test;
        there is intentionally no unsafe rank-local fallback.
        """
        d = self.dma
        rows = local.shape[0]
        splits, offsets, owned, row_bytes = layout(rows, width, self.rank)
        main = torch.cuda.current_stream(d.device)
        if (
            local.dtype != torch.uint8
            or tuple(local.shape) != (rows, row_bytes)
            or not local.is_contiguous()
            or local.device != d.device
            or d._closed
            or self.busy
            or self.poisoned
            or torch.cuda.is_current_stream_capturing()
            or self.generations >= MAX_GENERATIONS
        ):
            raise ValueError("Invalid borrowed-channel call or generation exhausted")
        if self.stream is not None and self.stream != main.cuda_stream:
            raise ValueError("Borrowed DMA channel is bound to one compute stream")
        self.stream = main.cuda_stream
        self.busy = True
        self.generations += 1
        peers = [(self.rank + offset) % 4 for offset in (1, 2, 3)]
        try:
            # Prior ring scratch is not globally dead until every rank is ready.
            self._barrier(READY, peers)
            if producer_delay_cycles:
                torch.cuda._sleep(producer_delay_cycles)
            d._input_ready.record(main)
            with torch.cuda.stream(d._copy_stream):
                d._copy_stream.wait_event(d._input_ready)
                for index, peer in enumerate(peers):
                    d._kernels.dma_copy(
                        d._scratch_base[peer] + self.rank * splits[peer] * row_bytes,
                        local.data_ptr() + offsets[peer] * row_bytes,
                        splits[peer] * row_bytes,
                    )
                    d._copied_events[index].record(d._copy_stream)
                d._kernels.dma_copy(
                    d._scratch_base[self.rank] + self.rank * owned * row_bytes,
                    local.data_ptr() + offsets[self.rank] * row_bytes,
                    owned * row_bytes,
                )
                d._copied_events[3].record(d._copy_stream)
            with torch.cuda.stream(d._flag_stream):
                for index, peer in enumerate(peers):
                    d._flag_stream.wait_event(d._copied_events[index])
                    self._publish(DATA, peer)
            for peer in peers:
                self._wait(DATA, peer)
            main.wait_event(d._copied_events[3])
            received = self.tensor_view(
                d._scratch_base[self.rank],
                (4, owned, row_bytes),
                dtype=torch.uint8,
                device=d.device,
            )
            if consumer_delay_cycles:
                torch.cuda._sleep(consumer_delay_cycles)
            result = consume(received)
            if not isinstance(result, (torch.Tensor, bool)):
                raise TypeError("consume must return an independent Tensor or bool")
            if isinstance(result, torch.Tensor):
                base = d._scratch_base[self.rank]
                if base <= result.data_ptr() < base + 6 * d.shard_capacity:
                    raise ValueError("Borrowed scratch must not escape consume")
            # All scratch reads precede DONE; no peer can reenter the ring early.
            del received
            main.wait_stream(d._copy_stream)
            main.wait_stream(d._flag_stream)
            self._barrier(DONE, peers)
            return result
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.busy = False
