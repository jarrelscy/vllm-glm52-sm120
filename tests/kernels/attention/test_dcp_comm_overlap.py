# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exactness + capture-legality tests for VLLM_GLM_COMM_OVERLAP
(vllm/v1/attention/ops/dcp_comm_overlap.py).

Single-GPU: the DCP all-gather is faked with a deterministic per-rank
transform executed with the SAME stream/event structure ProcessGroupNCCL
uses (collective kernels on a side "comm" stream forked from the current
stream; ``work.wait()`` blocks the current stream on the comm stream).
A long ``torch.cuda._sleep`` is injected before the fake collective's
kernels so that any missing dependency edge would deterministically expose
a race (the overlapped compute would read stale data).

Checks:
  1. AsyncAllGather.wait() output is bit-identical to the serial
     (sync-collective) reference, with independent compute interleaved
     while the collective is in flight.
  2. The deferred indexer-merge protocol (stash/consume) produces a
     bit-identical topk buffer vs the serial merge order, including the
     stale-stash self-heal path.
  3. The whole overlapped schedule records and replays inside
     torch.cuda.CUDAGraph (multi-stream capture with cross-stream events is
     capture-legal), replay results bit-identical to serial eager for
     fresh inputs.
"""

import importlib.util
import os
import sys

import torch

try:
    import pytest
except ImportError:  # standalone mode (e.g. prod container)
    class _FakeMark:
        def skipif(self, *a, **k):
            return lambda f: f

    class _FakePytest:
        mark = _FakeMark()

        @staticmethod
        def fixture(*a, **k):
            return lambda f: f

    pytest = _FakePytest()

# Import the module under test directly by path so the test also runs in
# environments whose installed vllm predates it (e.g. the prod container).
_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD_PATH = os.path.join(
    _HERE, "..", "..", "..", "vllm", "v1", "attention", "ops", "dcp_comm_overlap.py"
)
_MOD_PATH = os.path.normpath(_MOD_PATH)
if os.environ.get("DCP_COMM_OVERLAP_MODULE"):
    _MOD_PATH = os.environ["DCP_COMM_OVERLAP_MODULE"]

spec = importlib.util.spec_from_file_location("dcp_comm_overlap_ut", _MOD_PATH)
dcp_comm_overlap = importlib.util.module_from_spec(spec)
sys.modules["dcp_comm_overlap_ut"] = dcp_comm_overlap
spec.loader.exec_module(dcp_comm_overlap)

AsyncAllGather = dcp_comm_overlap.AsyncAllGather
stash_pending_dcp_merge = dcp_comm_overlap.stash_pending_dcp_merge
consume_pending_dcp_merge = dcp_comm_overlap.consume_pending_dcp_merge

WORLD_SIZE = 4
SLEEP_CYCLES = 2_000_000  # ~1ms at ~2GHz: dwarfs the interleaved compute


def _rank_transform(x: torch.Tensor, rank: int) -> torch.Tensor:
    # Deterministic, dtype-preserving stand-in for "rank r's shard".
    return x * (rank + 1) + rank


class _FakeWork:
    """Mimics ProcessGroupNCCL Work: collective ran on a comm stream; wait()
    blocks the caller's current stream on it."""

    def __init__(self, comm_stream: torch.cuda.Stream):
        self._comm_stream = comm_stream

    def wait(self):
        torch.cuda.current_stream().wait_stream(self._comm_stream)


class _FakeGroup:
    """Single-GPU stand-in for the DCP GroupCoordinator.

    all_gather_into_tensor semantics: out[r*B:(r+1)*B] = rank r's tensor.
    Sync and async paths compute rank shards with the SAME ops in the SAME
    order, so their results are directly comparable bit-for-bit; only the
    scheduling differs (async runs on a forked comm stream after a long
    sleep, exposing any missing dependency edge as stale data).
    """

    def __init__(self, sleep: bool = True):
        self.world_size = WORLD_SIZE
        self.rank_in_group = 0
        self.device_group = self  # AsyncAllGather passes this to dist.*
        self.comm_stream = torch.cuda.Stream()
        self.sleep = sleep

    def _fill(self, output: torch.Tensor, input_: torch.Tensor):
        b = input_.shape[0]
        for r in range(WORLD_SIZE):
            output[r * b : (r + 1) * b].copy_(_rank_transform(input_, r))

    # --- sync reference (mirrors DeviceCommunicatorBase.all_gather) ---
    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * WORLD_SIZE,) + input_size[1:]
        output = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
        self._fill(output, input_)
        output = output.reshape((WORLD_SIZE,) + input_size)
        output = output.movedim(0, dim)
        return output.reshape(
            input_size[:dim]
            + (WORLD_SIZE * input_size[dim],)
            + input_size[dim + 1 :]
        )

    # --- async fake dist.all_gather_into_tensor ---
    def fake_all_gather_into_tensor(self, output, input_, group=None, async_op=False):
        assert group is self
        if not async_op:
            self._fill(output, input_)
            return None
        main = torch.cuda.current_stream()
        self.comm_stream.wait_stream(main)  # fork: comm waits for input ready
        with torch.cuda.stream(self.comm_stream):
            if self.sleep:
                torch.cuda._sleep(SLEEP_CYCLES)
            self._fill(output, input_)
        return _FakeWork(self.comm_stream)


@pytest.fixture(autouse=True)
def _patch_dist(monkeypatch):
    """Route the module-under-test's dist.all_gather_into_tensor to the fake
    when the group is a _FakeGroup (identified by duck-typing)."""

    def router(output, input_, group=None, async_op=False):
        assert isinstance(group, _FakeGroup)
        return group.fake_all_gather_into_tensor(
            output, input_, group=group, async_op=async_op
        )

    monkeypatch.setattr(dcp_comm_overlap.dist, "all_gather_into_tensor", router)
    yield
    # Never leave a pending merge behind for the next test.
    consume_pending_dcp_merge()


def _independent_compute(topk: torch.Tensor, block_table: torch.Tensor):
    """Stand-in for the fills + triton index-convert that run while the AG is
    in flight (integer ops, same inputs both paths; capture-safe: no
    host syncs)."""
    conv = torch.where(
        topk >= 0,
        topk * 2 + block_table.flatten()[0],
        torch.full_like(topk, -1),
    )
    empty_rows = (conv == -1).all(dim=-1)
    return conv, empty_rows


def _serial_reference(group, q, topk, block_table, w):
    gathered = group.all_gather(q, dim=1)
    conv, empty_rows = _independent_compute(topk, block_table)
    out = torch.matmul(gathered.reshape(q.shape[0], -1), w)
    return gathered, conv, empty_rows, out


def _overlapped(group, q, topk, block_table, w):
    ag = AsyncAllGather(group, q, dim=1)
    # Independent compute interleaved while the fake collective (delayed by
    # the sleep) is still running on the comm stream.
    conv, empty_rows = _independent_compute(topk, block_table)
    gathered = ag.wait()
    out = torch.matmul(gathered.reshape(q.shape[0], -1), w)
    return gathered, conv, empty_rows, out


def _make_inputs(device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((4, 16, 576), dtype=torch.float32, device=device, generator=g)
    q = q.to(torch.bfloat16)
    topk = torch.randint(
        -1, 1000, (4, 64), dtype=torch.int32, device=device, generator=g
    )
    block_table = torch.randint(
        0, 100, (4, 8), dtype=torch.int32, device=device, generator=g
    )
    w = torch.randn(
        (WORLD_SIZE * 16 * 576, 32), dtype=torch.float32, device=device, generator=g
    ).to(torch.bfloat16)
    return q, topk, block_table, w


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_async_allgather_bit_exact_vs_serial():
    device = "cuda"
    group = _FakeGroup(sleep=True)
    q, topk, block_table, w = _make_inputs(device)

    ref = _serial_reference(group, q, topk, block_table, w)
    got = _overlapped(group, q, topk, block_table, w)
    torch.cuda.synchronize()

    for name, a, b in zip(("gathered", "conv", "empty_rows", "out"), ref, got):
        assert torch.equal(a, b), f"{name} mismatch (overlap vs serial)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_deferred_merge_protocol_bit_exact():
    device = "cuda"
    group = _FakeGroup(sleep=True)
    g = torch.Generator(device=device).manual_seed(1)
    packed = torch.randn((4, 64, 2), dtype=torch.float32, device=device, generator=g)

    # Serial: gather + "merge" (a deterministic selection into the buffer).
    buf_ref = torch.empty((4, 64), dtype=torch.float32, device=device)
    gathered_ref = group.all_gather(packed, dim=1)
    buf_ref.copy_(gathered_ref[:, :64, 0])

    # Deferred: async AG, stash the merge, run "independent" work, consume.
    buf = torch.empty_like(buf_ref)
    ag = AsyncAllGather(group, packed, dim=1)

    def _finish():
        gathered = ag.wait()
        buf.copy_(gathered[:, :64, 0])

    stash_pending_dcp_merge(_finish)
    _ = torch.ones((512, 512), device=device) @ torch.ones((512, 512), device=device)
    consume_pending_dcp_merge()
    torch.cuda.synchronize()
    assert torch.equal(buf, buf_ref)

    # Stale-stash self-heal: stashing twice completes the first merge.
    buf2 = torch.zeros_like(buf_ref)
    ag2 = AsyncAllGather(group, packed, dim=1)
    stash_pending_dcp_merge(lambda: buf2.copy_(ag2.wait()[:, :64, 1]))
    ag3 = AsyncAllGather(group, packed, dim=1)
    stash_pending_dcp_merge(lambda: ag3.wait())  # forces heal of the first
    torch.cuda.synchronize()
    assert torch.equal(buf2, group.all_gather(packed, dim=1)[:, :64, 1])
    consume_pending_dcp_merge()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_overlap_schedule_cudagraph_capture_replay():
    """The overlapped schedule (side-stream collective + deferred wait +
    interleaved compute) must record into a CUDA graph and replay
    bit-exactly. Cross-stream events inside capture are legal; this
    exercises exactly the structure used under FULL_AND_PIECEWISE."""
    device = "cuda"
    group = _FakeGroup(sleep=True)
    q, topk, block_table, w = _make_inputs(device)

    # Static input/output buffers for graph replay.
    q_static = q.clone()
    topk_static = topk.clone()

    # Warmup on a non-default stream (required before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            _ = _overlapped(group, q_static, topk_static, block_table, w)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = _overlapped(group, q_static, topk_static, block_table, w)
    torch.cuda.synchronize()

    for seed in (7, 8, 9):
        q2, topk2, _, _ = _make_inputs(device, seed=seed)
        q_static.copy_(q2)
        topk_static.copy_(topk2)
        graph.replay()
        torch.cuda.synchronize()
        ref = _serial_reference(group, q2, topk2, block_table, w)
        torch.cuda.synchronize()
        for name, a, b in zip(("gathered", "conv", "empty_rows", "out"), ref, out_static):
            assert torch.equal(a, b), f"replay {name} mismatch @seed={seed}"


if __name__ == "__main__":
    # Allow running standalone (e.g. inside the prod container) without pytest.
    def router(output, input_, group=None, async_op=False):
        return group.fake_all_gather_into_tensor(
            output, input_, group=group, async_op=async_op
        )

    dcp_comm_overlap.dist.all_gather_into_tensor = router
    test_async_allgather_bit_exact_vs_serial()
    consume_pending_dcp_merge()
    test_deferred_merge_protocol_bit_exact()
    consume_pending_dcp_merge()
    test_overlap_schedule_cudagraph_capture_replay()
    print("ALL TESTS PASSED (bit-exact overlap + capture/replay)")
