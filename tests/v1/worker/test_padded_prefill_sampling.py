# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded restored-prefill inputs and target-only sampling regressions."""

from itertools import accumulate

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def run(fn, query_lengths, new_request, graph):
    # Batch order differs from request-state order, like a real mixed admission.
    count = len(query_lengths)
    mapping = torch.arange(count - 1, -1, -1, device="cuda", dtype=torch.int32)
    query = torch.tensor(
        [0] + list(accumulate(query_lengths)),
        device="cuda",
        dtype=torch.int32,
    )
    seq = torch.tensor(
        [
            8191 + q if fresh else 8200 + q
            for q, fresh in zip(query_lengths, new_request)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    prefill = torch.full((count,), 8192, device="cuda", dtype=torch.int32)
    last = torch.tensor(
        [18000 + i for i in range(count)], device="cuda", dtype=torch.int32
    )
    drafts = torch.tensor(
        [[19000 + i * 3 + j for j in range(3)] for i in range(count)],
        device="cuda",
        dtype=torch.int32,
    )
    prepared = torch.full(
        (sum(query_lengths),), 23456, device="cuda", dtype=torch.int32
    )
    expected = []
    for batch, (q, fresh) in enumerate(zip(query_lengths, new_request)):
        state = count - 1 - batch
        if fresh:
            last[state] = 8190
            drafts[state].zero_()
            prepared[int(query[batch])] = 8191
            expected.extend([8191] + [0] * (q - 1))
        else:
            expected.extend(
                [18000 + state] + [19000 + state * 3 + j for j in range(q - 1)]
            )
    ids = prepared.clone()

    def call():
        return fn(ids, mapping, last, query, seq, prefill, drafts, query, len(ids))

    logits = call()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            logits = call()
        ids.copy_(prepared)
        capture.replay()
    torch.accelerator.synchronize()
    return dict(
        query_lengths=query_lengths,
        new_request=new_request,
        graph=graph,
        actual=ids.cpu().tolist(),
        expected=expected,
        input_equal=ids.cpu().tolist() == expected,
        logits=logits.cpu().tolist(),
        logits_equal=logits.cpu().tolist() == list(range(len(ids))),
    )


def partial(fn, graph):
    ids = torch.tensor([700, 701, 702], device="cuda", dtype=torch.int32)
    mapping = torch.tensor([0], device="cuda", dtype=torch.int32)
    last = torch.tensor([699], device="cuda", dtype=torch.int32)
    query = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    seq = torch.tensor([703], device="cuda", dtype=torch.int32)
    prefill = torch.tensor([8192], device="cuda", dtype=torch.int32)
    drafts = torch.zeros((1, 3), device="cuda", dtype=torch.int32)
    cumulative = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    def call():
        return fn(ids, mapping, last, query, seq, prefill, drafts, cumulative, 1)

    logits = call()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            logits = call()
        capture.replay()
    torch.accelerator.synchronize()
    return dict(
        partial_prefill=True,
        graph=graph,
        actual=ids.cpu().tolist(),
        input_equal=ids.cpu().tolist() == [700, 701, 702],
        logits=logits.cpu().tolist(),
        logits_equal=logits.cpu().tolist() == [2],
    )


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "lengths,fresh",
    [
        ([1], [True]),
        ([4, 4, 4, 4], [False, True, True, True]),
        ([4, 4], [False, False]),
    ],
)
def test_combine_preserves_prefill_input(lengths, fresh, graph):
    from vllm.v1.worker.gpu.input_batch import combine_sampled_and_draft_tokens

    result = run(combine_sampled_and_draft_tokens, lengths, fresh, graph)
    assert result["input_equal"], result
    assert result["logits_equal"], result


@pytest.mark.parametrize("graph", [False, True])
def test_combine_partial_prefill(graph):
    from vllm.v1.worker.gpu.input_batch import combine_sampled_and_draft_tokens

    result = partial(combine_sampled_and_draft_tokens, graph)
    assert result["input_equal"], result
    assert result["logits_equal"], result


@pytest.mark.parametrize("temp", [0.0, 0.7, 1.0])
@pytest.mark.parametrize("draftmode", ["random", "nan", "none"])
@pytest.mark.parametrize("block", [False, True])
def test_padded_prefill_uses_target_distribution(temp, draftmode, block):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample as new,
    )

    torch.manual_seed(173)
    D = "cuda"
    B = 4
    V = 257
    mapping = torch.tensor([3, 0, 2, 1], device=D, dtype=torch.int32)
    cu = torch.arange(0, 17, 4, device=D, dtype=torch.int32)
    expanded = mapping.repeat_interleave(4)
    local = torch.arange(4, device=D, dtype=torch.int32).repeat(4)
    pos = torch.arange(8191, 8195, device=D, dtype=torch.int64).repeat(4)
    seed = torch.tensor([173, 174, 175, 176], device=D, dtype=torch.int64)
    mask = torch.tensor([False, True, True, True], device=D)
    target = torch.randn(16, V, device=D)
    sampled = torch.zeros(16, device=D, dtype=torch.int64)
    temperature = torch.full((4,), temp, device=D)
    draft = None if draftmode == "none" else torch.randn(4, 3, V, device=D)
    if draftmode == "nan":
        assert draft is not None
        draft[mapping[1:].long()] = float("nan")
    args = (
        target,
        draft,
        sampled,
        cu,
        pos,
        mapping,
        expanded,
        local,
        temperature,
        seed,
        3,
    )
    kw = {"use_block_verification": block}
    actual, count = new(*args, **kw, padded_prefill=mask)
    baseline, basecount = new(*args, **kw)
    # Ordinary ongoing row must retain old exact behavior.
    n = int(count[0])
    assert n == int(basecount[0])
    assert torch.equal(actual[0, :n], baseline[0, :n])
    refargs = (
        target[::4].contiguous(),
        None,
        sampled[::4].contiguous(),
        torch.arange(5, device=D, dtype=torch.int32),
        pos[::4].contiguous(),
        mapping,
        mapping,
        torch.zeros(4, device=D, dtype=torch.int32),
        temperature,
        seed,
        3,
    )
    reference, refcount = new(*refargs, **kw)
    assert torch.equal(count[1:], torch.ones(3, device=D, dtype=count.dtype))
    assert torch.equal(actual[1:, 0], reference[1:, 0]), (
        temp,
        draftmode,
        block,
        actual[:, 0].tolist(),
        reference[:, 0].tolist(),
    )
    # None optional mask leaves every ordinary row unchanged.
    if draftmode != "nan":
        plain, pcount = new(*args, **kw)
        assert torch.equal(pcount, basecount)
        for r in range(B):
            assert torch.equal(
                plain[r, : int(pcount[r])], baseline[r, : int(pcount[r])]
            )
    # Capture/replay exercises the actual optional-mask Triton variants.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout, gcount = new(*args, **kw, padded_prefill=mask)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(gcount, count)
    for r in range(B):
        assert torch.equal(gout[r, : int(count[r])], actual[r, : int(count[r])])
