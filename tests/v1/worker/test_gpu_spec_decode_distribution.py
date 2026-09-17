# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual draft generation and rejection kernels as one sampler."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def _sample(p, q, *, steps=1, temperature=1.0, use_fp64=False, greedy_draft=False):
    n = 32768
    device = "cuda"
    p = torch.tensor(p, dtype=torch.float32, device=device)
    q = torch.tensor(q, dtype=torch.float32, device=device)
    vocab = p.numel()
    # Non-contiguous state slots and different positions exercise both mappings.
    slots = torch.arange(n, 0, -1, dtype=torch.int32, device=device)
    temps = torch.full((n + 2,), temperature, device=device)
    seeds = torch.full((n + 2,), 0xABCD, dtype=torch.int64, device=device)
    positions = 1000 + torch.arange(n, device=device, dtype=torch.int64) * 8
    q_logits = q.log().expand(n, -1)
    cache = torch.zeros(n + 2, steps, vocab, device=device)
    fake = SimpleNamespace(
        model=SimpleNamespace(compute_logits=lambda hidden: hidden),
        use_fp64_gumbel=use_fp64,
        _greedy_sample_draft=lambda hidden: hidden.argmax(dim=-1),
    )
    proposals = []
    for step in range(steps):
        proposals.append(
            DraftModelSpeculator.sample_draft(
                fake,
                q_logits,
                positions + step - 1,
                slots,
                temps,
                seeds,
                torch.tensor(step, dtype=torch.int64, device=device),
                None if greedy_draft else cache,
            )
        )
    draft_tokens = torch.zeros(n, steps + 1, dtype=torch.int64, device=device)
    draft_tokens[:, 1:] = torch.stack(proposals, dim=1)
    target = p.log()
    if temperature:
        target = target / temperature
    target = target.expand(n * (steps + 1), -1)
    pos = (positions[:, None] + torch.arange(steps + 1, device=device)).flatten()
    sampled, lengths = rejection_sample(
        target_logits=target,
        draft_logits=None if greedy_draft else cache,
        draft_sampled=draft_tokens.flatten(),
        cu_num_logits=torch.arange(
            0, (n + 1) * (steps + 1), steps + 1, dtype=torch.int32, device=device
        ),
        pos=pos,
        idx_mapping=slots,
        expanded_idx_mapping=slots.repeat_interleave(steps + 1),
        expanded_local_pos=torch.arange(
            steps + 1, dtype=torch.int32, device=device
        ).repeat(n),
        temperature=temps,
        seed=seeds,
        num_speculative_steps=steps,
        use_fp64=use_fp64,
    )
    expected = target[0].softmax(-1).double()
    return sampled, lengths, expected


@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_probabilistic_composition_matches_target(use_fp64, steps, temperature):
    sampled, lengths, expected = _sample(
        [0.1, 0.3, 0.6],
        [0.7, 0.2, 0.1],
        steps=steps,
        temperature=temperature,
        use_fp64=use_fp64,
    )
    for step in range(steps + 1):
        tokens = sampled[lengths > step, step]
        observed = torch.bincount(tokens, minlength=3).double()
        n = tokens.numel()
        sigma = (n * expected * (1 - expected)).sqrt().clamp_min(1)
        z = (observed - n * expected).abs() / sigma
        assert z.max().item() < 8, (
            f"step={step}, observed={(observed / n).tolist()}, "
            f"expected={expected.tolist()}, z={z.tolist()}"
        )


def test_identical_distributions_accept_all_drafts():
    _, lengths, _ = _sample([0.1, 0.3, 0.6], [0.1, 0.3, 0.6], steps=3)
    assert torch.all(lengths == 4)


def test_zero_temperature_returns_target_argmax():
    sampled, lengths, _ = _sample(
        [0.1, 0.3, 0.6], [0.7, 0.2, 0.1], steps=3, temperature=0.0
    )
    assert torch.all(lengths == 1)
    assert torch.all(sampled[:, 0] == 2)


def test_one_hot_greedy_proposals_match_target():
    sampled, _, expected = _sample([0.1, 0.3, 0.6], [0.7, 0.2, 0.1], greedy_draft=True)
    observed = torch.bincount(sampled[:, 0], minlength=3).double()
    n = sampled.shape[0]
    sigma = (n * expected * (1 - expected)).sqrt()
    assert torch.all((observed - n * expected).abs() < 8 * sigma)
