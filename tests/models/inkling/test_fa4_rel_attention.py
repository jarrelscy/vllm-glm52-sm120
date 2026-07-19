# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the Inkling FA4 relative-attention score-mod kernel.

Checks ``inkling_fa4_rel_attention`` against a pure-PyTorch reference that
implements the relative bias exactly as documented in the Inkling architecture
guide::

    logit(i, j, h) = (1 / head_dim) * dot(q[i, h], k[j, h]) + rel_bias(i, j, h)
    rel_bias(i, j, h) = rel_logits[i, h, i - j]   if 0 <= i - j < rel_extent
                      = 0                          otherwise

with causal (and optionally sliding-window) masking handled by the backend.
"""

import pytest
import torch

from vllm.models.inkling.nvidia.attention import (
    InklingAttention,
    compute_log_scaling_tau,
)
from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_num_splits,
    inkling_fa4_rel_attention,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

_cap = current_platform.get_device_capability() if current_platform.is_cuda() else None

NUM_HEADS = [
    (4, 4),
    (8, 2),
    # Production TP4 per-rank head counts (task #101/#104 reopened gap): the
    # real Inkling-512k-NVFP4-AQLM-hybrid checkpoint's full-attention layers
    # (num_attention_heads=64, num_key_value_heads=8) shard to 16 q-heads /
    # 2 kv-heads per rank at tensor_parallel_size=4 -- qhead_per_kvhead=8,
    # a ratio never previously exercised by this file's real-kernel tests
    # (only inkling_fa4_num_splits' scheduling-heuristic tests used (16, 2)
    # as scalar args, never through _run_case/the actual kernel). The SWA
    # layers (swa_num_attention_heads=64, swa_num_key_value_heads=16) shard
    # to 16 q-heads / 4 kv-heads per rank -- ratio 4, same ratio as (8, 2)
    # but double the absolute head count, worth confirming separately since
    # Pack-GQA addressing may be sensitive to absolute head count, not just
    # the ratio.
    (16, 2),  # prod full-attention, TP4
    (16, 4),  # prod SWA/local attention, TP4
]  # (num_heads, num_kv_heads)
GLOBAL_REL_EXTENTS = [128, 1024]
LOCAL_REL_EXTENTS = [128, 256]
HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16


def test_log_scaling_tau_matches_reference():
    positions = torch.tensor([0, 127999, 128000, 999999], dtype=torch.int64)
    actual = compute_log_scaling_tau(positions, 128000, 0.1)
    expected = 1.0 + 0.1 * torch.log(
        torch.clamp((positions + 1).float() / 128000.0, min=1.0)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_split_packed_kv_cache():
    attention = InklingAttention.__new__(InklingAttention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 8
    attention.kv_cache = torch.arange(2 * 3 * 4 * 16).reshape(2, 3, 4, 16)

    key_cache, value_cache = attention._split_kv_cache()

    assert key_cache.shape == value_cache.shape == (2, 4, 3, 8)
    torch.testing.assert_close(key_cache, attention.kv_cache[..., :8].transpose(1, 2))
    torch.testing.assert_close(value_cache, attention.kv_cache[..., 8:].transpose(1, 2))


def test_num_splits_hopper_is_unsplit(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=9, minor=0),
    )
    assert (
        inkling_fa4_num_splits(
            is_local=False,
            batch_size=1,
            max_query_len=1,
            num_heads=16,
            num_kv_heads=2,
            max_kv_len=1_048_576,
        )
        == 1
    )


@pytest.fixture
def blackwell_platform(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=10, minor=0),
    )


@pytest.mark.parametrize(
    ("batch_size", "max_query_len", "expected"),
    [
        (1, 1, (16, 32, 128, 128)),
        (8, 1, (2, 4, 8, 16)),
        (32, 1, (1, 1, 2, 4)),
        (1, 128, (2, 4, 8, 16)),
        (1, 2048, (1, 1, 1, 1)),
    ],
)
def test_num_splits_all_tp(blackwell_platform, batch_size, max_query_len, expected):
    actual = tuple(
        inkling_fa4_num_splits(
            is_local=False,
            batch_size=batch_size,
            max_query_len=max_query_len,
            num_heads=64 // tp,
            num_kv_heads=8 // tp,
            max_kv_len=131072,
        )
        for tp in (1, 2, 4, 8)
    )
    assert actual == expected


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_num_splits_local_is_unsplit(tp):
    assert (
        inkling_fa4_num_splits(
            is_local=True,
            batch_size=1,
            max_query_len=1,
            num_heads=64 // tp,
            num_kv_heads=16 // tp,
            max_kv_len=512,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("max_kv_len", "expected"),
    [(8192, 32), (65536, 64), (1048576, 128)],
)
@pytest.mark.parametrize("tp", [4, 8])
def test_num_splits_long_context_bound(blackwell_platform, tp, max_kv_len, expected):
    assert (
        inkling_fa4_num_splits(
            is_local=False,
            batch_size=1,
            max_query_len=1,
            num_heads=64 // tp,
            num_kv_heads=8 // tp,
            max_kv_len=max_kv_len,
        )
        == expected
    )


def _ref_rel_attn(
    q: torch.Tensor,  # [total_q, H, D]
    key_cache: torch.Tensor,  # [num_blocks, block, Hkv, D]
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,  # [total_q, H, rel_extent]
    *,
    q_lens: list[int],
    kv_lens: list[int],
    block_table: torch.Tensor,
    scale: float,
    rel_extent: int,
    window_left: int | None,
) -> torch.Tensor:
    num_kv_heads = key_cache.shape[2]
    num_heads = q.shape[1]
    g = num_heads // num_kv_heads
    bt = block_table.cpu().numpy()
    out = torch.empty_like(q)

    start = 0
    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        qi = q[start : start + ql].float()  # [ql, H, D]
        rl = rel_logits[start : start + ql].float()  # [ql, H, rel_extent]

        nblk = (kl + BLOCK_SIZE - 1) // BLOCK_SIZE
        blk = bt[i, :nblk]
        k = key_cache[blk].reshape(-1, num_kv_heads, HEAD_DIM)[:kl].float()
        v = value_cache[blk].reshape(-1, num_kv_heads, HEAD_DIM)[:kl].float()
        k = k.repeat_interleave(g, dim=1)  # [kl, H, D]
        v = v.repeat_interleave(g, dim=1)

        # [H, ql, kl]
        scores = torch.einsum("qhd,khd->hqk", qi, k) * scale

        dev = q.device
        qpos = torch.arange(ql, device=dev).view(ql, 1) + (kl - ql)  # query pos
        kpos = torch.arange(kl, device=dev).view(1, kl)
        dist = qpos - kpos  # [ql, kl] = i - j

        # Relative bias: rel_logits[i, h, dist] when 0 <= dist < rel_extent.
        in_rng = (dist >= 0) & (dist < rel_extent)  # [ql, kl]
        idx = dist.clamp(0, rel_extent - 1)
        # gather per head: bias[h, i, j] = rl[i, h, idx[i, j]]
        bias = rl.permute(1, 0, 2).gather(  # [H, ql, rel_extent]
            2, idx.unsqueeze(0).expand(num_heads, -1, -1)
        )  # [H, ql, kl]
        bias = torch.where(in_rng.unsqueeze(0), bias, torch.zeros_like(bias))
        scores = scores + bias

        mask = dist < 0  # causal
        if window_left is not None:
            mask = mask | (dist > window_left)
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        out[start : start + ql] = torch.einsum("hqk,khd->qhd", probs, v).to(q.dtype)
        start += ql
    return out


def _run_case(
    seq_lens,
    num_heads,
    num_kv_heads,
    rel_extent,
    window_left,
    seed=0,
    shared_block=False,
    bucketed_max_seqlen_q=False,
):
    # bucketed_max_seqlen_q=True mirrors the REAL call site
    # (attention.py:301: `max_seqlen_q = bucket_max_seqlen_q(md.max_query_len)`)
    # exactly: the scheduling bound handed to inkling_fa4_num_splits AND the
    # kernel itself is rounded UP to the next power of two, not the raw
    # per-batch max query length. Every pre-existing caller of this function
    # defaults to False (the harness's original, unbucketed behavior) so as
    # not to silently change what those tests exercise; new tests that want
    # production-faithful scheduling should pass True explicitly.
    torch.manual_seed(seed)
    device = "cuda"
    q_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    total_q = sum(q_lens)
    num_seqs = len(seq_lens)

    scale = 1.0 / HEAD_DIM
    sched_max_seqlen_q = (
        bucket_max_seqlen_q(max(q_lens)) if bucketed_max_seqlen_q else max(q_lens)
    )

    # q/k are RMS-normed in the model (unit-ish norm); normalize here so the
    # logit magnitudes are realistic and the bias is not numerically dwarfed.
    q = torch.randn(total_q, num_heads, HEAD_DIM, device=device, dtype=DTYPE)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)

    # Paged KV cache.
    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * max_blocks + 1
    key_cache = torch.randn(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )
    key_cache = torch.nn.functional.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )

    # Distinct blocks per sequence (block 0 left as a never-referenced pad),
    # UNLESS shared_block: this replicates GPUModelRunner.prepare_dummy_attn's
    # actual dummy-run addressing (block_table.py get_dummy_block_tables slices
    # the persistent, all-zero-initialized input_block_tables buffer -- so on
    # the very first flashinfer_autotune warmup call, before any real request
    # has ever populated it, EVERY concurrent request's block_table row is all
    # zeros, i.e. all requests alias physical block 0). This is qualitatively
    # different from this harness's normal distinct-valid-blocks-per-sequence
    # construction and is untested elsewhere in this file (task #101/#104).
    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
    if not shared_block:
        for i in range(num_seqs):
            block_table[i] = torch.arange(
                1 + i * max_blocks, 1 + (i + 1) * max_blocks, dtype=torch.int32
            )

    cu_seqlens_q = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    rel_logits = torch.randn(total_q, num_heads, rel_extent, device=device, dtype=DTYPE)

    window_size = (-1, -1) if window_left is None else (window_left, 0)

    num_splits = inkling_fa4_num_splits(
        is_local=window_left is not None,
        batch_size=num_seqs,
        max_query_len=sched_max_seqlen_q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_kv_len=max(kv_lens),
    )

    preallocated_out = torch.empty_like(q)
    out = inkling_fa4_rel_attention(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=sched_max_seqlen_q,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        num_splits=num_splits,
        out=preallocated_out,
    )
    assert out.data_ptr() == preallocated_out.data_ptr()
    out = out.view(total_q, num_heads, HEAD_DIM)

    ref = _ref_rel_attn(
        q,
        key_cache,
        value_cache,
        rel_logits,
        q_lens=q_lens,
        kv_lens=kv_lens,
        block_table=block_table,
        scale=scale,
        rel_extent=rel_extent,
        window_left=window_left,
    )

    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(64, 64)],  # single full prefill
        [(64, 64), (33, 33), (17, 17)],  # ragged prefill batch
        [(512, 512)],  # seq_len >> rel_extent (most keys get zero bias)
        [(300, 300), (512, 512), (129, 129)],  # large ragged batch
    ],
)
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@torch.inference_mode()
def test_full_attention(seq_lens, num_heads, rel_extent):
    # rel_extent=128 exercises the out-of-range (zero bias) path; 1024 covers all.
    # With the 512-token cases and rel_extent=128, query/seq lengths are far
    # larger than rel_extent so the vast majority of (i, j) pairs are out of
    # range and must contribute zero bias.
    _run_case(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(200, 512)],  # chunked prefill: q_len=200 (> rel_extent), 312 cached
        [(200, 512), (50, 300), (1, 400)],  # mixed chunked + decode
    ],
)
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@torch.inference_mode()
def test_chunked_prefill(seq_lens, num_heads, rel_extent):
    # q_len < kv_len with q_len itself larger than rel_extent (for the 128 case):
    # exercises the seqlen_k - seqlen_q offset together with the out-of-range path.
    _run_case(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(64, 64), (40, 40)],  # seq_len > window
        [(512, 512), (300, 300)],  # seq_len/query_len >> window
        [(1, 512)],  # decode with kv_len >> window
    ],
)
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@torch.inference_mode()
def test_sliding_window(seq_lens, num_heads, local_extent):
    # Local layers use window_size=(local_extent-1, 0) and rel_extent==local_extent.
    # With the 512-token cases, query/seq lengths far exceed the window so most
    # keys are masked out by the sliding window.
    _run_case(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent=local_extent,
        window_left=local_extent - 1,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 50)],
        [(1, 50), (1, 7), (1, 200)],
        [(1, 512), (1, 333)],  # kv_len >> rel_extent
    ],
)
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@torch.inference_mode()
def test_decode(seq_lens, num_heads, rel_extent):
    # q_len=1 with kv_len>q_len: the score-mod's seqlen_k - seqlen_q offset path.
    _run_case(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


# task #101/#104 (reopened): attempt9's real cudaErrorIllegalAddress crash traced
# to vllm/model_executor/warmup/kernel_warmup.py's flashinfer_autotune ->
# GPUModelRunner._dummy_run(num_tokens=scheduler_config.max_num_batched_tokens),
# which (when TP>1, so use_persistent_cache=False) builds
# num_reqs=min(num_tokens, max_num_reqs) requests of num_tokens//num_reqs tokens
# each. With the smoke test's defaults (max_num_batched_tokens=2048,
# max_num_seqs=128) that is *128 concurrent 16-token prefill requests in one
# varlen batch* -- a request-COUNT extreme no existing seq_lens parametrization
# in this file comes close to (max prior was 3-4 concurrent sequences). This is
# a batch-composition axis independent of (and in addition to) the ratio=8/4
# Pack-GQA gap; test it at both production head ratios.
@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@torch.inference_mode()
def test_full_attention_autotune_dummy_run_batch_shape(num_heads, rel_extent):
    # Exact shape of flashinfer_autotune's dummy prefill: 128 reqs x 16 tokens,
    # full (non-causal-truncated) prefill, q_len == kv_len for every request.
    seq_lens = [(16, 16)] * 128
    _run_case(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@torch.inference_mode()
def test_sliding_window_autotune_dummy_run_batch_shape(num_heads, local_extent):
    # Same 128x16 dummy-run shape, but through the sliding-window/local path
    # (production SWA layers are what actually run this ratio in the real
    # 66-layer forward that crashed).
    seq_lens = [(16, 16)] * 128
    _run_case(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent=local_extent,
        window_left=local_extent - 1,
    )


# Sharper variant of the above: on the real, very-first flashinfer_autotune
# dummy run, GPUModelRunner.prepare_dummy_attn's block_table isn't just a
# generic 128-request batch -- it's block_table.py's get_dummy_block_tables(),
# which slices the persistent input_block_tables buffer that is torch.zeros_like
# at init and has never yet been written by any real request. So all 128
# concurrent requests' block_table rows are literally all-zero (all alias
# physical KV-cache block 0), simultaneously with slot_mappings all == -1
# (PAD_SLOT_ID, i.e. cache writes skipped). No existing test in this file
# exercises many concurrent requests aliasing the SAME physical block.
@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@torch.inference_mode()
def test_full_attention_shared_zero_block_dummy_run(num_heads, rel_extent):
    seq_lens = [(16, 16)] * 128
    _run_case(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent,
        window_left=None,
        shared_block=True,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@torch.inference_mode()
def test_sliding_window_shared_zero_block_dummy_run(num_heads, local_extent):
    seq_lens = [(16, 16)] * 128
    _run_case(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent=local_extent,
        window_left=local_extent - 1,
        shared_block=True,
    )


# task #101/#104 (reopened again): re-tracing kernel_warmup.py against the
# CURRENT source shows the *_autotune_dummy_run_batch_shape / *_zero_block
# tests above target a call site that, per the actual code, never reaches the
# FA4 kernel at all: flashinfer_autotune()'s own runner._dummy_run(num_tokens=
# max_num_batched_tokens, is_profile=True) passes neither force_attention=True
# nor a cudagraph_runtime_mode that resolves to FULL, so GPUModelRunner.
# _dummy_run's `if force_attention or cudagraph_runtime_mode == FULL:` guard
# (gpu_model_runner.py ~5935) is False -- attn_metadata is never built (stays
# None) and InklingAttention.forward()'s `if not isinstance(attn_metadata,
# dict): attn_output.zero_()` guard (attention.py:241-242) short-circuits
# before ever calling inkling_fa4_rel_attention. So that specific 128-req
# dummy-run path can't be attempt9's crash site as literally written today.
#
# The actual FIRST call in kernel_warmup() that forces attn_metadata to be
# built as a real dict -- and therefore the first place the real FA4 kernel
# can fire during warmup -- is the very next step, "Warming up FlashInfer
# attention" (kernel_warmup.py ~147-156): runner._dummy_run(num_tokens=16,
# force_attention=True, create_mixed_batch=True). Per _dummy_run's
# create_mixed_batch branch (gpu_model_runner.py ~5824-5833) with
# max_num_seqs=128 (attempt9's default) and num_tokens=16:
#   num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2) = min(127, 8) = 8
#   num_prefill_tokens = num_tokens - num_decode_tokens = 8
#   num_reqs = num_decode_tokens + 1 = 9
#   seq_lens (cache_seqlens) = [1]*8 + [num_prefill_tokens + 1] = [1]*8 + [9]
#   query lens              = [1]*8 + [num_prefill_tokens]     = [1]*8 + [8]
# i.e. 8 single-token DECODE requests (q_len=1, kv_len=1) mixed with ONE
# 8-token PREFILL request (q_len=8, kv_len=9) in the SAME varlen batch. This
# is also the first force_attention=True call of the whole warmup sequence
# (it runs after flashinfer_autotune and all the earlier no-attention warmup
# steps), so block_table rows for all 9 requests are still the persistent,
# never-yet-written all-zero buffer -- same shared-physical-block-0 condition
# as test_*_shared_zero_block_dummy_run above, but combined with a genuine
# decode+prefill-mixed batch composition that no existing test in this file
# exercises (test_decode is all-decode-uniform; the ragged/chunked-prefill
# tests are all-prefill; nothing mixes single-token decode rows with a
# multi-token prefill row in one call). Test at both production head ratios,
# both block-table conditions, and with production-faithful bucketed
# max_seqlen_q scheduling (bucket_max_seqlen_q(8) == 8, a no-op here, but
# passed explicitly for fidelity/documentation).
@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@pytest.mark.parametrize("shared_block", [False, True])
@torch.inference_mode()
def test_kernel_warmup_flashinfer_attention_mixed_batch_full_attention(
    num_heads, rel_extent, shared_block
):
    # Exact shape of kernel_warmup.py's "Warming up FlashInfer attention" step
    # (num_tokens=16, force_attention=True, create_mixed_batch=True) at
    # attempt9's real max_num_seqs=128 default: 8 decode rows + 1 prefill row.
    seq_lens = [(1, 1)] * 8 + [(8, 9)]
    _run_case(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent,
        window_left=None,
        shared_block=shared_block,
        bucketed_max_seqlen_q=True,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@pytest.mark.parametrize("shared_block", [False, True])
@torch.inference_mode()
def test_kernel_warmup_flashinfer_attention_mixed_batch_sliding_window(
    num_heads, local_extent, shared_block
):
    # Same mixed decode+prefill shape, through the SWA/local path -- this is
    # the ratio (16, 4) confirmed from attempt9's own debug log
    # (pack_gqa=True, qhead_per_kvhead=4), so this specific parametrization
    # is the closest single-kernel match to the real crashing call found so
    # far.
    seq_lens = [(1, 1)] * 8 + [(8, 9)]
    _run_case(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent=local_extent,
        window_left=local_extent - 1,
        shared_block=shared_block,
        bucketed_max_seqlen_q=True,
    )


# task #88 (MTP speculative decoding, ns=8 gate): the target model's verify
# pass is structurally a chunked prefill -- each of `num_seqs` concurrent
# sequences appends MTP_NS freshly-drafted tokens on top of its OWN existing
# KV context in a single batched causal forward (q_len == ns fixed per
# request, kv_len == existing_ctx + ns, heterogeneous across the batch since
# real decode batches are never context-uniform: every request sits at its
# own point in its own generation). This is a distinct axis from the
# autotune_dummy_run_batch_shape tests above, which correctly model a
# from-scratch *profiling* pass (q_len == kv_len, all requests identical) --
# untested until now was many concurrent requests each doing a >1-token
# causal chunked-prefill step with irregular, non-uniform kv_lens, at the
# production Pack-GQA ratios.
MTP_NS = 8  # 8 draft tokens per verify step


def _mtp_verify_seq_lens(
    num_seqs: int, base_ctx: int, ns: int = MTP_NS
) -> list[tuple[int, int]]:
    # Irregular (non-arithmetic-run) per-request jitter so no two requests in
    # the batch share a kv_len/cu_seqlens_q boundary by construction -- a
    # genuinely non-trivial varlen pattern, not num_seqs copies of one tuple.
    jitter_mod = max(base_ctx // 2, 16)
    return [(ns, base_ctx + ns + (i * 37) % jitter_mod) for i in range(num_seqs)]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@pytest.mark.parametrize(
    ("num_seqs", "base_ctx"),
    [
        (32, 0),  # MTP kicking in on freshly-started requests (no prior ctx)
        (32, 511),  # mid-length existing context
        (32, 8191),  # long existing context, well past rel_extent=128's range
        (128, 511),  # production-scale concurrent-request count (cf. the
        # 128-request dummy-run tests above), but a chunked-prefill verify
        # shape instead of a from-scratch prefill
    ],
)
@torch.inference_mode()
def test_mtp_verify_ns8_full_attention(num_heads, rel_extent, num_seqs, base_ctx):
    seq_lens = _mtp_verify_seq_lens(num_seqs, base_ctx)
    _run_case(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@pytest.mark.parametrize(
    ("num_seqs", "base_ctx"),
    [
        (32, 0),
        (32, 511),
        (32, 8191),  # kv_len far exceeds the sliding window itself
        (128, 511),
    ],
)
@torch.inference_mode()
def test_mtp_verify_ns8_sliding_window(num_heads, local_extent, num_seqs, base_ctx):
    # Production SWA layers run at ratio (16,4) -- same absolute head count
    # class as full attention's (16,2) but double the kv heads.
    seq_lens = _mtp_verify_seq_lens(num_seqs, base_ctx)
    _run_case(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent=local_extent,
        window_left=local_extent - 1,
    )


def _mtp_ragged_seq_lens(base_ctx: int) -> list[tuple[int, int]]:
    # A single batch mixing MTP-verify requests at different draft lengths
    # (ns=8, ns=4 -- e.g. a shorter continuation) with plain single-token
    # decode requests (q_len=1) -- the realistic production case where MTP is
    # enabled but not every in-flight request is on a full ns=8 verify step
    # (a draft was rejected down to 1 token last round, or that request never
    # uses speculation at all). Distinct axis from the uniform-ns tests above:
    # this is the non-trivial *batch-composition* pattern, mirroring how the
    # 128-request/shared-zero-block tests above probed batch composition for
    # the pure-prefill dummy-run case.
    draft_lens = [8, 8, 4, 1, 8, 1, 1, 4]
    return [(dl, base_ctx + dl + i * 53) for i, dl in enumerate(draft_lens)]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@pytest.mark.parametrize("base_ctx", [0, 511, 8191])
@torch.inference_mode()
def test_mtp_ragged_verify_batch_full_attention(num_heads, rel_extent, base_ctx):
    seq_lens = _mtp_ragged_seq_lens(base_ctx)
    _run_case(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


# --- Real production q_len=9 (not q_len=8) MTP-verify shape ---
#
# Every "ns8" test above/below uses `_mtp_verify_seq_lens(..., ns=MTP_NS=8)`,
# i.e. q_len=8 per request. But the REAL engine's uniform-decode CUDA-graph
# shape is `uniform_decode_query_len = 1 + num_spec_tokens` (gpu_model_runner
# .py:854) -- for ns=8 MTP that is q_len=9 (8 draft tokens + 1 bonus/anchor
# token), NOT 8. And `InklingAttention._attention` buckets that scheduling
# bound up via `bucket_max_seqlen_q()` (attention.py:301) before calling the
# FA4 kernel: bucket_max_seqlen_q(8) == 8 is a no-op (already a power of two)
# but bucket_max_seqlen_q(9) == 16 is NOT -- a real 7-token-per-request
# scheduling-bound/actual-length gap. Every "ns8" q_len=8 test above is
# therefore accidentally exercising a bucketing no-op and has never tested
# the real padded-to-16 shape. These q_len=9 tests close that gap, using
# `_mtp_verify_seq_lens(..., ns=9)` (that helper's `ns` param is used
# directly as q_len, so ns=9 IS the real q_len=9 case, not a renamed
# ns=8) plus `bucketed_max_seqlen_q=True` to force the real
# bucket_max_seqlen_q(9)==16 scheduling bound through `_run_case`.
def _mtp_ragged_seq_lens_real_qlen(base_ctx: int) -> list[tuple[int, int]]:
    # Same batch-composition INTENT as `_mtp_ragged_seq_lens` above (full
    # ns=8 verify requests, partial ns=4 requests, plain q_len=1 decode
    # requests with zero draft tokens) but with the off-by-one fixed: a
    # request with `ns` draft tokens has a REAL total query length of
    # ns + 1 (the draft tokens plus the bonus/anchor token), so ns=8 -> 9,
    # ns=4 -> 5, and ns=0 (plain decode, no speculation at all) -> 1 (0+1,
    # already consistent -- there is no separate "ns=1" case here).
    draft_ns = [8, 8, 4, 0, 8, 0, 0, 4]
    q_lens = [n + 1 for n in draft_ns]
    return [(ql, base_ctx + ql + i * 53) for i, ql in enumerate(q_lens)]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@pytest.mark.parametrize(
    ("num_seqs", "base_ctx"),
    [(32, 0), (32, 511), (32, 8191), (128, 511)],
)
@torch.inference_mode()
def test_mtp_verify_real_qlen9_full_attention(num_heads, rel_extent, num_seqs, base_ctx):
    seq_lens = _mtp_verify_seq_lens(num_seqs, base_ctx, ns=9)
    _run_case(
        seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None,
        bucketed_max_seqlen_q=True,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@pytest.mark.parametrize(
    ("num_seqs", "base_ctx"),
    [(32, 0), (32, 511), (32, 8191), (128, 511)],
)
@torch.inference_mode()
def test_mtp_verify_real_qlen9_sliding_window(num_heads, local_extent, num_seqs, base_ctx):
    seq_lens = _mtp_verify_seq_lens(num_seqs, base_ctx, ns=9)
    _run_case(
        seq_lens, num_heads[0], num_heads[1], rel_extent=local_extent,
        window_left=local_extent - 1, bucketed_max_seqlen_q=True,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@pytest.mark.parametrize("base_ctx", [0, 511, 8191])
@torch.inference_mode()
def test_mtp_ragged_verify_batch_real_qlen9_full_attention(num_heads, rel_extent, base_ctx):
    seq_lens = _mtp_ragged_seq_lens_real_qlen(base_ctx)
    _run_case(
        seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None,
        bucketed_max_seqlen_q=True,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@pytest.mark.parametrize("base_ctx", [0, 511, 8191])
@torch.inference_mode()
def test_mtp_ragged_verify_batch_sliding_window(num_heads, local_extent, base_ctx):
    seq_lens = _mtp_ragged_seq_lens(base_ctx)
    _run_case(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent=local_extent,
        window_left=local_extent - 1,
    )


# task #88 (MTP ns=8 gate, CUDA-graph axis): none of the tests above have
# exercised the kernel under CUDA graph capture/replay at all -- they all run
# eagerly via torch.inference_mode(). The real engine never calls FA4 eagerly
# once graphs are warmed up: vllm/v1/worker/gpu/cudagraph_utils.py's
# CUDAGraphManager.capture() runs one warmup forward_fn(CUDAGraphMode.NONE)
# call (outside any graph, "with fresh attention state" for FULL mode) against
# a fixed set of static input tensors, then wraps a SECOND, identical-shape
# call in `with torch.cuda.graph(graph, self.pool): forward_fn(...)`, and
# every subsequent step is just `self.graphs[desc].replay()` -- literally
# re-running the exact same recorded kernel launches against whatever the
# static input buffers now contain (run_fullgraph()'s comment: "replay could
# overwrite static buffers while those copies are still in flight" makes
# explicit that replay depends on in-place-mutated content in the SAME
# addresses, never reallocation). PackGQA's compute_ptr indexes physical KV
# blocks through exactly the block_table/cache_seqlens tensors that change
# every decode step in real MTP verify -- so the risk this gate targets is:
# does the captured kernel launch correctly re-read FRESH block_table/
# cache_seqlens/q/k/v content on each replay, or did capture bake in
# capture-time pointer arithmetic / addressing that goes stale (silently
# wrong results, not necessarily a crash) once the underlying data changes
# but the tensors' addresses/shapes do not?
def _make_permuted_block_table(
    num_seqs: int, max_blocks: int, perm_seed: int
) -> torch.Tensor:
    # Distinct, randomly-permuted physical block ids per replay trial (never
    # block 0, which is reserved as a never-referenced pad elsewhere in this
    # file) -- same logical shape every trial (required for a valid replay),
    # but a genuinely different physical-block mapping each time, so a kernel
    # that incorrectly reused capture-time addressing would read the WRONG
    # cache content and diverge from the reference computed on the same
    # permutation.
    g = torch.Generator().manual_seed(perm_seed)
    total = num_seqs * max_blocks
    perm = torch.randperm(total, generator=g) + 1
    return perm.view(num_seqs, max_blocks).to(torch.int32)


def _run_case_cudagraph(
    seq_lens,
    num_heads,
    num_kv_heads,
    rel_extent,
    window_left,
    num_replays=4,
    seed=0,
    bucketed_max_seqlen_q=False,
):
    """CUDA-graph counterpart of `_run_case`: allocates static, fixed-address
    input/output tensors once, runs a warmup pass, captures ONE
    `inkling_fa4_rel_attention` call in a `torch.cuda.graph()` (mirroring
    CUDAGraphManager.capture()'s FULL-mode pattern), then replays it
    `num_replays` times, mutating the SAME static buffers in place (fresh
    random q/k/v/rel_logits content plus a fresh block_table permutation)
    before each replay, and checks each replay's output against
    `_ref_rel_attn` computed on that trial's actual data. A capture-time-only
    correctness bug (right on the first call, wrong under replay) would pass
    every other test in this file but fail here.

    bucketed_max_seqlen_q=True mirrors the REAL FULL-mode uniform-decode
    graph-capture call site exactly: `_warmup_and_capture` passes
    `max_query_len = uniform_decode_query_len = 1 + num_spec_tokens` into
    `_dummy_run`, and `InklingAttention._attention` buckets that up via
    `bucket_max_seqlen_q()` before it ever reaches
    `inkling_fa4_num_splits`/`inkling_fa4_rel_attention` -- for ns=8 MTP that
    is bucket_max_seqlen_q(9) == 16, a real 7-slot scheduling-bound/actual-
    per-request-length gap that no unbucketed (default False) caller of this
    function has ever exercised. Defaults to False so pre-existing callers'
    behavior is unchanged.
    """
    torch.manual_seed(seed)
    device = "cuda"
    q_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    total_q = sum(q_lens)
    num_seqs = len(seq_lens)
    scale = 1.0 / HEAD_DIM
    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * max_blocks + 1
    window_size = (-1, -1) if window_left is None else (window_left, 0)
    sched_max_seqlen_q = (
        bucket_max_seqlen_q(max(q_lens)) if bucketed_max_seqlen_q else max(q_lens)
    )

    # Static buffers: allocated once, fixed addresses for the lifetime of the
    # graph. All content is written via copy_()/in-place ops from here on,
    # never reassigned -- reassigning any of these would silently break the
    # graph (replay would keep using the ORIGINAL tensor's memory).
    q = torch.empty(total_q, num_heads, HEAD_DIM, device=device, dtype=DTYPE)
    key_cache = torch.empty(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )
    value_cache = torch.empty(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )
    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
    cache_seqlens = torch.zeros(num_seqs, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    rel_logits = torch.empty(total_q, num_heads, rel_extent, device=device, dtype=DTYPE)
    out = torch.empty(total_q, num_heads, HEAD_DIM, device=device, dtype=DTYPE)

    num_splits = inkling_fa4_num_splits(
        is_local=window_left is not None,
        batch_size=num_seqs,
        max_query_len=sched_max_seqlen_q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_kv_len=max(kv_lens),
    )

    def _fill(trial_seed, kv_lens_trial, block_table_trial):
        g = torch.Generator(device="cpu").manual_seed(trial_seed)

        def _randn(*shape):
            return torch.randn(*shape, generator=g).to(device=device, dtype=DTYPE)

        q.copy_(
            torch.nn.functional.normalize(_randn(total_q, num_heads, HEAD_DIM).float(), dim=-1).to(DTYPE)
        )
        key_cache.copy_(
            torch.nn.functional.normalize(
                _randn(num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM).float(), dim=-1
            ).to(DTYPE)
        )
        value_cache.copy_(_randn(num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM))
        rel_logits.copy_(_randn(total_q, num_heads, rel_extent))
        block_table.copy_(block_table_trial)
        cache_seqlens.copy_(torch.tensor(kv_lens_trial, dtype=torch.int32, device=device))

    # --- Warmup: at least one full eager call against the SAME static
    # buffers/addresses that capture will use, per CUDAGraphManager.capture()
    # (forward_fn(CUDAGraphMode.NONE) before wrapping the capture call).
    _fill(seed, kv_lens, _make_permuted_block_table(num_seqs, max_blocks, seed))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            inkling_fa4_rel_attention(
                q,
                key_cache,
                value_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=sched_max_seqlen_q,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                rel_extent=rel_extent,
                rel_logits=rel_logits,
                num_splits=num_splits,
                out=out,
            )
    torch.cuda.current_stream().wait_stream(s)

    # --- Capture: exactly one call, wrapped in torch.cuda.graph(), against
    # the same static tensors -- mirroring `with torch.cuda.graph(graph,
    # self.pool): forward_fn(...)`.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = inkling_fa4_rel_attention(
            q,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=sched_max_seqlen_q,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            rel_extent=rel_extent,
            rel_logits=rel_logits,
            num_splits=num_splits,
            out=out,
        )
    assert captured_out.data_ptr() == out.data_ptr(), (
        "kernel reallocated the output tensor instead of writing in place -- "
        "this would silently desync from the captured graph under replay"
    )

    # --- Replay: mutate the SAME static buffers with fresh data (new random
    # q/k/v/rel_logits, new block_table permutation, small in-bounds kv_lens
    # jitter within the fixed max_blocks/num_blocks sizing decided above) and
    # call graph.replay() -- exactly `self.graphs[desc].replay()`. Verify
    # each replay's numerics, not just that it runs without crashing.
    for trial in range(num_replays):
        trial_seed = seed + 1000 + trial
        kv_lens_trial = [max(q_lens[i], kv_lens[i] - trial * 3) for i in range(num_seqs)]
        bt_trial = _make_permuted_block_table(num_seqs, max_blocks, trial_seed)
        _fill(trial_seed, kv_lens_trial, bt_trial)

        graph.replay()
        torch.cuda.synchronize()

        ref = _ref_rel_attn(
            q,
            key_cache,
            value_cache,
            rel_logits,
            q_lens=q_lens,
            kv_lens=kv_lens_trial,
            block_table=block_table,
            scale=scale,
            rel_extent=rel_extent,
            window_left=window_left,
        )
        torch.testing.assert_close(
            out.float(),
            ref.float(),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m: f"replay #{trial} (trial_seed={trial_seed}) mismatch: {m}",
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@pytest.mark.parametrize(("num_seqs", "base_ctx"), [(32, 511), (128, 511)])
def test_mtp_verify_ns8_cuda_graph_capture_replay(num_heads, rel_extent, num_seqs, base_ctx):
    seq_lens = _mtp_verify_seq_lens(num_seqs, base_ctx)
    _run_case_cudagraph(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@pytest.mark.parametrize(("num_seqs", "base_ctx"), [(32, 511), (128, 511)])
def test_mtp_verify_ns8_sliding_window_cuda_graph_capture_replay(
    num_heads, local_extent, num_seqs, base_ctx
):
    seq_lens = _mtp_verify_seq_lens(num_seqs, base_ctx)
    _run_case_cudagraph(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent=local_extent,
        window_left=local_extent - 1,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
def test_mtp_ragged_verify_batch_cuda_graph_capture_replay(num_heads, rel_extent):
    # Ragged batch composition (mixed draft lengths incl. plain q_len=1
    # decode requests) under graph capture/replay -- the combination of the
    # two riskiest axes (non-uniform cu_seqlens_q AND graph replay) at once.
    seq_lens = _mtp_ragged_seq_lens(base_ctx=511)
    _run_case_cudagraph(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


# --- Real production q_len=9, bucketed to max_seqlen_q=16, under CUDA-graph
# capture/replay -- the highest-fidelity repro of the actual FULL-mode
# uniform-decode MTP-verify graph (`_warmup_and_capture` -> `_dummy_run`
# -> `InklingAttention._attention`'s `bucket_max_seqlen_q(9) == 16`) that
# this harness has ever built. See the q_len=9 comment block above
# `_mtp_ragged_seq_lens_real_qlen` for the full derivation.
@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@pytest.mark.parametrize(("num_seqs", "base_ctx"), [(32, 511), (128, 511)])
def test_mtp_verify_real_qlen9_cuda_graph_capture_replay(num_heads, rel_extent, num_seqs, base_ctx):
    seq_lens = _mtp_verify_seq_lens(num_seqs, base_ctx, ns=9)
    _run_case_cudagraph(
        seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None,
        bucketed_max_seqlen_q=True,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@pytest.mark.parametrize(("num_seqs", "base_ctx"), [(32, 511), (128, 511)])
def test_mtp_verify_real_qlen9_sliding_window_cuda_graph_capture_replay(
    num_heads, local_extent, num_seqs, base_ctx
):
    seq_lens = _mtp_verify_seq_lens(num_seqs, base_ctx, ns=9)
    _run_case_cudagraph(
        seq_lens, num_heads[0], num_heads[1], rel_extent=local_extent,
        window_left=local_extent - 1, bucketed_max_seqlen_q=True,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
def test_mtp_ragged_verify_batch_real_qlen9_cuda_graph_capture_replay(num_heads, rel_extent):
    seq_lens = _mtp_ragged_seq_lens_real_qlen(base_ctx=511)
    _run_case_cudagraph(
        seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None,
        bucketed_max_seqlen_q=True,
    )


# task #88 (MTP ns=8 gate, multi-shape/shared-pool axis): the single-shape
# CUDA-graph tests above each capture exactly one graph into its own implicit
# pool. The real CUDAGraphManager.capture() (cudagraph_utils.py:315-353) is
# never that simple: it captures MANY different BatchExecutionDescriptors
# (varying token counts, PIECEWISE mode then FULL mode) one after another,
# ALL via `torch.cuda.graph(graph, self.pool)` -- explicitly the SAME shared
# memory pool across every capture ("PIECEWISE has larger activations so
# FULL activations should fit in already allocated buffers in the graph
# pool", per that file's own comment). A single-shape test cannot expose a
# bug where capturing a SECOND, differently-shaped graph corrupts a FIRST,
# already-captured graph's static buffers via mismanaged pool memory reuse
# (two graphs' logically-distinct tensors aliasing the same physical
# addresses when they shouldn't, since both may need to be replayed later,
# non-concurrently but in arbitrary order, as real request batches vary
# shape from step to step). This is a different-in-kind risk from anything
# tested above and is the last kernel-harness-reachable synthetic hypothesis
# identified for attempt9's cudaErrorIllegalAddress signature before the only
# remaining paths are TP4/distributed interactions or a134's live run.
def _run_case_multi_shape_shared_pool(shape_specs, num_heads, num_kv_heads, seed=0):
    """Capture multiple DIFFERENTLY-SHAPED MTP verify batches into one shared
    CUDA graph pool (torch.cuda.graphs.graph_pool_handle()), then replay them
    in an INTERLEAVED order (0, 1, ..., 0, 1, ...) across several rounds,
    mutating each shape's own static buffers (fresh q/k/v/rel_logits,
    freshly-permuted block_table, small in-bounds cache_seqlens jitter)
    between replays -- verifying every replay of every shape against
    `_ref_rel_attn`, not just the first replay right after its own capture.
    shape_specs: list of (seq_lens, rel_extent, window_left) tuples, OR
    (seq_lens, rel_extent, window_left, bucketed_max_seqlen_q) 4-tuples --
    the 4th element defaults to False (pre-existing, unbucketed behavior) if
    omitted, so old callers are unaffected. Passing True on a shape mirrors
    the real per-descriptor `max_query_len = uniform_decode_query_len`
    scheduling bound getting bucketed up via `bucket_max_seqlen_q()` inside
    `InklingAttention._attention`, independently per shape sharing one pool.
    """
    torch.manual_seed(seed)
    device = "cuda"
    pool = torch.cuda.graphs.graph_pool_handle()
    scale = 1.0 / HEAD_DIM

    contexts = []
    for spec_idx, spec in enumerate(shape_specs):
        seq_lens, rel_extent, window_left = spec[0], spec[1], spec[2]
        bucketed_max_seqlen_q = spec[3] if len(spec) > 3 else False
        q_lens = [s[0] for s in seq_lens]
        kv_lens = [s[1] for s in seq_lens]
        total_q = sum(q_lens)
        num_seqs = len(seq_lens)
        max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = num_seqs * max_blocks + 1
        window_size = (-1, -1) if window_left is None else (window_left, 0)
        sched_max_seqlen_q = (
            bucket_max_seqlen_q(max(q_lens)) if bucketed_max_seqlen_q else max(q_lens)
        )

        q = torch.empty(total_q, num_heads, HEAD_DIM, device=device, dtype=DTYPE)
        key_cache = torch.empty(
            num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
        )
        value_cache = torch.empty(
            num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
        )
        block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
        cache_seqlens = torch.zeros(num_seqs, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor(
            [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        rel_logits = torch.empty(
            total_q, num_heads, rel_extent, device=device, dtype=DTYPE
        )
        out = torch.empty(total_q, num_heads, HEAD_DIM, device=device, dtype=DTYPE)

        def _fill(trial_seed, kv_lens_trial, block_table_trial, *, _q=q, _kc=key_cache,
                  _vc=value_cache, _rl=rel_logits, _bt=block_table, _cs=cache_seqlens,
                  _tq=total_q, _nh=num_heads, _nkv=num_kv_heads, _nb=num_blocks,
                  _re=rel_extent):
            g = torch.Generator(device="cpu").manual_seed(trial_seed)

            def _randn(*shape):
                return torch.randn(*shape, generator=g).to(device=device, dtype=DTYPE)

            _q.copy_(
                torch.nn.functional.normalize(_randn(_tq, _nh, HEAD_DIM).float(), dim=-1).to(DTYPE)
            )
            _kc.copy_(
                torch.nn.functional.normalize(
                    _randn(_nb, BLOCK_SIZE, _nkv, HEAD_DIM).float(), dim=-1
                ).to(DTYPE)
            )
            _vc.copy_(_randn(_nb, BLOCK_SIZE, _nkv, HEAD_DIM))
            _rl.copy_(_randn(_tq, _nh, _re))
            _bt.copy_(block_table_trial)
            _cs.copy_(torch.tensor(kv_lens_trial, dtype=torch.int32, device=device))

        _fill(seed + spec_idx, kv_lens, _make_permuted_block_table(num_seqs, max_blocks, seed + spec_idx))
        num_splits = inkling_fa4_num_splits(
            is_local=window_left is not None,
            batch_size=num_seqs,
            max_query_len=sched_max_seqlen_q,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_kv_len=max(kv_lens),
        )

        # Warmup against the SAME static buffers, on a side stream.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                inkling_fa4_rel_attention(
                    q, key_cache, value_cache, block_table=block_table,
                    cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=sched_max_seqlen_q, softmax_scale=scale, causal=True,
                    window_size=window_size, rel_extent=rel_extent,
                    rel_logits=rel_logits, num_splits=num_splits, out=out,
                )
        torch.cuda.current_stream().wait_stream(s)

        contexts.append(dict(
            q=q, key_cache=key_cache, value_cache=value_cache, block_table=block_table,
            cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q, rel_logits=rel_logits,
            out=out, q_lens=q_lens, kv_lens=kv_lens, max_blocks=max_blocks,
            rel_extent=rel_extent, window_left=window_left, window_size=window_size,
            max_seqlen_q=sched_max_seqlen_q, num_splits=num_splits, num_seqs=num_seqs,
            fill=_fill, graph=None,
        ))

    # Capture EVERY shape into the SAME pool, in order -- mirroring the real
    # capture loop's single shared self.pool reused across many descriptors.
    for spec_idx, ctx in enumerate(contexts):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            captured_out = inkling_fa4_rel_attention(
                ctx["q"], ctx["key_cache"], ctx["value_cache"],
                block_table=ctx["block_table"], cache_seqlens=ctx["cache_seqlens"],
                cu_seqlens_q=ctx["cu_seqlens_q"], max_seqlen_q=ctx["max_seqlen_q"],
                softmax_scale=scale, causal=True, window_size=ctx["window_size"],
                rel_extent=ctx["rel_extent"], rel_logits=ctx["rel_logits"],
                num_splits=ctx["num_splits"], out=ctx["out"],
            )
        assert captured_out.data_ptr() == ctx["out"].data_ptr(), (
            f"shape#{spec_idx}: kernel reallocated output instead of writing in place"
        )
        ctx["graph"] = graph

    # Replay in an INTERLEAVED order across several rounds -- if capturing a
    # later shape corrupted an earlier shape's static-buffer addressing via
    # bad pool reuse, replaying the earlier shape AGAIN (after later shapes
    # were captured/replayed in between) is what would expose it.
    for round_idx in range(3):
        for spec_idx, ctx in enumerate(contexts):
            trial_seed = seed + 5000 + round_idx * 100 + spec_idx
            kv_lens_trial = [
                max(ctx["q_lens"][i], ctx["kv_lens"][i] - round_idx * 3)
                for i in range(ctx["num_seqs"])
            ]
            bt_trial = _make_permuted_block_table(ctx["num_seqs"], ctx["max_blocks"], trial_seed)
            ctx["fill"](trial_seed, kv_lens_trial, bt_trial)

            ctx["graph"].replay()
            torch.cuda.synchronize()

            ref = _ref_rel_attn(
                ctx["q"], ctx["key_cache"], ctx["value_cache"], ctx["rel_logits"],
                q_lens=ctx["q_lens"], kv_lens=kv_lens_trial, block_table=ctx["block_table"],
                scale=scale, rel_extent=ctx["rel_extent"], window_left=ctx["window_left"],
            )
            torch.testing.assert_close(
                ctx["out"].float(),
                ref.float(),
                atol=2e-2,
                rtol=2e-2,
                msg=lambda m, si=spec_idx, ri=round_idx: (
                    f"shape#{si} round#{ri} (of {len(contexts)} shapes sharing one pool) mismatch: {m}"
                ),
            )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
def test_mtp_verify_ns8_multi_shape_shared_pool_full_attention(num_heads, rel_extent):
    # Three genuinely different batch shapes (32 seqs, 128 seqs, and the
    # ragged mixed-draft-length batch) captured into ONE shared graph pool --
    # mirroring the real engine's many-descriptors-one-pool capture loop.
    shape_specs = [
        (_mtp_verify_seq_lens(32, 511), rel_extent, None),
        (_mtp_verify_seq_lens(128, 511), rel_extent, None),
        (_mtp_ragged_seq_lens(511), rel_extent, None),
    ]
    _run_case_multi_shape_shared_pool(shape_specs, num_heads[0], num_heads[1])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
def test_mtp_verify_ns8_multi_shape_shared_pool_sliding_window(num_heads, local_extent):
    shape_specs = [
        (_mtp_verify_seq_lens(32, 511), local_extent, local_extent - 1),
        (_mtp_verify_seq_lens(128, 511), local_extent, local_extent - 1),
        (_mtp_ragged_seq_lens(511), local_extent, local_extent - 1),
    ]
    _run_case_multi_shape_shared_pool(shape_specs, num_heads[0], num_heads[1])


# --- Real production q_len=9 (bucketed to max_seqlen_q=16) variant of the
# multi-shape-shared-pool tests above -- three DIFFERENTLY-bucketed shapes
# (all q_len=9 -> bucket 16 here, vs. the ns8 tests' q_len=8 -> bucket 8
# no-op above) sharing one CUDA-graph pool, replayed interleaved. See the
# q_len=9 derivation comment above `_mtp_ragged_seq_lens_real_qlen`.
@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
def test_mtp_verify_real_qlen9_multi_shape_shared_pool_full_attention(num_heads, rel_extent):
    shape_specs = [
        (_mtp_verify_seq_lens(32, 511, ns=9), rel_extent, None, True),
        (_mtp_verify_seq_lens(128, 511, ns=9), rel_extent, None, True),
        (_mtp_ragged_seq_lens_real_qlen(511), rel_extent, None, True),
    ]
    _run_case_multi_shape_shared_pool(shape_specs, num_heads[0], num_heads[1])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4)])
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
def test_mtp_verify_real_qlen9_multi_shape_shared_pool_sliding_window(num_heads, local_extent):
    shape_specs = [
        (_mtp_verify_seq_lens(32, 511, ns=9), local_extent, local_extent - 1, True),
        (_mtp_verify_seq_lens(128, 511, ns=9), local_extent, local_extent - 1, True),
        (_mtp_ragged_seq_lens_real_qlen(511), local_extent, local_extent - 1, True),
    ]
    _run_case_multi_shape_shared_pool(shape_specs, num_heads[0], num_heads[1])
