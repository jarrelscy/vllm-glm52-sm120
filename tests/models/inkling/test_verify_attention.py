# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the virtual-sequence verify-attention path (task #111).

attention.py routes uniform small-query batches (spec-decode verify at
qlen = ns+1, MTP draft window passes) to the split-KV decode kernel by
treating each query row as its own virtual sequence: row j of request i
shares request i's block-table row with kv_len = seq_len_i - (qlen-1) + j.
This checks that expansion against the pure-PyTorch relative-bias reference
(same one the FA4 kernel is tested against), including the strided
_split_kv_cache-style K/V views and the sliding-window layers.
"""

import pytest
import torch

from vllm.models.inkling.nvidia.ops.triton_decode_attention import (
    triton_rel_decode_attention,
)
from vllm.platforms import current_platform

from test_fa4_rel_attention import (  # same-directory import (no package)
    BLOCK_SIZE,
    DTYPE,
    HEAD_DIM,
    _ref_rel_attn,
)

NUM_HEADS = [(16, 2), (16, 4)]  # prod TP4 full-attention / SWA per-rank


def _make_paged_kv(num_seqs, max_kv, num_kv_heads, device, strided):
    max_blocks = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * max_blocks + 1
    if strided:
        # _split_kv_cache layout: K and V packed in the last dim of one
        # buffer -> token stride HKV*2D, head stride 2D.
        buf = torch.randn(
            num_blocks, BLOCK_SIZE, num_kv_heads, 2 * HEAD_DIM,
            device=device, dtype=DTYPE,
        )
        key_cache = buf[..., :HEAD_DIM]
        value_cache = buf[..., HEAD_DIM:]
    else:
        key_cache = torch.randn(
            num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM,
            device=device, dtype=DTYPE,
        )
        value_cache = torch.randn_like(key_cache)
    key_cache.copy_(
        torch.nn.functional.normalize(key_cache.float(), dim=-1).to(DTYPE)
    )
    block_table = torch.zeros(
        num_seqs, max_blocks, dtype=torch.int32, device=device
    )
    for i in range(num_seqs):
        block_table[i] = torch.arange(
            1 + i * max_blocks, 1 + (i + 1) * max_blocks, dtype=torch.int32
        )
    return key_cache, value_cache, block_table


def _run_verify_case(
    kv_lens, qlen, num_heads, num_kv_heads, rel_extent, window_left,
    strided=False, seed=0,
):
    torch.manual_seed(seed)
    device = "cuda"
    num_seqs = len(kv_lens)
    total_q = num_seqs * qlen
    scale = 1.0 / HEAD_DIM

    q = torch.randn(total_q, num_heads, HEAD_DIM, device=device, dtype=DTYPE)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)
    key_cache, value_cache, block_table = _make_paged_kv(
        num_seqs, max(kv_lens), num_kv_heads, device, strided
    )
    rel_logits = torch.randn(
        total_q, num_heads, rel_extent, device=device, dtype=DTYPE
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    # The exact expansion attention.py's verify branch performs.
    sk = (
        seq_lens[:, None]
        - (qlen - 1)
        + torch.arange(qlen, device=device, dtype=seq_lens.dtype)[None, :]
    ).view(-1)
    out = torch.empty_like(q)
    triton_rel_decode_attention(
        q,
        key_cache,
        value_cache,
        block_table=block_table.repeat_interleave(qlen, dim=0),
        cache_seqlens=sk,
        rel_logits=rel_logits,
        softmax_scale=scale,
        rel_extent=rel_extent,
        window_left=-1 if window_left is None else window_left,
        num_splits=8 if window_left is not None else 64,
        out=out,
    )

    ref = _ref_rel_attn(
        q,
        key_cache,
        value_cache,
        rel_logits,
        q_lens=[qlen] * num_seqs,
        kv_lens=kv_lens,
        block_table=block_table,
        scale=scale,
        rel_extent=rel_extent,
        window_left=window_left,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("qlen", [2, 3, 8])
@pytest.mark.parametrize("kv_lens", [[515], [515, 67], [2049, 131, 515]])
@pytest.mark.parametrize("rel_extent", [128, 1024])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_verify_full_attention(num_heads, qlen, kv_lens, rel_extent, strided):
    _run_verify_case(
        kv_lens, qlen, *num_heads, rel_extent, window_left=None,
        strided=strided,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("qlen", [2, 3, 8])
@pytest.mark.parametrize("kv_lens", [[515], [2049, 131, 515]])
@pytest.mark.parametrize("window_left", [128, 512])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_verify_sliding_window(num_heads, qlen, kv_lens, window_left, strided):
    _run_verify_case(
        kv_lens, qlen, *num_heads, 128, window_left=window_left,
        strided=strided,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@torch.inference_mode()
def test_verify_kv_shorter_than_window_and_qlen_edge():
    # kv_len barely above qlen (fresh request verified right after prefill
    # of a tiny prompt) and kv shorter than the window.
    _run_verify_case([9, 12], 8, 16, 2, 128, window_left=None)
    _run_verify_case([9, 12], 8, 16, 4, 128, window_left=512)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@torch.inference_mode()
def test_fused_combine_matches_torch_epilogue():
    """INKLING_ATTN_FUSED_COMBINE replaces the torch split-merge epilogue
    with one Triton kernel. Kernel partials are identical between runs, so
    fp32 outputs may differ only by fp32 reduction-order rounding: gate at
    a tolerance orders of magnitude tighter than the reference tests'
    2e-2 (a masking/indexing bug would blow straight past it). Includes a
    kv_len < num_splits case so EMPTY splits exercise the -inf guard, and
    both bf16 and fp32 out dtypes."""
    from vllm.models.inkling.nvidia.ops import (
        triton_decode_attention as tda,
    )

    torch.manual_seed(5)
    device = "cuda"
    cases = [
        # (num_heads, num_kv_heads, kv_lens, window_left, out_dtype)
        (16, 2, [515, 67], None, torch.float32),
        (16, 4, [2049, 131, 515], 128, torch.float32),
        (16, 2, [7, 9], None, torch.bfloat16),  # kv < num_splits: empty splits
        (16, 2, [515], None, torch.bfloat16),
    ]
    for num_heads, num_kv_heads, kv_lens, window_left, out_dtype in cases:
        num_seqs = len(kv_lens)
        scale = 1.0 / HEAD_DIM
        q = torch.randn(
            num_seqs, num_heads, HEAD_DIM, device=device, dtype=DTYPE
        )
        key_cache, value_cache, block_table = _make_paged_kv(
            num_seqs, max(kv_lens), num_kv_heads, device, strided=True
        )
        rel_logits = torch.randn(
            num_seqs, num_heads, 128, device=device, dtype=DTYPE
        )
        seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        outs = {}
        orig = tda._FUSED_COMBINE
        try:
            for fused in (False, True):
                tda._FUSED_COMBINE = fused
                torch.manual_seed(7)  # identical everything
                out = torch.empty(
                    num_seqs, num_heads, HEAD_DIM,
                    device=device, dtype=out_dtype,
                )
                triton_rel_decode_attention(
                    q,
                    key_cache,
                    value_cache,
                    block_table=block_table,
                    cache_seqlens=seq_lens,
                    rel_logits=rel_logits,
                    softmax_scale=scale,
                    rel_extent=128,
                    window_left=-1 if window_left is None else window_left,
                    num_splits=8 if window_left is not None else 64,
                    out=out,
                )
                outs[fused] = out
        finally:
            tda._FUSED_COMBINE = orig

        if out_dtype == torch.float32:
            torch.testing.assert_close(
                outs[True], outs[False], rtol=1e-5, atol=1e-6,
                msg=f"fused combine fp32 mismatch (kv={kv_lens})",
            )
        else:
            # bf16 store: fp32 rounding differences may flip the last bf16
            # ulp; anything larger is a real bug.
            torch.testing.assert_close(
                outs[True].float(), outs[False].float(),
                rtol=1e-2, atol=1e-2,
                msg=f"fused combine bf16 mismatch (kv={kv_lens})",
            )
