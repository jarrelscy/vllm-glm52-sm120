# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse only consumed bytes of an owned temporary rank-major KV gather.

Must execute on the attention stream after each peer's attention call. Never
pass the resident KV cache: this function overwrites the disposable gather.
"""


def stash_consumed(allkv, parts, consumed_peers, stashed):
    import torch

    assert allkv.dtype == torch.uint8 and allkv.is_contiguous()
    assert parts and allkv.shape[0] == 4
    assert 0 <= stashed <= len(parts) <= consumed_peers <= allkv.shape[0]
    assert all(p.dtype == torch.bfloat16 and p.device == allkv.device for p in parts)
    part = parts[0]
    assert part.dtype == torch.bfloat16 and part.is_contiguous()
    size = part.numel() * part.element_size()
    consumed_bytes = consumed_peers * allkv[0].numel()
    flat = allkv.view(-1)
    while stashed < len(parts) and (stashed + 1) * size <= consumed_bytes:
        original = parts[stashed]
        assert original.shape == part.shape and original.is_contiguous()
        slot = flat[stashed * size : (stashed + 1) * size].view(part.dtype)
        slot = slot.view(part.shape)
        slot.copy_(original)
        parts[stashed] = slot
        stashed += 1
    return stashed
