# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture compact native attention operands without changing computation."""

from pathlib import Path

import torch


def install(root, active):
    import vllm.utils.flashinfer as fi

    original = fi.flashinfer_trtllm_batch_decode_with_kv_cache_mla

    def wrapped(*args, **kw):
        result = original(*args, **kw)
        layer = active.get("layer", -1)
        pos = active.get("position", -1)
        if (
            active["records"] is not None
            and layer in (0, 42, 52)
            and pos in (31, 703, 831, 994, 995, 996)
        ):
            indices = kw["block_tables"].flatten(1)
            used = indices[indices >= 0].unique(sorted=True).long()
            raw = kw["kv_cache"].reshape(-1, 656)
            cache = torch.zeros(
                ((len(used) + 63) // 64, 64, 656), device=raw.device, dtype=torch.uint8
            )
            cache.reshape(-1, 656)[: len(used)].copy_(raw[used])
            remapped = (
                torch.searchsorted(used, indices.long())
                .int()
                .masked_fill(indices < 0, -1)
            )
            output = result[0] if isinstance(result, tuple) else result
            torch.save(
                dict(
                    q=kw["query"].squeeze(1).cpu(),
                    cache=cache.cpu(),
                    indices=remapped.cpu(),
                    seq_lens=kw["seq_lens"].cpu()
                    if kw["seq_lens"] is not None
                    else None,
                    scale=kw["bmm1_scale"],
                    output=output.squeeze(1).cpu(),
                ),
                Path(root) / f"attention-layer{layer}-pos{pos}.pt",
            )
        return result

    fi.flashinfer_trtllm_batch_decode_with_kv_cache_mla = wrapped
