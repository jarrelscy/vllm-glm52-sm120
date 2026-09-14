# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Only eligible hot prefill routes enter the paired kernel."""

import torch


def partition(batches, tokens, hot_ids, native_cold):
    paired_batches = []
    for route_indices, split in batches:
        if tokens in (2048, 4096) and route_indices.numel() >= 512:
            hot_local = hot_ids[route_indices].long()
            hot_valid = (hot_local >= 0) & (native_cold[route_indices] < 0)
            counts_hot = torch.bincount(hot_local[hot_valid], minlength=256)
            selected_hot = hot_valid & (counts_hot[hot_local.clamp_min(0)] >= 32)
            hot_slots = route_indices[selected_hot]
            # Existing nonzero/dynamic indexing already synchronizes in
            # grouped eager execution. All overhead is included in timing.
            if hot_slots.numel() >= 512:
                order = torch.argsort(hot_ids[hot_slots], stable=True)
                paired_batches.append((hot_slots[order], split, True))
                paired_batches.append((route_indices[~selected_hot], split, False))
            else:
                paired_batches.append((route_indices, split, False))
        else:
            paired_batches.append((route_indices, split, False))
    return paired_batches
