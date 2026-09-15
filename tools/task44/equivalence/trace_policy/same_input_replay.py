# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay PP last-row inputs into independent MLP and O-projection operations."""

import os
from pathlib import Path

import torch


def install(model, active):
    reference = Path("/opt/task44-traces") / os.environ["VLLM_TASK44_SAME_INPUT_ARM"]
    cached = {"position": None, "records": {}}

    def records():
        position = active["position"]
        if cached["position"] != position:
            merged = {}
            files = sorted(reference.glob(f"rank*-step*-pos{position}.pt"))
            if len(files) != 4:
                raise RuntimeError(
                    f"Expected 4 PP traces at position {position}, got {len(files)}"
                )
            for path in files:
                payload = torch.load(path, weights_only=True, map_location="cpu")
                merged.update(payload["records"])
            cached.update(position=position, records=merged)
        return cached["records"]

    def replay(name, column_sharded):
        def hook(module, args):
            if active["records"] is None:
                return
            source = records()[name]
            x = args[0]
            if column_sharded:
                assert source.numel() == x.shape[-1] * module.tp_size
                source = source.chunk(module.tp_size)[module.tp_rank]
            assert source.shape == x[-1].shape, (name, source.shape, x.shape)
            replaced = x.clone()
            replaced[-1].copy_(source.to(device=x.device, dtype=x.dtype))
            return (replaced, *args[1:])

        return hook

    full_attention = os.environ.get("VLLM_TASK44_REPLAY_ATTENTION") == "1"

    def replay_attention(name):
        def hook(module, args):
            if active["records"] is None:
                return
            source = records()[name + ":normalized_input:full"]
            assert source.shape == args[1].shape, (name, source.shape, args[1].shape)
            replacement = source.to(device=args[1].device, dtype=args[1].dtype)
            return (args[0], replacement, *args[2:])

        return hook

    count = 0
    for name, child in model.named_modules():
        if name.endswith(".mlp"):
            child.register_forward_pre_hook(replay(name + ":normalized_input", False))
            count += 1
        elif full_attention and type(child).__name__ == "DeepseekV2MLAAttention":
            child.register_forward_pre_hook(replay_attention(name))
            count += 1
        elif not full_attention and name.endswith(".self_attn.o_proj"):
            child.register_forward_pre_hook(replay(name + ":input", True))
            count += 1
    print(f"TASK44 same-input replay installed: {count} operations", flush=True)
