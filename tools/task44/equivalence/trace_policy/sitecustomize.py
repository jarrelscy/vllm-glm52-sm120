# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager, opt-in layer boundary snapshots; inactive without a marker file."""

import os
from pathlib import Path

if os.environ.get("VLLM_TASK44_LAYER_TRACE"):
    import torch

    if os.environ.get("VLLM_DIAGNOSTIC_BF16_FP32_ACCUM") == "1":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    root = Path(os.environ["VLLM_TASK44_LAYER_TRACE"])
    root.mkdir(parents=True, exist_ok=True)
    active = {"records": None, "step": 0}

    def snapshot(value):
        if isinstance(value, torch.Tensor):
            return (
                value[-1].detach().float().cpu().clone()
                if value.ndim
                else value.detach().cpu().clone()
            )
        if isinstance(value, (tuple, list)):
            return [snapshot(v) for v in value]
        return None

    def before(label, position):
        def hook(module, args):
            if active["records"] is not None and len(args) > position:
                active["records"][label] = snapshot(args[position])
                if os.environ.get(
                    "VLLM_TASK44_FULL_ATTN_INPUTS"
                ) == "1" and label.endswith(".self_attn:normalized_input"):
                    active["records"][label + ":full"] = (
                        args[position].detach().cpu().clone()
                    )

        return hook

    def after(label):
        def hook(module, args, output):
            if active["records"] is not None:
                active["records"][label] = snapshot(output)

        return hook

    def finish(module, args, output):
        if active["records"] is None:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        payload = dict(active)
        payload["root_output"] = (
            snapshot(output) if isinstance(output, (torch.Tensor, tuple)) else None
        )
        torch.save(
            payload,
            root / f"rank{rank}-step{active['step']:04d}-pos{active['position']}.pt",
        )
        active["records"] = None
        active["step"] += 1

    def start(module, args):
        if module.__class__.__name__ == "DeepseekV2DecoderLayer":
            if args:
                active["layer"] = getattr(module, "_task44_layer_index", -1)
            if (
                active["records"] is not None
                and active.get("position_pending")
                and args
            ):
                active["position"] = int(args[0][-1].item())
                active["num_tokens"] = args[0].numel()
                active["position_pending"] = False
            return
        # DeepseekV2Model's compilation decorator bypasses nn.Module.__call__.
        # Observe its ordinary causal-LM parent instead, even in eager mode.
        if module.__class__.__name__ not in (
            "GlmMoeDsaForCausalLM",
            "DeepseekV2ForCausalLM",
        ):
            return
        if not getattr(module, "_task44_hooks_installed", False):
            if (
                os.environ.get("VLLM_TASK44_REFERENCE_CHUNK")
                and os.environ.get("VLLM_ARVQ_REFERENCE_WEIGHTS") == "1"
            ):
                from reference_chunk import install as install_reference_chunk

                install_reference_chunk()
            if os.environ.get("VLLM_TASK44_CAPTURE_ATTENTION") == "1":
                from capture_attention import install as install_capture

                install_capture(root, active)
            if os.environ.get("VLLM_TASK44_FP32_PARTIALS") == "1":
                from tp_fp32_reference import install

                install(module)
            if os.environ.get("VLLM_TASK44_SAME_INPUT_ARM"):
                from same_input_replay import install as install_replay

                install_replay(module, active)
            for name, child in module.named_modules():
                kind = child.__class__.__name__
                if kind == "DeepseekV2MLAAttention":
                    child.register_forward_pre_hook(
                        before(name + ":normalized_input", 1)
                    )
                    child.register_forward_hook(after(name + ":output"))
                elif name.endswith(".mlp"):
                    child.register_forward_pre_hook(
                        before(name + ":normalized_input", 0)
                    )
                    child.register_forward_hook(after(name + ":output"))
                elif kind == "DeepseekV2DecoderLayer":
                    child._task44_layer_index = int(name.split(".")[-1])
                    child.register_forward_hook(after(name + ":output_and_residual"))
                elif name.endswith(
                    (
                        "fused_qkv_a_proj",
                        "q_a_layernorm",
                        "kv_a_layernorm",
                        "q_b_proj",
                        "o_proj",
                        "embed_tokens",
                    )
                ):
                    if name.endswith("o_proj"):
                        child.register_forward_pre_hook(before(name + ":input", 0))
                    child.register_forward_hook(after(name + ":output"))
            module.register_forward_hook(finish)
            module._task44_hooks_installed = True
        if (root / "enabled").exists():
            active["records"] = {}
            active["position_pending"] = True
            active["position"] = -1
            active["num_tokens"] = -1

    torch.nn.modules.module.register_module_forward_pre_hook(start)
