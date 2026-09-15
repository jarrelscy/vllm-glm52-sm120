# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in target-model row tracing; no mutation of model inputs or outputs."""

import os
from pathlib import Path

if os.environ.get("VLLM_TASK44_ROW_TRACE"):
    import torch

    root = Path(os.environ["VLLM_TASK44_ROW_TRACE"])
    root.mkdir(parents=True, exist_ok=True)
    active = {"records": None, "step": 0, "layer": -1}

    def snapshot(value):
        if isinstance(value, torch.Tensor):
            # Every verification row; only the final eight rows of large prefill.
            selected = value if value.ndim == 0 or value.shape[0] <= 32 else value[-8:]
            return selected.detach().cpu().clone()
        if isinstance(value, (list, tuple)):
            return [snapshot(x) for x in value]
        return None

    def before(label, index):
        def hook(module, args):
            if active["records"] is not None and len(args) > index:
                active["records"][label] = snapshot(args[index])

        return hook

    def after(label):
        def hook(module, args, output):
            if active["records"] is not None:
                active["records"][label] = snapshot(output)
                if label.endswith("self_attn:output"):
                    buffer = getattr(module.mla_attn, "topk_indices_buffer", None)
                    if buffer is not None:
                        active["records"][label + ":selected_indices"] = snapshot(
                            buffer[: active["num_tokens"]]
                        )

        return hook

    def finish(module, args, output):
        if active["records"] is None:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        payload = dict(active)
        payload["root_output"] = snapshot(output)
        torch.save(payload, root / f"rank{rank}-step{active['step']:04d}.pt")
        active["records"] = None
        active["step"] += 1

    def install_routes():
        from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as arvq

        original = arvq.arvq_mlp

        def call(x, weights, ids, lookups, tensors, alphas, chunk_tokens):
            if active["records"] is not None:
                label = f"model.layers.{active['layer']}.mlp:routes"
                active["records"][label + ":ids"] = snapshot(ids)
                active["records"][label + ":weights"] = snapshot(weights)
                active["records"][label + ":input"] = snapshot(x)
            return original(x, weights, ids, lookups, tensors, alphas, chunk_tokens)

        arvq.arvq_mlp = call

    def start(module, args):
        kind = type(module).__name__
        if kind == "DeepseekV2DecoderLayer":
            active["layer"] = getattr(module, "_task44_layer_index", -1)
            if active["records"] is not None and args:
                active["positions"] = snapshot(args[0])
                active["num_tokens"] = args[0].numel()
            return
        if kind not in ("GlmMoeDsaForCausalLM", "DeepseekV2ForCausalLM"):
            return
        if not getattr(module, "_task44_rows_installed", False):
            if os.environ.get("VLLM_ARVQ_REFERENCE_WEIGHTS") == "1":
                from reference_chunk import install

                install()
            install_routes()
            for name, child in module.named_modules():
                child_kind = type(child).__name__
                if child_kind == "DeepseekV2DecoderLayer":
                    child._task44_layer_index = int(name.split(".")[-1])
                    child.register_forward_hook(after(name + ":output_and_residual"))
                elif child_kind == "DeepseekV2MLAAttention":
                    child.register_forward_pre_hook(
                        before(name + ":normalized_input", 1)
                    )
                    child.register_forward_hook(after(name + ":output"))
                elif name.endswith(".mlp"):
                    child.register_forward_pre_hook(
                        before(name + ":normalized_input", 0)
                    )
                    child.register_forward_hook(after(name + ":output"))
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
                    if name.endswith("embed_tokens"):
                        child.register_forward_pre_hook(before("input_token_ids", 0))
                    if name.endswith("o_proj"):
                        child.register_forward_pre_hook(before(name + ":input", 0))
                    child.register_forward_hook(after(name + ":output"))
            module.register_forward_hook(finish)
            module._task44_rows_installed = True
        if (root / "enabled").exists():
            active["records"] = {}
            active["probe"] = (root / "enabled").read_text().strip()
            active["positions"] = None
            active["num_tokens"] = -1

    torch.nn.modules.module.register_module_forward_pre_hook(start)
