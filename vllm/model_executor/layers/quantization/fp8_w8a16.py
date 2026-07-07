# SPDX-License-Identifier: Apache-2.0
"""v4: e4m3 W8A16 linear method for the GLM-5.2 bf16-side projections.

Applied (via the hybrid quant config's get_quant_method, see integration
diff) to the attention/shared-expert Linear layers that the NVFP4 ignore-list
otherwise keeps in bf16:

    *.self_attn.o_proj          (K=16384 -> M=6144, 201 MB bf16 — the big one)
    *.self_attn.q_b_proj        (K=2048  -> M=16384, 67 MB)
    *.self_attn.fused_qkv_a_proj / q_a_proj+kv_a_proj_with_mqa (32 MB)
    *.mlp.shared_experts.{gate_up,down}_proj (75 MB)

Weights become e4m3 [M,K] + per-output-channel fp32 scale; the bf16 copy is
freed. Decode (batch-1, full-cudagraph) uses the custom hw-cvt gemv in
aqlm_moe_v2.cu's sibling fp8_linear_v4.cu; prefill uses torch._scaled_mm
(RowWise fp8 tensor cores) with dynamic per-row activation quant.

UNDO PATH: this is a load-time transform of already-loaded bf16 params — no
checkpoint change. Removing the get_quant_method hook (or the pattern match)
restores full bf16 on next server start. Nothing on disk is modified.

Measured (SM120, cold-L2, N=1): o_proj 1.81x, q_b 1.72x, shared_gate_up
1.62x, shared_down 1.48x, fused_qkv_a 1.28x; ~2.17 ms/stage decode saving.
Per-projection numerics vs bf16: decode ~2.6e-2 rel (e4m3 mantissa floor,
does not average down in a dot product), prefill ~3.9e-2 (adds fp8 act quant).
This is the e4m3 weight cost; the hot experts already run NVFP4 (4-bit) and
cold experts AQLM (2-bit), so 8-bit attention/shared weights are the most
precise tier — end-to-end coherence gates the decision.
"""
import pathlib

import torch

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        src = (
            pathlib.Path(__file__).resolve().parents[4]
            / "csrc" / "quantization" / "aqlm_moe" / "fp8_linear_v4.cu"
        )
        _ext = load(name="fp8_w8a16_ext", sources=[str(src)],
                    extra_cuda_cflags=["-O3"], verbose=False)
    return _ext


# Name patterns whose bf16 Linear weights are converted to e4m3.
TARGET_SUFFIXES = (
    # Only projections consumed exclusively through this method's apply().
    # The MLA q/kv projections (q_b_proj, q_a_proj, fused_qkv_a_proj,
    # kv_a_proj_with_mqa) are read as raw .weight tensors during MLA weight
    # absorption (bf16 matmul outside apply) — quantizing them to uint8 breaks
    # that path, so they stay bf16. o_proj (201MB, the largest single win) and
    # the shared experts are standard Linear layers, safe to fp8.
    "self_attn.o_proj",
    "shared_experts.gate_up_proj",
    "shared_experts.down_proj",
)


def matches(prefix: str) -> bool:
    # Escape hatch to isolate other features (e.g. TP bring-up): when
    # VLLM_DISABLE_FP8_W8A16=1, this method never claims any layer, so the
    # attention/shared projections fall back to bf16.
    import os as _os

    if _os.environ.get("VLLM_DISABLE_FP8_W8A16") == "1":
        return False
    return any(prefix.endswith(s) for s in TARGET_SUFFIXES)


class Fp8W8A16LinearMethod(LinearMethodBase):
    """Load bf16 -> convert to e4m3 per-channel after load; fp8 at runtime."""

    # Decode gemv is used up to this many tokens and always under graph
    # capture; above it, prefill takes the _scaled_mm path.
    DECODE_MAX_TOKENS = 64
    ACT_SHIFT = 6  # must equal fp8_linear_v4.cu's ACT_SHIFT

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra):
        # Register a bf16 weight so the checkpoint loads unchanged.
        weight_loader = extra.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes),
                             input_size_per_partition, dtype=params_dtype),
            input_dim=1, output_dim=0, weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra)
        layer._fp8_ready = False

    def process_weights_after_loading(self, layer) -> None:
        w = layer.weight.data
        if w.dtype == torch.uint8:  # already converted
            return
        wf = w.float()
        amax = wf.abs().amax(dim=1).clamp_min(1e-8)
        scale = (amax / 448.0)                       # [M] dequant multiplier
        q = (wf / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        packed = q.view(torch.uint8).contiguous()    # [M, K]

        del layer.weight
        layer.register_parameter(
            "weight", torch.nn.Parameter(packed, requires_grad=False))
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(scale.float().contiguous(),
                               requires_grad=False))
        # fp8 view of the packed bytes, for _scaled_mm (prefill).
        layer._w_fp8 = packed.view(torch.float8_e4m3fn)
        layer._fp8_ready = True

    def apply(self, layer, x, bias=None):
        ext = _get_ext()
        M = layer.weight.shape[0]
        orig_dtype = x.dtype
        num_tokens = x.numel() // x.shape[-1]
        x2d = x.reshape(-1, x.shape[-1])

        if (num_tokens <= self.DECODE_MAX_TOKENS
                or torch.cuda.is_current_stream_capturing()):
            xh = (x2d.to(torch.float32) * (2.0 ** -self.ACT_SHIFT)).half()
            xh = xh.contiguous()
            y = ext.fp8_w8a16_gemv(xh, layer.weight, layer.weight_scale)
        else:
            # Prefill: RowWise fp8 x fp8. Dynamic per-row act quant.
            xa = x2d.to(torch.float32)
            xamax = xa.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            xs = (xamax / 448.0)
            xq = (xa / xs).clamp(-448, 448).to(torch.float8_e4m3fn)
            y = torch._scaled_mm(
                xq, layer._w_fp8.t(),
                scale_a=xs.float().contiguous(),
                scale_b=layer.weight_scale.view(1, M).contiguous(),
                out_dtype=torch.bfloat16,
            )
        y = y.to(orig_dtype)
        if bias is not None:
            y = y + bias
        return y.view(*x.shape[:-1], M)
