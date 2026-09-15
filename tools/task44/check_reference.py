# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run in the serving image with the repository mounted at /work."""

import importlib.util
import json

import torch

spec = importlib.util.spec_from_file_location(
    "reference",
    "/work/vllm/model_executor/layers/quantization/arvq_reference.py",
)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import (  # noqa: E402
    _projection,
)
from vllm.model_executor.layers.quantization.nvfp4_arvq_prefill import (  # noqa: E402
    dequantize_cold,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(44015)
device = "cuda"
results = []
for bits in (7, 8):
    for n, k in ((32, 128), (32, 256), (32, 7168), (2048, 128)):
        experts = 2
        words = 4 * (8 + bits)
        cw = torch.randint(
            -(2**31),
            2**31 - 1,
            (experts * (n // 16) * (k // 64) * words + 1,),
            dtype=torch.int32,
            device=device,
        )
        cb = torch.randint(
            -(2**31),
            2**31 - 1,
            (256 + (1 << bits),),
            dtype=torch.int32,
            device=device,
        )
        cs = torch.randint(
            32, 65, (experts, n // 16, k // 128, 16), dtype=torch.uint8, device=device
        )
        logical_codes = torch.randint(0, 16, (experts, n, k), device=device)
        packed = sum(logical_codes[..., i::8] << (i * 4) for i in range(8))
        hw = (
            packed.to(torch.int32)
            .reshape(experts, n // 16, 2, 8, k // 64, 2, 4)
            .permute(0, 1, 4, 5, 2, 3, 6)
            .contiguous()
            .reshape(experts, n // 16, k // 64, 4, 32)
        )
        logical_scales = torch.randint(32, 65, (experts, n, k // 16), device=device)
        hs = sum(logical_scales[..., i::4] << (8 * i) for i in range(4)).to(torch.int32)
        hg = torch.tensor([[0.7, 1.3], [0.9, 1.1]], device=device)
        for expert in range(experts):
            expected_hot = ref.fp4(logical_codes[expert]) * ref.e4m3(
                logical_scales[expert]
            ).repeat_interleave(16, -1)
            torch.testing.assert_close(
                ref.hot_rows(hw, hs, n, k, expert), expected_hot, rtol=0, atol=0
            )
            offset = expert * (n // 16) * (k // 64) * words
            native = dequantize_cold(cw[offset:], cb, cs[expert], 0.8, n, k)
            decoded = ref.cold_rows(cw, cb, cs, n, k, expert) * 0.8
            torch.testing.assert_close(decoded.half(), native, rtol=0, atol=0)
        for slots in (1, 4, 9):
            x = (torch.randn(slots, k, device=device) * 0.4).half()
            cold = torch.tensor(
                [i % 2 if i % 3 == 0 else -1 for i in range(slots)],
                device=device,
                dtype=torch.int32,
            )
            hot = torch.tensor(
                [i % 2 if i % 3 == 1 else -1 for i in range(slots)],
                device=device,
                dtype=torch.int32,
            )
            tensors = [cw, cb, cs, hw, hs, hg]
            actual = _projection(x, cold, hot, tensors, 0.8, n, 2, 2)
            expected = ref.projection(x, cold, hot, tensors, 0.8, n, 2, 2)
            error = (actual - expected).abs()
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=0.003)
            results.append(
                dict(bits=bits, k=k, slots=slots, max_abs=error.max().item())
            )

q = torch.randn(3, 4, 16, device=device)
keys = torch.randn(7, 16, device=device)
values = torch.randn(7, 8, device=device)
indices = torch.tensor([[1, 2, 1, -1], [-1, -1, -1, -1], [0, 3, 5, 6]], device=device)
out, lse = ref.sparse_attention(q, keys, values, indices, 0.25)
assert torch.equal(out[1], torch.zeros_like(out[1]))
assert torch.isneginf(lse[1]).all()
for row in (0, 2):
    idx = indices[row][indices[row] >= 0]
    scores = (q[row].double() @ keys[idx].double().T) * 0.25
    expected = scores.softmax(-1) @ values[idx].double()
    torch.testing.assert_close(out[row].double(), expected, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(
        lse[row].double(),
        scores.logsumexp(-1) / 0.6931471805599453,
        rtol=2e-5,
        atol=1e-6,
    )
from vllm import _custom_ops as ops  # noqa: E402
from vllm.utils.flashinfer import (  # noqa: E402
    flashinfer_trtllm_batch_decode_with_kv_cache_mla,
)

cache = torch.zeros(1, 64, 656, device=device, dtype=torch.uint8)
latent = torch.randn(7, 512, device=device, dtype=torch.bfloat16)
rope = torch.randn(7, 64, device=device, dtype=torch.bfloat16)
ops.concat_and_cache_mla(
    latent,
    rope,
    cache,
    torch.arange(7, device=device),
    kv_cache_dtype="fp8_ds_mla",
    scale=torch.tensor(1.0, device=device),
)
query = torch.randn(3, 64, 576, device=device, dtype=torch.bfloat16)
physical = torch.full((3, 2048), -1, device=device, dtype=torch.int32)
physical[:, :4] = indices.int()
lengths = torch.tensor([4, 0, 4], device=device, dtype=torch.int32)
expected, expected_lse = ref.paged_attention(query, cache, physical, 1 / 16, lengths)
native_out = torch.empty(3, 1, 64, 512, device=device, dtype=torch.bfloat16)
_, native_lse = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
    query=query.unsqueeze(1),
    kv_cache=cache.unsqueeze(1),
    workspace_buffer=torch.empty(128 * 1024**2, device=device, dtype=torch.uint8),
    qk_nope_head_dim=192,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    block_tables=physical.unsqueeze(1),
    seq_lens=lengths,
    max_seq_len=2048,
    out=native_out,
    bmm1_scale=1 / 16,
    bmm2_scale=1.0,
    sparse_mla_top_k=2048,
    kv_scale_format="arbitrary_fp32",
    return_lse=True,
)
torch.testing.assert_close(
    native_out[:, 0].float(), expected.float(), rtol=0.03, atol=0.025
)
native_lse = native_lse.reshape(3, 64)
torch.testing.assert_close(native_lse[[0, 2]], expected_lse[[0, 2]], rtol=0, atol=5e-5)
print(
    json.dumps(
        {
            "projection_cases": results,
            "attention": "PASS",
            "paged_attention": "PASS",
            "paged_lse_max_abs": (native_lse[[0, 2]] - expected_lse[[0, 2]])
            .abs()
            .max()
            .item(),
        },
        indent=2,
    )
)
