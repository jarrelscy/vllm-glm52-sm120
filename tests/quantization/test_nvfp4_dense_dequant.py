# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native-fragment decode handles all FP4 values and finite FP8 scale ranges."""

import ctypes
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM120 GPU")
def test_native_hot_dequant_exact_bf16_and_capture():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM120 GPU")
    import vllm.model_executor.layers.quantization.nvfp4_p4_linear as runtime

    path = Path(
        os.environ.get(
            "VLLM_NVFP4_P4_DENSE_LIB",
            str(Path(runtime.__file__).with_name("arvq") / "dense.so"),
        )
    )
    library = ctypes.CDLL(str(path))
    library.nvfp4_dense_dequant.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    )
    library.nvfp4_dense_dequant.restype = ctypes.c_int
    n, k = 32, 128
    codes = torch.arange(n * k).reshape(n, k).remainder(16).to(torch.uint8)
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    packed = (codes[:, ::2] | codes[:, 1::2] << 4).contiguous()
    words = packed.view(torch.int32).reshape(1, n // 16, 16, k // 64, 8)
    words = words.permute(0, 1, 3, 2, 4)
    fragment = torch.stack(
        [
            words[
                :, :, :, 8 * (j % 2) : 8 * (j % 2) + 8, 4 * (j // 2) : 4 * (j // 2) + 4
            ].reshape(1, n // 16, k // 64, 32)
            for j in range(4)
        ],
        dim=3,
    ).contiguous()
    # Include zero, subnormal, minimum normal, unity, and maximum finite scales.
    raw_scales = torch.tensor(
        [0, 1, 7, 8, 0x30, 0x38, 0x70, 0x7E], dtype=torch.uint8
    ).repeat(n, 1)
    global_scale = torch.tensor([0.0137])
    expected = (
        levels[codes.long()]
        * raw_scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        * global_scale
    ).bfloat16()
    weights = tuple(
        t.cuda() for t in (fragment, raw_scales.view(torch.int32), global_scale)
    )
    output = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)

    def launch():
        result = library.nvfp4_dense_dequant(
            *[ctypes.c_void_p(t.data_ptr()) for t in (*weights, output)],
            n,
            k,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        assert result == 0

    launch()
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    output.zero_()
    graph.replay()
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM120 GPU")
def test_dense_custom_op_native_and_prefill_graphs():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM120 GPU")
    from vllm.model_executor.layers.quantization import nvfp4_p4_linear as dense

    torch.manual_seed(42)
    packed = tuple(
        t.cuda() for t in dense.quantize_weight(torch.randn(64, 128).bfloat16())
    )
    decoded = dense._dequantize(*packed, 64, 128)
    routes = torch.tensor([[-1] * 4, [0] * 4], device="cuda", dtype=torch.int32)

    def run(x):
        return dense.dense_p4(x, *packed, routes, 4)

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    for tokens in (1, 4, 5):
        x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(x, decoded)
        actual = compiled(x)
        if tokens <= 4:
            # Native computes the FP32 encoded weights; fallback rounds the
            # same reconstruction to BF16. This bound covers that rounding.
            relative = (
                actual.float() - expected.float()
            ).norm() / expected.float().norm()
            assert relative < 0.006
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(x)
        graph.replay()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
