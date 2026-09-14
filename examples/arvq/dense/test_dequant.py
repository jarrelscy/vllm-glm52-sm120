# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native-hot dequant correctness and full prefill fallback timing."""

import ctypes
import json
import os
from pathlib import Path

import torch
from nvfp4_dense import load_weight, quantize


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    # Initialize GEMM before occupying the measured weight pool.
    torch.mm(torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
    lib = ctypes.CDLL(os.environ["VLLM_NVFP4_P4_DENSE_LIB"])
    lib.nvfp4_dense_dequant.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    )
    lib.nvfp4_dense_dequant.restype = ctypes.c_int

    def ptr(x):
        return ctypes.c_void_p(x.data_ptr())

    def launch(weights, out):
        result = lib.nvfp4_dense_dequant(
            *[ptr(t) for t in (*weights, out)],
            out.shape[0],
            out.shape[1],
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        assert result == 0, result

    # Exact test includes zero/subnormal/fullrange FP8 block scales and all16
    # FP4 nibble codes, independent of the matrix quantizer.
    n, k = 32, 128
    codes = torch.arange(n * k).reshape(n, k).remainder(16).to(torch.uint8)
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    packed = (codes[:, ::2] | codes[:, 1::2] << 4).contiguous()
    words = (
        packed.view(torch.int32)
        .reshape(1, n // 16, 16, k // 64, 8)
        .permute(0, 1, 3, 2, 4)
    )
    fragment = torch.stack(
        [
            words[
                :, :, :, 8 * (j % 2) : 8 * (j % 2) + 8, 4 * (j // 2) : 4 * (j // 2) + 4
            ].reshape(1, n // 16, k // 64, 32)
            for j in range(4)
        ],
        dim=3,
    ).contiguous()
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
    out = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    launch(weights, out)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch(weights, out)
    out.zero_()
    graph.replay()
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
    del graph, weights, out

    source = load_weight("self_attn.o_proj.weight")
    base, decoded = quantize(source)
    expected = decoded.bfloat16()
    weights = tuple(t.cuda() for t in base)
    out = torch.empty_like(source, device="cuda")
    launch(weights, out)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
    del decoded, source
    # >272MiB packed weights, distinct physical allocations for rotation.
    pool = [weights] + [tuple(t.clone() for t in weights) for _ in range(20)]
    torch.cuda.empty_cache()

    def measure(fn):
        for w in pool:
            fn(w)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for w in pool:
                fn(w)
        for _ in range(3):
            graph.replay()
        times = []
        for _ in range(5):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(10):
                graph.replay()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) * 1000 / (10 * len(pool)))
        return sorted(times)[2]

    report = {
        "exact_bf16_edge_oracle": True,
        "exact_bf16_actual_weight_oracle": True,
        "cuda_graph": True,
        "pool_bytes": sum(t.nbytes for t in weights) * len(pool),
        "dequant_us": measure(lambda w: launch(w, out)),
        "fallback": [],
    }
    for tokens in (4, 8, 16, 32, 128):
        x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)

        def fallback(w, x=x):
            launch(w, out)
            return torch.nn.functional.linear(x, out)

        report["fallback"].append(
            {"tokens": tokens, "rotating_dequant_plus_bf16_gemm_us": measure(fallback)}
        )
    (Path(os.environ.get("DENSE_OUTPUT_DIR", ".")) / "dequant.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
