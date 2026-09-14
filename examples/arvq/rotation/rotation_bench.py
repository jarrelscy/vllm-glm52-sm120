# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conditional cold Hadamard plus P4 activation packing, independent oracles."""

# ruff: noqa: B023
# Benchmark closures are executed synchronously before the loop advances.
import argparse
import ctypes
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def hadamard(block):
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < block:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    return matrix / block**0.5


def pack_reference(value):
    slots, k = value.shape
    residual = value.float().clone()
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    all_packed, all_scales = [], []
    for _ in range(4):
        maximum = residual.reshape(slots, -1, 16).abs().amax(-1)
        exponent = torch.ceil(torch.log2((maximum / 6).clamp_min(2**-20))).clamp(-6, 8)
        scale = torch.exp2(exponent).repeat_interleave(16, -1)
        magnitude = residual.abs() / scale
        code = sum(
            (magnitude > t).long() for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
        )
        code |= (residual < 0).long() << 3
        decoded = levels[code] * scale
        residual = (residual - decoded) * 16
        all_packed.append(
            (code.reshape(slots, k // 8, 8) << (torch.arange(8) * 4)).sum(-1).int()
        )
        all_scales.append(((exponent.long() + 7) << 3).to(torch.uint8))
    return torch.stack(all_packed, 1), torch.stack(all_scales, 1)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(413)
    library = ctypes.CDLL(str(ROOT / "rotation.so"))
    base = ctypes.CDLL(str(ROOT / "hybrid.so"))
    library.rotation_pack.argtypes = (
        [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    library.rotation_transform.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    base.hybrid_pack.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    for function in (
        library.rotation_pack,
        library.rotation_transform,
        base.hybrid_pack,
    ):
        function.restype = ctypes.c_int

    def pointer(tensor):
        return ctypes.c_void_p(tensor.data_ptr())

    def stream():
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def check(result):
        assert result == 0, result

    def measure(call):
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(100):
                call()
        for _ in range(3):
            graph.replay()
        values = []
        for _ in range(5):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return sorted(values)[2]  # ms *1000/(100*10) -> us

    results = []
    for projection, k in (("gateup", 6144), ("down", 512)):
        generator = torch.Generator().manual_seed(
            91426 + 3 * 2 + (projection == "down")
        )
        signs_cpu = (torch.randint(0, 2, (k,), generator=generator) * 2 - 1).to(
            torch.int8
        )
        signs = signs_cpu.cuda()
        for slots in (8, 32):
            value_cpu = torch.randn(slots, k).half()
            value = value_cpu.cuda()
            packed = torch.empty((slots, 4, k // 8), device="cuda", dtype=torch.int32)
            scales = torch.empty((slots, 4, k // 16), device="cuda", dtype=torch.uint8)
            transformed = torch.empty((slots, k), device="cuda")
            base_q, base_s = torch.empty_like(packed), torch.empty_like(scales)

            def baseline():
                check(
                    base.hybrid_pack(
                        pointer(value),
                        pointer(base_q),
                        pointer(base_s),
                        k,
                        slots,
                        4,
                        stream(),
                    )
                )

            base_us = measure(baseline)
            for mixed in (False, True):
                cold_cpu = torch.arange(slots, dtype=torch.int32)
                if mixed:
                    cold_cpu[6::8] = cold_cpu[7::8] = -1
                cold = cold_cpu.cuda()
                for block in (0, 32, 128):

                    def transform():
                        check(
                            library.rotation_transform(
                                *map(pointer, (value, signs, cold, transformed)),
                                k,
                                slots,
                                block,
                                stream(),
                            )
                        )

                    def pack():
                        check(
                            library.rotation_pack(
                                *map(pointer, (value, signs, cold, packed, scales)),
                                k,
                                slots,
                                block,
                                stream(),
                            )
                        )

                    transform()
                    actual = transformed.cpu()
                    expected = value_cpu.double()
                    if block:
                        rotated = (
                            (expected * signs_cpu).reshape(slots, -1, block)
                            @ hadamard(block)
                        ).reshape(slots, k)
                        expected = torch.where(
                            (cold_cpu >= 0)[:, None], rotated, expected
                        )
                    relative = float(
                        (actual.double() - expected).norm() / expected.norm()
                    )
                    assert relative < 2e-7, (k, slots, block, relative)
                    # Validate P4 coding against transform-only FP32 outputs;
                    # this isolates coding correctness from transform rounding.
                    expected_q, expected_s = pack_reference(actual)
                    pack()
                    assert torch.equal(packed.cpu(), expected_q)
                    assert torch.equal(scales.cpu(), expected_s)
                    if mixed:
                        hot = cold_cpu < 0
                        assert torch.equal(packed.cpu()[hot], base_q.cpu()[hot])
                        assert torch.equal(scales.cpu()[hot], base_s.cpu()[hot])
                    if block == 0:
                        assert torch.equal(packed, base_q)
                        assert torch.equal(scales, base_s)
                    # Independent dot-product preservation check, before lossy P4.
                    algebra_error = 0.0
                    if block:
                        w = torch.randn(2, k, dtype=torch.float64)
                        wr = (
                            (w * signs_cpu).reshape(2, -1, block) @ hadamard(block)
                        ).reshape(2, k)
                        xr = (
                            (value_cpu.double() * signs_cpu).reshape(slots, -1, block)
                            @ hadamard(block)
                        ).reshape(slots, k)
                        reference = value_cpu.double() @ w.T
                        algebra_error = float(
                            (xr @ wr.T - reference).norm() / reference.norm()
                        )
                        assert algebra_error < 1e-12
                    row = {
                        "projection": projection,
                        "K": k,
                        "slots": slots,
                        "mixed_6cold_2hot": mixed,
                        "block": block,
                        "transform_relative_l2": relative,
                        "algebra_relative_l2": algebra_error,
                        "packed_exact": True,
                        "hot_unchanged": True,
                        "baseline_pack_us": base_us,
                        "transform_only_us": measure(transform),
                        "fused_rotation_pack_us": measure(pack),
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)
    (ROOT / "rotation_bench.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "seed_rule": "91426+layer*2+(projection==down)",
                "layer": 3,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
