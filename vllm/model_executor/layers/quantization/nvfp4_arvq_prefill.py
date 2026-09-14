# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in grouped cold prefill, reconstructing only temporary FP16 weights.

The caller gates token count, memory budget and CUDA capture. Cold weights
remain serialized ARVQ; hot experts retain the native P4 implementation.
"""

import ctypes
import os
from collections.abc import Callable
from pathlib import Path

import torch

_LIB = None


def _compact_prefill_enabled():
    """Diagnostic marker mode must remain fixed for the entire TP request."""
    mode = os.environ.get("VLLM_ARVQ_COMPACT_PREFILL", "0")
    return mode == "1" or (
        mode == "toggle" and Path("/dev/shm/vllm_arvq_compact_prefill_on").exists()
    )


def _sort_native_prefill_enabled():
    """Diagnostic marker must stay fixed for the entire TP request."""
    mode = os.environ.get("VLLM_ARVQ_SORT_NATIVE_PREFILL", "0")
    return mode == "1" or (
        mode == "toggle" and Path("/dev/shm/vllm_arvq_sort_native_prefill_on").exists()
    )


def dequantize_cold(packed, codebooks, scales, global_scale, n, k, dtype=torch.float16):
    """Decode one expert into an ephemeral 16-bit [N,K] matrix."""
    from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import _layout

    words, entries = _layout(codebooks)
    required = (n // 16) * (k // 64) * words + (entries == 384)
    if n <= 0 or k <= 0 or n % 16 or k % 128 or packed.numel() < required:
        raise ValueError("Invalid ARVQ prefill packed layout")
    global _LIB
    if _LIB is None:
        path = Path(
            os.environ.get(
                "VLLM_ARVQ_PREFILL_LIB",
                str(Path(__file__).with_name("arvq") / "prefill.so"),
            )
        )
        _LIB = ctypes.CDLL(str(path))
        _LIB.arvq_dequant.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _LIB.arvq_dequant.restype = ctypes.c_int
        _LIB.arvq_dequant_fp16.argtypes = _LIB.arvq_dequant.argtypes
        _LIB.arvq_dequant_fp16.restype = ctypes.c_int
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("ARVQ prefill decode requires FP16 or BF16 output")

    pointer = lambda tensor: ctypes.c_void_p(tensor.data_ptr())
    name = "arvq_dequant_fp16" if dtype == torch.float16 else "arvq_dequant"
    if entries == 512:
        name += "_8x8"
    try:
        decoder = getattr(_LIB, name)
    except AttributeError as missing_symbol:
        raise RuntimeError(
            f"Missing {name}; rebuild arvq/build_prefill.sh for ARVQ 8+8"
        ) from missing_symbol
    decoder.argtypes = _LIB.arvq_dequant.argtypes
    decoder.restype = ctypes.c_int
    output = torch.empty((n, k), device=packed.device, dtype=dtype)
    error = decoder(
        pointer(packed),
        pointer(codebooks),
        pointer(scales),
        global_scale,
        pointer(output),
        n,
        k,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if error:
        raise RuntimeError(f"ARVQ prefill weight decode failed with CUDA error {error}")
    return output


def grouped_cold_prefill(
    x,
    topk_weights,
    topk_ids,
    lookups,
    tensors,
    alphas,
    *,
    projection,
    min_expert_tokens=32,
    chunk_tokens=128,
    max_routed_bytes=1024**3,
):
    if x.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Grouped cold prefill cannot run in CUDA capture")
    tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    slots = tokens * top_k
    if slots * hidden * 4 > max_routed_bytes:
        raise RuntimeError("Grouped cold prefill exceeds its routed-output budget")
    n13 = tensors[3].shape[1] * 16

    def decode(name, expert):
        offset = 0 if name == "w13" else 6
        n, k = (n13, hidden) if offset == 0 else (hidden, n13 // 2)
        packed, codebooks, scales = tensors[offset : offset + 3]
        from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import _layout

        tile_words, _ = _layout(codebooks)
        words = (n // 16) * (k // 64) * tile_words
        # Packed storage has one final guard; internal expert boundaries also
        # provide the readable guard required by the cross-word extraction.
        return dequantize_cold(
            packed.reshape(-1)[expert * words : expert * words + words + 1],
            codebooks,
            scales[expert],
            alphas[offset // 6],
            n,
            k,
        )

    ids = lookups[:, topk_ids.reshape(-1).long()]
    cold_ids, hot_ids = ids[0], ids[1]
    cold_slots = torch.nonzero(cold_ids >= 0, as_tuple=False).flatten()
    if cold_slots.numel():
        local_ids = cold_ids[cold_slots].long()
        sorted_slots = cold_slots[torch.argsort(local_ids, stable=True)]
        counts = torch.bincount(local_ids)
        # One explicit count transfer; nonzero also synchronizes on CUDA.
        counts_cpu = counts.cpu().tolist()
        selected = counts >= min_expert_tokens
        native_cold = cold_ids.clone()
        native_cold[cold_slots[selected[local_ids]]] = -1
    else:
        sorted_slots = cold_slots
        counts_cpu = []
        native_cold = cold_ids

    inverse = None
    if (
        os.environ.get("VLLM_ARVQ_DIRECT_COLD_OUTPUT", "0") == "1"
        and tokens == 4096
        and n13 == 1024
        and top_k == 8
        and hidden == 6144
        and chunk_tokens == 128
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype == torch.bfloat16
        and topk_weights.dtype == torch.float32
        and topk_weights.is_contiguous()
        and _compact_prefill_enabled()
        and os.environ.get("VLLM_ARVQ_FUSED_ROUTE_SUM", "0") == "1"
        and 2 * cold_slots.numel() >= slots
    ):
        from vllm.model_executor.layers.quantization import (
            nvfp4_arvq_direct_output as direct_output,
        )

        # Its last reader above has been queued on the same stream. Reuse the
        # allocation only after native_cold has consumed local_ids completely.
        inverse = local_ids.view(torch.int32)[:slots]
        direct_output.make_inverse(inverse, cold_slots, sorted_slots, slots)

    # FP32 preserves the native hot outputs and final route-sum ordering.
    # T4096,H6144,top8 =>768MiB, without a full repeated-input allocation.
    routed = torch.empty((slots, hidden), device=x.device, dtype=torch.float32)
    if _compact_prefill_enabled():
        # Drop only grouped-cold rows. Original both-negative routes remain
        # native so their defined zero outputs still initialize routed storage.
        grouped_mask = (cold_ids >= 0) & (native_cold < 0)
        native_slots = torch.nonzero(~grouped_mask, as_tuple=False).flatten()
        normal_split = 16 if chunk_tokens * top_k <= 32 else 8
        tail_tokens = tokens % chunk_tokens
        if normal_split == 8 and 0 < tail_tokens * top_k <= 32:
            # Preserve the original tiny last chunk's split/reduction order.
            cutoff = (tokens - tail_tokens) * top_k
            batches = [
                (native_slots[native_slots < cutoff], normal_split),
                (native_slots[native_slots >= cutoff], 16),
            ]
        else:
            batches = [(native_slots, normal_split)]
        # Decode and non-target/tiny-tail shapes never enter the candidate.
        from vllm.model_executor.layers.quantization.nvfp4_arvq_paired_policy import (
            partition,
        )

        # CPU-known upper bound: avoid candidate kernels when too few hot
        # routes can qualify (including all-cold and tiny-hot controls).
        if (
            os.environ.get("VLLM_ARVQ_PAIRED_HOT_PREFILL", "0") == "1"
            and slots - cold_slots.numel() >= 512
        ):
            paired_batches = partition(batches, tokens, hot_ids, native_cold)
        else:
            paired_batches = [(rows, split, False) for rows, split in batches]
        active_projection: Callable[..., torch.Tensor]
        for route_indices, split, use_pairs in paired_batches:
            if (
                not use_pairs
                and route_indices.numel() >= 512
                and _sort_native_prefill_enabled()
            ):
                # Each route is independent; preserve the original split
                # segmentation and scatter destinations while reusing weights.
                cold_key = native_cold[route_indices].long()
                hot_key = hot_ids[route_indices].long()
                key = torch.where(cold_key >= 0, cold_key, 256 + hot_key)
                route_indices = route_indices[torch.argsort(key, stable=True)]
            for begin in range(0, route_indices.numel(), chunk_tokens * top_k):
                route_slots = route_indices[begin : begin + chunk_tokens * top_k]
                from vllm.model_executor.layers.quantization import (
                    nvfp4_arvq_route_pack as route_pack,
                )

                xr = (
                    (x, route_slots, top_k)
                    if route_pack.eligible(x, route_slots, top_k, use_pairs)
                    else x[route_slots // top_k].half()
                )
                cold, hot = native_cold[route_slots], hot_ids[route_slots]
                if use_pairs:
                    from vllm.model_executor.layers.quantization import (
                        nvfp4_arvq_paired_runtime as paired_runtime,
                    )

                    if os.environ.get("VLLM_ARVQ_WIDE_HOT_PREFILL", "0") == "1":
                        from vllm.model_executor.layers.quantization import (
                            nvfp4_arvq_wide_runtime as wide_runtime,
                        )

                        # Use the whole scheduled prefill count, not the size
                        # of this compact route chunk or per-expert group.
                        active_projection = wide_runtime.WideHot(
                            wide_runtime.mode_for_tokens(tokens)
                        )
                    else:
                        active_projection = paired_runtime.PairedHot()
                else:
                    active_projection = projection
                h13 = active_projection(
                    xr, cold, hot, tensors[:6], alphas[0], n13, split, 2
                ).half()
                gate, up = h13.chunk(2, dim=-1)
                act = (torch.nn.functional.silu(gate) * up).half()
                if inverse is None:
                    routed[route_slots] = active_projection(
                        act, cold, hot, tensors[6:], alphas[1], hidden, 2, 1
                    )
                else:
                    native_out = active_projection(
                        act, cold, hot, tensors[6:], alphas[1], hidden, 2, 1
                    )
                    direct_output.scatter(routed, route_slots, native_out, inverse)
                    del native_out
    else:
        _native_prefill_routes(
            x,
            native_cold,
            hot_ids,
            tensors,
            alphas,
            routed,
            projection,
            chunk_tokens,
            top_k,
            n13,
            hidden,
        )

    from vllm.model_executor.layers.quantization import (
        nvfp4_arvq_cold_scatter as cold_scatter,
    )

    scatter_cold = (
        cold_scatter.prepare(x, sorted_slots, routed) if inverse is None else None
    )
    start = 0
    for expert, count in enumerate(counts_cpu):
        stop = start + count
        if count >= min_expert_tokens:
            route_slots = sorted_slots[start:stop]
            if (
                os.environ.get("VLLM_ARVQ_FUSED_COLD_GATHER", "0") == "1"
                and tokens in (2048, 4096)
                and top_k == 8
                and hidden == 6144
                and chunk_tokens == 128
                and n13 == 1024
                and x.dtype == torch.bfloat16
                and x.is_contiguous()
                and tensors[1].numel() == 512
            ):
                from vllm.model_executor.layers.quantization import (
                    nvfp4_arvq_cold_gather as cold_gather,
                )
                from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import (
                    _layout,
                )

                words_per_tile, _ = _layout(tensors[1])
                words = (n13 // 16) * (hidden // 64) * words_per_tile
                rows, w13 = cold_gather.decode_gather(
                    x,
                    route_slots,
                    tensors[0].reshape(-1)[expert * words : expert * words + words + 1],
                    tensors[1],
                    tensors[2][expert],
                    alphas[0],
                    n13,
                    hidden,
                    top_k,
                )
            else:
                rows = x[route_slots // top_k].to(torch.float16)
                w13 = decode("w13", expert)
            if os.environ.get("VLLM_ARVQ_FUSED_COLD_ACTIVATION", "0") == "1":
                from vllm.model_executor.layers.quantization import (
                    nvfp4_arvq_cold_activation as cold_activation,
                )

                h13 = torch.mm(rows, w13.T, out_dtype=torch.float32)
                del rows, w13
                act = cold_activation.run(h13)
                del h13
            else:
                h13 = torch.mm(rows, w13.T, out_dtype=torch.float32).half()
                del rows, w13
                gate, up = h13.chunk(2, dim=-1)
                act = (torch.nn.functional.silu(gate) * up).to(torch.float16)
                del h13, gate, up
            w2 = decode("w2", expert)
            if inverse is None:
                out = torch.mm(act, w2.T, out_dtype=torch.float32)
                # Preserve the existing prepared batch-scatter fallback.
                if scatter_cold is None:
                    routed[route_slots] = out
                else:
                    scatter_cold(out, route_slots)
                del out
            else:
                destination = routed[start:stop]
                assert destination.is_contiguous()
                assert destination.stride() == (hidden, 1)
                torch.ops.aten.mm.dtype_out(act, w2.T, torch.float32, out=destination)
                del destination
            del act, w2
        start = stop

    if (
        os.environ.get("VLLM_ARVQ_FUSED_ROUTE_SUM", "0") == "1"
        and tokens in (2048, 4096)
        and top_k == 8
        and hidden == 6144
        and chunk_tokens == 128
        and x.dtype == torch.bfloat16
        and routed.is_contiguous()
        and topk_weights.dtype == torch.float32
        and topk_weights.is_contiguous()
    ):
        from vllm.model_executor.layers.quantization.nvfp4_arvq_route_sum import run

        if inverse is not None:
            return direct_output.sum_routes(routed, topk_weights, inverse, x.dtype)
        return run(routed, topk_weights, x.dtype)

    outputs = []
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        tile = routed[start * top_k : stop * top_k].view(stop - start, top_k, hidden)
        outputs.append(
            (tile * topk_weights[start:stop, :, None].float()).sum(1).to(x.dtype)
        )
    return torch.cat(outputs, dim=0)


def _native_prefill_routes(
    x,
    native_cold,
    hot_ids,
    tensors,
    alphas,
    routed,
    projection,
    chunk_tokens,
    top_k,
    n13,
    hidden,
):
    """Original uncompressed route schedule, retained for controlled A/B."""
    tokens = x.shape[0]
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        route_slice = slice(start * top_k, stop * top_k)
        xr = x[start:stop].half().repeat_interleave(top_k, 0)
        cold, hot = native_cold[route_slice], hot_ids[route_slice]
        h13 = projection(
            xr,
            cold,
            hot,
            tensors[:6],
            alphas[0],
            n13,
            16 if xr.shape[0] <= 32 else 8,
            2,
        ).half()
        gate, up = h13.chunk(2, dim=-1)
        act = (torch.nn.functional.silu(gate) * up).half()
        routed[route_slice] = projection(
            act,
            cold,
            hot,
            tensors[6:],
            alphas[1],
            hidden,
            2,
            1,
        )
