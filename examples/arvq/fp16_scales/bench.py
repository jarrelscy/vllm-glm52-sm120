# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes as C
import json
import statistics
from pathlib import Path

import numpy as np
import torch

W = Path("/work")
torch.manual_seed(18)
torch.backends.cuda.matmul.allow_tf32 = False
libs = [C.CDLL(str(W / (s + ".so"))) for s in ["baseline", "fp16"]]
for lib in libs:
    lib.hybrid_launch_8x8_expert.argtypes = (
        [C.c_void_p] * 12 + [C.c_float] + [C.c_int] * 6 + [C.c_void_p]
    )
    lib.hybrid_pack.argtypes = [C.c_void_p] * 3 + [C.c_int] * 3 + [C.c_void_p]


def ptr(ts):
    return [C.c_void_p(t.data_ptr()) for t in ts]


def stream():
    return C.c_void_p(torch.cuda.current_stream().cuda_stream)


levels = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)
results = []
for N, K in [(1024, 6144), (6144, 512), (4096, 6144), (6144, 2048)]:
    rng = np.random.default_rng(123)
    E = 2
    pairs = rng.integers(0, 65536, (E, N, K // 8), dtype=np.uint16)
    pairs[..., 0] = 0
    pairs[..., -1] = 65535
    books = rng.integers(0, 2**32, (E, 512), dtype=np.uint32)
    frag = (
        pairs.reshape(E, N // 16, 2, 8, K // 64, 2, 4)
        .transpose(0, 1, 4, 5, 2, 3, 6)
        .copy()
        .reshape(E, N // 16, K // 64, 128)
    )
    packed = frag[..., ::2].astype(np.uint32) | (
        frag[..., 1::2].astype(np.uint32) << 16
    )
    sc = rng.integers(32, 57, (E, N, K // 128), dtype=np.uint8)
    sf = ((1 + (sc.astype(int) & 7) / 8) * np.exp2((sc.astype(int) >> 3) - 7)).astype(
        np.float16
    )

    def storage(s, E=E, N=N, K=K):
        return torch.from_numpy(
            s.reshape(E, N // 16, 16, K // 128).transpose(0, 1, 3, 2).copy()
        ).cuda()

    cw = torch.from_numpy(packed).cuda()
    cb = torch.from_numpy(books).cuda()
    s8 = storage(sc)
    s16 = storage(sf)
    dummy = torch.zeros(1, device="cuda")
    split = 8
    for slots in [8, 32]:
        s16.copy_(storage(sf))
        x = torch.randn(slots, K, device="cuda", dtype=torch.float16)
        q = torch.empty(slots, 4, K // 8, device="cuda", dtype=torch.int32)
        qs = torch.empty(slots, 4, K // 16, device="cuda", dtype=torch.uint8)
        ids = (torch.arange(slots, device="cuda", dtype=torch.int32) % 2).flip(0)
        hot = torch.full_like(ids, -1)
        part = torch.empty(slots, N, split, device="cuda")
        outs = [torch.empty(slots, N, device="cuda") for _ in libs]
        assert libs[0].hybrid_pack(*ptr([x, q, qs]), K, slots, 4, stream()) == 0

        def call(
            v,
            args=(
                cw,
                cb,
                s8,
                s16,
                dummy,
                q,
                qs,
                ids,
                hot,
                part,
                outs,
                N,
                K,
                slots,
                split,
            ),
        ):
            cw, cb, s8, s16, dummy, q, qs, ids, hot, part, outs, N, K, slots, split = (
                args
            )
            err = libs[v].hybrid_launch_8x8_expert(
                *ptr(
                    [
                        cw,
                        cb,
                        [s8, s16][v],
                        dummy,
                        dummy,
                        dummy,
                        q,
                        qs,
                        ids,
                        hot,
                        part,
                        outs[v],
                    ]
                ),
                1.0,
                N,
                K,
                slots,
                split,
                4,
                1,
                stream(),
            )
            assert err == 0

        call(0)
        call(1)
        torch.accelerator.synchronize()
        rel = float(
            torch.linalg.vector_norm(outs[1] - outs[0])
            / torch.linalg.vector_norm(outs[0])
        )
        mx = float((outs[1] - outs[0]).abs().max())
        assert rel < 1e-5, (N, K, rel)
        graphs = []
        for v in range(2):
            for _ in range(3):
                call(v)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(10):
                    call(v)
            graphs.append(g)
        times = [[], []]
        for repeat in range(6):
            for v in [0, 1] if repeat % 2 == 0 else [1, 0]:
                a, b = (
                    torch.Event(device="cuda", enable_timing=True),
                    torch.Event(device="cuda", enable_timing=True),
                )
                a.record()
                graphs[v].replay()
                b.record()
                b.synchronize()
                times[v].append(a.elapsed_time(b) * 100)
        # Non-FP8 scales: independently decode the packed activation planes.
        snew = (sf.astype(np.float32) * 1.023).astype(np.float16)
        s16.copy_(storage(snew))
        call(1)
        words = q.cpu().numpy().astype(np.uint32)
        av = levels[
            (words[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15
        ].reshape(slots, 4, K)
        ac = qs.cpu().numpy().astype(np.int32)
        asc = np.where(
            ac >> 3 == 0,
            (ac & 7) * 2.0**-9,
            (1 + (ac & 7) / 8) * np.exp2((ac >> 3) - 7),
        )
        ax = (
            (
                av
                * np.repeat(asc, 16, axis=-1)
                / np.array([1, 16, 256, 4096])[None, :, None]
            )
            .sum(axis=1)
            .astype(np.float32)
        )
        bv = levels[(books[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15]
        refs = []
        for i, e in enumerate(ids.cpu().tolist()):
            ww = (bv[e, pairs[e] & 255] + bv[e, 256 + (pairs[e] >> 8)]).reshape(
                N, K
            ) * np.repeat(snew[e].astype(np.float32), 128, axis=1)
            refs.append(ww @ ax[i])
        ref = torch.from_numpy(np.stack(refs)).cuda()
        r = float(
            torch.linalg.vector_norm(outs[1] - ref) / torch.linalg.vector_norm(ref)
        )
        m = float((outs[1] - ref).abs().max())
        assert r < 2e-5, r
        med = [statistics.median(t) for t in times]
        row = dict(
            N=N,
            K=K,
            slots=slots,
            fp8_us=med[0],
            fp16_us=med[1],
            overhead_pct=100 * (med[1] / med[0] - 1),
            matched_rel_l2=rel,
            matched_max_abs=mx,
            fp16_oracle_rel_l2=r,
            fp16_oracle_max_abs=m,
            samples_us=times,
        )
        results.append(row)
        print(json.dumps(row), flush=True)
        (W / "results.json").write_text(json.dumps(results, indent=2))
        del graphs, ref, part, outs, x, q, qs, ids, hot
    del cw, cb, s8, s16
    torch.accelerator.empty_cache()
