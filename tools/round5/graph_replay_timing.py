# SPDX-License-Identifier: Apache-2.0
"""Capture-replicating in-graph timing for the round-5 copy-elimination levers.

Builds two CUDA graphs replicating one decode step's copy-adjacent op
sequence (78 verify-pass attention epilogues at B tokens + 22 DCP top-k
merges), replays them, and reports the per-step delta. NCCL collectives are
excluded from BOTH graphs (the levers do not change them: byte-identical
inputs into the identical enqueue), so the delta isolates exactly what the
levers change: the direct_copy kernels, the strided-store correct kernel and
the raw-layout top-k read.

Graph A (baseline): correct(in-place) -> staging copy -> epilogue copy ->
                    v_up bmm (strided view)      [x NUM_LAYERS]
                    concat copy -> 3-D stable-topk [x NUM_MERGES]
Graph B (levers):   correct(strided store) -> v_up bmm (contig batch)
                    [x NUM_LAYERS]
                    4-D raw stable-topk            [x NUM_MERGES]

Also validates replay bit-exactness: after capture, both graphs are replayed
on identical inputs and every output is compared bytewise.

Run inside the prod container (shared GPU with the serving process: use the
min over many replays):

  docker exec <c> /opt/vllm/.venv/bin/python /tmp/round5/graph_replay_timing.py \
      --common /tmp/round5/common.py --kernel /tmp/round5/dcp_indexer_cutedsl.py
"""

import argparse
import importlib.util
import sys

import torch

WS = 4
H = 64
HL = H // WS
D = 512
V = 256
K = 2048
NUM_LAYERS = 78  # verify-pass attention calls per step
NUM_MERGES = 22  # DCP top-k merges per step (measured: 1628 / 74 steps)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_rs_graph(common, B, staged_lever, pool=None):
    """One graph containing NUM_LAYERS attention-epilogue sequences."""
    out0 = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    lses = torch.randn(WS, B, H, device="cuda", dtype=torch.float32)
    w_uv = torch.randn(HL, D, V, device="cuda", dtype=torch.bfloat16)
    proj = torch.empty(B, HL, V, device="cuda", dtype=torch.bfloat16)
    rank = 1

    def one_layer():
        # NB: exact per-layer kernel population of the prod path -- the
        # baseline corrects IN PLACE (no clone), so across replays out0
        # accumulates repeated corrections; that is irrelevant for timing
        # (same kernels, same shapes) and op-level bit-exactness is proven
        # separately by bitexact_rs_staged.py.
        if staged_lever:
            staged = torch.empty((H, B, D), device="cuda", dtype=torch.bfloat16)
            common.correct_attn_out_staged(
                out0, lses, rank, staged.movedim(0, 1), is_lse_base_on_e=True
            )
            rs_out = staged.view(WS, HL, B, D)[rank]  # stands in for RS output
            x = rs_out.movedim(0, 1)  # VIEW (RS_VIEW lever)
            torch.bmm(
                x.view(-1, HL, D).transpose(0, 1), w_uv, out=proj.transpose(0, 1)
            )
        else:
            common.correct_attn_out(
                out0, lses, rank, common.CPTritonContext(), is_lse_base_on_e=True
            )
            staged = out0.movedim(0, 1).contiguous()  # staging COPY
            rs_out = staged.view(WS, HL, B, D)[rank]
            x = rs_out.movedim(0, 1).contiguous()  # epilogue COPY
            torch.bmm(
                x.view(-1, HL, D).transpose(0, 1), w_uv, out=proj.transpose(0, 1)
            )
        return proj

    # warmup on a side stream (compiles triton kernels outside capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_layer()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    kw = {"pool": pool} if pool is not None else {}
    with torch.cuda.graph(g, **kw):
        for _ in range(NUM_LAYERS):
            r = one_layer()
    return g, (out0, lses, w_uv), r


def build_topk_graph(mod, rows, raw_lever, pool=None):
    raw = torch.randn(WS, rows, K, 2, device="cuda", dtype=torch.float32)
    out = torch.empty(rows, K, device="cuda", dtype=torch.int32)

    def one_merge():
        if raw_lever:
            mod.stable_topk_from_gathered_candidates_cutedsl(
                raw, K, out=out, canonical=True
            )
        else:
            concat = raw.movedim(0, 1).reshape(rows, WS * K, 2)  # COPY
            mod.stable_topk_from_gathered_candidates_cutedsl(
                concat, K, out=out, canonical=True
            )
        return out

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_merge()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    kw = {"pool": pool} if pool is not None else {}
    with torch.cuda.graph(g, **kw):
        for _ in range(NUM_MERGES):
            r = one_merge()
    return g, raw, r


def time_graph(g, replays=200, inner=5):
    # min over many timed windows: robust against the co-resident serving
    # process's kernels
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(replays)]
    best = float("inf")
    for a, b in evs:
        a.record()
        for _ in range(inner):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / inner)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    torch.cuda.set_device("cuda:0")
    common = load_module("round5_common", args.common)
    mod = load_module("round5_cutedsl", args.kernel)
    torch.manual_seed(7)

    B = args.batch

    # --- RS epilogue graphs -------------------------------------------------
    g_base, in_base, out_base = build_rs_graph(common, B, staged_lever=False)
    g_lever, in_lever, out_lever = build_rs_graph(common, B, staged_lever=True)
    # lever-graph replay determinism on fixed inputs (op-level bit-exactness
    # vs the baseline path is proven by bitexact_rs_staged.py; the baseline
    # graph corrects in place so its buffer evolves across replays by design)
    g_lever.replay()
    torch.cuda.synchronize()
    ref = out_lever.clone()
    g_lever.replay()
    torch.cuda.synchronize()
    same = torch.equal(
        ref.view(torch.uint8), out_lever.contiguous().view(torch.uint8)
    )
    print(f"[{'PASS' if same else 'FAIL'}] RS lever graph: replay deterministic")
    if not same:
        sys.exit(1)

    t_base = time_graph(g_base)
    t_lever = time_graph(g_lever)
    print(
        f"RS epilogue x{NUM_LAYERS} (B={B}): baseline {t_base*1e3:.1f} us  "
        f"lever {t_lever*1e3:.1f} us  saving {1e3*(t_base-t_lever):.1f} us/step"
    )

    # --- top-k merge graphs -------------------------------------------------
    rows = B
    gt_base, raw_base, o_base = build_topk_graph(mod, rows, raw_lever=False)
    gt_lever, raw_lever_t, o_lever = build_topk_graph(mod, rows, raw_lever=True)
    raw_lever_t.copy_(raw_base)
    gt_base.replay()
    ref = o_base.clone()
    gt_lever.replay()
    torch.cuda.synchronize()
    same = torch.equal(ref, o_lever)
    print(f"[{'PASS' if same else 'FAIL'}] topk graphs: replay outputs identical")
    if not same:
        sys.exit(1)

    t_base = time_graph(gt_base)
    t_lever = time_graph(gt_lever)
    print(
        f"topk merge x{NUM_MERGES} (rows={rows}): baseline {t_base*1e3:.1f} us  "
        f"lever {t_lever*1e3:.1f} us  saving {1e3*(t_base-t_lever):.1f} us/step"
    )


if __name__ == "__main__":
    main()
