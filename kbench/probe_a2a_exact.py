#!/usr/bin/env python3
"""4-rank NCCL probe + bit-exactness proof + microbench for VLLM_DCP_A2A_EXACT.

Per decode shape [B, H=64, D=512] (B = padded decode tokens, capture sizes
1/2/4/8/16):

  1. ORDER PROBE -- run torch.distributed reduce_scatter_tensor on the same
     flattened layout the ag_rs epilogue uses ([H, B, D] contiguous, chunk r =
     head shard r), with wide-exponent random bf16, and identify the unique
     4-leaf reduction tree (with per-node bf16 rounding) that reproduces the
     received chunk bit-for-bit. Reports per-element uniformity.
  2. EXACTNESS -- run the full ag_rs epilogue (AG(lse) + correct kernel + RS)
     vs the exact-a2a epilogue (pack + all_to_all_single + exact combine) on
     random + adversarial inputs; assert bitwise equality of the outputs.
  3. MICROBENCH -- time both epilogues back-to-back over many iterations
     (CUDA events, barrier-synced), report per-layer us and the projected
     per-step saving (x81 MLA layers).

Run inside the prod image (same libnccl) with prod NCCL env, co-tenant safe
(<1 GiB/GPU):

  docker run --rm --gpus all --entrypoint /bin/bash --shm-size 8g \
    -e NCCL_MAX_NCHANNELS=4 -e NCCL_BUFFSIZE=1048576 -e NCCL_ALGO=RING,TREE \
    -v /home/jarrelscy/glm52/wt-round3:/work glm52-vision-sm120:latest -c \
    'source /opt/vllm/.venv/bin/activate && torchrun --nproc-per-node=4 \
       /work/kbench/probe_a2a_exact.py'
"""

import itertools
import os
import sys

import torch
import torch.distributed as dist

WORLD = 4
H_TOTAL = 64
D = 512
DTYPE = torch.bfloat16
BATCHES = [1, 2, 4, 8, 16]
BENCH_ITERS = 400
WARMUP = 50


def log(rank, *a):
    print(f"[rank{rank}]", *a, flush=True)


def enumerate_trees():
    trees = []
    for perm in itertools.permutations(range(4)):
        p0, p1, p2, p3 = perm
        if p0 < p1:
            trees.append((0, perm))
    for perm in itertools.permutations(range(4)):
        p0, p1, p2, p3 = perm
        if p0 < p1 and p2 < p3 and p0 < p2:
            trees.append((1, perm))
    return trees


def apply_tree(vals, tree):
    bal, (p0, p1, p2, p3) = tree
    dt = vals[0].dtype

    def add(a, b):
        return (a.float() + b.float()).to(dt)

    if bal == 0:
        return add(add(add(vals[p0], vals[p1]), vals[p2]), vals[p3])
    return add(add(vals[p0], vals[p1]), add(vals[p2], vals[p3]))


def make_wide(shape, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mant = torch.rand(shape, generator=gen, dtype=torch.float32) * 1.9 + 0.05
    sign = torch.where(torch.rand(shape, generator=gen) < 0.5, -1.0, 1.0)
    expo = torch.randint(-6, 7, shape, generator=gen).to(torch.float32)
    return (mant * sign * (2.0**expo)).to(DTYPE).to(device)


def probe_order(rank, group, B, device):
    """Mirror CudaCommunicator.reduce_scatter's layout: RS input [H, B, D]."""
    x_bhd = make_wide((B, H_TOTAL, D), 0x5EED + 7919 * rank + B, device)
    # exact layout transformation of the prod path: movedim(0, 1).contiguous()
    x_rs = x_bhd.movedim(0, 1).contiguous()  # [H, B, D]

    xs = torch.empty((WORLD, H_TOTAL, B, D), dtype=DTYPE, device=device)
    dist.all_gather_into_tensor(xs.view(WORLD, -1), x_rs.view(-1), group=group)

    out = torch.empty((H_TOTAL // WORLD, B, D), dtype=DTYPE, device=device)
    dist.reduce_scatter_tensor(out, x_rs, group=group)
    torch.cuda.synchronize()

    h_lo, h_hi = rank * (H_TOTAL // WORLD), (rank + 1) * (H_TOTAL // WORLD)
    vals = [xs[r, h_lo:h_hi] for r in range(WORLD)]

    trees = enumerate_trees()
    full, partial = [], {}
    for tree in trees:
        ref = apply_tree(vals, tree)
        eq = ref == out
        frac = eq.float().mean().item()
        if frac == 1.0:
            full.append(tree)
        elif frac > 0:
            partial[tree] = frac
    log(rank, f"B={B}: full matches={full} partial={ {k: round(v,4) for k,v in sorted(partial.items(), key=lambda i:-i[1])[:4]} }")
    if len(full) == 1:
        return full[0]
    if len(full) == 0 and partial:
        # segmentation diagnostic: per-element best tree id map along flat dim
        best = torch.full((out.numel(),), -1, dtype=torch.int32)
        flat_out = out.view(-1)
        for i, tree in enumerate(trees):
            ref = apply_tree(vals, tree).view(-1)
            m = (ref == flat_out) & (best.to(out.device) == -1)
            best = torch.where(m.cpu(), i, best)
        # report transitions
        b = best.numpy()
        trans = [(0, int(b[0]))]
        for j in range(1, len(b)):
            if b[j] != b[j - 1]:
                trans.append((j, int(b[j])))
        log(rank, f"B={B}: SEGMENTED order, {len(trans)} segments, first 12: {trans[:12]}")
    return None


# --------------- epilogue implementations (mirror prod) ---------------


def ag_rs_epilogue(out, lse, group, correct_fn):
    """cp_lse_ag_out_rs with raw dist ops (identical layouts/collectives)."""
    B, H, Dd = out.shape
    lse_c = lse.contiguous()
    lses = torch.empty((WORLD,) + lse_c.shape, dtype=lse_c.dtype, device=lse_c.device)
    dist.all_gather_into_tensor(lses.view(WORLD, -1), lse_c.view(-1), group=group)
    o, _ = correct_fn(out, lses.view((WORLD,) + lse_c.shape))
    x_rs = o.movedim(0, 1).contiguous()  # [H, B, D]
    res = torch.empty((H // WORLD, B, Dd), dtype=o.dtype, device=o.device)
    dist.reduce_scatter_tensor(res, x_rs, group=group)
    return res.movedim(0, 1).contiguous()  # [B, H/4, D]


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD

    from vllm.v1.attention.ops.common import CPTritonContext, correct_attn_out
    import vllm.v1.attention.ops.dcp_alltoall as a2a

    results = {}
    for B in BATCHES:
        # ---------- 1. order probe ----------
        tree = probe_order(rank, group, B, device)
        ok = torch.tensor([1 if tree is not None else 0], device=device)
        dist.all_reduce(ok, group=group)
        if ok.item() != WORLD:
            log(rank, f"B={B}: NO unique uniform order on some rank -- exact a2a not viable at this shape")
            continue

        # ---------- 2. bit-exactness ----------
        # ctx=None -> fresh CPTritonContext per call, exactly like prod's
        # cp_lse_ag_out_rs(ctx=None) (the cached-handle branch of
        # CPTritonContext.call_kernel is never exercised in prod and is
        # broken on this Triton version).
        def correct_fn(o, lses):
            return correct_attn_out(o, lses, rank, None, is_lse_base_on_e=True)

        n_mismatch = 0
        for t in range(8):
            out = make_wide((B, H_TOTAL, D), 0xA2A + 104729 * rank + t, device)
            gen = torch.Generator(device="cpu").manual_seed(0x15E + 31 * rank + t)
            lse = (torch.randn((B, H_TOTAL), generator=gen) * 4.0).to(device)
            if t == 0:
                lse.view(-1)[0] = float("inf")
                lse.view(-1)[1] = float("-inf")
                lse.view(-1)[2] = float("nan")
            if t == 1:
                lse.fill_(float("-inf"))  # all ranks -inf -> lse_max clamp path

            ref = ag_rs_epilogue(out.clone(), lse.clone(), group, correct_fn)
            # exact-a2a epilogue from the kernel pieces (same as prod impl)
            lse_pack_dim = a2a._dcp_a2a_lse_pack_dim(out.dtype)
            send, recv = a2a._dcp_a2a_send_recv_buffers(
                (WORLD, B, H_TOTAL // WORLD, D + lse_pack_dim), device, out.dtype
            )
            a2a._dcp_a2a_pack_send(
                out, lse.contiguous(), send, WORLD, H_TOTAL // WORLD, D, lse_pack_dim
            )
            dist.all_to_all_single(recv.view(-1), send.view(-1), group=group)
            test = a2a._dcp_a2a_unpack_combine_exact(
                recv, D, lse_pack_dim, True, tree
            )
            if not torch.equal(ref, test):
                n_mismatch += 1
                bad = (ref != test).sum().item()
                log(rank, f"B={B} trial{t}: MISMATCH {bad} elems")
        status = "BIT-EXACT" if n_mismatch == 0 else f"FAILED ({n_mismatch}/8 trials)"
        log(rank, f"B={B}: exactness {status}, tree={tree}")

        # ---------- 3. microbench ----------
        out = make_wide((B, H_TOTAL, D), 0xBE7 + rank, device)
        gen = torch.Generator(device="cpu").manual_seed(0xBE8 + rank)
        lse = (torch.randn((B, H_TOTAL), generator=gen) * 4.0).to(device)
        lse_pack_dim = a2a._dcp_a2a_lse_pack_dim(out.dtype)

        def run_agrs():
            return ag_rs_epilogue(out, lse, group, correct_fn)

        def run_a2a():
            send, recv = a2a._dcp_a2a_send_recv_buffers(
                (WORLD, B, H_TOTAL // WORLD, D + lse_pack_dim), device, out.dtype
            )
            a2a._dcp_a2a_pack_send(
                out, lse, send, WORLD, H_TOTAL // WORLD, D, lse_pack_dim
            )
            dist.all_to_all_single(recv.view(-1), send.view(-1), group=group)
            return a2a._dcp_a2a_unpack_combine_exact(recv, D, lse_pack_dim, True, tree)

        times = {}
        for name, fn in (("ag_rs", run_agrs), ("a2a_exact", run_a2a)):
            for _ in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(BENCH_ITERS):
                fn()
            e.record()
            torch.cuda.synchronize()
            times[name] = s.elapsed_time(e) * 1000 / BENCH_ITERS  # us
            dist.barrier()
        d_us = times["ag_rs"] - times["a2a_exact"]
        log(
            rank,
            f"B={B}: ag_rs={times['ag_rs']:.2f}us a2a_exact={times['a2a_exact']:.2f}us "
            f"delta={d_us:+.2f}us/layer -> {d_us*81/1000:+.3f}ms/step (x81)",
        )
        results[B] = (status, times)

    dist.barrier()
    if rank == 0:
        log(rank, "SUMMARY:", {k: (v[0], {n: round(t, 2) for n, t in v[1].items()}) for k, v in results.items()})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
