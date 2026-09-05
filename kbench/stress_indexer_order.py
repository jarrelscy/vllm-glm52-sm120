#!/usr/bin/env python3
"""DSA indexer top-k ORDER-determinism stress test.

The DCP sparse-indexer decode path has two atomic-append stages whose output
ORDER (not set) is scheduling-dependent:
  1. StableTopKFromGatheredCandidatesKernel: selected keys land at
     `dst = atomic_add(committed_count)` -- order = warp-scheduling order.
  2. _convert_req_index_to_global_index_kernel COMPACT_TO_FRONT: cross-tile
     prefix base via `atomic_add(valid_count_ptr)` -- documented "Prefix
     order is unspecified (only the set matters)".
The trtllm-gen sparse attention accumulates in index order, so an unspecified
order = fp reduction-order nondeterminism at temp 0.

This test runs both kernels on FIXED inputs many times (optionally with side
-stream traffic) and reports: set changes (must be 0 = selection is sound)
and order changes (the nondeterminism under audit). With
VLLM_DSA_CANONICAL_TOPK=1 both must be 0.
"""
import argparse
import os
import sys

import torch


def side_traffic(dev, n_streams=3, mb=48):
    streams = [torch.cuda.Stream(device=dev) for _ in range(n_streams)]
    n = mb * 1024 * 1024 // 2
    bufs = [(torch.randn(n, dtype=torch.float16, device=dev),
             torch.empty(n, dtype=torch.float16, device=dev))
            for _ in streams]
    mm = [torch.randn(1536, 1536, dtype=torch.float16, device=dev)
          for _ in streams]
    return streams, bufs, mm


def pump(streams, bufs, mm, depth=8):
    for s, (a, b), m in zip(streams, bufs, mm):
        with torch.cuda.stream(s):
            for _ in range(depth):
                b.copy_(a, non_blocking=True)
                torch.mm(m, m)


def run_repeated(fn, out_fn, iters, traffic):
    """Returns (set_changes, order_changes) vs iteration 0."""
    fn()
    torch.cuda.synchronize()
    ref = out_fn().clone()
    ref_sets = [set(r[r >= 0].tolist()) for r in ref.cpu()]
    set_changes = order_changes = 0
    for it in range(iters):
        if traffic and it % 8 == 0:
            pump(*traffic)
        fn()
        torch.cuda.synchronize()
        cur = out_fn()
        if not torch.equal(cur, ref):
            order_changes += 1
            cur_sets = [set(r[r >= 0].tolist()) for r in cur.cpu()]
            if cur_sets != ref_sets:
                set_changes += 1
    if traffic:
        for s in traffic[0]:
            s.synchronize()
    return set_changes, order_changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--dcp", type=int, default=4)
    ap.add_argument("--no-traffic", action="store_true")
    args = ap.parse_args()

    dev = "cuda:0"
    torch.cuda.set_device(dev)
    torch.manual_seed(0)

    sys.path.insert(0, os.getcwd())
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        stable_topk_from_gathered_candidates_cutedsl,
    )
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _CANONICAL_TOPK,
        _canonicalize_topk_order,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index,
    )
    canonical = _CANONICAL_TOPK

    traffic = None if args.no_traffic else side_traffic(dev)

    # ---- Stage 1: StableTopK merge selector -------------------------------
    # gathered: [rows, dcp*topk, 2] (score fp32, global_id fp32), unique ids,
    # scores drawn so ~topk*2 candidates are competitive (realistic ties in
    # high radix bins).
    ncand = args.dcp * args.topk
    scores = torch.randn(args.rows, ncand, device=dev)
    ids = torch.arange(ncand, device=dev, dtype=torch.float32)
    ids = ids.unsqueeze(0).expand(args.rows, -1)
    gathered = torch.stack([scores, ids], dim=-1).contiguous()
    out = torch.empty((args.rows, args.topk), dtype=torch.int32, device=dev)

    def run_merge():
        # Mirrors _merge_dcp_topk_global: selector + (flag-gated) canonical
        # ordering.
        stable_topk_from_gathered_candidates_cutedsl(
            gathered, args.topk, out=out)
        if canonical:
            _canonicalize_topk_order(out)

    sc, oc = run_repeated(run_merge, lambda: out, args.iters, traffic)
    print(f"StableTopK merge:  set_changes={sc}  ORDER_changes={oc}/"
          f"{args.iters}")
    r1 = oc

    # ---- Stage 2: DCP filter + compaction ---------------------------------
    # Simulate the post-merge global ids for this rank: interleave groups of
    # cp_interleave across dcp ranks; seq of 200K tokens, block_size 64.
    n_tok = args.rows
    seq_len = 200_000
    block_size = 64
    n_blocks = (seq_len + block_size - 1) // block_size
    block_table = torch.randperm(n_blocks * 2, device=dev,
                                 dtype=torch.int32)[:n_blocks]
    block_table = block_table.unsqueeze(0).expand(n_tok, -1).contiguous()
    req_id = torch.arange(n_tok, device=dev, dtype=torch.int32)
    # unique global token ids in range; ~1/dcp of them owned by rank 0
    g = torch.Generator(device="cpu").manual_seed(7)
    tok = torch.stack([
        torch.randperm(seq_len, generator=g)[:args.topk]
        for _ in range(n_tok)
    ]).int().to(dev)

    def run_filter():
        res = triton_filter_and_convert_dcp_index(
            req_id, block_table, tok,
            dcp_size=args.dcp, dcp_rank=0,
            cp_kv_cache_interleave_size=64,
            BLOCK_SIZE=block_size, NUM_TOPK_TOKENS=args.topk,
            return_valid_counts=True)
        run_filter.out = res[0]

    sc, oc = run_repeated(run_filter, lambda: run_filter.out,
                          args.iters, traffic)
    print(f"DCP filter+compact: set_changes={sc}  ORDER_changes={oc}/"
          f"{args.iters}")

    bad = (r1 + oc) if canonical else 0
    if canonical:
        print("CANONICAL mode:", "PASS (bit-stable)" if bad == 0 else "FAIL")
        sys.exit(1 if bad else 0)
    print("(sets must never change; order changes demonstrate the "
          "scheduling-dependent nondeterminism)")
    sys.exit(1 if sc else 0)


if __name__ == "__main__":
    main()
