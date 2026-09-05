#!/usr/bin/env python3
"""VLLM_DSA_CANONICAL_TOPK=logical / =inkernel verification.

Demonstrates the block-layout dependence of the physical-sort canonical mode
(VLLM_DSA_CANONICAL_TOPK=1) and verifies the "logical" mode fixes it:

  Given the SAME logical top-k selection per row and two DIFFERENT KV block
  tables (same logical->KV mapping, physically permuted blocks — exactly what
  the block pool produces after different predecessor requests):

    * physical mode: the compacted prefix order (mapped back to logical token
      ids) DIFFERS between the two layouts -> the sparse attention accumulates
      the same values in a different fp order -> temp-0 flips that track
      predecessor history.
    * logical mode: the prefix order is identical for both layouts (descending
      logical token id), same selected set, same valid_counts -> accumulation
      order is a pure function of request content.

  Both modes must also be repeat-deterministic on fixed inputs.

VLLM_DSA_CANONICAL_TOPK=inkernel (follow-up): same canonical order as
"logical" with the sorts moved INTO the producing kernels. This script
additionally verifies:

  * filter path: "inkernel" passes every "logical" check AND its output is
    bit-identical to "logical" mode's (same canonical order definition);
  * merge kernel (StableTopKFromGatheredCandidatesKernel canonical=True):
    output equals the descending sort of the base kernel's output (same set,
    canonical order), is invariant under permutation of the CANDIDATE input
    order (pure function of content — covers nondeterministic upstream
    per-rank top-k order), and is repeat-deterministic;
  * timing: canonical-vs-base merge kernel delta and deterministic-vs-atomic
    filter delta, against the ~63us/layer torch.sort of "logical" mode.

Run inside the prod image:
  python kbench/check_canonical_logical.py
"""
import importlib.util
import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
SU_PATH = HERE.parent / "vllm/v1/attention/backends/mla/sparse_utils.py"


def load_sparse_utils(mode: str):
    os.environ["VLLM_DSA_CANONICAL_TOPK"] = mode
    name = f"sparse_utils_{mode}"
    spec = importlib.util.spec_from_file_location(name, str(SU_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_case(dev, rows=8, topk=2048, block_size=64, dcp_size=4, dcp_rank=1,
               interleave=1, max_len=200_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    req_id = torch.arange(rows, dtype=torch.int32, device=dev)
    n_blocks = (max_len // (dcp_size * block_size)) + 2
    # layout A: sequential physical blocks; layout B: a permutation of the
    # same physical blocks (what a churned block pool hands out)
    bt_a = torch.arange(rows * n_blocks, dtype=torch.int32,
                        device=dev).reshape(rows, n_blocks)
    perm = torch.randperm(rows * n_blocks, generator=g)
    bt_b = bt_a.reshape(-1)[perm.to(dev)].reshape(rows, n_blocks).contiguous()

    # same LOGICAL top-k selection per row: unique global token ids,
    # descending (what the logical-canonical merge emits); physical mode gets
    # a fixed permutation of the same set (merge order is unspecified there)
    tk = torch.full((rows, topk), -1, dtype=torch.int32)
    for r in range(rows):
        n_valid = topk - 37 * r
        ids = torch.randperm(max_len, generator=g)[:n_valid]
        tk[r, :n_valid] = torch.sort(ids, descending=True).values.int()
    return req_id, bt_a, bt_b, tk.to(dev), block_size, dcp_size, dcp_rank, \
        interleave


def logical_of(out, bt, block_size, dcp_size, dcp_rank, interleave, counts):
    """Map physical prefix entries back to logical global token ids."""
    res = []
    inv = {}
    for req in range(bt.shape[0]):
        for lb, pb in enumerate(bt[req].tolist()):
            inv[(req, pb)] = lb
    for r in range(out.shape[0]):
        c = int(counts[r])
        row = out[r, :c].tolist()
        logi = []
        for phys in row:
            pb, off = phys // block_size, phys % block_size
            lb = inv[(r, pb)]
            local = lb * block_size + off
            glob = ((local // interleave) * dcp_size + dcp_rank) * interleave \
                + local % interleave
            logi.append(glob)
        res.append(logi)
    return res


def load_dcp_indexer():
    path = HERE.parent / (
        "vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py")
    spec = importlib.util.spec_from_file_location("dcp_indexer_wt", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dcp_indexer_wt"] = mod
    spec.loader.exec_module(mod)
    return mod


def time_cuda(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # us


def check_filter_modes(dev):
    (req_id, bt_a, bt_b, tk, block_size, dcp_size, dcp_rank,
     interleave) = build_case(dev)

    failures = 0
    canonical_outs = {}
    for mode in ("1", "logical", "inkernel"):
        su = load_sparse_utils(mode)
        run = lambda bt: su.triton_filter_and_convert_dcp_index(  # noqa: E731
            req_id, bt, tk, dcp_size=dcp_size, dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=interleave, BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=tk.shape[1], return_valid_counts=True)

        out_a, cnt_a = run(bt_a)
        out_b, cnt_b = run(bt_b)
        # repeat determinism on fixed inputs
        rep_bad = sum(
            0 if torch.equal(run(bt_a)[0], out_a) else 1 for _ in range(25))
        assert torch.equal(cnt_a, cnt_b), "valid_counts must match layouts"
        log_a = logical_of(out_a.cpu(), bt_a.cpu(), block_size, dcp_size,
                           dcp_rank, interleave, cnt_a.cpu())
        log_b = logical_of(out_b.cpu(), bt_b.cpu(), block_size, dcp_size,
                           dcp_rank, interleave, cnt_b.cpu())
        sets_equal = all(set(a) == set(b) for a, b in zip(log_a, log_b))
        order_equal = log_a == log_b
        # padding contiguity: everything past valid_count must be -1
        pad_ok = all(
            bool((out_a[r, int(cnt_a[r]):] == -1).all()) and
            bool((out_b[r, int(cnt_b[r]):] == -1).all())
            for r in range(out_a.shape[0]))
        if mode in ("logical", "inkernel"):
            # canonical order must be descending and layout-invariant
            desc = all(all(x > y for x, y in zip(row, row[1:]))
                       for row in log_a)
            ok = sets_equal and order_equal and pad_ok and desc \
                and rep_bad == 0
            canonical_outs[mode] = (log_a, cnt_a.cpu())
            print(f"[{'ok  ' if ok else 'FAIL'}] {mode} mode: set_eq="
                  f"{sets_equal} order_layout_invariant={order_equal} "
                  f"descending={desc} pad_ok={pad_ok} repeat_bad={rep_bad}")
            failures += not ok
        else:
            ok = sets_equal and pad_ok and rep_bad == 0
            print(f"[{'ok  ' if ok else 'FAIL'}] physical mode: set_eq="
                  f"{sets_equal} pad_ok={pad_ok} repeat_bad={rep_bad}; "
                  f"order_layout_invariant={order_equal} "
                  f"(False EXPECTED = the layout dependence being fixed)")
            failures += not ok
            if order_equal:
                print("  NOTE: physical order matched layouts here — "
                      "permutation too gentle to show the dependence?")

    # inkernel must define the SAME canonical order as logical (bit-identical
    # logical sequences => bit-identical physical rows per layout).
    same = canonical_outs["logical"][0] == canonical_outs["inkernel"][0] and \
        torch.equal(canonical_outs["logical"][1], canonical_outs["inkernel"][1])
    print(f"[{'ok  ' if same else 'FAIL'}] inkernel == logical (bit-identical "
          f"canonical order + counts): {same}")
    failures += not same

    # ---- timing: deterministic-base vs atomic compaction ------------------
    su_base = load_sparse_utils("0")
    su_ink = load_sparse_utils("inkernel")
    su_log = load_sparse_utils("logical")
    for label, su in (("base(atomic)", su_base), ("logical", su_log),
                      ("inkernel", su_ink)):
        t = time_cuda(lambda: su.triton_filter_and_convert_dcp_index(
            req_id, bt_a, tk, dcp_size=dcp_size, dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=interleave, BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=tk.shape[1], return_valid_counts=True))
        print(f"  filter timing [{tk.shape[0]}x{tk.shape[1]}] {label}: "
              f"{t:.1f} us")
    return failures


def check_merge_kernel(dev, rows=8, topk=2048, dcp=4, reps=25):
    """Canonical (in-kernel bitonic) selector: order = pure fn of content."""
    dcpk = load_dcp_indexer()
    g = torch.Generator().manual_seed(3)
    ncand = dcp * topk
    scores = torch.randn(rows, ncand, generator=g).to(dev)
    # unique global ids; a tail of pad (-1) candidates like a short row
    ids = torch.stack([torch.randperm(4 * ncand, generator=g)[:ncand]
                       for _ in range(rows)]).float()
    npad = 173
    scores[:, -npad:] = -float("inf")
    ids[:, -npad:] = -1.0
    # row 0: fewer valid candidates than topk, so the OUTPUT carries -1
    # padding too (the selector may commit surplus duplicate pad keys in
    # scheduling-dependent ways; all decode to -1, so the canonical row must
    # still be bit-stable).
    nshort = max(topk - 548, 1)
    scores[0, nshort:] = -float("inf")
    ids[0, nshort:] = -1.0
    gathered = torch.stack([scores, ids.to(dev)], dim=-1).contiguous()

    run = lambda gath, canon: (  # noqa: E731
        dcpk.stable_topk_from_gathered_candidates_cutedsl(
            gath, topk, canonical=canon))

    out_base = run(gathered, False)
    ref = torch.sort(out_base, dim=-1, descending=True).values
    out_canon = run(gathered, True)
    eq_ref = torch.equal(out_canon, ref)
    print(f"[{'ok  ' if eq_ref else 'FAIL'}] merge canonical == "
          f"sort_desc(base): {eq_ref}")

    # candidate-order invariance: permute candidates (same content) — output
    # must be bit-identical (upstream per-rank top-k order can't leak through)
    fails = 0
    for rep in range(reps):
        perm = torch.randperm(ncand, generator=g).to(dev)
        out_p = run(gathered[:, perm].contiguous(), True)
        fails += not torch.equal(out_p, ref)
    print(f"[{'ok  ' if fails == 0 else 'FAIL'}] merge canonical invariant "
          f"under {reps} candidate permutations: bad={fails}")

    # repeat determinism
    rep_bad = sum(0 if torch.equal(run(gathered, True), ref) else 1
                  for _ in range(reps))
    print(f"[{'ok  ' if rep_bad == 0 else 'FAIL'}] merge canonical repeat "
          f"determinism: bad={rep_bad}/{reps}")

    # ---- timing ------------------------------------------------------------
    t_base = time_cuda(lambda: run(gathered, False))
    t_canon = time_cuda(lambda: run(gathered, True))
    t_sort = time_cuda(
        lambda: torch.sort(out_base, dim=-1, descending=True).values)
    print(f"  merge timing [{rows}x{ncand}->{topk}]: base={t_base:.1f} us  "
          f"canonical={t_canon:.1f} us (delta {t_canon - t_base:+.1f})  vs "
          f"post-hoc torch.sort={t_sort:.1f} us")
    return (not eq_ref) + fails + rep_bad, t_canon - t_base, t_sort


def main():
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    failures = check_filter_modes(dev)
    f2, merge_delta, sort_cost = check_merge_kernel(dev)
    failures += f2
    # prefill-shaped merge (many rows) — cost scaling check only
    f3, merge_delta_pf, sort_cost_pf = check_merge_kernel(dev, rows=512)
    failures += f3
    print(f"SUMMARY: failures={failures}  decode merge delta="
          f"{merge_delta:+.1f} us vs torch.sort {sort_cost:.1f} us; "
          f"prefill(512-row) delta={merge_delta_pf:+.1f} us vs "
          f"torch.sort {sort_cost_pf:.1f} us")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
