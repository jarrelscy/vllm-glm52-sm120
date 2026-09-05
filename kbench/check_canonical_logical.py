#!/usr/bin/env python3
"""VLLM_DSA_CANONICAL_TOPK=logical verification.

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


def main():
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    (req_id, bt_a, bt_b, tk, block_size, dcp_size, dcp_rank,
     interleave) = build_case(dev)

    failures = 0
    for mode in ("1", "logical"):
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
        if mode == "logical":
            # logical order must be descending and layout-invariant
            desc = all(all(x > y for x, y in zip(row, row[1:]))
                       for row in log_a)
            ok = sets_equal and order_equal and pad_ok and desc \
                and rep_bad == 0
            print(f"[{'ok  ' if ok else 'FAIL'}] logical mode: set_eq="
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
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
