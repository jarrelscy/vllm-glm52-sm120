# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness proof for VLLM_GLM_DCP_AG_RAW_TOPK.

Same-process A/B on one GPU (run inside the prod container with the
deployment's cutedsl/cutlass/torch):

  docker exec <c> /opt/vllm/.venv/bin/python /tmp/round5/bitexact_ag_raw_topk.py \
      --kernel /tmp/round5/dcp_indexer_cutedsl.py

Feeds IDENTICAL candidate data through:
  baseline: 3-D concat layout [rows, ws*k, 2] (the movedim+reshape copy of
            the raw AG output), 3-D kernel;
  lever:    4-D raw rank-major layout [ws, rows, k, 2] (zero-copy view of
            the AG output), raw-layout kernel.

With canonical=True (prod: VLLM_DSA_CANONICAL_TOPK=inkernel) the output row
is fully deterministic (descending token id), so the outputs must be
BYTE-identical. With canonical=False the output order is atomics-scheduling
dependent even between two runs of the SAME kernel, so only set equality is
checked there.
"""

import argparse
import importlib.util
import sys

import torch

WS = 4
K = 2048  # index_topk / per-rank candidate count


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_candidates(rows, device, gen, n_valid_per_rank):
    """Raw rank-major candidates [WS, rows, K, 2] fp32 (score, global_id).

    Global ids are unique across ranks (interleave-1 ownership: id % WS ==
    rank), scores random with deliberate duplicate values (uniqueness of the
    stable key comes from the id). Invalid slots carry id -1, score -inf --
    same convention the pack kernel emits.
    """
    raw = torch.empty(WS, rows, K, 2, device=device, dtype=torch.float32)
    for r in range(WS):
        for row in range(rows):
            nv = int(n_valid_per_rank[r, row])
            # unique ids owned by rank r
            perm = torch.randperm(8 * K, generator=gen, device=device)[:nv]
            ids = (perm * WS + r).to(torch.float32)
            scores = torch.randn(nv, generator=gen, device=device)
            # deliberate score ties
            if nv > 16:
                scores[3] = scores[7]
                scores[10:14] = 0.25
            raw[r, row, :nv, 0] = scores
            raw[r, row, :nv, 1] = ids
            raw[r, row, nv:, 0] = float("-inf")
            raw[r, row, nv:, 1] = -1.0
    return raw


def rows_sets(t: torch.Tensor):
    return [set(row[row >= 0].tolist()) for row in t.cpu()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--topk", type=int, default=2048)
    args = ap.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(device)
    mod = load_module("round5_cutedsl", args.kernel)
    gen = torch.Generator(device=device)
    gen.manual_seed(20260906)

    for rows in (1, 2, 4, 8):
        n_valid = torch.randint(
            K // 2, K + 1, (WS, rows), generator=gen, device=device
        )
        n_valid[0, 0] = K  # at least one fully-populated row
        if rows > 1:
            # under-filled row: total valid < topk -> -1 padding in output
            n_valid[:, 1] = 100
        raw = make_candidates(rows, device, gen, n_valid)
        concat = raw.movedim(0, 1).reshape(rows, WS * K, 2).contiguous()

        # canonical (prod config): byte-identical required
        out_ref = torch.full((rows, args.topk), -7, dtype=torch.int32, device=device)
        out_raw = torch.full((rows, args.topk), -8, dtype=torch.int32, device=device)
        mod.stable_topk_from_gathered_candidates_cutedsl(
            concat, args.topk, out=out_ref, canonical=True
        )
        mod.stable_topk_from_gathered_candidates_cutedsl(
            raw, args.topk, out=out_raw, canonical=True
        )
        torch.cuda.synchronize()
        ok = torch.equal(out_ref, out_raw)
        print(f"[{'PASS' if ok else 'FAIL'}] canonical bytes rows={rows}")
        if not ok:
            diff = (out_ref != out_raw).nonzero()
            print("first diffs:", diff[:8].tolist())
            sys.exit(1)

        # non-canonical: selected SET equality (order is atomics-dependent
        # in BOTH paths; prod does not use this mode)
        out_ref2 = torch.empty_like(out_ref)
        out_raw2 = torch.empty_like(out_raw)
        mod.stable_topk_from_gathered_candidates_cutedsl(
            concat, args.topk, out=out_ref2, canonical=False
        )
        mod.stable_topk_from_gathered_candidates_cutedsl(
            raw, args.topk, out=out_raw2, canonical=False
        )
        torch.cuda.synchronize()
        ok = rows_sets(out_ref2) == rows_sets(out_raw2)
        print(f"[{'PASS' if ok else 'FAIL'}] non-canonical sets rows={rows}")
        if not ok:
            sys.exit(1)

        # canonical output must equal sorted(non-canonical set) desc: guards
        # that the raw path did not perturb the canonical order machinery
        srt = torch.sort(out_raw2, dim=-1, descending=True).values
        ok = torch.equal(srt, out_raw)
        print(f"[{'PASS' if ok else 'FAIL'}] canonical order preserved rows={rows}")
        if not ok:
            sys.exit(1)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
