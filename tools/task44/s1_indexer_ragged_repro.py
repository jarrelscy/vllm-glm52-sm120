# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S1 repro: ragged decode batch in the sparse-attn indexer padded-native branch.

Claim under test (sparse_attn_indexer.py ~line 745/808): when
decode_metadata.requires_padding is True, Q is packed per-request via
pack_seq_triton into (B, Lmax, H, D) but `weights[:num_padded_tokens]` is
passed UNPACKED token-major; fp8_fp4_paged_mqa_logits reads weight row
b*Lmax+t, correct row is starts[b]+t. Additionally the native-MTP seq_lens
formula seq_lens[b,j] = L_b - Lmax + 1 + j is wrong for requests with
decode_len < Lmax (their tokens are the LAST decode_len tokens, so the sole
token of a dl=1 request should see L_b, not L_b - Lmax + 1).

Reachability on SM120 (this box): indexer.py sets
use_flattening = not family(100) and next_n not in (1,2). Production
tp4-1m-mtp has NSDEF=3 -> next_n=4 -> flatten path -> requires_padding never
True. Native mode is available on SM120 for next_n==2 (NUM_SPEC=1) and on
SM100 for any next_n. The dispatcher also requires uniform decode lengths;
this direct ragged-helper repro does not prove live scheduler reachability.
Repro geometry: next_n=2, decode_lens=[1,2] / [2,1].

Method: run fp8_fp4_paged_mqa_logits exactly as the padded branch does
(production form), then with corrected weights packing, corrected seq_lens,
both, and compare against per-request isolated runs (uniform path = ground
truth). Single GPU.
"""

import math
import sys

import torch

torch.manual_seed(0)
DEV = "cuda:0"

H = 32  # GLM-5.3 index_n_heads
D = 128  # index_head_dim
BLOCK = 64
TOPK_EVAL = 256


def build_cache(seq_lens, max_blocks):
    """fp8 indexer cache [num_blocks, 64, 1, 132] + per-request block tables."""
    total_blocks = sum(math.ceil(s / BLOCK) for s in seq_lens) + 2
    perm = torch.randperm(total_blocks)
    cache = torch.zeros(total_blocks, BLOCK, D + 4, dtype=torch.uint8, device=DEV)
    bts, ks = [], []
    o = 0
    for s in seq_lens:
        nb = math.ceil(s / BLOCK)
        bt = torch.zeros(max_blocks, dtype=torch.int32)
        bt[:nb] = perm[o : o + nb].to(torch.int32)
        o += nb
        k = torch.randn(s, D, device=DEV) * 0.7
        from vllm import _custom_ops as ops

        t = torch.arange(s, device=DEV)
        slots = bt.to(DEV)[t // BLOCK].long() * BLOCK + t % BLOCK
        ops.indexer_k_quant_and_cache(k.to(torch.bfloat16), cache, slots, 128, "fp32")
        bts.append(bt)
        ks.append(k)
    return cache.unsqueeze(-2), torch.stack(bts).to(DEV), ks


def mqa_logits(q, cache, w, ctx, bt):
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_paged_mqa_logits_metadata,
    )

    num_sms = torch.cuda.get_device_properties(DEV).multi_processor_count
    sched = get_paged_mqa_logits_metadata(ctx, BLOCK, num_sms)
    return fp8_fp4_paged_mqa_logits(
        (q, None),
        cache,
        w,
        ctx,
        bt,
        sched,
        max_model_len=4096,
        clean_logits=False,
    )


def topk_set(logit_row, n_ctx, k=TOPK_EVAL):
    v = logit_row[:n_ctx]
    return set(v.topk(min(k, n_ctx)).indices.tolist())


def run(decode_lens, L):
    from vllm.v1.attention.ops.common import pack_seq_triton

    B = len(decode_lens)
    Lmax = max(decode_lens)  # == next_n in the native branch
    nd_tokens = sum(decode_lens)
    n_pad = B * Lmax
    max_blocks = max(math.ceil(s / BLOCK) for s in L)

    cache, bt, _ = build_cache(L, max_blocks)

    # token-major decode Q/weights exactly as the indexer buffers hold them,
    # plus trailing buffer rows (prefill tokens' weights in a mixed batch).
    # Weights are DISTINCTIVE per row (one dominant head, different head per
    # row): a row-misaligned read flips the dominant head -> logits change
    # completely. iid-random weights would be statistically inconclusive
    # (positive sums over 32 heads correlate heavily).
    q_tok = (torch.randn(nd_tokens, H, D, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    w_all = torch.full((nd_tokens + 8, H), 0.01, device=DEV)
    for i in range(nd_tokens + 8):
        w_all[i, (5 * i + 3) % H] = 3.0

    dl = torch.tensor(decode_lens, dtype=torch.int32, device=DEV)
    q_packed = pack_seq_triton(q_tok, dl)  # [B, Lmax, H, D]
    assert q_packed.shape == (B, Lmax, H, D)

    # production seq_lens (indexer.py:570-585): L_b - Lmax + 1 + j
    seq_prod = torch.tensor(
        [[length - Lmax + 1 + j for j in range(Lmax)] for length in L],
        dtype=torch.int32,
        device=DEV,
    )
    # corrected seq_lens: token j of request b (packed at slot j, j<dl_b) is
    # the request's (dl_b - 1 - j)-th token from the end -> ctx = L_b-dl_b+1+j
    seq_fix = torch.tensor(
        [
            [
                L[b] - decode_lens[b] + 1 + min(j, decode_lens[b] - 1)
                for j in range(Lmax)
            ]
            for b in range(B)
        ],
        dtype=torch.int32,
        device=DEV,
    )

    w_prod = w_all[:n_pad].contiguous()  # production form
    w_fix = pack_seq_triton(w_all[:nd_tokens], dl, pad_value=0).reshape(n_pad, H)

    arms = {
        "prod (raw weights, prod seq_lens)": (w_prod, seq_prod),
        "fixW (packed weights, prod seq_lens)": (w_fix, seq_prod),
        "fixS (raw weights, fixed seq_lens)": (w_prod, seq_fix),
        "fixWS (packed weights + fixed seq_lens)": (w_fix, seq_fix),
    }
    logits = {
        name: mqa_logits(q_packed.view(B, Lmax, H, D), cache, w, s, bt)
        for name, (w, s) in arms.items()
    }

    # ground truth: each request isolated (uniform decode, no padding branch)
    iso = []
    starts = [0]
    for d in decode_lens:
        starts.append(starts[-1] + d)
    for b in range(B):
        dlb = decode_lens[b]
        qb = q_tok[starts[b] : starts[b + 1]].view(1, dlb, H, D)
        wb = w_all[starts[b] : starts[b + 1]].contiguous()
        sb = torch.tensor(
            [[L[b] - dlb + 1 + j for j in range(dlb)]], dtype=torch.int32, device=DEV
        )
        iso.append(mqa_logits(qb, cache, wb, sb, bt[b : b + 1]))

    print(f"\n### decode_lens={decode_lens} L={L} next_n={Lmax} num_padded={n_pad}")
    ok_all = {}
    for name, (w_arm, seq_arm) in arms.items():
        lg = logits[name]
        worst_rel, worst_jac, rows = 0.0, 1.0, []
        trunc = []
        for b in range(B):
            for t in range(decode_lens[b]):
                row = b * Lmax + t
                ctx_true = L[b] - decode_lens[b] + 1 + t
                ctx_seen = int(seq_arm[b, t].item())  # what the kernel used
                if ctx_seen != ctx_true:
                    trunc.append(
                        f"b{b}t{t}:ctx {ctx_seen} != {ctx_true}"
                        f" (last {ctx_true - ctx_seen} KV incl. self"
                        " invisible)"
                    )
                # weights-effect isolation: compare on the ctx both computed
                ctx = min(ctx_seen, ctx_true)
                got = lg[row, :ctx].float()
                ref = iso[b][t, :ctx].float()
                assert torch.isfinite(got).all() and torch.isfinite(ref).all()
                bad = ~(torch.isfinite(got) & torch.isfinite(ref))
                got, ref = got.clone(), ref.clone()
                got[bad] = 0
                ref[bad] = 0
                rel = ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
                js, rs = topk_set(got, ctx), topk_set(ref, ctx)
                jac = len(js & rs) / len(js | rs)
                worst_rel = max(rel, worst_rel)
                worst_jac = min(jac, worst_jac)
                rows.append(f"b{b}t{t}:rel={rel:.1e},jacc={jac:.3f}")
        ok = worst_rel < 2e-2 and worst_jac > 0.99 and not trunc
        ok_all[name] = ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name:45s} "
            f"weights-effect max_rel={worst_rel:.3e} "
            f"min_topk_jaccard={worst_jac:.3f} ctx_trunc={len(trunc)}"
        )
        if not ok:
            print("        " + "  ".join(rows))
            for s in trunc:
                print("        TRUNC " + s)
    return ok_all


def main():
    # reachability check for the LIVE config (next_n=4, SM120)
    from vllm.platforms import current_platform

    fam100 = current_platform.is_device_capability_family(100)
    for nn in (2, 3, 4):
        flat = (not fam100) and nn not in (1, 2)
        print(
            f"[reachability] SM120 next_n={nn}: use_flattening={flat} -> "
            f"padded native branch {'DEAD' if flat else 'native mode eligible'}"
        )

    r1 = run([1, 2], [500, 700])
    r2 = run([2, 1], [700, 500])
    r3 = run([2, 2], [512, 640])  # uniform: padding branch not taken in prod,
    # but packed==raw here -> sanity: all arms PASS

    print("\n==== SUMMARY ====")
    print("uniform sanity (all arms should PASS):", all(r3.values()))
    verdict_bug = (
        not r1["prod (raw weights, prod seq_lens)"]
        and r1["fixWS (packed weights + fixed seq_lens)"]
    )
    print("S1 confirmed (prod form FAIL, fixWS PASS):", verdict_bug)
    assert verdict_bug
    assert r2["fixWS (packed weights + fixed seq_lens)"]
    sys.exit(0 if r3 and all(r3.values()) else 1)


if __name__ == "__main__":
    main()
