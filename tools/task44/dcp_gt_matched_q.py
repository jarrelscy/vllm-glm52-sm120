# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical ground-truth harness for the GLM-5.3 SM120 sparse-MLA DCP4 decode path.

torchrun --standalone --nproc-per-node 4 dcp_gt_harness.py [--quick]

Each scenario/flag cell is compared against a plain-torch FP32 reference
using the actual fp8_ds_mla cache bytes and kernel-quantized FP8 Q:

  stage F  triton_filter_and_convert_dcp_index  (per-rank shard filter)
  stage K  flashinfer trtllm sparse-MLA SM120 kernel out + LSE (per rank)
  stage C  cp_lse_ag_out_rs combine (ag_rs; STAGED/VIEW copyfree vs baseline)
  stage E  end-to-end final output vs full-KV fp32 ground truth

Production geometry (read from tree + model config, not guessed):
  64 q heads total (TP4 -> DCP all-gathers query to all 64 heads/rank),
  kv_lora_rank 512, qk_nope 192, qk_rope 64 (MQA head dim 576),
  topk 2048, kernel block 64, interleave 1, fp8_ds_mla entry 656 B
  (512 fp8 latent + 4 fp32 per-128-tile scales + 64 bf16 rope),
  kv_scale_format arbitrary_fp32, LSE base-2 (lse_base_on_e=False).
"""

import argparse
import json
import math
import os
from types import SimpleNamespace

# ---- env that must be set before vllm imports (module-level reads) ----
os.environ.setdefault("VLLM_DSA_CANONICAL_TOPK", "inkernel")  # production
# production NCCL knobs (comm behavior parity)
os.environ.setdefault("NCCL_ALGO", "RING,TREE")
os.environ.setdefault("NCCL_BUFFSIZE", "1048576")
os.environ.setdefault("NCCL_MAX_NCHANNELS", "4")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
# indexer-side experimental flags: set for parity; inert in this isolation
os.environ.setdefault("VLLM_EXPERIMENT_DCP_BYTEPACK", "1")
os.environ.setdefault("VLLM_EXPERIMENT_DCP_BYTEPACK_OWNER", "1")
os.environ.setdefault("VLLM_GLM_DCP_AG_RAW_TOPK", "1")
os.environ.setdefault("VLLM_DCP_A2A_EXACT", "0")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

LN2 = math.log(2.0)

H_TOTAL = 64  # query heads after DCP all-gather
H_PER_RANK = 16  # TP4 shard (combine reduce-scatters back to this)
KV_LORA = 512
ROPE = 64
QK_DIM = KV_LORA + ROPE  # 576 (absorbed MQA)
TOPK = 2048
BLOCK = 64
ENTRY = 656
WORLD = 4
INTERLEAVE = 1
SCALE = 1.0 / math.sqrt(192 + 64)  # bmm1 softmax scale (qk head dim 256)

DT = torch.bfloat16


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


# --------------------------------------------------------------------------
# scenario synthesis (CPU, seed-deterministic, identical on all ranks)
# --------------------------------------------------------------------------


def make_scenario(name, reqs, seed):
    """reqs: list of (ctx_len, q_len). Returns dict with all host-side data."""
    g = torch.Generator().manual_seed(seed)
    B = len(reqs)
    q_lens = [q for _, q in reqs]
    s_lens = [c + q for c, q in reqs]
    num_tokens = sum(q_lens)

    # per-request full KV with wildly varying per-token/per-tile magnitudes
    kv_c, k_pe = [], []
    for i, s in enumerate(s_lens):
        base = torch.randn(s, KV_LORA, generator=g)
        # log-uniform magnitudes ~1/8 .. 8 varying per (token, 128-tile):
        # 64x per-tile scale spread (a pow2/e8m0 scale misread => up to 2x
        # dequant error => out rel ~O(1), far above the 0.2 fail band) while
        # keeping attention logits realistic (post-norm latents are O(1);
        # huge spreads make softmax an argmax lottery where bf16-vs-fp32
        # legitimately flips winners).
        mag = torch.exp(
            (torch.rand(s, 4, generator=g) * 2 - 1) * math.log(8.0)
        ).repeat_interleave(128, dim=1)
        kv_c.append((base * 0.7 * mag).to(DT))
        k_pe.append((torch.randn(s, ROPE, generator=g) * 0.7).to(DT))

    # queries: post-absorb, post-all-gather form [T, 64, 576]
    q = (torch.randn(num_tokens, H_TOTAL, QK_DIM, generator=g) * 0.3).to(DT)

    # top-k rows, cycling edge-case patterns
    req_id_per_token = torch.empty(num_tokens, dtype=torch.int32)
    positions = torch.empty(num_tokens, dtype=torch.int64)
    topk = torch.full((num_tokens, TOPK), -1, dtype=torch.int32)
    patterns = []
    tok = 0
    for i, (c, ql) in enumerate(reqs):
        for j in range(ql):
            pos = c + j
            req_id_per_token[tok] = i
            positions[tok] = pos
            pat = tok % 6
            avail = torch.arange(pos + 1)
            if pat == 0:  # uniform random
                sel = avail[torch.randperm(pos + 1, generator=g)[:TOPK]]
                pname = "uniform"
            elif pat == 1:  # clustered: all on rank (tok % WORLD)
                r = tok % WORLD
                cand = avail[avail % WORLD == r]
                sel = cand[torch.randperm(len(cand), generator=g)[:TOPK]]
                if len(sel) == 0:
                    sel = avail[-1:]
                pname = f"all_on_rank{r}"
            elif pat == 2:  # one rank owns nothing
                r = (tok + 1) % WORLD
                cand = avail[avail % WORLD != r]
                sel = cand[torch.randperm(len(cand), generator=g)[:TOPK]]
                if len(sel) == 0:
                    sel = avail[-1:]
                pname = f"empty_on_rank{r}"
            elif pat == 3:  # block-edge / interleave-boundary indices
                loc = torch.arange((pos + 1) // WORLD + 1)
                edges = loc[(loc % BLOCK <= 1) | (loc % BLOCK == BLOCK - 1)]
                cand = []
                for r in range(WORLD):
                    t = edges * WORLD + r
                    cand.append(t[t <= pos])
                sel = torch.cat(cand + [avail[:4], avail[-4:]]).unique()
                sel = sel[torch.randperm(len(sel), generator=g)[:TOPK]]
                pname = "block_edges"
            elif pat == 4:  # duplicates + boundary
                k = min(64, pos + 1)
                b = avail[torch.randperm(pos + 1, generator=g)[:k]]
                sel = torch.cat([b, b[: k // 2], b[:8], avail[-1:]])[:TOPK]
                pname = "duplicates"
            else:  # singleton (most recent token only)
                sel = avail[-1:]
                pname = "singleton"
            sel = sel.sort(descending=True).values  # canonical (inkernel) order
            topk[tok, : len(sel)] = sel.to(torch.int32)
            patterns.append(pname)
            tok += 1

    # per-rank paged caches: block tables with shuffled physical blocks,
    # DIFFERENT layout per rank (catches cross-rank table mixups)
    blocks_per_req = [
        [math.ceil(max(1, math.ceil((s - r) / WORLD)) / BLOCK) for s in s_lens]
        for r in range(WORLD)
    ]
    max_blocks = max(max(b) for b in blocks_per_req)
    block_tables, num_blocks = [], []
    for r in range(WORLD):
        total = sum(blocks_per_req[r]) + 8
        perm = torch.randperm(total, generator=g)[: sum(blocks_per_req[r])]
        bt = torch.zeros(B, max_blocks, dtype=torch.int32)
        o = 0
        for i in range(B):
            n = blocks_per_req[r][i]
            bt[i, :n] = perm[o : o + n].to(torch.int32)
            o += n
        block_tables.append(bt)
        num_blocks.append(total)

    return dict(
        name=name,
        reqs=reqs,
        B=B,
        num_tokens=num_tokens,
        s_lens=s_lens,
        kv_c=kv_c,
        k_pe=k_pe,
        q=q,
        req_id_per_token=req_id_per_token,
        positions=positions,
        topk=topk,
        patterns=patterns,
        block_tables=block_tables,
        num_blocks=num_blocks,
        max_blocks=max_blocks,
    )


# --------------------------------------------------------------------------
# cache build + plain-torch dequant
# --------------------------------------------------------------------------


def build_full_cache(sc, device):
    """Non-DCP control: one cache holding ALL tokens (local_idx == t)."""
    from vllm import _custom_ops as ops

    g = torch.Generator().manual_seed(777)
    nblocks = sum(math.ceil(s / BLOCK) for s in sc["s_lens"]) + 4
    perm = torch.randperm(nblocks, generator=g)
    bt = torch.zeros(
        sc["B"], max(math.ceil(s / BLOCK) for s in sc["s_lens"]), dtype=torch.int32
    )
    cache = torch.zeros(nblocks, BLOCK, ENTRY, dtype=torch.uint8, device=device)
    o = 0
    for i, s in enumerate(sc["s_lens"]):
        nb = math.ceil(s / BLOCK)
        bt[i, :nb] = perm[o : o + nb].to(torch.int32)
        o += nb
        t = torch.arange(s)
        slots = (bt[i, t // BLOCK].long() * BLOCK + t % BLOCK).to(device)
        ops.concat_and_cache_mla(
            sc["kv_c"][i].to(device),
            sc["k_pe"][i].to(device),
            cache,
            slots,
            kv_cache_dtype="fp8_ds_mla",
            scale=torch.tensor(1.0, dtype=torch.float32, device=device),
        )
    return cache, bt.to(device)


def run_nodcp_control(sc, impl, q_dev, device, ref_out, ref_lse2):
    """Same rows through the same kernel WITHOUT DCP (full cache, global
    index conversion). Calibrates the kernel-intrinsic noise floor."""
    cache, bt = build_full_cache(sc, device)
    meta = SimpleNamespace(
        req_id_per_token=sc["req_id_per_token"].to(device),
        block_table=bt,
        block_size=BLOCK,
        topk_tokens=TOPK,
        cp_kv_cache_interleave_size=INTERLEAVE,
    )
    save = (impl.dcp_world_size, impl.dcp_rank)
    impl.dcp_world_size, impl.dcp_rank = 1, 0
    try:
        with torch.inference_mode():
            out, lse = impl.forward_mqa(q_dev, cache, meta, None)
    finally:
        impl.dcp_world_size, impl.dcp_rank = save
    kres = summarize_diff(out, ref_out, "nodcp_kernel", 0.20)
    dl = (lse.float() - ref_lse2).abs()
    del cache
    return kres, dl.max().item()


def build_rank_cache(sc, r, device):
    """Quantize rank r's owned tokens into its paged fp8_ds_mla cache."""
    from vllm import _custom_ops as ops

    cache = torch.zeros(
        sc["num_blocks"][r], BLOCK, ENTRY, dtype=torch.uint8, device=device
    )
    for i, s in enumerate(sc["s_lens"]):
        t = torch.arange(s)
        own = t[t % WORLD == r]
        if len(own) == 0:
            continue
        loc = own // WORLD
        slots = (
            sc["block_tables"][r][i, loc // BLOCK].long() * BLOCK + loc % BLOCK
        ).to(device)
        ops.concat_and_cache_mla(
            sc["kv_c"][i][own].to(device),
            sc["k_pe"][i][own].to(device),
            cache,
            slots,
            kv_cache_dtype="fp8_ds_mla",
            scale=torch.tensor(1.0, dtype=torch.float32, device=device),
        )
    return cache


def dequant_cache(cache):
    """Plain torch dequant of the whole fp8_ds_mla cache -> (K[nb*64,576] fp32, V)."""
    lat8 = cache[..., :KV_LORA].view(torch.float8_e4m3fn).float()
    scales = (
        cache[..., KV_LORA : KV_LORA + 16]
        .contiguous()
        .view(torch.float32)
        .reshape(*cache.shape[:2], 4)
    )
    latent = lat8 * scales.repeat_interleave(128, dim=-1)
    rope = (
        cache[..., KV_LORA + 16 :]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(*cache.shape[:2], ROPE)
        .float()
    )
    K = torch.cat([latent, rope], dim=-1).reshape(-1, QK_DIM)
    return K, latent.reshape(-1, KV_LORA)


def slot_of(sc, req, tok_ids, r):
    loc = tok_ids // WORLD
    bt = sc["block_tables"][r]
    return bt[req, (loc // BLOCK).cpu()].long() * BLOCK + (loc % BLOCK).cpu()


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------


def reference(sc, Ks, Vs, q, device, permute=False):
    """Full fp32 reference from dequantized cache bytes.

    Returns: out [T,64,512] fp32, lse2 [T,64] fp32 (base-2),
             per-rank partial outs [4,T,64,512], partial lse2 [4,T,64].
    """
    T = sc["num_tokens"]
    out = torch.zeros(T, H_TOTAL, KV_LORA, device=device)
    lse2 = torch.full((T, H_TOTAL), float("-inf"), device=device)
    pout = torch.zeros(WORLD, T, H_TOTAL, KV_LORA, device=device)
    plse2 = torch.full((WORLD, T, H_TOTAL), float("-inf"), device=device)
    qf = q.float().clone()
    # Match the actual kernel's Q FP8 quantization, not just its KV bytes.
    tiles = qf[..., :KV_LORA].reshape(T, H_TOTAL, 4, 128)
    scale = torch.exp2(
        torch.ceil(
            torch.log2(tiles.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)
        )
    )
    qf[..., :KV_LORA] = (
        (tiles / scale).to(torch.float8_e4m3fn).float() * scale
    ).reshape(T, H_TOTAL, KV_LORA)
    for tok in range(T):
        req = int(sc["req_id_per_token"][tok])
        row = sc["topk"][tok]
        idx = row[row >= 0].long()
        if permute:
            gp = torch.Generator().manual_seed(999 + tok)
            idx = idx[torch.randperm(len(idx), generator=gp)]
        if len(idx) == 0:
            continue
        owner = (idx % WORLD).to(device)
        K = torch.empty(len(idx), QK_DIM, device=device)
        V = torch.empty(len(idx), KV_LORA, device=device)
        for r in range(WORLD):
            m = owner == r
            if m.any():
                sl = slot_of(sc, req, idx[m.cpu()], r).to(device)
                K[m] = Ks[r][sl]
                V[m] = Vs[r][sl]
        logits = (qf[tok] @ K.T) * SCALE  # [64, n]
        for r in [None] + list(range(WORLD)):
            if r is None:
                lg, vv = logits, V
            else:
                m = owner == r
                if not m.any():
                    continue
                lg, vv = logits[:, m], V[m]
            mx = lg.max(dim=-1).values
            p = torch.exp(lg - mx[:, None])
            s = p.sum(dim=-1)
            o = (p @ vv) / s[:, None]
            l2 = (mx + torch.log(s)) / LN2
            if r is None:
                out[tok], lse2[tok] = o, l2
            else:
                pout[r, tok], plse2[r, tok] = o, l2
    return out, lse2, pout, plse2


def ref_combine2(pouts, plses):
    """fp32 LSE-weighted combine of per-rank kernel outputs, base-2 LSE.

    pouts [4,T,H,D] fp32, plses [4,T,H] fp32 (may contain -inf / -1e30 / nan).
    Mirrors _correct_attn_cp_out_kernel math (incl. nan/inf -> -inf, all-empty
    row -> 0). Returns combined [T,H,D], global lse [T,H].
    """
    log_sums = plses.clone()
    log_sums[(log_sums != log_sums) | (log_sums == float("inf"))] = float("-inf")
    mx = log_sums.max(dim=0).values
    mx_safe = torch.where(mx == float("-inf"), torch.zeros_like(mx), mx)
    w = torch.exp2(log_sums - mx_safe[None])
    tot = w.sum(dim=0)
    glse = torch.log2(tot) + mx_safe
    factor = torch.exp2(log_sums - glse[None])
    factor[(factor != factor) | (factor == float("inf"))] = 0.0
    # mirror the kernel's `where(factor == 0, 0, out * factor)`: a rank with
    # zero weight contributes exact 0 even if its payload is garbage/NaN
    contrib = torch.where(
        factor[..., None] == 0.0, torch.zeros_like(pouts), pouts * factor[..., None]
    )
    comb = contrib.sum(dim=0)
    return comb, glse


# --------------------------------------------------------------------------
# optimized path (production code)
# --------------------------------------------------------------------------


def make_impl(topk_buffer, skip_empty_fill):
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        FlashInferMLASparseSM120Impl,
    )

    os.environ["VLLM_GLM_SKIP_EMPTY_FILL"] = "1" if skip_empty_fill else "0"
    if True:
        impl = FlashInferMLASparseSM120Impl(
            num_heads=H_PER_RANK,
            head_size=QK_DIM,
            scale=SCALE,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            indexer=SimpleNamespace(topk_indices_buffer=topk_buffer),
            kv_lora_rank=KV_LORA,
            qk_nope_head_dim=192,
            qk_rope_head_dim=ROPE,
        )
    assert impl.dcp_world_size == WORLD, impl.dcp_world_size
    assert impl.kv_scale_format == "arbitrary_fp32"
    assert impl.need_to_return_lse_for_decode
    impl._skip_empty_fill = bool(skip_empty_fill)
    return impl


def run_filter_check(sc, meta, rank, device):
    """Compare filter against torch; return (ok, msg, canonical_order_ok)."""
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index,
    )

    topk_dev = sc["topk"].to(device)
    got, counts = triton_filter_and_convert_dcp_index(
        meta.req_id_per_token,
        meta.block_table,
        topk_dev,
        dcp_size=WORLD,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=INTERLEAVE,
        BLOCK_SIZE=BLOCK,
        NUM_TOPK_TOKENS=TOPK,
        return_valid_counts=True,
    )
    tok = sc["topk"].long()
    valid = tok >= 0
    own = valid & ((tok % WORLD) == rank)
    loc = torch.where(own, tok // WORLD, torch.zeros_like(tok))
    inb = (loc // BLOCK) < sc["max_blocks"]
    own &= inb
    bt = sc["block_tables"][rank]
    req = sc["req_id_per_token"].long()
    slots = bt[req[:, None].expand_as(tok), loc // BLOCK].long() * BLOCK + loc % BLOCK
    ref_counts = own.sum(-1).to(torch.int32)
    if not torch.equal(counts.cpu(), ref_counts):
        return (
            False,
            f"valid_counts mismatch: got {counts.cpu().tolist()} "
            f"ref {ref_counts.tolist()}",
            False,
        )
    gc = got.cpu()
    order_ok = True
    for t in range(sc["num_tokens"]):
        n = int(ref_counts[t])
        pref = gc[t, :n].long()
        refsl = slots[t][own[t]]
        if not torch.equal(pref.sort().values, refsl.sort().values):
            return False, f"row {t}: slot set mismatch", False
        if (gc[t, n:] != -1).any():
            return False, f"row {t}: tail not -1", False
        if not torch.equal(pref, refsl):  # inkernel mode: order preserved
            order_ok = False
    return True, "", order_ok


def run_kernel(impl, q_dev, cache, meta):
    out, lse = impl.forward_mqa(q_dev, cache, meta, None)
    return out, lse


def summarize_diff(got, ref, tag, tol_rel, tol_abs_floor=1e-4):
    """Row-normalized comparison. Returns dict + fail bool."""
    g, r = got.float(), ref.float()
    d = (g - r).abs()
    denom = r.abs().amax(dim=-1, keepdim=True).clamp_min(tol_abs_floor)
    rel = d / denom
    bad = rel > tol_rel
    nan = (~torch.isfinite(g)).sum().item()
    # zero-where-nonzero structural check
    zwn = ((g.abs().amax(dim=-1) == 0) & (r.abs().amax(dim=-1) > 1e-2)).sum().item()
    res = dict(
        tag=tag,
        max_abs=d.max().item(),
        max_rel=rel.max().item(),
        mean_rel=rel.mean().item(),
        frac_rows_over_tol=(
            bad.any(dim=-1).float().mean().item()
            if bad.ndim > 1
            else bad.float().mean().item()
        ),
        nonfinite=nan,
        zero_where_nonzero=zwn,
    )
    res["fail"] = nan > 0 or zwn > 0 or res["frac_rows_over_tol"] > 0.001
    return res


def worst_rows(got, ref, sc, k=5):
    d = (got.float() - ref.float()).abs().amax(dim=-1)  # [T,H]
    dr = d / ref.float().abs().amax(dim=-1).clamp_min(1e-4)
    v, flat = dr.flatten().topk(min(k, dr.numel()))
    rows = []
    for val, f in zip(v.tolist(), flat.tolist()):
        t, h = f // dr.shape[1], f % dr.shape[1]
        rows.append(
            dict(
                tok=t,
                head=h,
                rel=val,
                req=int(sc["req_id_per_token"][t]),
                pattern=sc["patterns"][t],
                owner_rank_of_head_shard=h // H_PER_RANK,
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="/work/results.json")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local)
    device = f"cuda:{local}"

    # a real (empty) VllmConfig with a GLM model_config, held for the whole
    # run: custom ops / communicators / the SM120 impl all read it.
    from vllm.config import VllmConfig, set_current_vllm_config

    vcfg = VllmConfig()
    vcfg.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa", index_topk=TOPK),
        is_moe=True,
    )  # GLM-5.3 is MoE; read by initialize_model_parallel
    import contextlib

    stack = contextlib.ExitStack()
    stack.enter_context(set_current_vllm_config(vcfg))

    from vllm.distributed.parallel_state import (
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=WORLD,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=WORLD, decode_context_model_parallel_size=WORLD
    )
    dcp = get_dcp_group()
    assert dcp.world_size == WORLD and dcp.rank_in_group == rank
    pynccl = getattr(dcp.device_communicator, "pynccl_comm", None)
    log(
        rank,
        "[init] dcp group up, pynccl="
        + ("OK" if pynccl and not pynccl.disabled else "MISSING"),
    )

    small = [
        ("B1_q1_small", [(1500, 1)], 101),
        ("B1_q4_small", [(1500, 4)], 102),
        ("B4_q4_mixed", [(300, 4), (1500, 4), (70, 4), (4000, 4)], 103),
        (
            "B7_mixed",
            [(65, 1), (900, 4), (2500, 4), (140, 1), (5800, 4), (330, 1), (1200, 4)],
            104,
        ),
        (
            "B8_q4_mixed",
            [
                (256, 4),
                (511, 4),
                (513, 4),
                (1024, 4),
                (2048, 4),
                (63, 4),
                (129, 4),
                (3000, 4),
            ],
            105,
        ),
    ]
    big = [
        ("B1_q1_128k", [(131072, 1)], 201),
        ("B4_128k_mixed", [(131072, 4), (65536, 4), (98304, 1), (8192, 4)], 202),
        (
            "B7_128k_mixed",
            [
                (131072, 4),
                (8192, 1),
                (32768, 4),
                (131071, 1),
                (65536, 4),
                (16384, 1),
                (98304, 4),
            ],
            203,
        ),
    ]
    scenarios = small if args.quick else small + big

    combine_arms = [
        (
            "prod_staged_view",
            dict(VLLM_GLM_DCP_RS_STAGED="1", VLLM_GLM_DCP_RS_VIEW="1"),
        ),
        ("baseline_off", dict(VLLM_GLM_DCP_RS_STAGED="0", VLLM_GLM_DCP_RS_VIEW="0")),
        ("staged_only", dict(VLLM_GLM_DCP_RS_STAGED="1", VLLM_GLM_DCP_RS_VIEW="0")),
        ("view_only", dict(VLLM_GLM_DCP_RS_STAGED="0", VLLM_GLM_DCP_RS_VIEW="1")),
    ]

    results = []
    topk_buffer = torch.full((4096, TOPK), -1, dtype=torch.int32, device=device)

    for name, reqs, seed in scenarios:
        sc = make_scenario(name, reqs, seed)
        T = sc["num_tokens"]
        log(rank, f"\n=== scenario {name}: B={sc['B']} T={T} s_lens={sc['s_lens']}")

        caches = [build_rank_cache(sc, r, device) for r in range(WORLD)]
        # cross-rank cache determinism check
        csum = torch.stack([c.view(torch.int64).sum() for c in caches]).to(device)
        sums = [torch.zeros_like(csum) for _ in range(WORLD)]
        dist.all_gather(sums, csum)
        assert all(torch.equal(s, csum) for s in sums), (
            "per-rank cache bytes differ across GPUs (quant nondeterminism)"
        )

        Ks, Vs = zip(*[dequant_cache(c) for c in caches])
        q_dev = sc["q"].to(device)

        ref_out, ref_lse2, ref_pout, ref_plse2 = reference(sc, Ks, Vs, q_dev, device)
        # reference self-consistency: permuted summation order
        ref_out_p, _, _, _ = reference(sc, Ks, Vs, q_dev, device, permute=True)
        sc_res = summarize_diff(ref_out_p, ref_out, "ref_selfcheck", 2**-16)
        log(
            rank,
            f"  [ref-selfcheck] max_rel={sc_res['max_rel']:.2e} (permuted summation)",
        )
        assert sc_res["max_rel"] < 1e-4, "reference not self-consistent"

        meta = SimpleNamespace(
            req_id_per_token=sc["req_id_per_token"].to(device),
            block_table=sc["block_tables"][rank].to(device),
            block_size=BLOCK,
            topk_tokens=TOPK,
            cp_kv_cache_interleave_size=INTERLEAVE,
        )
        topk_buffer[:T].copy_(sc["topk"].to(device))
        topk_buffer[T:].fill_(-1)

        # ---- stage F
        fok, fmsg, order_ok = run_filter_check(sc, meta, rank, device)
        f_flag = torch.tensor([int(fok), int(order_ok)], device=device)
        dist.all_reduce(f_flag, op=dist.ReduceOp.MIN)
        log(
            rank,
            f"  [F filter] {'PASS' if f_flag[0] else 'FAIL ' + fmsg} "
            f"(canonical order preserved: {bool(f_flag[1])})",
        )

        # which ranks are empty per row (host truth)
        tokm = sc["topk"].long()
        own_any = [(tokm >= 0) & ((tokm % WORLD) == r) for r in range(WORLD)]
        empty_for = torch.stack([~o.any(-1) for o in own_any])  # [4,T]

        ctl_k, ctl_lse = None, None
        for skip_fill in (True, False):
            impl = make_impl(topk_buffer, skip_fill)
            if ctl_k is None:
                # no-DCP control: kernel-intrinsic noise floor on these rows
                ctl_k, ctl_lse = run_nodcp_control(
                    sc, impl, q_dev, device, ref_out, ref_lse2
                )
                cstat = torch.tensor([ctl_k["max_rel"], ctl_lse], device=device)
                dist.all_reduce(cstat, op=dist.ReduceOp.MAX)
                ctl_k["max_rel"], ctl_lse = cstat[0].item(), cstat[1].item()
                log(
                    rank,
                    f"  [no-DCP control] kernel max_rel="
                    f"{ctl_k['max_rel']:.3e} lse={ctl_lse:.2e} "
                    f"(intrinsic kernel noise floor)",
                )
            with torch.inference_mode():
                out_r, lse_r = run_kernel(impl, q_dev, caches[rank], meta)
                # VLLM_GLM_COMM_OVERLAP path: precompute_mqa_indices ->
                # forward_mqa(precomputed_indices=...) must be BIT-exact
                # vs the serial path (production runs with overlap ON).
                pre = impl.precompute_mqa_indices(meta, T)
                out_p, lse_p = impl.forward_mqa(
                    q_dev, caches[rank], meta, None, precomputed_indices=pre
                )
                overlap_ok = torch.equal(out_p, out_r) and torch.equal(lse_p, lse_r)
                ov = torch.tensor([float(overlap_ok)], device=device)
                dist.all_reduce(ov, op=dist.ReduceOp.MIN)
                if not bool(ov.item()):
                    log(
                        rank,
                        "  [overlap] BIT-EXACTNESS FAIL "
                        "(precompute_mqa_indices path != serial)",
                    )
                del out_p, lse_p
            # ---- stage K: per-rank kernel vs reference partial.
            # Noise floor measured empirically (single-row probe, exact-dequant
            # fp32 ref vs kernel): ~8e-2 max-rel — the kernel keeps K in fp8
            # in-kernel. Structural bugs (scale misread, wrong rows) give >=0.5.
            kres = summarize_diff(out_r, ref_pout[rank], "kernel_out", 0.20)
            myempty = empty_for[rank].to(device)
            lse_valid = ~myempty
            lres = dict(tag="kernel_lse")
            if lse_valid.any():
                dl = (lse_r.float()[lse_valid] - ref_plse2[rank][lse_valid]).abs()
                lres.update(
                    max_abs=dl.max().item(),
                    mean_abs=dl.mean().item(),
                    fail=dl.max().item() > 0.005,
                )
            else:
                lres.update(max_abs=0.0, mean_abs=0.0, fail=False)
            valid_rows = torch.where(lse_valid)[0]
            details = []
            if len(valid_rows):
                row_errors = (
                    (lse_r.float()[lse_valid] - ref_plse2[rank][lse_valid])
                    .abs()
                    .amax(-1)
                )
                for j in row_errors.argsort(descending=True)[:8].tolist():
                    t = int(valid_rows[j])
                    details.append(
                        dict(
                            row=t,
                            pattern=sc["patterns"][t],
                            position=int(sc["positions"][t]),
                            max_abs=float(row_errors[j]),
                        )
                    )
            all_lse_details = [None] * WORLD
            dist.all_gather_object(
                all_lse_details, dict(rank=rank, stats=lres.copy(), worst=details)
            )
            # empty-row sentinel behavior
            sentinel = dict(tag="empty_rows", n_empty=int(myempty.sum()))
            if myempty.any():
                eo = out_r[myempty].float().abs().max().item()
                el = lse_r[myempty].float()
                sentinel.update(
                    out_abs_max=eo,
                    lse_min=el.min().item(),
                    lse_max=el.max().item(),
                    out_zero=eo == 0.0,
                    lse_all_neg_huge=bool((el <= -1e29).all()),
                )
                sentinel["fail"] = not (
                    sentinel["out_zero"] and sentinel["lse_all_neg_huge"]
                )
            else:
                sentinel["fail"] = False

            # gather kernel outs/lses from all ranks (diagnostic only)
            outs_g = [torch.empty_like(out_r) for _ in range(WORLD)]
            lses_g = [torch.empty_like(lse_r) for _ in range(WORLD)]
            dist.all_gather(outs_g, out_r.contiguous())
            dist.all_gather(lses_g, lse_r.contiguous())
            outs_g = torch.stack(outs_g).float()
            lses_g = torch.stack(lses_g).float()
            comb_of_kernel, _ = ref_combine2(outs_g, lses_g)

            for arm_name, arm_env in combine_arms:
                for k, v in arm_env.items():
                    os.environ[k] = v
                import vllm.envs as envs

                assert (
                    arm_env["VLLM_GLM_DCP_RS_STAGED"] == "1"
                ) == envs.VLLM_GLM_DCP_RS_STAGED
                final = None
                from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

                with torch.inference_mode():
                    final = cp_lse_ag_out_rs(
                        out_r.clone(),
                        lse_r.clone(),
                        dcp,
                        is_lse_base_on_e=impl.lse_base_on_e,
                    )
                # my head shard of ground truth
                hs = slice(H_PER_RANK * rank, H_PER_RANK * (rank + 1))
                eres = summarize_diff(final, ref_out[:, hs], "end_to_end", 0.20)
                cres = summarize_diff(
                    final, comb_of_kernel[:, hs], "combine_vs_kernelcombine", 0.03
                )
                # structural-vs-noise verdict: compare against the no-DCP
                # control (same kernel, same rows, no DCP anywhere)
                noise_band = 0.03
                lse_band = 0.005
                kern_fail = (
                    kres["nonfinite"] > 0
                    or kres["zero_where_nonzero"] > 0
                    or kres["max_rel"] > noise_band
                )
                e2e_fail = (
                    eres["nonfinite"] > 0
                    or eres["zero_where_nonzero"] > 0
                    or eres["max_rel"] > noise_band
                )
                lse_fail = lres["max_abs"] > lse_band
                cell_fail = (
                    e2e_fail
                    or cres["fail"]
                    or kern_fail
                    or lse_fail
                    or sentinel["fail"]
                    or not bool(ov.item())
                    or not bool(f_flag[0])
                )
                wr = worst_rows(final, ref_out[:, hs], sc) if cell_fail else []

                # reduce PASS/FAIL + max stats across ranks
                stats = torch.tensor(
                    [
                        eres["max_abs"],
                        eres["max_rel"],
                        cres["max_rel"],
                        float(cell_fail),
                    ],
                    device=device,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
                cell = dict(
                    scenario=name,
                    skip_empty_fill=skip_fill,
                    arm=arm_name,
                    filter_pass=bool(f_flag[0]),
                    overlap_bitexact=bool(ov.item()),
                    nodcp_control=dict(max_rel=ctl_k["max_rel"], lse=ctl_lse),
                    kernel=kres,
                    kernel_lse=lres,
                    all_rank_lse=all_lse_details,
                    empty_rows=sentinel,
                    end_to_end=eres,
                    combine_isolated=cres,
                    global_max_abs=stats[0].item(),
                    global_max_rel=stats[1].item(),
                    combine_max_rel=stats[2].item(),
                    FAIL=bool(stats[3].item()),
                    worst=wr,
                )
                results.append(cell)
                log(
                    rank,
                    f"  [{arm_name:>16s} skip_fill={int(skip_fill)}] "
                    f"{'FAIL' if cell['FAIL'] else 'PASS'} "
                    f"e2e max_rel={stats[1].item():.3e} "
                    f"abs={stats[0].item():.3e} "
                    f"combine max_rel={stats[2].item():.3e} "
                    f"k_rel={kres['max_rel']:.3e} "
                    f"lse={lres['max_abs']:.2e} "
                    f"empty(n={sentinel['n_empty']},"
                    f"ok={not sentinel['fail']})",
                )
            del impl, outs_g, lses_g, comb_of_kernel
        del caches, Ks, Vs, ref_out, ref_pout, ref_plse2
        torch.accelerator.empty_cache()

    # ---- targeted -1e30 sentinel property through the copyfree combine ----
    log(rank, "\n=== targeted -1e30 sentinel test through STAGED/VIEW combine")
    for arm_name, arm_env in combine_arms:
        for k, v in arm_env.items():
            os.environ[k] = v
        from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

        Tn = 8
        g = torch.Generator().manual_seed(7)
        out = torch.randn(Tn, H_TOTAL, KV_LORA, generator=g).to(DT).to(device)
        lse = (torch.randn(Tn, H_TOTAL, generator=g) * 3).float().to(device)
        # rank 1 fully empty via kernel sentinel -1e30 + out garbage (must be
        # suppressed by factor==0 -> 0 exactly); rank 2 empty via true -inf
        if rank == 1:
            lse.fill_(-1e30)
            out.normal_(generator=None)  # garbage payload
        if rank == 2:
            lse.fill_(float("-inf"))
            out.fill_(float("nan"))  # worst case: nan payload with -inf lse
        with torch.inference_mode():
            got = cp_lse_ag_out_rs(
                out.clone(), lse.clone(), dcp, is_lse_base_on_e=False
            )
        # reference: only ranks 0,3 contribute
        outs = [torch.empty_like(out) for _ in range(WORLD)]
        lses = [torch.empty_like(lse) for _ in range(WORLD)]
        dist.all_gather(outs, out.contiguous())
        dist.all_gather(lses, lse.contiguous())
        ref, _ = ref_combine2(torch.stack(outs).float(), torch.stack(lses))
        hs = slice(H_PER_RANK * rank, H_PER_RANK * (rank + 1))
        finite = torch.isfinite(got.float()).all().item()
        d = summarize_diff(got, ref[:, hs], "sentinel", 0.03)
        okv = torch.tensor(
            [
                float(
                    finite
                    and not d["fail"]
                    and math.isfinite(d["max_rel"])
                    and d["max_rel"] < 0.03
                )
            ],
            device=device,
        )
        dist.all_reduce(okv, op=dist.ReduceOp.MIN)
        ok = bool(okv.item())
        results.append(
            dict(
                scenario="sentinel_-1e30_-inf_nan",
                arm=arm_name,
                FAIL=not ok,
                detail=d,
                finite=finite,
            )
        )
        log(
            rank,
            f"  [{arm_name:>16s}] {'PASS' if ok else 'FAIL'} "
            f"max_rel={d['max_rel']:.3e} finite={finite}",
        )

    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=str)
        nf = sum(1 for r in results if r["FAIL"])
        print(f"\n==== DONE: {len(results)} cells, {nf} FAIL ====", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
