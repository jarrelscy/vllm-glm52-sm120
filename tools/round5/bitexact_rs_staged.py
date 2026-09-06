# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness proof for VLLM_GLM_DCP_RS_STAGED / VLLM_GLM_DCP_RS_VIEW.

Runs single-GPU, in one process (same-process A/B: valid bit-exactness
methodology -- no cross-boot comparison). Intended to run inside the prod
container (same torch/cuBLAS/triton/GPU as the deployment):

  docker exec <container> /opt/vllm/.venv/bin/python /tmp/round5/bitexact_rs_staged.py \
      --common /tmp/round5/common.py

Parts:
  A. Strided correct kernel vs baseline correct kernel + movedim.contiguous
     copy: staging buffer must be BYTE-identical, final lse byte-identical.
     => the ReduceScatter input is unchanged, so the collective output is
     unchanged (identical bytes into the identical enqueue).
  B. cuBLAS bmm stride invariance for the RS_VIEW epilogue: bmm on
     (N, B, L) as a contiguous batch (new path: view of the RS output's
     natural layout) vs as a strided view of a [B, N, L] contiguous copy
     (baseline path). Output must be byte-identical for every deployed
     shape, else RS_VIEW must not ship.
  C. End-to-end layout algebra on a simulated deterministic 4-rank
     reduce-scatter: per-element operand multisets and rank order are
     identical between the two stagings (they are the same bytes), and the
     view epilogue exposes exactly the same elements the copy epilogue
     copied.
"""

import argparse
import importlib.util
import sys

import torch

WS = 4
H = 64  # total mqa heads after DCP AG(q)
D = 512  # kv_lora_rank
V = 256  # v_head_dim
HL = H // WS


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        sys.exit(1)


def part_a(common, device):
    torch.manual_seed(1234)
    for B in (1, 2, 4, 8, 16):
        for base_e in (True, False):
            out0 = torch.randn(B, H, D, device=device, dtype=torch.bfloat16)
            lses = torch.randn(WS, B, H, device=device, dtype=torch.float32) * 4
            # exercise the NaN/inf handling branches
            lses[0, 0, 0] = float("nan")
            if B > 1:
                lses[:, 1, 3] = float("-inf")  # all -inf -> factor 0 path
                lses[1, 1, 5] = float("inf")

            # baseline: in-place correct + staging copy
            out_ref = out0.clone()
            ctx = common.CPTritonContext()
            out_ref, lse_ref = common.correct_attn_out(
                out_ref, lses.clone(), 2, ctx, is_lse_base_on_e=base_e
            )
            staged_ref = out_ref.movedim(0, 1).contiguous()  # [H, B, D]

            # lever: strided write into rank-major staging
            out_in = out0.clone()
            staged = torch.empty((H, B, D), device=device, dtype=out0.dtype)
            lse_new = common.correct_attn_out_staged(
                out_in,
                lses.clone(),
                2,
                staged.movedim(0, 1),
                is_lse_base_on_e=base_e,
            )
            torch.cuda.synchronize()
            check(
                f"A: staging bytes identical (B={B}, base_e={base_e})",
                torch.equal(bits(staged_ref), bits(staged)),
            )
            check(
                f"A: lse bytes identical (B={B}, base_e={base_e})",
                torch.equal(bits(lse_ref), bits(lse_new)),
            )


def part_b(device):
    torch.manual_seed(4321)
    all_ok = True
    for B in (1, 2, 3, 4, 8, 16, 32):
        w_uv = torch.randn(HL, D, V, device=device, dtype=torch.bfloat16)
        vals = torch.randn(HL, B, D, device=device, dtype=torch.bfloat16)

        # baseline: x is a strided (N, B, L) view of a [B, N, L] contiguous
        # buffer (the epilogue copy's output)
        x_base_buf = vals.movedim(0, 1).contiguous()  # [B, HL, D]
        x_base = x_base_buf.view(-1, HL, D).transpose(0, 1)  # strided (HL,B,D)

        # lever: x is the movedim VIEW of the RS output's natural
        # [HL, B, D] contiguous buffer; after _v_up_proj's transpose it is
        # the contiguous [HL, B, D] buffer itself.
        x_view_buf = vals.clone()  # [HL, B, D] contiguous
        x_view = x_view_buf.movedim(0, 1).view(-1, HL, D).transpose(0, 1)

        assert torch.equal(x_base.contiguous(), x_view.contiguous())
        # For B == 1 the baseline movedim.contiguous() is itself a no-op
        # (size-1 dim), so both paths present identical strides and the
        # check is trivially satisfied; require distinct strides otherwise.
        assert B == 1 or x_base.stride() != x_view.stride(), (
            B,
            x_base.stride(),
            x_view.stride(),
        )

        out_a = torch.empty(B, HL, V, device=device, dtype=torch.bfloat16)
        out_b = torch.empty(B, HL, V, device=device, dtype=torch.bfloat16)
        torch.bmm(x_base, w_uv, out=out_a.transpose(0, 1))
        torch.bmm(x_view, w_uv, out=out_b.transpose(0, 1))
        torch.cuda.synchronize()
        same = torch.equal(bits(out_a), bits(out_b))
        print(
            f"[{'PASS' if same else 'FAIL'}] B: bmm stride-invariance "
            f"B={B} strides base={x_base.stride()} view={x_view.stride()}"
        )
        all_ok &= same
    if not all_ok:
        print("RS_VIEW MUST NOT SHIP: bmm output depends on input strides")
        sys.exit(1)


def part_c(common, device):
    """Simulated deterministic reduce-scatter over both stagings."""
    torch.manual_seed(99)
    B = 4
    per_rank_staged = []
    for _ in range(WS):
        out0 = torch.randn(B, H, D, device=device, dtype=torch.bfloat16)
        lses = torch.randn(WS, B, H, device=device, dtype=torch.float32)
        staged = torch.empty((H, B, D), device=device, dtype=out0.dtype)
        common.correct_attn_out_staged(
            out0, lses, 1, staged.movedim(0, 1), is_lse_base_on_e=True
        )
        per_rank_staged.append(staged)
    torch.cuda.synchronize()

    # Deterministic rank-order elementwise sum of chunk r (stands in for the
    # per-element NCCL ring accumulation; both layouts feed IDENTICAL bytes,
    # asserted in part A, so this simulation applies to both paths equally).
    r = 1
    chunk = [s.view(WS, HL, B, D)[r].float() for s in per_rank_staged]
    acc = chunk[0]
    for c in chunk[1:]:
        acc = acc + c
    rs_out = acc.to(torch.bfloat16)  # [HL, B, D] natural RS output

    # baseline epilogue: copy to [B, HL, D]; lever epilogue: movedim view.
    ref = rs_out.movedim(0, 1).contiguous()
    view = rs_out.movedim(0, 1)
    check(
        "C: view epilogue exposes exactly the copied elements",
        torch.equal(bits(ref), bits(view.contiguous()))
        and torch.equal(ref, view),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", required=True, help="path to modified common.py")
    args = ap.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(device)
    common = load_module("round5_common", args.common)

    part_a(common, device)
    part_b(device)
    part_c(common, device)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
