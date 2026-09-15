# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
from pathlib import Path

import torch
from attention_rounding_math import emu, error, sparse_mla_sm120_decode_dsv3_2

results = []
for file in sorted(Path(sys.argv[1]).glob("attention-*.pt")):
    d = torch.load(file, weights_only=True, map_location="cuda")
    q, raw, idx = d["q"], d["cache"], d["indices"]
    T, H, _ = q.shape
    K = idx.shape[-1]
    ns = K // 64
    mid = torch.empty(T, H, ns, 512, device="cuda", dtype=torch.bfloat16)
    ml = torch.empty(T, H, ns, device="cuda")
    out = torch.empty(T, H, 512, device="cuda", dtype=torch.bfloat16)
    ol = torch.empty(T, H, device="cuda")
    found = []
    for cpb in range(1, ns + 1):
        sparse_mla_sm120_decode_dsv3_2(
            q,
            raw,
            idx,
            mid,
            ml,
            out,
            ol,
            d["scale"],
            topk_length=d["seq_lens"],
            model_type=2,
            chunks_per_block=cpb,
        )
        if torch.equal(out[-1], d["output"][-1]):
            found.append(cpb)
    if not found:
        raise RuntimeError(f"No native tactic reproduces captured output: {file}")
    # Captured single-sequence last rows have only trailing negative padding.
    indices = idx[-1]
    valid = indices >= 0
    assert not bool((valid[1:] & ~valid[:-1]).any()), (
        "nontrailing padding needs explicit emulation"
    )
    selected = raw.reshape(-1, 656)[indices[valid].long()]
    row = dict(
        file=file.name,
        tokens=T,
        heads=H,
        selected=len(selected),
        matching_cpb=found,
        arms={},
    )
    for wp, bp in ((False, False), (True, False), (False, True), (True, True)):
        ref, lse = emu(q[-1], selected, found[0], wp, bp)
        row["arms"][f"fp8w={wp},bf16partial={bp}"] = error(d["output"][-1], ref)
    ref, lse = emu(q[-1], selected, found[0], True, True, source_order=True)
    row["arms"]["plus_source_qk_order"] = error(d["output"][-1], ref)
    results.append(row)
    print(json.dumps(row), flush=True)
Path(sys.argv[2]).write_text(json.dumps(results, indent=2))
