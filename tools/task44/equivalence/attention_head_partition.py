# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import torch
from flashinfer.mla._sparse_mla_sm120 import sparse_mla_sm120_decode_dsv3_2 as native

root = Path("/opt/task44-traces")
cpbs = {
    r["file"]: r["matching_cpb"][0]
    for r in json.loads((root / "attention_real_rounding_results.json").read_text())
}
results = []
for f in sorted((root / "arvq-task44-b26-pp-attention-capture").glob("attention-*.pt")):
    d = torch.load(f, weights_only=True, map_location="cuda")
    q = d["q"]
    T, H, D = q.shape
    K = d["indices"].shape[-1]
    row = dict(file=f.name, arms={})
    for mode, cpb in [("automatic", None), ("matched_split", cpbs[f.name])]:
        parts = []
        for part in q.chunk(4, dim=1):
            part = part.contiguous()
            h = part.shape[1]
            mid = torch.empty(T, h, K // 64, 512, device="cuda", dtype=torch.bfloat16)
            ml = torch.empty(T, h, K // 64, device="cuda")
            out = torch.empty(T, h, 512, device="cuda", dtype=torch.bfloat16)
            ol = torch.empty(T, h, device="cuda")
            native(
                part,
                d["cache"],
                d["indices"],
                mid,
                ml,
                out,
                ol,
                d["scale"],
                topk_length=d["seq_lens"],
                model_type=2,
                chunks_per_block=cpb,
            )
            parts.append(out)
        got = torch.cat(parts, 1)[-1]
        ref = d["output"][-1]
        row["arms"][mode] = dict(
            unequal=int((got != ref).sum()),
            rel_l2=float((got.float() - ref.float()).norm() / ref.float().norm()),
        )
    results.append(row)
    print(json.dumps(row), flush=True)
(root / "attention_head_partition_results.json").write_text(
    json.dumps(results, indent=2)
)
