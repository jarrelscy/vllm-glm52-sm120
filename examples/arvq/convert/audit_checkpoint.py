# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Header-only complete checkpoint inventory, independent of builder bookkeeping."""

import argparse
import json
import os
import struct
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument(
    "--model",
    type=Path,
    default=Path(
        os.environ.get(
            "ARVQ_OUTPUT_MODEL", str(Path.cwd() / "GLM-5.3-Vision-NVFP4-ARVQ-hybrid")
        )
    ),
)
P = ap.parse_args().model
index = json.loads((P / "model.safetensors.index.json").read_text())
wm = index["weight_map"]
cfg = json.loads((P / "config.json").read_text())
total = 0
seen = {}
arvq = []
for file in sorted(set(wm.values())):
    with open(P / file, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    end = 0
    for k, v in h.items():
        if k == "__metadata__":
            continue
        assert k in wm and wm[k] == file and k not in seen, k
        a, b = v["data_offsets"]
        assert b >= a >= 0
        total += b - a
        end = max(end, b)
        seen[k] = v
    assert (P / file).stat().st_size == 8 + n + end, file
assert set(seen) == set(wm) and total == index["metadata"]["total_size"]
for ls, book in cfg["quantization_config"]["aqlm_layer_books"].items():
    layer = int(ls)
    E = book["n_cold"]
    p = f"model.layers.{layer}.mlp.experts."
    for family, N, K in [("arvq_w13", 4096, 6144), ("arvq_w2", 6144, 2048)]:
        spec = {
            "packed": ("U32", [E, N // 16, K // 64, 60]),
            "scales": ("U8", [E, N // 16, K // 128, 16]),
            "codebooks": ("U32", [384]),
            "global": ("F32", [1]),
        }
        nbytes = 0
        for suffix, (dtype, shape) in spec.items():
            v = seen[p + family + "_" + suffix]
            assert v["dtype"] == dtype and v["shape"] == shape
            nbytes += v["data_offsets"][1] - v["data_offsets"][0]
        bpw = nbytes * 8 / (E * N * K)
        assert bpw < 2
        arvq.append(
            {
                "layer": layer,
                "family": family,
                "bpw": bpw,
                "weights": E * N * K,
                "tensor_bytes": nbytes,
            }
        )
    assert not any(
        k.startswith(p + old) for k in seen for old in ["w13_", "w2c_", "w2m_"]
    )
assert cfg["quantization_config"]["arvq"]["format"] == "rvq256_128x8"
assert (
    cfg["text_config"]["quantization_config"]["arvq"]
    == cfg["quantization_config"]["arvq"]
)
report = {
    "status": "passed",
    "files": len(set(wm.values())),
    "tensors": len(wm),
    "tensor_bytes": total,
    "cold_projection_count": len(arvq),
    "cold_weights": sum(r["weights"] for r in arvq),
    "cold_tensor_bytes": sum(r["tensor_bytes"] for r in arvq),
    "cold_bpw_min": min(r["bpw"] for r in arvq),
    "cold_bpw_max": max(r["bpw"] for r in arvq),
    "mtp_tensors": sum(k.startswith("model.layers.78.") for k in seen),
    "vision_tensors": sum(
        k.startswith(("vision_tower.", "mm_projector.")) for k in seen
    ),
}
(P / "arvq_checkpoint_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
