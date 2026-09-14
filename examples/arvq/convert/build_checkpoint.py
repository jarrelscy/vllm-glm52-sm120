# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a complete HF ARVQ checkpoint without modifying the AQLM source.
Retained tensors are copied as raw byte ranges. Cold expert conversion is chunked
on GPU, with independently decoded native-layout index checks before writing.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import time
from pathlib import Path

import numpy as np
import torch
from assign_api import translate_pack
from fit_codebooks import MODEL, ROOT
from format_utils import pack_cb
from safetensors import safe_open
from safetensors.torch import save_file

TARGET = Path(
    os.environ.get(
        "ARVQ_OUTPUT_MODEL", str(Path.cwd() / "GLM-5.3-Vision-NVFP4-ARVQ-hybrid")
    )
)
COLD = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(w13|w2c|w2m)_(codes|codebooks|scales)$"
)


def atomic_json(path, obj):
    p = Path(str(path) + ".partial")
    p.write_text(json.dumps(obj, indent=2) + "\n")
    p.replace(path)


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def retain_shard(name, names):
    target = TARGET / name
    record = TARGET / "build_records" / f"{name}.json"
    if record.exists() and target.exists():
        r = json.loads(record.read_text())
        assert target.stat().st_size == r["file_bytes"]
        return r
    t = time.perf_counter()
    h, start = header(MODEL / name)
    out = {}
    offset = 0
    for k in sorted(names, key=lambda k: h[k]["data_offsets"][0]):
        v = dict(h[k])
        n = v["data_offsets"][1] - v["data_offsets"][0]
        v["data_offsets"] = [offset, offset + n]
        out[k] = v
        offset += n
    out["__metadata__"] = {
        "format": "pt",
        "provenance": "original non-cold tensor bytes preserved",
    }
    hb = json.dumps(out, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    tmp = Path(str(target) + ".partial")
    digest = hashlib.sha256()
    with open(MODEL / name, "rb") as src, open(tmp, "wb") as dst:
        dst.write(struct.pack("<Q", len(hb)))
        dst.write(hb)
        for k in sorted(names, key=lambda k: h[k]["data_offsets"][0]):
            a, b = h[k]["data_offsets"]
            src.seek(start + a)
            remaining = b - a
            while remaining:
                block = src.read(min(16 << 20, remaining))
                assert block
                dst.write(block)
                digest.update(block)
                remaining -= len(block)
        dst.flush()
        os.fsync(dst.fileno())
    tmp.replace(target)
    with safe_open(target, framework="pt") as f:
        assert set(f.keys()) == set(names)
    r = dict(
        kind="retained",
        file=name,
        tensors=len(names),
        tensor_bytes=offset,
        file_bytes=target.stat().st_size,
        raw_tensor_sha256=digest.hexdigest(),
        seconds=time.perf_counter() - t,
    )
    atomic_json(record, r)
    print(json.dumps(r), flush=True)
    return r


def check_indices(packed, old, mapping):
    # Independently decode all128 IDs from first/last64-wide tiles and rows,
    # first/last experts in each conversion chunk. Exercises cross-word IDs.
    p = packed.numpy()
    old = old.numpy().view(np.uint16)
    mapping = mapping.numpy()
    E, N, K8 = old.shape
    checks = 0
    for e in sorted({0, E - 1}):
        for tile in sorted({0, N // 16 - 1}):
            for g in sorted({0, K8 // 8 - 1}):
                words = p[e, tile, g]
                for pos in range(128):
                    bit = pos * 15
                    word = bit // 32
                    shift = bit % 32
                    v = int(words[word]) >> shift
                    if shift > 17:
                        v |= int(words[word + 1]) << (32 - shift)
                    j, lane = divmod(pos, 32)
                    q, c = divmod(lane, 4)
                    row = tile * 16 + q + 8 * (j & 1)
                    kg = g * 8 + (j // 2) * 4 + c
                    assert (v & 32767) == int(mapping[old[e, row, kg]])
                    checks += 1
    return checks


def convert(layer, projection, wm, chunk):
    stem = "w13" if projection == "gateup" else "w2c"
    newstem = "arvq_w13" if projection == "gateup" else "arvq_w2"
    prefix = f"model.layers.{layer}.mlp.experts."
    name = f"arvq-layer-{layer:03d}-{projection}.safetensors"
    target = TARGET / name
    record = TARGET / "build_records" / f"{name}.json"
    fitpath = ROOT / f"fit_l{layer}_{projection}.pt"
    fitsha = hashlib.sha256(fitpath.read_bytes()).hexdigest()
    if record.exists() and target.exists():
        r = json.loads(record.read_text())
        assert target.stat().st_size == r["file_bytes"] and r["fit_sha256"] == fitsha
        return r
    t = time.perf_counter()
    fit = torch.load(fitpath, weights_only=True)
    assert fit["layer"] == layer and fit["projection"] == projection
    mapping_cpu = fit["translation"].to(torch.int32)
    assert int(mapping_cpu.max()) < 32768
    mapping = mapping_cpu.cuda().to(torch.uint16)
    cb = pack_cb(fit).cpu().view(torch.uint32)
    with safe_open(MODEL / wm[prefix + stem + "_codes"], framework="pt") as f:
        codes = f.get_tensor(prefix + stem + "_codes")
        assert codes.shape[1] == 1
        codes = codes[:, 0]
        E, N, K8 = codes.shape
        K = K8 * 8
        packed = torch.empty((E, N // 16, K // 64, 60), dtype=torch.uint32)
        checks = 0
        torch.cuda.reset_peak_memory_stats()
        for first in range(0, E, chunk):
            old = codes[first : first + chunk].contiguous()
            cudaold = old.cuda()
            cudaout = translate_pack(cudaold, mapping)
            cpuout = cudaout[:-1].reshape(old.shape[0], N // 16, K // 64, 60).cpu()
            checks += check_indices(cpuout, old, mapping_cpu)
            packed[first : first + old.shape[0]].copy_(cpuout)
            del cudaold, cudaout, cpuout
    with safe_open(MODEL / wm[prefix + stem + "_scales"], framework="pt") as f:
        oldsc = f.get_tensor(prefix + stem + "_scales").float()
    assert oldsc.shape == (E, N)
    rows = (oldsc.cuda() * fit["beta"] / fit["global_scale"]).to(torch.float8_e4m3fn)
    assert bool(torch.isfinite(rows.float()).all())
    scaled = (
        rows.view(torch.uint8)[:, :, None]
        .expand(E, N, K // 128)
        .reshape(E, N // 16, 16, K // 128)
        .permute(0, 1, 3, 2)
        .contiguous()
        .cpu()
    )
    underflow = int(((rows.float() == 0) & (oldsc.cuda() != 0)).sum().item())
    tensors = {
        prefix + newstem + "_packed": packed,
        prefix + newstem + "_scales": scaled,
        prefix + newstem + "_codebooks": cb,
        prefix + newstem + "_global": torch.tensor(
            [fit["global_scale"]], dtype=torch.float32
        ),
    }
    size = sum(v.numel() * v.element_size() for v in tensors.values())
    bpw = size * 8 / (E * N * K)
    assert bpw < 2
    tmp = Path(str(target) + ".partial")
    save_file(
        tensors,
        str(tmp),
        metadata={
            "format": "pt",
            "arvq_format": "rvq256_128x8",
            "source": "AQLM transcode; not original donor weights",
            "fit_sha256": fitsha,
            "tp_layout": (
                "global; shard w13 row tiles and w2 K groups; runtime appends guard"
            ),
        },
    )
    with open(tmp, "rb") as sync:
        os.fsync(sync.fileno())
    tmp.replace(target)
    with safe_open(target, framework="pt") as f:
        assert set(f.keys()) == set(tensors)
        assert torch.equal(f.get_tensor(prefix + newstem + "_codebooks"), cb)
    r = dict(
        kind="arvq",
        file=name,
        layer=layer,
        projection=projection,
        tensors=list(tensors),
        shapes={k: list(v.shape) for k, v in tensors.items()},
        tensor_bytes=size,
        file_bytes=target.stat().st_size,
        weights=E * N * K,
        bpw=bpw,
        fit_sha256=fitsha,
        dictionary_relative_l2=fit["relative_l2"],
        global_scale=fit["global_scale"],
        scale_rows_underflow=underflow,
        packed_indices_verified=checks,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.perf_counter() - t,
    )
    atomic_json(record, r)
    print(json.dumps(r), flush=True)
    del packed, scaled, tensors, rows, mapping
    torch.cuda.empty_cache()
    return r


def main():
    global ROOT, MODEL, TARGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-experts", type=int, default=16)
    ap.add_argument("--layers", type=int, nargs="*")
    ap.add_argument("--cold-only", action="store_true")
    ap.add_argument("--source-model", type=Path, default=MODEL)
    ap.add_argument("--fit-dir", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path, default=TARGET)
    args = ap.parse_args()
    ROOT = args.fit_dir
    MODEL = args.source_model
    TARGET = args.output
    if args.chunk_experts < 1:
        ap.error("--chunk-experts must be positive")
    if TARGET.resolve() == MODEL.resolve():
        ap.error("Output must differ from source checkpoint")
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "build_records").mkdir(exist_ok=True)
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    config = json.loads((MODEL / "config.json").read_text())
    books = config["quantization_config"]["aqlm_layer_books"]
    assert all(v["n_base"] == 0 for v in books.values())
    layers = sorted(map(int, books)) if args.layers is None else args.layers
    atomic_json(
        TARGET / "BUILD_IN_PROGRESS.json",
        dict(source=str(MODEL), target=str(TARGET), pid=os.getpid(), layers=layers),
    )
    # Publish one complete cold layer early for loader development.
    records = []
    for p in ["gateup", "down"]:
        records.append(convert(layers[0], p, wm, args.chunk_experts))
    if not args.cold_only:
        for p in MODEL.iterdir():
            if (
                p.is_file()
                and p.name
                not in ["README.md", "model.safetensors.index.json", "config.json"]
                and not p.name.endswith(".safetensors")
            ):
                shutil.copy2(p, TARGET / p.name)
        retained = {
            k: v
            for k, v in wm.items()
            if not (COLD.match(k) and int(COLD.match(k)[1]) in set(map(int, books)))
        }
        grouped = {}
        for k, v in retained.items():
            grouped.setdefault(v, []).append(k)
        for filename, names in sorted(grouped.items()):
            records.append(retain_shard(filename, names))
    for layer in layers[1:]:
        for p in ["gateup", "down"]:
            records.append(convert(layer, p, wm, args.chunk_experts))
    if args.cold_only:
        return
    assert layers == sorted(map(int, books)), (
        "Partial layers cannot finalize checkpoint"
    )
    config["quantization_config"]["arvq"] = {
        "format": "rvq256_128x8",
        "activation_planes": 4,
        "weight_scale_group": 128,
        "version": 1,
    }
    if "text_config" in config:
        config["text_config"].setdefault(
            "quantization_config", dict(config["quantization_config"])
        )
        config["text_config"]["quantization_config"]["arvq"] = dict(
            config["quantization_config"]["arvq"]
        )
    config["_name_or_path"] = "jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid"
    atomic_json(TARGET / "config.json", config)
    outwm = dict(retained)
    for r in records:
        if r["kind"] == "arvq":
            outwm.update({k: r["file"] for k in r["tensors"]})
    for filename in set(outwm.values()):
        h, _ = header(TARGET / filename)
        assert set(k for k in h if k != "__metadata__") == set(
            k for k, v in outwm.items() if v == filename
        )
    total = sum(r["tensor_bytes"] for r in records)
    atomic_json(
        TARGET / "model.safetensors.index.json",
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(outwm.items()))},
    )
    fits = TARGET / "arvq_frozen_fits"
    fits.mkdir(exist_ok=True)
    for layer in layers:
        for p in ["gateup", "down"]:
            shutil.copy2(ROOT / f"fit_l{layer}_{p}.pt", fits / f"fit_l{layer}_{p}.pt")
    shutil.copy2(__file__, TARGET / "build_checkpoint.py")
    shutil.copy2(MODEL / "README.md", TARGET / "SOURCE_MODEL_CARD.md")
    report = dict(
        source=str(MODEL),
        source_model="jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m",
        source_revision=MODEL.name,
        format="rvq256_128x8",
        quantization_provenance=(
            "Existing decoded AQLM dictionary transcoded into two"
            " FP4-constrained additive codebooks; not original "
            "BF16 donor quantization"
        ),
        layer_count=len(layers),
        tensor_count=len(outwm),
        tensor_bytes=total,
        records=records,
        noncold_preservation=(
            "All original non-cold tensor payloads copied "
            "byte-for-byte, including hot NVFP4, vision, "
            "attention and MTP; obsolete n_base=0 w2m tensors "
            "removed"
        ),
        quality=(
            "No calibrated end-to-end accuracy claim. Dictionary "
            "fit errors are proxy distortion, not perplexity or "
            "task accuracy."
        ),
    )
    atomic_json(TARGET / "arvq_build_report.json", report)
    dictionary_errors = [
        r["dictionary_relative_l2"] for r in records if r["kind"] == "arvq"
    ]
    model_card = (
        "---\nlibrary_name: vllm\nbase_model: "
        "jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m\ntags:\n"
        "- arvq\n- nvfp4\n- experimental\n---\n# GLM-5.3 Vision "
        "NVFP4–ARVQ hybrid\n\n**Untuned research checkpoint; "
        "not accuracy-ready.** This complete checkpoint "
        "targets the custom SM120 vLLM ARVQ backend. All 75 "
        "cold MoE layers were transcoded from the existing "
        "AQLM dictionaries into two additive FP4-constrained "
        "codebooks. This is **not quantization from the "
        "original BF16 donor**. Hot NVFP4 experts and all "
        "other model tensors, including vision and MTP, "
        "retain their original payloads.\n\nThe measured "
        "unweighted dictionary relative L2 error against the "
        "decoded source AQLM dictionaries ranges from "
        "**__ERROR_MIN__ to __ERROR_MAX__** across the 150 "
        "layer projections. This is additional transcoding "
        "distortion, not perplexity or task accuracy. No "
        "calibrated end-to-end accuracy claim is made. See "
        "`arvq_build_report.json` for individual fit results "
        "and `arvq_fit_provenance.json` for the seed and "
        "traversal order. Frozen fits are included to "
        "reproduce the conversion exactly.\n\n## Format and "
        "execution\n\nEach 8-weight vector has an 8-bit first "
        "index and a 7-bit residual index, tightly packed in "
        "native MMA fragment order. FP8 E4M3 scales per 128 "
        "weights give 1.9375 bits per weight before the small"
        " shared dictionaries and global scale; complete cold"
        " tensors remain below 2 bits per weight. Hot experts"
        " preserve their original NVFP4 4.5-bit block format."
        "\n\nThe cold ARVQ path issues two FP4 MMA instructions"
        " per K tile, one for each additive codebook, while "
        "the hot NVFP4 path issues one. **Both paths place "
        "four residual activation planes (P4) into spare "
        "output columns of those same MMA tiles.** The four "
        "planes require no additional MMA instructions; their"
        " outputs are combined with weights 1, 1/16, 1/256, "
        "and 1/4096.\n\nTensors are serialized with global "
        "dimensions. The loader slices gate/up row tiles and "
        "down input groups for TP4, then appends the kernel "
        "guard word. The `arvq` marker is present in both the"
        " root and nested text quantization configurations. "
        "The matching loader selects ARVQ automatically from "
        "these explicit markers; no enable environment "
        "variable is required. Standard upstream vLLM does "
        "not support this checkpoint format.\n\n## Launch\n\nUse "
        "the matching [ARVQ SM120 "
        "branch](https://github.com/jarrelscy/vllm-glm52-sm120/tree/arvq-hybrid-sm120),"
        " which includes the standalone "
        "`examples/arvq/compose.yaml` configuration:\n\n```bash"
        "\ngit clone --branch arvq-hybrid-sm120 "
        "https://github.com/jarrelscy/vllm-glm52-sm120.git\ncd"
        " vllm-glm52-sm120\n"
        "ARVQ_MODEL_DIR=/path/to/GLM-5.3-Vision-NVFP4-ARVQ-hybrid"
        " \\\n  docker compose -f examples/arvq/compose.yaml up"
        " -d --build\n```\n\nThe configuration targets four "
        "SM120 GPUs and serves port 8001. Consult the branch "
        "deployment records for measured settings and "
        "validation status; checkpoint construction alone "
        "does not establish serving performance or "
        "long-context accuracy.\n\nConversion tools and "
        "reproduction instructions are in "
        "`examples/arvq/convert` on the same branch. The "
        "source model card is preserved as "
        "`SOURCE_MODEL_CARD.md`.\n"
    )
    (TARGET / "README.md").write_text(
        model_card.replace("__ERROR_MIN__", f"{min(dictionary_errors):.2%}").replace(
            "__ERROR_MAX__", f"{max(dictionary_errors):.2%}"
        )
    )
    (TARGET / "BUILD_IN_PROGRESS.json").unlink()
    atomic_json(
        TARGET / "BUILD_COMPLETE.json",
        {
            "tensor_count": len(outwm),
            "tensor_bytes": total,
            "layers": len(layers),
            "status": "checkpoint complete; serving validation separate",
        },
    )
    print(
        json.dumps(
            {"complete": str(TARGET), "tensor_count": len(outwm), "tensor_bytes": total}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
