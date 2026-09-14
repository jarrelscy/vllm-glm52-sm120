# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import pathlib

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = pathlib.Path(os.environ.get("ARVQ_LAB_ROOT", "/lab"))
INV = json.loads((ROOT / "scale_experiment/inventory.json").read_text())
MODEL = pathlib.Path(INV["checkpoint"])
L = next(a for a in INV["layers"] if a["layer_id"] == 3)
LEVELS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]


def tensor(name):
    v = L["tensors"][name]
    with safe_open(str(MODEL / v["file"]), framework="pt") as f:
        return f.get_tensor(v["key"])


def shard(t, proj, kind, rank=3):
    if proj == "gateup":
        axis = 2 if kind == "codes" else 1
        return torch.cat(
            (t.narrow(axis, rank * 512, 512), t.narrow(axis, 2048 + rank * 512, 512)),
            dim=axis,
        ).contiguous()
    if kind == "codes":
        return t[:, :, :, rank * 64 : (rank + 1) * 64].contiguous()
    if kind == "packed":
        return t[:, :, rank * 256 : (rank + 1) * 256].contiguous()
    if kind == "bscale":
        return t[:, :, rank * 32 : (rank + 1) * 32].contiguous()
    return t.contiguous()


def native_fragment(w):
    E, N, K2 = w.shape
    G = K2 // 32
    n = w.view(torch.int32).reshape(E, N // 16, 16, G, 8).permute(0, 1, 3, 2, 4)
    return torch.stack(
        [
            n[
                :, :, :, 8 * (j % 2) : 8 * (j % 2) + 8, 4 * (j // 2) : 4 * (j // 2) + 4
            ].reshape(E, N // 16, G, 32)
            for j in range(4)
        ],
        dim=3,
    ).contiguous()


def load():
    rvq = load_file(
        str(ROOT / "scale_experiment/rvq_layer3_rank3.safetensors"), device="cuda"
    )
    all = {}
    for proj, prefix, nv in [
        ("gateup", "w13", "nvfp4_w13"),
        ("down", "w2c", "nvfp4_w2"),
    ]:
        codes = shard(tensor(prefix + "_codes"), proj, "codes").cuda()
        sc = shard(tensor(prefix + "_scales"), proj, "scales").cuda()
        cb = tensor(prefix + "_codebooks").cuda()
        pw = shard(tensor(nv + "_packed"), proj, "packed").cuda()
        bs = shard(tensor(nv + "_bscale"), proj, "bscale").cuda()
        s2 = tensor(nv + "_scale2").cuda()
        fit = torch.load(ROOT / f"scale_experiment/fit_l3_{proj}.pt", weights_only=True)
        all[proj] = {
            "codes": codes,
            "source_scales": sc,
            "source_cb": cb,
            "hot_packed": pw,
            "hot_bscale": bs,
            "hot_scale2": s2,
            "hot_fragment": native_fragment(pw),
            "hot_native_scales": bs.view(torch.int32),
            "cold_packed": rvq[proj + ".packed"],
            "cold_scales": rvq[proj + ".scales"],
            "cold_cb": rvq[proj + ".codebooks"],
            "cold_global": rvq[proj + ".global_scale"],
            "translation": fit["translation"].cuda().long(),
            "c0": fit["c0"].cuda(),
            "c1": fit["c1"].cuda(),
            "N": codes.shape[2],
            "K": codes.shape[3] * 8,
        }
    return all


def decoded_weights(data, ci, hi, rvq):
    result = []
    N = data["N"]
    K = data["K"]
    levels = torch.tensor(LEVELS, device="cuda")
    scales = data["cold_scales"]
    E = scales.shape[0]
    nsc = (
        scales.reshape(E, N // 16, K // 128, 16)
        .permute(0, 1, 3, 2)
        .reshape(E, N, K // 128)
        .view(torch.float8_e4m3fn)
        .float()
    )
    for c, h in zip(ci.tolist(), hi.tolist()):
        if c >= 0:
            ids = data["codes"][c, 0].long() & 65535
            if rvq:
                q = data["translation"][ids]
                w = (data["c0"][q & 255] + data["c1"][q >> 8]).reshape(N, K)
                w = w * nsc[c].repeat_interleave(128, -1) * data["cold_global"]
            else:
                w = (
                    data["source_cb"][0, ids].reshape(N, K).float()
                    * data["source_scales"][c, :, None].float()
                )
        else:
            q = data["hot_packed"][h]
            w = levels[torch.stack((q & 15, q >> 4), -1).long()].reshape(N, K)
            w = w * data["hot_bscale"][h].view(
                torch.float8_e4m3fn
            ).float().repeat_interleave(16, -1)
            s = data["hot_scale2"][h]
            w = w * (s.repeat_interleave(N // s.numel())[:, None])
        result.append(w)
    return torch.stack(result)


def reconstructed_activations(x, planes=4):
    levels = torch.tensor(LEVELS, device=x.device)
    res = x.float().clone()
    y = torch.zeros_like(res)
    for j in range(planes):
        v = res * (16**j)
        s = torch.pow(
            2.0,
            torch.ceil(
                torch.log2(v.reshape(x.shape[0], -1, 16).abs().amax(-1) / 6)
            ).clamp(-6, 8),
        )
        a = v / s.repeat_interleave(16, -1)
        code = sum(
            (a.abs() > z).int() for z in [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
        ) + ((a < 0).int() << 3)
        q = levels[code.long()] * s.repeat_interleave(16, -1) / (16**j)
        res -= q
        y += q
    return y
