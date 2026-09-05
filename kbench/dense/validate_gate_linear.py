"""Validate VLLM_SM120_ROUTER_GEMM GateLinear numerics + perf at layer level.

Runs the patched fork (mounted at /glm52/vllm-vision-build) against the image's
torch. Compares:
  A) default path (env off): fp32 weight, F.linear fp32 (prod behavior)
  B) env on: bf16 weight, Tier1 dsv3_router_gemm (M<=8) / Tier3 cublas (M>8)
against an fp64 reference, on the REAL gate weights of layer 3 and layer 40
from the GLM-5.3 checkpoint, plus random activations at several scales.
Also checks top-8 (num_experts_per_tok) selection with sigmoid scoring +
e_score_correction_bias, mimicking grouped_topk.
"""
import os
import sys
import glob
import json
import struct
import torch

SNAP = glob.glob(
    "/data/huggingface/hub/models--jarrelscy--GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m/snapshots/*/"
)[0]


def load_tensor(name):
    idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))
    shard = os.path.join(SNAP, idx["weight_map"][name])
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        dt = {"BF16": torch.bfloat16, "F32": torch.float32}[meta["dtype"]]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        buf = f.read(end - start)
    t = torch.frombuffer(bytearray(buf), dtype=dt).reshape(meta["shape"]).clone()
    return t


_DIST_READY = False


def ensure_dist():
    global _DIST_READY
    if _DIST_READY:
        return
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(
        world_size=1, rank=0,
        distributed_init_method="tcp://127.0.0.1:29877",
        local_rank=0, backend="nccl",
    )
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(1, 1)
    _DIST_READY = True


def build_gate(env_on):
    ensure_dist()
    os.environ["VLLM_SM120_ROUTER_GEMM"] = "1" if env_on else "0"
    # envs is lazy (module __getattr__ reads os.environ per access) in vLLM
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    torch.set_default_dtype(torch.bfloat16)
    g = GateLinear(
        6144, 256,
        params_dtype=torch.float32,
        out_dtype=torch.float32,
        force_fp32_compute=True,
        prefix="test.gate",
    ).cuda()
    torch.set_default_dtype(torch.float32)
    return g


def main():
    dev = "cuda:0"
    results = {}
    for lname in ["model.layers.3.mlp.gate.weight", "model.layers.40.mlp.gate.weight"]:
        w_bf16 = load_tensor(lname).to(dev)
        bias = load_tensor(lname.replace(".weight", ".e_score_correction_bias")).to(dev)
        for env_on in (False, True):
            g = build_gate(env_on)
            with torch.no_grad():
                g.weight.copy_(w_bf16.to(g.weight.dtype))
            assert g.weight.dtype == (torch.bfloat16 if env_on else torch.float32), g.weight.dtype
            if env_on:
                assert g.allow_dsv3_router_gemm, "dsv3 tier not enabled!"
            torch.manual_seed(1234)
            worst_logit = 0.0
            topk_mismatch = 0
            for trial in range(300):
                M = [1, 2, 4, 8, 64][trial % 5]
                scale = [0.5, 2.0, 20.0][trial % 3]
                x = (torch.randn(M, 6144, device=dev) * scale).to(torch.bfloat16)
                out, _ = g(x)
                assert out.dtype == torch.float32
                ref = x.double() @ w_bf16.double().t()
                worst_logit = max(worst_logit, (out.double() - ref).abs().max().item())
                # noaux_tc top-8: sigmoid + bias, topk on scores
                s = torch.sigmoid(out) + bias
                sref = torch.sigmoid(ref.float()) + bias
                t1 = s.topk(8, dim=-1).indices.sort(-1).values
                t2 = sref.topk(8, dim=-1).indices.sort(-1).values
                topk_mismatch += int((t1 != t2).any())
            results[(lname, env_on)] = (worst_logit, topk_mismatch)
            print(f"{lname} env_on={env_on}: weight dtype {g.weight.dtype}, "
                  f"max|logit-fp64ref|={worst_logit:.3e}, top8 mismatches vs fp64: {topk_mismatch}/300")
            # perf at M=4 with L2-defeating rotation
            gates = []
            for i in range(61):
                gi = build_gate(env_on)
                with torch.no_grad():
                    gi.weight.copy_(w_bf16.to(gi.weight.dtype))
                gates.append(gi)
            x4 = torch.randn(4, 6144, device=dev, dtype=torch.bfloat16)
            for gi in gates:
                gi(x4)
            torch.cuda.synchronize()
            s_ev = torch.cuda.Event(True); e_ev = torch.cuda.Event(True)
            s_ev.record()
            for i in range(61 * 40):
                gates[i % 61](x4)
            e_ev.record(); torch.cuda.synchronize()
            print(f"   M=4 layer-level: {s_ev.elapsed_time(e_ev)*1000/(61*40):.2f} us/call")
            del gates
            torch.cuda.empty_cache()
    print("PASS" if all(v[1] == 0 for v in results.values()) else "CHECK MISMATCHES")


if __name__ == "__main__":
    main()
