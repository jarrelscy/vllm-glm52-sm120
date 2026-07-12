# SPDX-License-Identifier: Apache-2.0
"""TIER 2 — q_b_proj replication vs sharded + DCP q all_gather (idea 3).

Pre-written for the planned optimization that replicates q_b_proj on
every DCP rank to remove the per-attention-layer
``get_dcp_group().all_gather(mqa_q, dim=1)`` (mla_attention.py — the
"unavoidable" query-head gather of ag_rs-style DCP).

Contract being gated: for the SAME input hidden states, computing the
FULL q projection locally (replicated weight) must equal computing the
rank's head-shard and all-gathering, up to GEMM reorder tolerance —
per rank, for every rank's shard.

Single-GPU part (no comms — the math): full GEMM vs concatenated
shard GEMMs on one device.  The gather itself is a bit-exact memcpy, so
this IS the numerical content of the equivalence; the torchrun part
additionally exercises the real all_gather plumbing.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")

# GLM-5.2 MLA q_b_proj-ish dims: hidden/kv_lora input -> (heads * head_dim)
# per-rank shard = total / dcp_world.  Use production-like sizes.
CASES = [
    # (B, K_in, H_total*D_out, world)
    (4, 1536, 96 * 192, 4),
    (1, 1536, 96 * 192, 4),
    (16, 6144, 48 * 128, 2),
]


@pytest.fixture(scope="module")
def gpu():
    from common.kernels import pick_gpu
    idx = pick_gpu()
    if idx is None:
        pytest.skip("no GPU with enough free memory")
    torch.cuda.set_device(idx)
    return f"cuda:{idx}"


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"B{c[0]}K{c[1]}M{c[2]}W{c[3]}")
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
def test_replicated_equals_sharded_gemm(case, dtype, gpu):
    b, k, m, world = case
    g = torch.Generator(device="cpu").manual_seed(11)
    x = (torch.randn(b, k, generator=g) * 0.05).to(dtype).to(gpu)
    w = (torch.randn(m, k, generator=g) * 0.02).to(dtype).to(gpu)

    full = x @ w.t()                       # replicated-q_b path
    shard_m = m // world
    shards = [x @ w[r * shard_m:(r + 1) * shard_m].t()
              for r in range(world)]
    gathered = torch.cat(shards, dim=1)    # == all_gather(mqa_q, dim=1)

    diff = (full.float() - gathered.float()).abs()
    # GEMM split-K / algorithm reorder bound: eps * sum|x_i * w_ij|
    bound = (torch.finfo(dtype).eps *
             (x.float().abs() @ w.t().float().abs()) * 2 + 1e-6)
    bad = (diff > bound).sum().item()
    assert bad == 0, \
        (f"replicated vs sharded q GEMM: {bad}/{diff.numel()} elements "
         f"outside reorder bound; max diff {diff.max().item():.3e} "
         f"(bound max {bound.max().item():.3e})")

    exact_frac = (full == gathered).float().mean().item()
    print(f"\n[info] B{b} K{k} M{m} W{world}: bitwise-equal fraction "
          f"{exact_frac:.4f} (informational; cuBLAS may pick different "
          "kernels for different N)")


@pytest.mark.skipif(os.environ.get("GLM_SM120_DIST_TESTS") != "1" or
                    "RANK" not in os.environ,
                    reason="full-comms variant: launch_dist_tests.sh")
def test_replicated_equals_sharded_with_allgather():
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.cuda.current_device()

    b, k, m = 4, 1536, 96 * 192
    g = torch.Generator(device="cpu").manual_seed(11)  # same on all ranks
    x = (torch.randn(b, k, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    w = (torch.randn(m, k, generator=g) * 0.02).to(torch.bfloat16).to(dev)

    full = x @ w.t()
    shard_m = m // world
    mine = x @ w[rank * shard_m:(rank + 1) * shard_m].t()
    parts = [torch.empty_like(mine) for _ in range(world)]
    dist.all_gather(parts, mine.contiguous())
    gathered = torch.cat(parts, dim=1)

    diff = (full.float() - gathered.float()).abs()
    bound = (torch.finfo(torch.bfloat16).eps *
             (x.float().abs() @ w.t().float().abs()) * 2 + 1e-6)
    assert bool((diff <= bound).all())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
