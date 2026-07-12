# SPDX-License-Identifier: Apache-2.0
"""TIER 2 — custom/P2P allreduce vs NCCL allreduce (torchrun, 2-4 GPUs).

Guards idea 4 (one-shot P2P allreduce replacing NCCL for small TP
messages).  Any custom allreduce registered in ``kernel_variants.json``
with ``kind: "allreduce"``-style compare hooks — or simply importable as
``vllm.distributed.device_communicators...`` — must produce elementwise
|diff| within the fp16/bf16 reorder bound of NCCL's result on the REAL
message sizes of the serving config.

Also documents determinism ACROSS REPEATS of the same implementation:
NCCL itself may vary run-to-run (ring reductions are not order-stable);
a custom one-shot P2P allreduce SHOULD be deterministic and is asserted
so when its registry entry declares ``"deterministic": true``.

Run with:  tests/sm120_correctness/tier2_dist/launch_dist_tests.sh
(which torchruns this file inside the container; needs GLM_SM120_DIST_TESTS=1)
"""

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.skipif(os.environ.get("GLM_SM120_DIST_TESTS") != "1",
                       reason="multi-GPU tier: launch via "
                       "launch_dist_tests.sh (GLM_SM120_DIST_TESTS=1)"),
    pytest.mark.skipif("RANK" not in os.environ,
                       reason="must run under torchrun"),
]

# Real decode-path message sizes on tp4-1m-mtp:
#   TP all-reduce of hidden states: [ntokens, 6144] bf16
#   ntokens = 4 (MTP verify), 1 (draft), 8/16 (small batch)
MESSAGE_SHAPES = [(1, 6144), (4, 6144), (8, 6144), (16, 6144)]
REPEATS = 20


@pytest.fixture(scope="module")
def dist_group():
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    yield dist
    # leave the group up for other tests in the same torchrun


def _reorder_bound(inputs_abs_sum: torch.Tensor, world: int,
                   dtype: torch.dtype) -> torch.Tensor:
    eps = torch.finfo(dtype).eps
    return (world - 1) * eps * inputs_abs_sum + 1e-8


def _discover_custom_allreduces():
    """Custom allreduce impls declared in kernel_variants.json with
    kind 'python_env' and a compare hook named '*allreduce*'."""
    from common.variant_registry import load_registry, resolve_compare
    out = []
    for v in load_registry():
        if v["kind"] == "python_env" and "allreduce" in v["name"]:
            out.append((v, resolve_compare(v["compare"])))
    return out


@pytest.mark.parametrize("shape", MESSAGE_SHAPES, ids=str)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_nccl_baseline_and_custom(shape, dtype, dist_group):
    dist = dist_group
    world = dist.get_world_size()
    rank = dist.get_rank()
    dev = torch.cuda.current_device()

    g = torch.Generator(device="cpu").manual_seed(1000 + rank)
    x = (torch.randn(shape, generator=g) * 0.5).to(dtype).to(dev)

    # exact fp64 reference across ranks (via all_gather of fp64 copies)
    xs = [torch.zeros(shape, dtype=torch.float64, device=dev)
          for _ in range(world)]
    dist.all_gather(xs, x.double())
    exact = torch.stack(xs).sum(0)
    abs_sum = torch.stack(xs).abs().sum(0)
    bound = _reorder_bound(abs_sum, world, dtype)

    # NCCL result within the bound of the exact sum
    y = x.clone()
    dist.all_reduce(y)
    diff = (y.double() - exact).abs()
    assert bool((diff <= bound).all()), \
        (f"NCCL allreduce outside reorder bound: max diff "
         f"{diff.max().item():.3e} vs bound {bound.max().item():.3e}")

    # NCCL repeat determinism: DOCUMENTED, not gated
    outs = set()
    for _ in range(REPEATS):
        z = x.clone()
        dist.all_reduce(z)
        outs.add(z.view(torch.uint8).cpu().numpy().tobytes())
    if rank == 0:
        print(f"\n[info] NCCL allreduce {shape} {dtype}: "
              f"{len(outs)} distinct bit-patterns over {REPEATS} repeats "
              "(NCCL is not required to be deterministic)")

    # custom implementations (auto-discovered)
    for variant, hook in _discover_custom_allreduces():
        ref, got = hook({"x": x, "group": dist.group.WORLD}, f"cuda:{dev}")
        d = (torch.as_tensor(got).double().to(dev) - exact).abs()
        assert bool((d <= bound).all()), \
            (f"{variant['name']}: outside fp reorder bound of the exact "
             f"sum (max {d.max().item():.3e})")
        if variant.get("deterministic"):
            reps = set()
            for _ in range(REPEATS):
                _, got2 = hook({"x": x, "group": dist.group.WORLD},
                               f"cuda:{dev}")
                reps.add(torch.as_tensor(got2).cpu().numpy().tobytes())
            assert len(reps) == 1, \
                (f"{variant['name']} declared deterministic but produced "
                 f"{len(reps)} distinct outputs over {REPEATS} repeats")


def test_dcp_a2a_vs_agrs_full_comms(dist_group):
    """Full-comms equivalence of the two DCP combine backends (idea 8)."""
    dist = dist_group
    world = dist.get_world_size()
    rank = dist.get_rank()
    dev = torch.cuda.current_device()
    if world < 2:
        pytest.skip("needs >= 2 ranks")

    from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
    from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

    class _Group:  # minimal GroupCoordinator shim for the two ops
        world_size = world
        rank_in_group = rank
        device_group = dist.group.WORLD

        @staticmethod
        def all_gather(t, dim=0):
            ts = [torch.empty_like(t) for _ in range(world)]
            dist.all_gather(ts, t.contiguous())
            return torch.cat(ts, dim=dim)

        @staticmethod
        def reduce_scatter(t, dim=1):
            n = t.shape[dim] // world
            dist.all_reduce(t)
            return t.narrow(dim, rank * n, n).contiguous()

    b, h, d = 4, 32, 128
    g = torch.Generator(device="cpu").manual_seed(7 + rank)
    out = torch.randn(b, h, d, generator=g).to(torch.bfloat16).to(dev)
    lse = (torch.randn(b, h, generator=g) * 3).float().to(dev)

    got_agrs = cp_lse_ag_out_rs(out.clone(), lse.clone(), _Group())
    got_a2a = dcp_a2a_lse_reduce(out.clone(), lse.clone(), _Group())

    a, b_ = got_agrs.float(), got_a2a.float()
    atol = 2 * 2.0**-8 * a.abs().max().item() + 1e-6
    torch.testing.assert_close(a, b_, atol=atol, rtol=2.0**-7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
