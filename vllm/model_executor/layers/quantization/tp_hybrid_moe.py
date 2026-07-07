# SPDX-License-Identifier: Apache-2.0
"""Path B: tensor-parallel HybridExpertsMoEMethod for the NVFP4+AQLM MoE.

Replaces the tp_size>1 hard-raise (nvfp4_aqlm_hybrid.py:251) with real TP
sharding. The v2 gemv KERNEL is unchanged — sharding is entirely load-time
weight-slicing + one all-reduce after the down projection. Correctness proven
bit-exact (gate_up) / 8e-4 (down) in tp_microbench.py.

Sharding for rank g of T = tp_size (see TP_DESIGN.md):
  gate_up (w13): COLUMN-parallel over the 2I output dim. Each rank owns
      I/T gate rows + I/T up rows, stored as [gate_slice ; up_slice] so SwiGLU
      pairs locally and NVFP4's gate/up scale2 split still lands. No comm.
  down   (w2):  ROW-parallel over the I input dim (the K/g / K/2 / K/16 axes).
      Per-output-channel scales are on the unsharded H dim -> replicated; the
      scale is linear so applying it to partial sums is exact. All-reduce the
      down output across the TP group.
  codebooks, NVFP4 scale2, hyb_kind: global -> REPLICATED.

Integration: in NvFp4AqlmHybridConfig.get_quant_method, return
TPHybridExpertsMoEMethod when tp_size>1 (drop the raise). See the diff in the
report. Requires a real TP serve to validate end-to-end (needs a 4-GPU lease +
Path A's PP fix for the TP2xPP2 1M config).
"""
import torch

from vllm.model_executor.utils import set_weight_attrs

# Import the deployed base method from the in-tree module.
from vllm.model_executor.layers.quantization.nvfp4_aqlm_hybrid import (
    HybridExpertsMoEMethod,
)


def _shard_range(dim_full: int, rank: int, world: int) -> slice:
    assert dim_full % world == 0, f"{dim_full} not divisible by TP={world}"
    c = dim_full // world
    return slice(rank * c, (rank + 1) * c)


def _gateup_loader(rank, world, i, axis):
    """Weight loader slicing the 2I output dim as [gate_slice ; up_slice].

    axis = the tensor dim holding 2I rows (2 for codes, 1 for scales/packed).
    """
    ii = i // world

    def load(param, loaded):
        # loaded has full 2I along `axis`: [0:I] gate, [I:2I] up.
        idx_g = [slice(None)] * loaded.dim()
        idx_u = [slice(None)] * loaded.dim()
        idx_g[axis] = slice(rank * ii, (rank + 1) * ii)
        idx_u[axis] = slice(i + rank * ii, i + (rank + 1) * ii)
        shard = torch.cat([loaded[tuple(idx_g)], loaded[tuple(idx_u)]], dim=axis)
        param.data.copy_(shard)

    return load


def _rowk_loader(rank, world, axis):
    """Weight loader slicing the down input (I) dim along `axis`."""
    def load(param, loaded):
        idx = [slice(None)] * loaded.dim()
        idx[axis] = _shard_range(loaded.shape[axis], rank, world)
        param.data.copy_(loaded[tuple(idx)])

    return load


def _replicate_loader():
    def load(param, loaded):
        param.data.copy_(loaded)

    return load


class TPHybridExpertsMoEMethod(HybridExpertsMoEMethod):
    """HybridExpertsMoEMethod with TP weight-sharding + down all-reduce."""

    def __init__(self, *a, tp_size: int, tp_rank: int, **kw):
        # Bypass the base __init__ tp==1 assertion by setting the flag first.
        self._tp = tp_size
        self._tpr = tp_rank
        # Base __init__ raises for tp_size>1 via moe_parallel_config; we call
        # the grandparent path by temporarily faking tp==1 on the moe config
        # is fragile, so we replicate the needed init here.
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )
        FusedMoEMethodBase.__init__(self, kw["moe_config"])
        self.layer_idx = kw["layer_idx"]
        self.n_nvfp4 = kw["n_nvfp4"]
        self.n_base = kw["n_base"]
        self.n_cold = kw["n_cold"]
        import os
        self._stats_dir = os.environ.get("VLLM_HYBRID_EXPERT_STATS")
        self._stats = None
        self._stats_calls = 0

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        # NOTE: vLLM already divides the intermediate by tp_size before calling
        # us (FusedMoEConfig, config.py:1315). So `ish` here is the per-rank
        # (sharded) intermediate; the full checkpoint dim is i_full = ish * T.
        # Our params are sized with ish; our loaders slice the full tensor down.
        T, g_rank = self._tp, self._tpr
        h = hidden_size                       # hidden is NOT sharded
        ish = intermediate_size_per_partition
        i_full = ish * T
        g = 8
        entries = 65536
        na, nm, nc = self.n_nvfp4, self.n_base, self.n_cold
        nb = nm + nc
        assert na + nb == num_experts

        def make(name, shape, dtype, loader):
            p = torch.nn.Parameter(torch.empty(*shape, dtype=dtype),
                                   requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, {"weight_loader": loader})

        rep = _replicate_loader()
        make("hyb_kind", (num_experts,), torch.int8, rep)

        # --- gate_up (w13): column-parallel, full 2*i_full -> local 2*ish,
        #     stored [gate_slice ; up_slice]. i_full passed to the loader. ---
        make("w13_codes", (nb, 1, 2 * ish, h // g), torch.int16,
             _gateup_loader(g_rank, T, i_full, axis=2))
        make("w13_codebooks", (1, entries, g), torch.float16, rep)
        make("w13_scales", (nb, 2 * ish), torch.float16,
             _gateup_loader(g_rank, T, i_full, axis=1))

        # --- down (w2): row-parallel, shard the I input dim to ish ---
        make("w2m_codes", (nm, 2, h, ish // g), torch.int16,
             _rowk_loader(g_rank, T, axis=3))
        make("w2m_codebooks", (2, entries, g), torch.float16, rep)
        make("w2m_scales", (nm, h), torch.float16, rep)          # H unsharded
        make("w2c_codes", (nc, 1, h, ish // g), torch.int16,
             _rowk_loader(g_rank, T, axis=3))
        make("w2c_codebooks", (1, entries, g), torch.float16, rep)
        make("w2c_scales", (nc, h), torch.float16, rep)

        if na > 0:
            make("nvfp4_w13_packed", (na, 2 * ish, h // 2), torch.uint8,
                 _gateup_loader(g_rank, T, i_full, axis=1))
            make("nvfp4_w13_bscale", (na, 2 * ish, h // 16), torch.uint8,
                 _gateup_loader(g_rank, T, i_full, axis=1))
            make("nvfp4_w13_scale2", (na, 2), torch.float32, rep)  # gate/up glob
            make("nvfp4_w2_packed", (na, h, ish // 2), torch.uint8,
                 _rowk_loader(g_rank, T, axis=2))
            make("nvfp4_w2_bscale", (na, h, ish // 16), torch.uint8,
                 _rowk_loader(g_rank, T, axis=2))
            make("nvfp4_w2_scale2", (na, 1), torch.float32, rep)

    # NOTE: no apply() override. The base apply produces the row-parallel
    # PARTIAL routed output (summed over this rank's I-shard). The MoE runner's
    # _maybe_reduce_final_output (moe_runner.py:458) all-reduces the combined
    # routed+shared output once at tp_size>1 (our method sets no moe_kernel, so
    # _fused_output_is_reduced is False and the late reduce fires). Adding an
    # all_reduce here would double-reduce.
