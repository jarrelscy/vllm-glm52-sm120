# TP4 fused remote push and copy-engine DMA

Both paths have a useful measured range on this four-GPU SM120 box, after correcting the one-token fused launch geometry. The recommended routing is:

- **One token, hidden size 6144:** fused remote push, capped at **12 KiB**, with `B12X_PCIE_TP4_REMOTE_PUSH=1`, `B12X_PCIE_FUSED_THREADS=512`, and `B12X_PCIE_FUSED_CTAS_PER_ROW=1`.
- **BF16 payloads at least 24 MiB:** copy-engine DMA, without compressed wire transport.
- **Intermediate sizes:** NCCL. Plain one-shot remains disabled.

These are collective microbench results, not measured end-to-end serving gains. The 24 MiB DMA floor is the smallest positive point tested, not a proven exact crossover.

## Production-toolchain confirmation

CUTLASS DSL 4.5.2, PyTorch 2.11.0+cu130, four RTX PRO 6000 Blackwell Max-Q GPUs:

| Operation | Tokens × 6144 | NCCL baseline µs | b12x µs | NCCL/b12x speed ratio |
| --- | ---: | ---: | ---: | ---: |
| Tuned fused AR + residual + RMSNorm | 1 | 15.532 | 12.410 | **1.252×** |
| Tuned fused AR + residual + RMSNorm | 2 | 16.691 | 31.490 | 0.530× |
| Tuned fused AR + residual + RMSNorm | 3 | 16.686 | 47.113 | 0.354× |
| BF16 DMA all-reduce, 24 MiB | 2048 | 934.711 | 728.443 | **1.283×** |
| BF16 DMA all-reduce, 48 MiB | 4096 | 1847.093 | 1464.615 | **1.261×** |

The default fused geometry was slower at every tested decode size. A bounded sweep of threads 128/512 and CTAs per row 1/2 rescued only the one-token case. With the winning geometry, two and three tokens remain slower; four tokens also lost in the sweep. The exact 12 KiB cap is therefore important.

| Default geometry, CUTLASS 4.6.2 | NCCL baseline µs | b12x µs |
| --- | ---: | ---: |
| One token | 15.48 | 18.02 |
| Four tokens | 16.50 | 63.07 |
| Five tokens | 17.11 | 80.21 |
| Sixteen tokens | 26.31 | 244.44 |

The initial CUTLASS 4.6.2 DMA results agree with production 4.5.2: 24 MiB was 934.79 versus 727.92 µs, and 48 MiB was 1846.79 versus 1463.53 µs.

## Method and correctness

The fused baseline calls production `vllm._custom_ops.fused_add_rms_norm` after NCCL all-reduce. The candidate calls `all_reduce_fused_add_rms_norm`; its actual resolved transport is asserted to be `stage_remote_push`. The TP4 remote-push flag defaults off upstream, so old benchmarks that omitted it did not qualify this path.

Both arms reset inputs and residuals inside every captured operation, including identical reset-copy overhead. DMA aliases input/output after the reset, retaining only two payload buffers. Eight operations are captured per graph. The initial 4.6.2 comparison and final tuned confirmation use 100 replays per sample and seven samples per arm, with balanced alternating A/B order. The 4.5.2 compatibility pass and geometry sweep use 30 replays and three samples. Every sample is the slowest rank; tables report sample medians. All relevant fused callables are prepared and warmed before the launcher cache is frozen; no cache misses occur during capture or replay.

Outputs are finite, nonzero, and bit-identical across all four ranks before and after repeated graph replay. Tuned fused output relative L2 difference against NCCL + production norm is approximately 0.0037–0.0039; residual difference is approximately 0.0032. BF16 DMA relative L2 difference against NCCL is approximately 0.00311, consistent with different BF16 reduction order. This is not bit-exact equivalence to NCCL. Maximum observed absolute difference is 0.0625. Peak DMA PyTorch allocation is 102.5 MB plus 75.5 MB raw IPC scratch, below 200 MB excluding CUDA context and small flag storage.

The full ARVQ server remained loaded but idle. These tests do not establish overlap benefits under concurrent expert compute. GPU identity, clocks, throttle flags, and memory snapshots are recorded. The initial run remained P1 with unchanged memory clocks and no throttle flags; SM clocks rose from 2272 MHz to 2317–2355 MHz. Clocks were not locked, so this is targeted diagnostic evidence, not formal fixed-clock release qualification.

## Provenance and reproduction

- b12x commit: `9043b448622764a598969518d413b3fd8b3c0c07`.
- Host worktree: `/home/jarrelscy/glm52/b12x-graft/b12x-src`; only the pre-existing `pyproject.toml` modification was present. Communication source hashes are recorded.
- `CUTE_DSL_ARCH=sm_120a`, `NCCL_P2P_LEVEL=SYS`, `NCCL_IB_DISABLE=1`.
- Script: `b12x_fused_bench.py`.
- Full evidence: `b12x_fused_bench_cutlass462.json`, `b12x_fused_bench_cutlass452.json`, four `b12x_fused_sweep_*.json` files, and `b12x_fused_tuned_confirm.json` in this directory. Raw logs remain in the local experiment directory.

Initial run in the existing CUTLASS 4.6.2 container:

```bash
docker exec -e NCCL_P2P_LEVEL=SYS -e NCCL_IB_DISABLE=1 \
  -e CUTE_DSL_ARCH=sm_120a kernel46-bench \
  /opt/vllm/.venv/bin/torchrun --standalone --nproc-per-node=4 \
  /graft/b12x_fused_bench.py \
  --output /graft/out/b12x_fused_bench_cutlass462.json
```

The production 4.5.2 pass uses the unchanged `cold-format-lab` image and a source-only b12x snapshot on `PYTHONPATH=/lab/b12x_src_snapshot`. No production packages or flags were changed by the microbench. With the branch's matching vLLM and b12x environment installed, reproduce the tuned confirmation from the repository root:

```bash
B12X_PCIE_FUSED_THREADS=512 B12X_PCIE_FUSED_CTAS_PER_ROW=1 \
  torchrun --standalone --nproc-per-node=4 examples/arvq/b12x_fused_bench.py \
  --output tuned.json --skip-dma --fused-tokens 1 2 3
```

For the default-geometry compatibility pass, omit the two geometry overrides
and use `--output compatibility.json --reps 3 --iters 30`. See
[the PCIe integration notes](../PCIE.md) for the runtime routing configuration.
