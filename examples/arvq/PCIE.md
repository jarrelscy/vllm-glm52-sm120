# PCIe communication in the ARVQ serving path

The image includes the raw `b12x` module at commit
`9043b448622764a598969518d413b3fd8b3c0c07`. Its revision and license are retained
under `/usr/local/share/arvq-b12x`. No `b12x` distribution is installed: this
avoids automatic vLLM plugin entry points and dependency upgrades. The image
asserts that no `b12x` vLLM plugin entry point is registered.

Both measured communication paths are enabled in the default example:

- **Single-row fused one-shot:** all-reduce, residual add, and RMSNorm at eligible
  attention-output and combined routed-plus-shared MLP boundaries. The ceiling
  is exactly 12,288 bytes: one BF16 row at hidden size 6144. The collective is
  not issued per expert. TP4 remote push, 512 threads, and one CTA per row are
  explicitly selected; upstream defaults did not win on this machine.
- **Large-payload copy-engine DMA:** reduce-scatter/all-gather for TP all-reduces
  starting at 24 MiB. BF16 transport is explicit; FP8 transport is off.

Plain one-shot and two-shot all-reduce remain off. Intermediate payloads fall
through to the existing communicator, normally NCCL on this box. Disabling
plain one-shot with an explicit zero preserves the independent fused channel.
Four-token verification exceeds the fused ceiling and uses the ordinary path.
The example sets `compile_sizes: [1]`. vLLM first traces ordinary all-reduce and
normalization, then a compiler pass substitutes the fused operation only in the
singleton `[1,1]` compile range. Larger ranges retain ordinary compiler fusion.
A Python shape branch alone is insufficient because vLLM reuses the traced
graph across shapes. The byte ceiling is fixed when a graph is compiled;
restart after changing it.
DCP attention all-gathers and reduce-scatters are not redirected by this TP
all-reduce adapter.

The policy is explicit in `compose.yaml`. The equivalent additional override is
also provided for deployments that otherwise disable the fused path:

```bash
docker compose -f examples/arvq/compose.yaml \
  -f examples/arvq/compose.pcie-fused.yaml up -d --build
```

| Setting | Value |
| --- | --- |
| `VLLM_ENABLE_PCIE_ALLREDUCE` | `1` |
| `VLLM_GLM_PCIE_FUSED_AR_RMS` | `1` |
| `B12X_PCIE_TP4_REMOTE_PUSH` | `1` |
| `B12X_PCIE_FUSED_THREADS` | `512` |
| `B12X_PCIE_FUSED_CTAS_PER_ROW` | `1` |
| `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE` | `0` |
| `VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE` | `12288` |
| `VLLM_PCIE_DMA_MIN_BYTES` | `24MB` (24 MiB) |
| `VLLM_PCIE_DMA_FP8` | `off` |

## Measurements and limitations

Four-rank CUDA-graph tests on the production CuTe DSL 4.5.2 dependency version
qualified the tuned single-row path and BF16 DMA:

| Operation | NCCL comparison | b12x |
| --- | ---: | ---: |
| One-row all-reduce + residual add + RMSNorm | 15.532 µs | 12.410 µs |
| 24 MiB BF16 all-reduce | 934.71 µs | 728.44 µs |
| 48 MiB BF16 all-reduce | 1847.09 µs | 1464.62 µs |

Two-row and three-row fused operations still lost to NCCL, so the default
ceiling deliberately excludes them. The untuned fused geometry also lost at
one, four, five, and sixteen rows. No smaller DMA crossover is claimed. These
are collective microbenchmarks, not full-model speedups.

BF16 transport avoids FP8 wire quantization but does not guarantee bitwise
NCCL equivalence. Earlier checks measured relative L2 differences of about
0.0031 for DMA and 0.0037 for fused one-shot, with matching outputs across ranks.
FP32 inputs transported as BF16 likewise require numerical validation.

## Capture and model guards

The adapter preserves graph warmup/channel planning and capture-stream
ownership. Fused warmup preplans the channel and executes the eager operation.
The model only defers its MLP reduction when the shared and routed outputs use
the combined late-reduction path. Single-stage PP, TP greater than one, no
sequence-parallel MoE, and no auxiliary hidden-state taps are required for this
fusion. Unsupported cases keep the ordinary reduction and normalization path.

DMA buffers are sized from the scheduler and model configuration, including
FP32 capacity and the draft model. They consume additional GPU memory, so a
previous KV-cache fit cannot simply be assumed to hold after enabling DMA.
The example does not force vLLM's unrelated stock PCIe custom all-reduce on.
Do not pass `--disable-custom-all-reduce`: that disables the TP adapter gate too.

The standalone example uses the module pinned inside the image. If a local
Compose deployment bind-mounts a raw `b12x` checkout over it, verify that the
checkout's HEAD equals the recorded commit and its `b12x/` tree has no changes;
otherwise the image revision no longer describes the executing kernels.

CPU policy and graph-warmup tests are in
`tests/distributed/test_arvq_pcie_policy.py`. Live profiling should verify which
communication kernels actually execute before attributing an end-to-end gain.

## Live serving validation

The final singleton-range compiler pass measured **138.401 tokens/s** with MTP
(136 input tokens, 512 output, one measured request after one warmup), versus the
previous ARVQ 138.197 and original hybrid 97.163 (three-run mean). All drafts were
accepted, giving 4.0 emitted tokens per step. This confirms preserved decode
throughput, not a further full-model gain from these collective microbenchmarks.

A bounded profile of short, exactly-one-token and fresh 4212-token prompts
confirmed both paths on all four ranks: 225 fused oneshot kernels and 1920 peer
copies per rank, with DMA add/flag kernels. The short request retained exactly
3962 GPU events per target invocation. An earlier opaque fallback regression
was resolved by preserving compiler-visible larger ranges and rewriting only
the singleton range. See [live results](results/new_pcie_NOTES.md) and
[actual path evidence](results/new_pcie_range_path_evidence.json).
