# Experimental NVFP4 P4 attention output

This optional load-time transform quantizes only the target model's
`self_attn.o_proj` weights in layers 0–77. The default example leaves it off.
ARVQ expert weights, shared experts, other attention projections, vision, and
MTP draft weights retain their existing formats. The loader explicitly excludes
both `mtp_block` prefixes and layer indices beyond the serialized target range.

The original BF16 attention weights remain unchanged in the checkpoint. After
loading and TP slicing, the optional method quantizes each local matrix on CPU
into native FP4 fragments, one E4M3 scale per 16 weights, and one FP32 global
scale. Resident storage is 4.5 bits per weight plus that global scale
(4.50000127 bpw for the measured 6144×4096 shard). It releases the resident BF16
parameter; restarting without the override restores ordinary BF16 loading.

For one to eight tokens in the example profile, the custom operator packs activations into four FP4
planes and runs the existing native hot-expert MMA path with one weight expert.
The split count is eight for one token, two for two to four tokens, and one
for five to eight tokens. Larger
calls reconstruct a temporary BF16 matrix and use BF16 linear multiplication.
This fallback uses the quantized weights, so it does not restore original BF16
accuracy. Token dispatch stays inside an opaque custom operator so Dynamo does
not freeze a Python shape branch across graph sizes.

## Paired FP4 tile execution

`VLLM_NVFP4_P4_PAIRED=1` enables a dense-only kernel that places two token
positions into the eight MMA columns, retaining four residual activation planes
for each. One-token calls keep the original path. The native token cap is a
separate setting: `VLLM_NVFP4_P4_MAX_TOKENS=16` allows the tested paired path
through 16 positions; larger calls retain temporary BF16 reconstruction.
The library defaults remain paired OFF and native cap four. This process-level
flag must be set before CUDA graph capture; changing it requires a restart and
fresh capture.

Paired split counts are eight at two positions, four at three/four, and two at
five through sixteen. On the actual layer-3 attention-output matrix with a
297 MB rotating weight pool, complete packing, projection, reduction, and output
cast fell from 24.82 to 18.47 us at four positions and 39.82 to 25.39 us at eight.
At sixteen positions the production paired kernel took 36.55 us versus 45.75 us
for a fresh dequantize-plus-BF16-GEMM control. Communication is excluded.

The serialized and resident weight formats are unchanged. Odd/even position
counts, changed-activation CUDA graph replay, independent encoded-weight/P4
oracles, and the 17-position fallback were checked. At sixteen positions,
enabling native execution replaces the old BF16 fallback arithmetic. The bounded
serving check accepted all drafts and reached total throughput 136.1/200.5/274.1
tokens/s at one/two/four streams, versus approximately 133.4/192.7/264.2 before.
Each configuration used the repeated 136-token prompt and 512 output tokens per
request; these are cross-boot comparisons with two measured batches per stream
count. Single-stream decode excluding initial latency measured 144.8 tokens/s.
These kernel speedups are not end-to-end throughput multipliers or acceptance
estimates for ordinary production traffic.

## Run the bounded experiment

From the repository root:

```bash
docker compose -f examples/arvq/compose.yaml \
  -f examples/arvq/compose.dense-p4.yaml up -d --build
```

The override enables `VLLM_ENABLE_NVFP4_P4_O_PROJ=1`, sets
`VLLM_NVFP4_P4_MAX_TOKENS=8`, selects no-MTP TP4 by default, and captures graph
sizes `[1,2,4,8,16]`. It preserves the singleton compile range used by the
PCIe fusion pass. The original `[1,2,4]` limit disabled graphs for concurrent
MTP verification at 8/16 positions. Raising only this limit increased total
short-prompt throughput from 72.726 to 184.806 tokens/s at two streams and
139.908 to 265.347 at four streams. Capture reported 0.88 GiB per rank;
there is no preserved old capture-allocation measurement for a memory delta.
Enabling the measured eight-position native path on top of corrected graphs
raises two-stream total throughput further to 192.740 tokens/s (+4.29%).
The Python method's default remains four unless the profile opts into eight.
Setting `PARALLEL=tp4-1m-mtp` requests MTP, whose own projection weights remain
unchanged. The bounded MTP result below covers the repeated-pangram workload;
acceptance on realistic traffic needs separate measurement.

The image builds `arvq/dense.so` with `build_dense.sh`. Runtime library overrides
are `VLLM_ARVQ_KERNEL_LIB` for the native P4 kernels and
`VLLM_NVFP4_P4_DENSE_LIB` for weight reconstruction. The production base example
does not enable attention weight quantization.

## Microbench evidence

Measured on an RTX PRO 6000 Blackwell Max-Q, SM120, with the full server idle.
Weights are actual layer-3 checkpoint tensors sliced for TP4. Graph replay uses
rotating weight pools of at least 272 MiB, exceeding L2. Each result is the
median of five timing batches, each containing 20 graph replays. Quantization
happens before timing; activation casts, P4 packing, split reduction, global
scaling, and output conversion are included. Collectives are excluded.

| M=1 projection, local N×K | BF16 eager | Existing FP8 W8A16 | NVFP4 P4 |
| --- | ---: | ---: | ---: |
| Attention output, 6144×4096 | 33.75 µs | 23.04 µs | 17.76 µs |
| Shared gate/up, 1024×6144 | 10.12 µs | 16.10 µs | 12.58 µs |
| Shared down, 6144×512 | 5.93 µs | 9.50 µs | 9.61 µs |

Shared projections regressed and are excluded from the runtime transform.
FP8 serves as a comparison and exceeds the 4.5-bpw weight budget. The
`bf16_inductor` fields in the raw results refer to standalone default
`torch.compile`, which selected behavior similar to eager. They are **not** a
reproduction of the serving graph's tuned Inductor reductions. The serving
profile measured approximately 32.4 µs per attention-output reduction, close
to this microbench's BF16 baseline; end-to-end measurements remain necessary.

| Attention-output tokens | Best native P4 | Split | Dequantize + BF16 fallback |
| --- | ---: | ---: | ---: |
| 1 | 17.76 µs | 8 | — |
| 4 | 24.80 µs | 2 | 43.90 µs |
| 8 | 39.73 µs | 1 | 44.61 µs |
| 16 | 89.08 µs | 4 | 48.69 µs |
| 32 | 168.99 µs | 4 | 48.71 µs |
| 128 | — | — | 61.83 µs |

Weight reconstruction alone takes 25.67 µs. Native M8 was measured but is not
part of the selected threshold. Native M16/M32 lose to reconstruction plus
BF16 multiplication.

These are untuned, lossy weights: attention-output weight relative L2 error is
9.51%, with roughly 9–10% output relative L2 on random activations. These local
errors are not model accuracy measurements. Native FP4 arithmetic agrees with
an independent quantized oracle to roughly 3×10⁻⁷ relative L2. Weight
reconstruction matches the rounded BF16 oracle exactly, including subnormal
scale and FP4 edge cases. Seven CPU loader/dispatch tests and two GPU tests
cover reconstruction, custom-op compilation, and CUDA graph replay.

## End-to-end bounded result

Three measured no-MTP runs, after one warmup, generated 512 tokens each from
the same nominal 128-token prompt (136 tokens after chat formatting). Decode
throughput increased from **50.439 to 53.276 tokens/s**, a **5.62%** gain. The
three P4 runs ranged from 53.260 to 53.287 tokens/s; mean time to first token
was 233.64 ms. Throughput counts tokens emitted after the first stream event
over the corresponding decode interval. Both configurations retain the same
ARVQ expert and PCIe communication paths.

The compiled graph contains exactly 78 dense-P4 calls, covering target layers
0–77. Rank-0 model-load memory fell from 69.74 to 67.13 GiB; this is reported
model-load memory, not total process memory or KV-cache capacity.

With MTP enabled, three measured runs reached **141.661 tokens/s**, versus
138.401 tokens/s for the preceding communication build's single measured run
(+2.36%). All 1152 proposed draft tokens were accepted across 384 steps,
corresponding to four emitted tokens/step. Mean time to first token was
255.17 ms. This repetitive prompt has unusually high acceptance and does not
establish expected production MTP throughput. See the
[MTP summary](results/dense_p4_mtp_on_summary.md) and
[speed inventory](SPEED_INVENTORY.md) for the full progression.

Eight fixed authored passages, totaling 541 scored tokens, gave teacher-forced
NLL 2.150433 with BF16 attention weights and 2.136026 with quantized weights
(delta −0.014408; perplexity ratio 0.985696). This small diagnostic found no
aggregate loss but does **not** establish general accuracy improvement or
preservation. It runs through large-prefill weight reconstruction and therefore
does not test the native P4 activation path's model-level quality. The feature
remains opt-in pending broader accuracy and workload testing.

## Reproduce the microbenches

Scripts live in `dense/`; original measured JSON is in `results/dense/`. The
multi-token JSON's historical `scope` string says M1; its explicit `tokens`
field is authoritative. Published scripts correct that label without changing
benchmark arithmetic. Run in the built CUDA image with this repository and the
checkpoint mounted, using the image's Python environment:

```bash
export CUDA_VISIBLE_DEVICES=1
export TORCH_CUDA_ARCH_LIST=12.0
export DENSE_MODEL_DIR=/models/arvq
export DENSE_OUTPUT_DIR=/tmp/dense-results
export VLLM_ARVQ_KERNEL_LIB=/opt/vllm/vllm/model_executor/layers/quantization/arvq/hybrid.so
export VLLM_NVFP4_P4_DENSE_LIB=/opt/vllm/vllm/model_executor/layers/quantization/arvq/dense.so
mkdir -p "$DENSE_OUTPUT_DIR"
/opt/vllm/.venv/bin/python examples/arvq/dense/dense_micro.py
/opt/vllm/.venv/bin/python examples/arvq/dense/nvfp4_dense.py
/opt/vllm/.venv/bin/python examples/arvq/dense/nvfp4_dense.py \
  --tokens 4 --oproj-only --output "$DENSE_OUTPUT_DIR/nvfp4_dense_m4.json"
/opt/vllm/.venv/bin/python examples/arvq/dense/test_dequant.py
```

Repeat the multi-token command with 8, 16, or 32 and a distinct output path.
Use an idle GPU window with enough memory for the rotating pool and temporary
buffers. The checkpoint may be the original AQLM hybrid or the published ARVQ
hybrid: both preserve these BF16 source projections identically.

The serving measurements use `bench_serving.py` in the parent example folder
with `--prompt-lengths 128 --max-tokens 512 --runs 3 --warmup-runs 1 --ignore-eos`.
To repeat the fixed diagnostic against each server configuration:

```bash
/opt/vllm/.venv/bin/python examples/arvq/dense/quality_probe.py \
  --label dense_p4_quality --dense-flag 1 \
  --output /tmp/dense_p4_quality.json
```

Set `OPENAI_API_KEY` when the endpoint requires authentication. Use
`--dense-flag 0` and a distinct output for the BF16 baseline. This argument
records provenance; it does not modify the server configuration.
