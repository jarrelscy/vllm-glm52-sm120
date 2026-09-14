---
library_name: vllm
base_model: jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m
tags:
- arvq
- nvfp4
- experimental
- work-in-progress
---
# GLM-5.3 Vision NVFP4–ARVQ hybrid

**Work in progress: untuned research checkpoint, not accuracy-ready.** Locally built and running on a custom SM120 vLLM backend. Cold MoE experts use additive residual vector quantization (ARVQ) with FP4-constrained codebooks; hot experts retain NVFP4. Vision and MTP tensors are preserved. **Rotation is retired; production and active tuning use unrotated ARVQ.**

**Publication status, September 14, 2026:** the complete checkpoint is available on the development host, but the full Hugging Face weight upload is still in progress. The benchmarks below use that complete local checkpoint. Metadata and `rotation-prototype-v1` samples alone are not a runnable model. Before downloading/running, verify that every shard referenced by `model.safetensors.index.json` is present. The rotation samples are historical experiments, not replacement model shards.

## Kernel and vLLM code

Standard upstream vLLM does **not** support this format. Use the matching fork:

- [ARVQ SM120 branch](https://github.com/jarrelscy/vllm-glm52-sm120/tree/arvq-hybrid-sm120)
- [CUDA kernels and build scripts](https://github.com/jarrelscy/vllm-glm52-sm120/tree/arvq-hybrid-sm120/vllm/model_executor/layers/quantization/arvq): `hybrid.cu`, `dense.cu`, and `prefill.cu`
- [Hybrid loader and dispatch](https://github.com/jarrelscy/vllm-glm52-sm120/blob/arvq-hybrid-sm120/vllm/model_executor/layers/quantization/nvfp4_arvq_hybrid.py)
- [Paired dense P4 implementation](https://github.com/jarrelscy/vllm-glm52-sm120/blob/arvq-hybrid-sm120/vllm/model_executor/layers/quantization/nvfp4_p4_linear.py)
- [Grouped/compact prefill implementation](https://github.com/jarrelscy/vllm-glm52-sm120/blob/arvq-hybrid-sm120/vllm/model_executor/layers/quantization/nvfp4_arvq_prefill.py)
- [Conversion tools](https://github.com/jarrelscy/vllm-glm52-sm120/tree/arvq-hybrid-sm120/examples/arvq/convert)

The measured deployment and raw results are recorded at code revision [`460457d2a`](https://github.com/jarrelscy/vllm-glm52-sm120/tree/460457d2a). The standalone optimized Compose override was added in [`43a92d728`](https://github.com/jarrelscy/vllm-glm52-sm120/tree/43a92d728).

## How to run

On the configured development host:

```bash
cd /home/jarrelscy/homeassistant
./switch.sh glm5.3-arvq
```

This selects the optimized hybrid with MTP, paired dense FP4 execution, grouped cold prefill, route compaction, LMCache, the full 1,048,576-token position limit, and eight request slots.

To use four slots instead:

```bash
MAX_NUM_SEQS=4 ARVQ_CAPTURE_SIZES='[1,2,4,8,16]' ./switch.sh glm5.3-arvq
``` The OpenAI-compatible API is at `http://localhost:8001/v1`, with model name `jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid`. Health is at `http://localhost:8001/health`; use the host's configured API key for inference.

For a separate deployment, the fork contains a standalone Compose profile:

```bash
git clone --branch arvq-hybrid-sm120 https://github.com/jarrelscy/vllm-glm52-sm120.git
cd vllm-glm52-sm120
export ARVQ_MODEL_DIR=/absolute/path/to/GLM-5.3-Vision-NVFP4-ARVQ-hybrid
export PARALLEL=tp4-1m-mtp

docker compose -f examples/arvq/compose.yaml \
  -f examples/arvq/compose.optimized.yaml up -d --build

curl --fail http://localhost:8001/health
```

**Build prerequisite:** [`Dockerfile.arvq`](https://github.com/jarrelscy/vllm-glm52-sm120/blob/arvq-hybrid-sm120/Dockerfile.arvq) layers this fork and its kernels over the compatible local base image `glm52-vision-sm120:latest`. That base supplies the compiled vLLM/SM120 and vision dependencies; this is not yet a self-contained public-image installation. The standalone Compose configuration has been validated, while the serving measurements use the [host deployment override](https://github.com/jarrelscy/vllm-glm52-sm120/tree/arvq-hybrid-sm120/examples/arvq/host). A new host needs the compatible base image, Docker GPU support, and the complete checkpoint before running the command.

The measured hardware is **4 × NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GiB each (SM120), PCIe without NVLink**. The speed measurements below used TP4 + DCP4, three MTP draft tokens, four concurrent requests, 4096 batched tokens, CUDA graph sizes `[1,2,4,8,16]`, memory utilization 0.96, and a 950,000-token request limit. **Current deployment:** the full **1,048,576-token** model position limit, **eight request slots**, LMCache ON, utilization **0.94**, and graph sizes `[1,2,4,8,16,32]` have been validated at startup and with bounded serving tests. The shared GPU KV pool is **1,073,920 tokens** across all requests, not a million tokens for each of eight simultaneous requests. Input and generated output both count toward the request limit; configured capacity does not establish full-length accuracy or a completed million-token request test.

The optimized override enables:

```text
VLLM_ENABLE_NVFP4_P4_O_PROJ=1
VLLM_NVFP4_P4_PAIRED=1
VLLM_NVFP4_P4_MAX_TOKENS=16
VLLM_ARVQ_GROUPED_PREFILL=1
VLLM_ARVQ_COMPACT_PREFILL=1
MAXLEN=1048576
MAX_NUM_SEQS=8
ENABLE_LMCACHE=1
UTIL=0.94
```

Set `PARALLEL=tp4-1m` before launching to disable MTP. Paired dense dispatch is fixed during CUDA graph capture; change its settings by restarting, not by changing a live marker. Prefill options must be enabled before startup memory profiling.

## Measured decode throughput

Current full-position/eight-slot profile, **LMCache ON**:

| Simultaneous streams | Total output tokens/s |
| --- | ---: |
| 1 | **136.2** |
| 4 | **277.3** |
| 8 | **344.9** |

Single-stream decode excluding initial latency measured **144.9 tokens/s**. The full-position four-slot reference with LMCache OFF measured 136.3/280.2 total tokens/s at one/four streams. The eight-slot configuration preserved those rates within about 1%; this comparison also changes LMCache and memory reservation, so it does not isolate the request-slot limit. All drafts were accepted on these short benchmark inputs. [Current configuration and raw results](https://github.com/jarrelscy/vllm-glm52-sm120/blob/67684eeac/examples/arvq/results/context_concurrency/NOTES.md).

Earlier paired-kernel comparison at 950k context, four slots, and LMCache OFF:

| Simultaneous streams | Previous total output tokens/s | Optimized total output tokens/s |
| --- | ---: | ---: |
| 1 | 133.4 | **136.1** |
| 2 | 192.7 | **200.5** |
| 4 | 264.2 | **274.1** |

**Earlier single-stream decode excluding initial latency: 144.8 tokens/s.** Total throughput above includes request startup/first-token latency and divides completed output tokens by the full batch wall time; it does not sum per-request rates.

Each stream used a 136-token repetitive prompt and generated 512 tokens, with one warmup and two measured batches per concurrency. **Every draft was accepted: four emitted tokens per speculative step.** These are high-acceptance synthetic-workload numbers, not a production acceptance forecast. Previous/current decode comparisons span server boots; some output sequences differed, so they do not establish identical model behavior or general accuracy preservation.

An earlier dense-P4 no-MTP run measured **53.28 decode tokens/s**. No-MTP was not rerun for this latest paired/compact profile; do not treat that historical measurement as a new result.

## Measured prefill throughput

Cold-cache requests, one generated output token, from the earlier 950k/four-slot/LMCache-OFF speed round:

| Input tokens | Latest request latency | Effective input tokens/s |
| --- | ---: | ---: |
| 1,024 | **1.905 s** | **537** |
| 4,096 | **4.165 s** | **984** |
| 8,192 | **8.571 s** | **956** |

Effective input throughput is input tokens divided by HTTP request wall time, including the first output token; it is not a pure GPU-prefill timing. Each length used one warmup per mode and three alternating measured pairs with identical input token IDs and a unique cache salt. The 8192-token case repeats the saved 4096-token input twice; these are bounded synthetic inputs.

Before grouped prefill and compaction, the 4096/8192-token requests took **7.192/15.595 seconds**. Latest latency is **4.165/8.571 seconds**, approximately **1.73×/1.82× faster across rounds**. Compaction alone was isolated in the final same-boot A/B: **4.688 → 4.165 seconds** and **9.724 → 8.571 seconds**, or **12.6–13.5% more input throughput**. The 1024-token control remains below the 2048-token grouped-prefill threshold and was unchanged within that A/B.

[Raw results, methodology, output comparisons, and the fresh decode profile](https://github.com/jarrelscy/vllm-glm52-sm120/blob/460457d2a/examples/arvq/results/prefill/PAIRED_COMPACT_NOTES.md) are available alongside the [speed inventory](https://github.com/jarrelscy/vllm-glm52-sm120/blob/460457d2a/examples/arvq/SPEED_INVENTORY.md).

## LMCache persistence

LMCache is enabled in the current full-position profile, alongside vLLM GPU prefix caching. The image contains the DCP4/multi-KV-group-aware [LMCache fork](https://github.com/jarrelscy/LMCache/tree/glm52-dcp-dsa), version `0.5.2.dev51`, revision `b339be5a7fd091109479e998c8716084d60fc4f9`. Stock LMCache was not used for these results.

The configuration uses V3 GPU connector, 256-token chunks, SHA256-CBOR hashing, CPU storage capped at 24 GiB per worker, and disk storage configured at 100 GiB per worker. The development host uses the dedicated `/data/lmcache/glm5.3-arvq` directory. Worker limits are not a single shared global quota. GPU utilization is reduced to 0.94 to leave room for LMCache staging and runtime workspaces.

A bounded persistence test stored a fresh **8192-token prompt**, stopped the server, and replayed the exact prompt as the **first generation after a clean restart**, generating 32 output tokens in each case:

| Request | End-to-end time |
| --- | ---: |
| Cold, no external/GPU cache hits | **8.284 s** |
| Restored after restart | **0.528 s** |

This was a **15.68× request-latency improvement** on one cache-reuse test, not a decode-throughput multiplier. The restore recorded **8191 external-cache hits and zero GPU-prefix-cache hits**; the last token is recomputed for logits. All four ranks retrieved their 2048-token shards. The 128 persisted chunks contained 448,331,776 data bytes, and the completion text was byte-identical. A full-million-token restore was not tested in this round.

[Persistence proof and per-rank evidence](https://github.com/jarrelscy/vllm-glm52-sm120/tree/arvq-hybrid-sm120/examples/arvq/results/lmcache) are published with the code.

CPU/disk persistence can avoid re-prefilling reusable context, but it does not turn the shared GPU pool into eight simultaneous million-token windows. Cache reuse must match the checkpoint and supported TP4/DCP4 topology; use a fresh cache namespace when changing weights.

## How the FP4 MMA tile is used

The native instruction is SM120 `mma.sync ... m16n8k64`: it multiplies a **16×64 weight tile** by a **64×8 activation tile** and accumulates a **16×8 FP32 output tile**. Here the 16 rows are output channels; the eight columns are independent activation representations. A single routed token would otherwise use just one of those columns. MTP provides several token positions, but MoE routing can send them to different experts, so they cannot automatically share one weight tile.

### Four residual activation planes per token

Instead of leaving most columns idle, the packer represents each activation vector as four successively refined FP4 planes. Let `r0 = x`. At plane `p`, quantize the current residual into FP4 E2M1 values with a block scale, decode that plane as `dp`, then compute:

```text
r[p+1] = 16 * (r[p] - d[p])
x ≈ d[0] + d[1]/16 + d[2]/256 + d[3]/4096
```

Each plane has its own scale per 16 activation values; the implementation chooses a bounded power-of-two scale representable in E4M3. Multiplying the remaining residual by 16 before the next quantization makes the smaller correction usable on the FP4 grid. These are four representations of the **same token**, not four speculative tokens and not four different weights.

For a single routed token, the activation operand uses:

```text
MMA column:   0      1      2      3      4   5   6   7
Contents:    d0     d1     d2     d3      0   0   0   0
```

One MMA computes all four `W*d[p]` columns simultaneously. The FP32 epilogue reconstructs the result with the same factors `1, 1/16, 1/256, 1/4096`. **Four planes do not require four times as many MMA instructions:** the instruction already computes eight columns. Packing four planes, loading their scales, and combining outputs still have real costs; only the MMA instruction count stays unchanged for a fixed weight tile and K step.

This reduces activation-quantization error. It does **not** undo errors already present in quantized weights, change cold/hot bits per weight, or involve a Hadamard rotation. The concrete implementation is [`pack_planes` and `hybrid_kernel`](https://github.com/jarrelscy/vllm-glm52-sm120/blob/arvq-hybrid-sm120/vllm/model_executor/layers/quantization/arvq/hybrid.cu).

### Two tokens fill all eight columns in dense projections

Dense attention-output projections use the same weight matrix for every token. The paired kernel can therefore assign four columns to token A and four to token B:

```text
MMA column:   0      1      2      3      4      5      6      7
Contents:   A.d0   A.d1   A.d2   A.d3   B.d0   B.d1   B.d2   B.d3
```

The weight tile is loaded once and one MMA produces both tokens' four-plane products. The epilogue reduces each group of four independently. An odd final token is guarded; single-token execution retains the original path. This pairing is implemented for dense `o_proj`, **not arbitrary MoE routes with different expert IDs**. See [`nvfp4_dense_paired_kernel`](https://github.com/jarrelscy/vllm-glm52-sm120/blob/arvq-hybrid-sm120/vllm/model_executor/layers/quantization/arvq/dense.cu).

### Additive cold codebooks stay on native FP4 MMA

For ARVQ, a cold weight vector is the scaled sum of two indexed FP4 codebook vectors. Rather than adding them into a higher-precision weight matrix during native decode, the kernel uses linearity:

```text
(W0 + W1) * d[p] = W0 * d[p] + W1 * d[p]
```

Packed indices select FP4 fragments from the two shared codebooks. Two native FP4 MMAs accumulate into the same FP32 output tile, one for each codebook component. Each MMA already covers all four activation planes. Thus the cold path uses **two MMAs per K tile**, while the hot NVFP4 path uses **one**; these are tile-level counts, not instruction counts for a complete projection or model layer.

### Prefill uses a different reuse opportunity

For long eligible prefills, many tokens visit the same expert. The current cold prefill path groups those routes, reconstructs one expert projection temporarily in FP16, and uses GEMM with FP32 outputs. That grouped cold path does **not** use the native P4 MMA trick. Hot routes and low-count cold routes retain native P4 execution, and compaction removes routes already handled by grouped GEMM. This distinction is part of why prefill and decode use different schedules while reading the same compressed checkpoint.

## Format, optimization scope, and accuracy status

Each eight-weight cold vector stores an 8-bit first index and a 7-bit residual index into two shared FP4 E2M1 codebooks of sizes 256 and 128. FP8 E4M3 scales per 128 weights give **1.9375 bits/weight** before the small shared dictionaries and global scale; complete cold tensors remain **below 2 bits/weight**. Hot experts remain **4.5-bit NVFP4**.

The native cold path uses two FP4 MMA instructions per K tile; hot NVFP4 uses one. Both retain **four residual activation planes**, combined with weights 1, 1/16, 1/256, and 1/4096. The paired dense kernel places two token positions into the eight MMA columns, retaining all four planes per token. Rotation is not used.

For eligible prefill batches, cold routes are grouped by expert, decoded into temporary FP16 matrices, and multiplied with FP32 outputs. No resident full-precision expert cache is added. Compaction removes already-grouped routes from the remaining native launches. Grouped prefill starts at 2048 tokens and bounds the routed output buffer to 1 GiB. These execution changes do not rewrite the checkpoint.

The speed profile also enables experimental load-time NVFP4 quantization of **target attention output projections (`self_attn.o_proj`, layers 0–77)**. The checkpoint retains their original BF16 payloads. MTP draft weights, vision tensors, shared experts, and other attention projections are not additionally quantized by that option. Vision configuration, processor files, and 335 vision/projector tensors remain present; image-input quality was not re-evaluated in this speed round.

All 75 cold MoE layers were transcoded from decoded AQLM dictionaries, **not quantized from the original BF16 donor with activation calibration**. Additional unweighted dictionary relative L2 error ranges from **34.21% to 35.83%** over 150 projections. This is not perplexity or task accuracy. Tuning and broader quality evaluation remain outstanding. Native FP4 and temporary-FP16/BF16 execution paths also differ numerically; speed validation is not an accuracy certification.

See `arvq_build_report.json` and `arvq_fit_provenance.json` for conversion provenance, and `SOURCE_MODEL_CARD.md` for the preserved source model card.
