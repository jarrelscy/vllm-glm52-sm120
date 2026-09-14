# Measured NVFP4–ARVQ hybrid layer prototype

`hybrid.cu` dispatches each routed slot to native NVFP4 or additive RVQ256+128 using the same FP4 MMA path. Both branches consume four residual activation planes. Cold weights require two MMA operations per K64 tile (one per codebook); hot weights require one. Dynamic activation scales and weighted plane reduction recover precision from otherwise unused MMA columns.

## Actual data and scope

`data.py` loads actual GLM-5.3 1M checkpoint layer 3, TP4 rank 3: 195 cold experts, 61 hot experts. Hot weights use their actual nibble payload, per 16 E4M3 block scales and separate global gate/up scales; down has its own global scale. Cold weights use `scale_experiment/rvq_layer3_rank3.safetensors`, including its serialized global scales. No hot weights are reconstructed from random synthetic nibbles. The cold weights are a lossy transcode of AQLM, not original donor weights.

The benchmark uses physical GPU 3 and synthetic random FP16 input vectors, with synthetic normalized routing weights. The routes select actual model expert IDs: six cold and two hot per token. Unique mode uses distinct experts across tokens; overlap mode shares the same expert IDs across all tokens. Gate/up keeps each token's own input even when experts overlap. Down uses each token/expert's own SwiGLU activation.

The entire measured CUDA graph chain is:

`input slot expansion → four-plane activation pack → mixed NVFP4/ARVQ gate+up → FP16 SwiGLU → four-plane pack → mixed NVFP4/ARVQ down → weighted routed reduction`.

Baseline uses the existing V4 fused AQLM+NVFP4 kernels with the same source checkpoint, routes, input, SwiGLU semantics and weighted reduction. All allocations in graphs use replay-stable buffers. A graph contains 48 layer invocations, repeated four times/sample; seven event samples are collected after warmup. Warm repeats the same routes. Rotating changes real expert IDs across the resident 195 cold/61 hot pool, covering far more than L2. It does not time a cache flush or duplicate the whole checkpoint 48 times. These are one-rank one-layer timings; TP all-reduce, real routing computation, attention, shared expert, MTP control and serving are excluded. Clocks were not pinned.

## Full-chain timing

| Tokens | Routing | New warm µs | New rotating µs | Existing warm µs | Existing rotating µs | Rotating speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | unique | 34.25 | 41.03 | 80.56 | 83.04 | 2.02× |
| 4 | unique | 70.85 | 86.74 | 257.61 | 279.98 | 3.23× |
| 4 | overlap | 71.08 | 82.09 | 260.43 | 272.22 | 3.32× |
| 5 | unique | 91.40 | 109.97 | 341.91 | 352.58 | 3.21× |
| 5 | overlap | 84.24 | 97.29 | 322.43 | 334.10 | 3.43× |

## Numerical validation and accuracy limits

Every selected slot is checked, including nonzero cold and hot local IDs. The independent oracle reconstructs weights from the source code indices and learned codebook translation (cold), or natural native FP4 bytes, block scales and separate global scales (hot). It reconstructs the four activation planes independently in Torch. Native output matches this oracle to ~3e-7 relative L2 for gate/up and ~1e-7 for down. This exercises the real expert-ID mapping, scale layout and global gate/up scale distinction.

The activation-only full MLP comparison fixes the NEW decoded weights and compares original FP16 inputs/intermediates with the four-plane path. The AQLM full MLP comparison separately includes the additional cold weight conversion error.

| Tokens | Routing | Activation-only output relative L2 | New versus decoded AQLM output relative L2 | Existing versus decoded AQLM output relative L2 |
| ---: | --- | ---: | ---: | ---: |
| 1 | unique | 0.000250 | 0.481698 | 0.000617 |
| 4 | unique | 0.000364 | 0.420167 | 0.000671 |
| 4 | overlap | 0.000417 | 0.396345 | 0.000677 |
| 5 | unique | 0.000328 | 0.361098 | 0.000768 |
| 5 | overlap | 0.000430 | 0.415647 | 0.000681 |

The roughly 36–48% full MLP output difference on synthetic inputs is substantial. Four activation planes fix activation precision loss; they do not fix the current cold weight codebook approximation. These measurements establish executable kernels and speed, not acceptable model quality, perplexity or tokens/second. No model accuracy benchmark or server integration was performed.

Cold payload plus 128-weight block scales is 1.9375 bpw, with small codebook/global overhead below 2 bpw at these real expert counts. Hot nibble plus block-scale storage is 4.5 bpw; this checkpoint additionally carries tiny per-expert global-scale metadata (two float32 gate/up values and one down). Global metadata must remain explicitly accounted for if 4.5 bpw is interpreted as a strict all-in cap.

Artifacts: `bench_hybrid.py`, `data.py`, `hybrid.cu`, `hybrid.so`, `results/hybrid.json`, `results_run.log`. Rerun: `docker exec -e CUDA_VISIBLE_DEVICES=3 cold-format-lab /opt/vllm/.venv/bin/python /lab/hybrid_experiment/bench_hybrid.py`. No serving files changed.
