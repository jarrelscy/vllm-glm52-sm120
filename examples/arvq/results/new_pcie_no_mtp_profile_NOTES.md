# No-MTP decode bottlenecks

The measured no-MTP baseline is **50.439 tokens/s** over three requests, about **19.827ms per post-first-token step**. A separate bounded profile captured 136 input tokens and 64 output tokens on the V1 runner. CUDA launch correlations select exactly 63 single-token decode invocations and exclude prefill/JIT activity. All four ranks show similar attribution.

Dense projections are the largest identified target. This includes Inductor reduction-based GEMVs as well as cuBLAS/CUTLASS kernels; classifying only names containing `gemm` would hide the largest kernels under generic Triton work.

| Category | Summed GPU ms/token, rank0 | Share of summed GPU duration |
| --- | ---: | ---: |
| Dense GEMM | 7.434 | 37.65% |
| Sparse attention/indexer | 2.584 | 13.09% |
| NCCL communication | 1.983 | 10.04% |
| Elementwise/norm/reduce (all model) | 1.864 | 9.44% |
| B12X fused all-reduce/RMSNorm | 1.803 | 9.13% |
| ARVQ/NVFP4 fused GEMV | 1.776 | 8.99% |
| Other Triton | 1.214 | 6.15% |
| Residual activation packing | 0.374 | 1.89% |
| Other | 0.348 | 1.76% |
| ARVQ split reduction/scales | 0.338 | 1.71% |
| Memory copy/set | 0.028 | 0.14% |

The graph emits 3625 GPU events per token. Rank0 GPU busy-union time is 18.071ms per token within a 20.354ms trace window per token. Summed GPU durations overlap across streams; they cannot be added as independent latency savings. The remaining window gaps do not establish CPU overhead by themselves.

| Actual kernel | Calls/token | Summed ms/token | Confirmed generated-source shape |
| --- | ---: | ---: | --- |
| `triton_red_fused_mm_t_0` | 78 | 2.530 | N6144, K4096 |
| `triton_red_fused_mm_t_2` | 56 | 1.196 | N2624, K6144 |
| `triton_red_fused_mm_rms_norm_split_with_sizes_t_5` | 56 | 0.683 | N4096, K2048; includes normalization |
| cuBLAS internal GEMV, 75-call variant | 75 | 0.814 | Not inferred from kernel name |
| cuBLAS internal GEMV, single-call variant | 1 | 0.312 | Not inferred from kernel name |
| CUTLASS BF16 WMMA 128x2 TN | 75 | 0.513 | Not inferred from kernel name |

The first three dense kernels total 4.409ms per token and are immediate candidates for meaningful shape-matched microbenchmarks. Generated code locations are recorded in `new_pcie_no_mtp_dense_sources.json`. The custom hybrid math takes 1.776ms, with another 0.374ms activation packing and 0.338ms split reduction; optimizing only its math has a smaller whole-model ceiling.

During the profiled request, 100ms NVIDIA-SMI samples reported median SM clocks 2272–2287MHz, memory clock 13365MHz and GPU utilization 98%. Maximum temperatures were 78/65/75/73°C across ranks, with mean sampled board power 157.6/140.3/149.0/156.4W. These samples include prefill and decode and do not include throttle-reason counters. They do not establish a thermal bottleneck.

Profiler was stopped after the request. Raw compressed traces remain local under `/data/profiles/glm5.3-arvq`; compact all-rank category and top-kernel data, clock samples and source mappings accompany this report. No serving configuration was changed by this profiling task.
