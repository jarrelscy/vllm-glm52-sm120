# Actual new-model MTP decode profile

The unprofiled headline is **138.197 actual decode tokens/s**,512completion tokens at128target/136actual prompttokens, after one warmup. TTFT249.533ms. All384of384drafttokens were accepted across128draftsteps, yielding4.0emitted tokens/step and28.890ms serverdecode/draftstep. OriginalMTP-on had97.163TPS and41.092ms/step at the same4.0emitted/step. The counter-normalized estimate is138.457TPS; it differs slightly from measuredTPS because the measured decode numerator excludes the first emitted token, while normalization uses theoretical emittedtokens/draftstep.

A separate `/start_profile` → one64token completion → `/stop_profile` capture succeeded; profiling is nowOFF and the server was left idle. Traces are22MBtotal compressed across fourranks, plus a tiny frontend trace. This is not the timing run. The profiler's active_iterations=5 does not cap this configuration: warmup=wait=0 causes the local fork to omit its scheduled profiler, so the request was explicitly bounded and stopped.

## Separating decode from prefill

The wholetrace includes large136token prefill batches, which substantially inflate MoE costs. Decode-only attribution uses CUDA correlationIDs: identify CUDA runtime launches inside CPU `execute_context_0(0)_generation_1(4)` ranges, then include the corresponding GPU kernels/memcpy/memset across all streams. It does not mistakenly assumeGPU execution occurs inside the CPUlaunch timeinterval. The independent geometry check finds2550fusedhybrid kernels, exactly17observed targetinvocations ×75layers ×2projections, allwith32routed slots; prefill kernels have1024or64slots and are excluded.

The request counters report16draftsteps/48accepted drafttokens/64emitted completiontokens, while the profiler contains17executed targetgeneration invocations (asynchronous scheduling can execute work beyond the successful draftcounter boundaries). The perinvocation table below divides by the observed17launches, not by16orbySSEchunks.

## Rank0 GPU attribution

| Category | SummedGPUms per observed target invocation | Share of summedGPUduration | Kernel events per invocation |
| --- | ---: | ---: | ---: |
| Dense GEMM | 9.186 | 34.68% | 840.0 |
| NCCL communication | 5.122 | 19.33% | 411.0 |
| ARVQ/NVFP4 fused GEMV | 4.905 | 18.52% | 150.0 |
| Elementwise/norm/reduce (all model) | 2.622 | 9.90% | 1174.0 |
| Sparse attention/indexer | 2.565 | 9.68% | 513.0 |
| Residual activation packing | 0.556 | 2.10% | 150.0 |
| ARVQ split reduction/scales | 0.394 | 1.49% | 150.0 |
| Other | 0.388 | 1.47% | 161.0 |
| Other Triton | 0.345 | 1.30% | 231.0 |
| Other MoE/expert GEMM | 0.229 | 0.86% | 75.0 |
| Memory copy/set | 0.178 | 0.67% | 107.0 |

Allfourranks show nearly identical category shares. RemainingGEMMs plusNCCL are now~54% of summedGPUduration; the fusedARVQ/NVFP4 GEMV itself is~18.5%. Activationpacking and splitreduction together are~3.6%. The elementwise/norm/reduction category includes the ENTIRE model, not only ARVQ glue, so it is not legitimate to claim that removing MoE glue would save its full2.62ms. Similarly, remainingGEMMs include routing/shared/MTP computations as well as dense attention projections; kernelname classification is not a perfect module decomposition.

These are summed GPU event durations, not criticalpath latency or additive predicted savings: streams can overlap, and collectives may include waiting on other ranks. CPUranges are inclusive and can overlap; cudaEventSynchronize time is primarily host waiting for GPU execution rather than independent CPU work. Real speed changes require an unprofiled serving A/B.

Artifacts: `new_mtp_on_profile_capture.json` records the request,usage,tracepaths and posthoccounterdelta; `new_mtp_on_profile_analysis.json` is wholecapture; `new_mtp_on_decode_profile.json` is correlation-filtered decode. `analyze_runtime.py --decode-only` reproduces the latter from the fourworkertracefiles under `/data/profiles/glm5.3-arvq`.
