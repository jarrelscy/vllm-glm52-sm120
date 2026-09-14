# Final PCIe communication validation

The final singleton-range compiler pass serves at **138.401 tokens/s**, after one warmup (138.426). The measured request used 136 input tokens and 512 output, temperature 0, seed 173 and ignoreEOS. TTFT was 251.330 ms, with 4.0 emitted tokens per draft step and 28.846 ms per step. The previous ARVQ build measured 138.197; the original AQLM hybrid MTP-on baseline averaged 97.163 over three runs. This repeated-pangram workload accepts every draft; full task accuracy remains unmeasured.

All four final-build GPU traces prove both communication paths: each contains 225 `_FusedOneshotLaunch` kernels, 1920 peer-to-peer GPU copies, 960 DMA `_AddLaunch` kernels and 4160 DMA `_FlagLaunch` kernels. The bounded profile contained 136-input/64-output, 1-input/1-output and fresh 4212-input/32-output requests. Compiler logs also record the singleton range and two fused all-reduce/RMSNorm pairs; execution claims rely on GPU events rather than those logs alone.

The first 17 target decode invocations belong to the short request, before the singleton and long requests. Correlating their CUDA runtime launch IDs with GPU activities yields 67354 events, exactly 3962 per invocation, matching the original graph. Elementwise/norm/reduce cost is 2.558 ms per invocation. This bounded selection avoids mixing long-context attention into the decode comparison. Summed GPU durations overlap across streams and are not additive critical-path savings.

The intermediate 126.431 TPS regression came with 5678 GPU events per invocation and doubled elementwise work. Restoring the compiler-visible fallback recovered 138.069 TPS but bypassed singleton fusion. The final dedicated singleton compiler range preserves the recovered decode graph and restores observed small-message fusion. Detailed intermediate artifacts are retained for reproducibility.

The microbenchmarked collective gains do not imply a further whole-model decode gain: this all-accepted MTP workload verifies four tokens together, above the one-row fused ceiling. BF16 DMA applies to large prefills. Profile traces prove selection but are not an unprofiled prefill speed comparison.
