# Dense NVFP4 with MTP

The dense attention-output NVFP4 build measured **141.661 tokens/s** over three requests after one warmup, versus the previous ARVQ communication build's **138.401 tokens/s** single measured request: **+2.36%**. All requests used the same 136-token repeated-pangram prompt, 512 output tokens, temperature 0, seed 173 and ignoreEOS.

| Measurement | Result |
| --- | ---: |
| Decode throughput, mean | 141.661 tokens/s |
| Measured range | 141.582–141.734 tokens/s |
| TTFT, mean | 255.171ms |
| Draft steps, total | 384 |
| Accepted / proposed draft tokens | 1152 / 1152 |
| Expected emitted tokens per step | 4.000 |
| Server decode time per draft-step proxy | 28.183ms |
| Normalized estimate at original 4 emitted tokens/step | 141.928 tokens/s |

All counter-isolation checks and returned-token-ID versus usage checks passed. The normalized estimate divides original expected emitted length by the new pooled server-time/step proxy; it is not a measured forced-acceptance rerun. The actual measured throughput excludes the first emitted token batch.

Target weights are identical to the dense-P4 no-MTP quality diagnostic and draft weights remain unchanged, so that diagnostic was not rerun. These repeated-pangram measurements have perfect draft acceptance and should not be generalized to typical serving traffic. The server remains running with dense NVFP4 enabled and MTP on.
