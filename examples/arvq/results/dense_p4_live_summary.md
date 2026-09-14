# Dense attention-output NVFP4 serving experiment

Converting only attention output projections in target layers 0–77 to NVFP4, with four residual activation planes for small native batches, increased no-MTP decode throughput from **50.439 to 53.276 tokens/s (+5.62%)**. Both profiles used one warmup and three measured requests with the same 136-token prompt, 512 output tokens, temperature 0, seed 173 and ignoreEOS. Returned token IDs matched usage, all counters were isolated, and no speculative drafts ran. New measured range was 53.260–53.287 tokens/s; mean TTFT was 233.641ms.

The eight fixed teacher-forced passages contain 541 scored tokens. Mean NLL changed from **2.150433 to 2.136026** (delta −0.014408); perplexity ratio was 0.985696. This small diagnostic showed no aggregate degradation. It does not establish task accuracy or an accuracy improvement. These long-prefill probes use BF16 multiplication with dequantized NVFP4 weights, so they diagnose the weight change; native one-token P4 activation error is not measured by this NLL test.

| Passage | Tokens | BF16 NLL | NVFP4 weight NLL | Delta |
| --- | ---: | ---: | ---: | ---: |
| 0 | 59 | 2.579402 | 2.563875 | -0.015527 |
| 1 | 75 | 1.759248 | 1.779269 | +0.020021 |
| 2 | 68 | 1.691404 | 1.660570 | -0.030834 |
| 3 | 82 | 1.621567 | 1.614109 | -0.007458 |
| 4 | 66 | 1.744331 | 1.724243 | -0.020088 |
| 5 | 56 | 3.455508 | 3.413775 | -0.041733 |
| 6 | 63 | 2.384078 | 2.372166 | -0.011912 |
| 7 | 72 | 2.395013 | 2.377535 | -0.017479 |

The current compiled graph contains exactly 78 `arvq_hybrid.dense_p4` calls, associated with output projections in target layers 0–77 and native maximum 4 tokens. The boot log reports 67.13GiB of model loading memory and 91.231s on rank0. This is the runner's load-time model allocation measurement, not total process VRAM or peak allocator reservation. No matched baseline memory reading is claimed here.

The experimental runtime flag was enabled and `PARALLEL=tp4-1m`. Original checkpoint files were not converted by this experiment; quantization happens at load time. Other dense and shared-expert weights are unchanged. Raw requests, paired token log-probabilities, safe configuration and compiled-method evidence accompany this summary.
