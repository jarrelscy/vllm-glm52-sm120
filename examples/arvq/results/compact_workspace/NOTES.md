# Live memory qualification

**The candidate remains opt-in.** The memory target passed, but the pooled four-stream decode result was 1.59% lower. The default `paired-compact` profile was restored under the user's no-tradeoff requirement. Both serving arms used the same **8+7 checkpoint**; these results do not measure 8+8 serving speed or accuracy.

Both arms retained the full 1,048,576 context, eight sequences, TP4/DCP4, MTP, GPU utilization 0.94, LMCache CPU 24 GiB/disk 100 GiB, dense paired P4 cap 16, and graph sizes 1/2/4/8/16/32. The candidate enables `VLLM_SM120_COMPACT_WORKSPACE` and shares the target language-model head with MTP. Diagnostic `VLLM_DEBUG_WORKSPACE` records allocation sizes.

## Memory

| Measurement | Baseline | Candidate |
| --- | --- | --- |
| Available KV budget, rounded | 13.69 GiB/rank | 18.26 GiB/rank |
| Cache blocks | 4,196 | 5,596 |
| Shared global token capacity | 1,074,176 | 1,432,576 |
| Model-loading memory, rounded | 69.2 GiB/rank | 68.75 GiB/rank |
| Graph capture memory, rounded | 1.02–1.03 GiB/rank | 1.02–1.03 GiB/rank |

The exact block-rounded KV increase is **4,903,628,800 bytes/rank**. Subtracting the independently audited 8+8 storage delta projects **1,287,424–1,287,680 cache tokens**, assuming unchanged other allocations. This is a storage calculation, not an 8+8 model deployment or a full-million-token input test.

The active backend is `FLASHINFER_MLA_SPARSE_SM120`. The measured saving comes from bounding indexer scratch (about 4.125 GiB) and eliminating a duplicate MTP head (about 0.44 GiB). The separate FlashMLA BF16-workspace guard is inactive on this backend and does not explain this machine's saving. New indexer scratch is logged at 1,057 MiB on all four ranks. The candidate memory figures reproduced on its second boot.

All four intended runtime file hashes were checked. An initial incomplete candidate boot was stopped before generation because a host indexer mount shadowed the image. The corrected overlay preserves host indexer scheduling and changes only allocation bounds; the original host source remains untouched.

## Decode

Decode uses 136 actual prompt tokens and 512 actual output tokens, temperature 0/seed 173, and one warmup per concurrency. There are two initial measured batches per arm. An unresolved initial concurrent-speed difference prompted one additional baseline/candidate cycle with three batches each at C4/C8. All cohorts are retained.

| Concurrency | Initial baseline | Initial candidate | Repeat baseline | Repeat candidate | Pooled baseline | Pooled candidate | Pooled change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 144.952 | 145.042 | — | — | 144.952 | 145.042 | +0.06% |
| 4 | 334.861 | 324.588 | 321.782 | 319.969 | 327.014 | 321.816 | -1.59% |
| 8 | 422.743 | 412.469 | 408.888 | 409.719 | 414.430 | 410.819 | -0.87% |

Values are aggregate **decode tokens/s**. The common interval is `(latest first emission, earliest completion]`; actual token IDs emitted inside it are counted. TTFT and first emission batches are excluded. This is not a sum of per-request rates. Single-stream rate uses the equivalent post-first-emission interval.

The C8 comparison changes sign in the repeat; C4 remains lower, although its repeat difference narrows to 0.56%. Existing clock samples and output/cohort variation are preserved, not used to normalize away measured differences. No additional clock experiment or clock setting change was made. The evidence does not establish that allocation changes caused a slowdown, but it does not satisfy the strict rollout gate.

Initial matched output token sequences were identical in 14 of 26 request pairs. Existing cross-boot nondeterminism prevents claiming full-model bitwise equivalence; raw token IDs and sampling/acceptance counters are retained. Candidate initial acceptance was 100%; initial baseline C8 acceptance was approximately 99.90%.

## Prefill and memory smoke

Cold prefill uses fresh UUID actual-token prefixes and one output token. The primary rate divides actual input tokens by the isolated server `request_prefill_time_seconds_sum` delta, requiring count 1 and zero GPU/external cache hits. It includes scheduled-to-first-token server work, not only GPU kernels. Fresh prefixes can change expert routes.

| Actual input tokens | Baseline prefill tokens/s | Candidate prefill tokens/s | Change |
| --- | --- | --- | --- |
| 2,048 | 871.955 | 871.028 | -0.11% |
| 4,096 | 967.258 | 965.273 | -0.21% |
| 8,192 | 954.873 | 957.079 | +0.23% |

There are three measured 4K requests and two measured requests per edge length per arm, following one warmup per length. All were isolated and cache cold.

Eight simultaneous, independent UUID-prefixed 8,192-token contexts completed 65,536 input tokens and eight output tokens in 68.33 seconds, with zero GPU/external hits and no errors. This is a memory/correctness smoke, not a throughput benchmark. Sampled peak device usage was 96,634–96,636 MiB/rank, leaving about 1.22 GiB physical margin.

## Artifacts

`summary.json`, `repeat_summary.json`, `memory_result.json`, and `runtime_hashes.json` contain compact results and provenance. Large raw JSON and boot logs are compressed losslessly as `.gz`; `timing_logs.tar.gz` contains timing/clock logs. `raw_manifest.json` records original byte counts and SHA256 hashes. No generated output was edited, and publication files were checked against configured credentials. `final_restore.json` records the final live default profile after the separate GPU microbenchmark.
