# Apparent C8 sorting regression: cause investigation

The original comparison was confounded by temperature-associated boost-clock drift under the power limit, plus one unusually synchronized OFF batch. The saved evidence does not establish a decode slowdown caused by sorting. The sorting branch is unreachable for this short workload, and no candidate was deployed by this investigation.

**Unchanged-code control:** ten consecutive C8 batches on the stable paired-compact image fell from **442.619 to 410.022 decode tokens/s**. GPU0 active clock averages fell from **1966 to 1759 MHz**, while temperature rose from **62.5 to 90.8°C**. This 7.36% drift exceeds the earlier 1.80% pooled OFF/ON gap. Software power-cap flags were active during load; software and hardware thermal-slowdown flags were not active. Therefore “temperature-associated boost/power-limit drift” is supported; an asserted thermal-slowdown flag is not.

**The hot OFF outlier:** its eight streams started within 0.494 ms and produced identical 512-token sequences in lockstep. All other original batches started over 1.555–1.727 seconds and had two full output sequences, with 3–5 distinct concurrent last-four-token states. The outlier had one concurrent state in all 142 sampled interior observations. Better expert/weight locality is plausible, but expert IDs and L2 behavior were not traced.

**Counting and cache controls:** trimming one second from both ends of each original common window retained the pooled difference; the fast final OFF batch remained 427.088 tokens/s. Every original batch had one-token initial emissions, zero GPU prefix-cache hits, zero external cache hits and 1088 cache queries. Thus larger initial MTP bursts, counted TTFT, cache-hit asymmetry and window edges do not explain the gap.

Three high OFF values explain the elevated pooled mean: the first two occurred during boost-clock warmup, and the final one had lockstep output alignment. The other three OFF values average 410.980 versus all six ON values at 411.562. This sensitivity calculation identifies influential observations; none were removed from the original qualification result.

## Request-arrival intervention

Four balanced alternating pairs compared eight independent HTTP requests with one completion request containing eight copies of the same prompt (`n=1`). Every choice had 512 verified token IDs, aggregate usage was 4096, server completed-request count was eight, and all draft proposals were accepted.

| Submission | Mean steady decode tokens/s | First-emission spread |
| --- | ---: | --- |
| Eight independent HTTP requests | 419.968 | 1.555–1.709 s |
| One eight-prompt HTTP request | 419.519 | 1.225–1.704 s |

The prompt-array request **did not force simultaneous GPU admission**: every batch still had staggered first emissions and two output sequences. Consequently this is a failed alignment control, not a causal demonstration of aligned-versus-staggered GPU execution. Submission styles differed by only 0.11% in mean throughput.

GPU0 cooled to 65°C during CPU preparation before this test. Active clocks again fell from roughly 1911 to 1769 MHz as temperature rose from 75.7 to 91°C; throughput fell from 436.941 to 411.797 tokens/s. The preparation gap and warming trend are retained explicitly rather than claiming a fully heat-soaked comparison.

## Method and limits

Primary decode throughput counts actual token IDs emitted in `(latest first emission, earliest final emission]` across streams and divides by that common interval. It excludes TTFT and every initial token batch; it does not sum whole-request rates. CSV clock summaries discard the first 40 interleaved rows (approximately two seconds at four GPUs and 200 ms sampling) before averaging GPU0. Raw timestamps permit alternative alignment.

All requests used the unchanged 1,048,576-token limit, eight sequence slots, MTP, LMCache, utilization 0.94 and existing weight/activation precision. No clocks, power limits, model settings or cache configuration were changed. The arrival test reused a 136-token prompt with 512 requested outputs per stream. The eight-prompt API path supports separate choice indices; the diagnostic reconstructs per-choice timelines and checks their sum against authoritative final usage.

Raw `arrival.json.gz` and `baseline_heatsoak.json.gz` are lossless gzip copies, with original byte sizes and SHA256 hashes in `raw_manifest.json`. `timing_logs.tar.gz` preserves clock samples and logs. Compact summaries and original-output alignment analysis remain plain JSON. The reproducer is `examples/arvq/sorting_cause_experiment/compare_arrival.py`.
