# Experimental DMA owner transport

This optional path preserves the current compressed candidate bytes and canonical
selector. It sends each query's candidates to one owner rank through the existing
B12X copy engine, then all-gathers only the selected IDs. The source defaults OFF.
These results establish the tested numerical cases, not strict decode-performance
nonregression or a general accuracy guarantee.

## Eligibility and coordination

`VLLM_EXPERIMENT_DCP_BYTEPACK_OWNER=1` requires the existing qualified bytepack
path, C1 prefill, 512–4096 query rows, TP4/DCP4 with identical rank order,
interleave 1, canonical top 2048, FP32 scores, and the reviewed SM120/NCCL setup.
Global context selects the six/seven-byte wire format uniformly. Unsupported
calls retain compressed AG or the outer original fallback. With the current
256 MiB global-context planner, contexts above 128K normally have fewer than 512
query rows and therefore fall back. This is not a 512K speedup claim.

The adapter borrows the initialized BF16 TP DMA ring's existing 144 MiB scratch;
it allocates no persistent CUDA storage, stream, event, or counter. READY, DATA
and DONE generation flags occupy slots 192–203, outside the existing ring's used
slots. The selected IDs leave the callback in independent storage. Per-call one
CPU/Gloo MIN status makes every rank choose the same fallback or transport;
startup readiness is separately agreed once.

This relies on serialized model compute and the original channel's stream
ordering. DCP overlap uses separate NCCL buffers. Arbitrary concurrent private
use of the TP DMA channel is unsupported. Changed-stream/unavailable-handle
conditions fall back collectively; an exception after protocol entry is fatal,
not a rank-local retry. Capture bypasses host collectives. The later CPU-tested
cleanup moves the capture query after the cheap ineligible guard; the live
measurements here precede that cleanup.

## Microbench evidence

[Raw four-rank results](micro/rank0.json) cover 48 fixtures per rank: both wire
widths, rows 496/512/768/1023/4032/4096, balanced scores, ties, nonfinite scores and
empty candidates. Every final ID matched bitwise; every peak allocation was
lower. The timing includes packing, readiness votes, peer-copy protocol,
unchanged selector, ID all-gather and final scatter. It is not full-model timing.

Rows 496 had regressions and are excluded. Rows 512 won all eight fixture medians.
One initial width 7 balanced timing was an outlier; its original sample remains
in the raw data. A [warm ten-pair repeat](micro_repeat/summary.json) won every
pair: baseline 0.70413 ms versus owner 0.56406 ms median, 1.248×, with 28,311,552 fewer
peak allocated bytes. Rows 4096 fixture medians were2.12–2.55× faster.

[Lifecycle checks](lifecycle/rank0.json) passed on all ranks for same-stream
reuse, one-rank changed-stream fallback, one-rank unavailable channel,
one-rank invalid width, and actual capture/replay bypass. Raw AR→owner→AR
checks with delayed producers/consumers are retained in the micro JSON files.

[Model shadow checks](shadow.json) logged 88 positive comparisons: 22 indexer
layers × 4 ranks, width 6. Shadow mode compares each eligible call and always
returns the baseline IDs; logs are deduplicated by layer/width. Shadow timing
must not be used as performance evidence.

## Live observations

The [live summary](live_summary.json) records metric definitions and links to
bounded raw requests. MAXLEN remained 1,048,576, max sequences 8, and LMCache
remained enabled; [boot provenance](live/boot_provenance.json) records the state.

| Observation | Tokens/s |
| --- | ---: |
| First cold 8K request, including first-use overhead | 1254.39 |
| Second independent cold 8K request | 2050.11 |
| Cold 128K request | 1911.36 |
| C1 decode, excluding first emitted token/time | 142.56 |
| C4 common-active decode, three runs | 316.12 / 322.33 / 315.73 |
| C8 common-active decode, three runs | 400.00 / 423.45 / 407.59 |

All cold-prefill requests had zero GPU/external prefix hits. Every fixed
concurrent request produced 512 IDs identical to its reference. Concurrent rates
count actual emitted tokens only during the interval when all streams remain
active; they are not sums of individual stream rates. The observed C4/C8 spread
does not establish strict performance nonregression. These runs do not establish
3,000-token/s prefill. First-use, process and input differences preclude treating
the historical profile delta as a controlled transport-only speedup.

`provenance.json` records source locations and hashes. Raw token streams and
metrics are retained alongside summaries; local source paths are provenance,
not portable launch instructions.
