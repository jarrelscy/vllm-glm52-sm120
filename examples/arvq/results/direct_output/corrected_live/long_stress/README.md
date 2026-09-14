# Corrected placement long-context stress

Two single-stream cases completed on the corrected placement image. The model
retained maximum context 1,048,576, max sequences 8, TP4/MTP, LMCache, and GPU
utilization setting 0.94. Exact image/configuration identifiers are in report.json.

| Prompt tokens | Cold prefill tokens/s | Warm decode tokens/s |
| ---: | ---: | ---: |
| 262,144 | 1801.24 | 141.04 |
| 524,288 | 1664.80 | 142.41 |

Health and request/metric validity checks passed, with no recorded request
errors. Both cold requests recorded zero GPU and external prefix-cache hits.
Warm decode reused the long prefix: GPU hits were 261,632/523,776 and external
hits 511 per request. Warm throughput is therefore not a cold-prefill measure.

These generated outputs were not compared to a numerical reference. Successful
completion and health are operational stress evidence, not proof of exact
long-context numerical equivalence. The earlier isolated C8 token discrepancy
remains unresolved, despite its clean repeat; no correctness promotion follows
from these long requests.

Raw metric/result JSON and compressed streamed events are retained. Prompt
archives are omitted to keep this evidence bounded. report.json also contains
a larger planned matrix: only the two completed cases above belong to this
stress run, and its pending cases must not be read as successful tests.
