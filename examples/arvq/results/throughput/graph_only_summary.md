# MTP graph-coverage correction

Extending capture sizes from `[1,2,4]` to `[1,2,4,8,16]`, with the same image and native dense maximum of **4**, restores MTP concurrency scaling. Each case uses the same 136-token prompt and 512 output tokens per stream, one warmup and two measured concurrent batches.

| Streams | Previous total tokens/s | Expanded graphs total tokens/s | Per-stream decode tokens/s | Median request seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 133.71 | 133.34 | 141.78 | 3.840 |
| 2 | 72.73 | 184.81 | 99.41 | 5.521 |
| 4 | 139.91 | 265.35 | 73.34 | 7.691 |

Total throughput is actual completed output tokens divided by the concurrent batch wall span, including prefill and final completion. All expanded-graph runs accepted every draft and passed counter isolation. No weight or native-M8 change was included in this test. Single-stream throughput is essentially unchanged; graph coverage explains the major multi-stream regression.

The V2 runner's boot log reports graph capture completing in 21 seconds using 0.88GiB per rank. The earlier MTP allocation log was not retained, so no matched memory delta is asserted. Full compilation configuration is included in the raw report and capture logs are retained separately. Native eight-token execution is a subsequent, separate experiment.
