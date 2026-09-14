# dense_p4_mtp_off

Token counts come from final streamed usage, never text-chunk counts. DecodeTPS excludes the first token batch when returned token IDs match usage. Counters are server-wide; normalization requires exactly one matching request and no counter reset.

| Prompt target | Run | Actual output | TTFTms | DecodeTPS | Emitted/draft | Serverms/draft proxy | NormalizedTPS estimate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0 | 512 | 232.309 | 53.287 | — | — | — |
| 128 | 1 | 512 | 233.587 | 53.279 | — | — | — |
| 128 | 2 | 512 | 235.026 | 53.260 | — | — | — |

`1+accepted/drafts` is the fork’s expected emitted length per speculative request-step. Stop truncation, prefill output and non-speculative steps can make it differ from actual completion_tokens/drafts. Server decode time divided by draft count is a request-level latency proxy, not GPU kernel latency. Normalization changes acceptance arithmetically; it does not measure a rerun at forced acceptance.
