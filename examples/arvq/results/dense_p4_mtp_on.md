# dense_p4_mtp_on

Token counts come from final streamed usage, never text-chunk counts. DecodeTPS excludes the first token batch when returned token IDs match usage. Counters are server-wide; normalization requires exactly one matching request and no counter reset.

| Prompt target | Run | Actual output | TTFTms | DecodeTPS | Emitted/draft | Serverms/draft proxy | NormalizedTPS estimate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0 | 512 | 255.063 | 141.734 | 4.000 | 28.167 | 142.010 |
| 128 | 1 | 512 | 254.748 | 141.666 | 4.000 | 28.184 | 141.925 |
| 128 | 2 | 512 | 255.701 | 141.582 | 4.000 | 28.199 | 141.850 |

`1+accepted/drafts` is the fork’s expected emitted length per speculative request-step. Stop truncation, prefill output and non-speculative steps can make it differ from actual completion_tokens/drafts. Server decode time divided by draft count is a request-level latency proxy, not GPU kernel latency. Normalization changes acceptance arithmetically; it does not measure a rerun at forced acceptance.
