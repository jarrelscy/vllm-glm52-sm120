# new_pcie_long_prefill

Token counts come from final streamed usage, never text-chunk counts. DecodeTPS excludes the first token batch when returned token IDs match usage. Counters are server-wide; normalization requires exactly one matching request and no counter reset.

| Prompt target | Run | Actual output | TTFTms | DecodeTPS | Emitted/draft | Serverms/draft proxy | NormalizedTPS estimate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 0 | 32 | 3384.234 | 114.767 | 4.000 | 33.909 | — |
| 4096 | 0 | 32 | 3876.407 | 114.218 | 4.000 | 34.143 | — |

`1+accepted/drafts` is the fork’s expected emitted length per speculative request-step. Stop truncation, prefill output and non-speculative steps can make it differ from actual completion_tokens/drafts. Server decode time divided by draft count is a request-level latency proxy, not GPU kernel latency. Normalization changes acceptance arithmetically; it does not measure a rerun at forced acceptance.
