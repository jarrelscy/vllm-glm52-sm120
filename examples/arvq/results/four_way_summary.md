# Four-way serving results

| Profile | 128-prompt decodeTPS | 4096-prompt decodeTPS | Normalized128 estimate | Normalized4096 estimate |
| --- | ---: | ---: | ---: | ---: |
| original_mtp_on_clean | 97.163 | 92.469 | — | — |
| original_mtp_off_clean | 44.349 | 44.553 | — | — |
| new_mtp_off_headline | 48.664 | — | — | — |
| new_mtp_on_headline | 138.197 | — | 138.457 | — |

| Profile | Prompttarget | Runs | TTFTms mean | Emitted/step | Stepms proxy | Draftacceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original_mtp_on_clean | 128 | 3 | 672.333 | 4.000 | 41.092 | 1.000 |
| original_mtp_on_clean | 4096 | 3 | 1456.065 | 3.954 | 42.519 | 0.985 |
| original_mtp_off_clean | 128 | 3 | 637.982 | 1.000 | 22.549 | — |
| original_mtp_off_clean | 4096 | 3 | 97.591 | 1.000 | 22.448 | — |
| new_mtp_off_headline | 128 | 1 | 234.436 | 1.000 | 20.550 | — |
| new_mtp_on_headline | 128 | 1 | 249.533 | 4.000 | 28.890 | 1.000 |

Measured decodeTPS is arithmetic mean of per-request token-ID-verified rates, excluding first emissionbatch.

MTPstep duration is pooled server requestdecode time / pooled draftrequest steps, not GPU kernel latency.

MTPoff assumes one forward per token after firstemission, and uses serverdecode/postfirsttokens.

NormalizedTPS is a counter-derived estimate at original pooled1+accepted/drafts; it is not a measured forced-acceptance rerun.

Naturalprofile MTPoff usesV1 and MTPon usesV2; original/new matchedpairs share runner andFULL_AND_PIECEWISE. On/off comparisons therefore include a runner difference.

Throughput on repeatedpangram input can reflect nearly100% MTPacceptance; do not generalize to realistic workload quality.
