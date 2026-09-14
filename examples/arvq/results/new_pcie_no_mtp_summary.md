# Four-way serving results

| Profile | 128-prompt decodeTPS | 4096-prompt decodeTPS | Normalized128 estimate | Normalized4096 estimate |
| --- | ---: | ---: | ---: | ---: |
| new_pcie_mtp_off | 50.439 | — | — | — |

| Profile | Prompttarget | Runs | TTFTms mean | Emitted/step | Stepms proxy | Draftacceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| new_pcie_mtp_off | 128 | 3 | 233.429 | 1.000 | 19.827 | — |

Measured decodeTPS is arithmetic mean of per-request token-ID-verified rates, excluding first emissionbatch.

MTPstep duration is pooled server requestdecode time / pooled draftrequest steps, not GPU kernel latency.

MTPoff assumes one forward per token after firstemission, and uses serverdecode/postfirsttokens.

NormalizedTPS is a counter-derived estimate at original pooled1+accepted/drafts; it is not a measured forced-acceptance rerun.

Naturalprofile MTPoff usesV1 and MTPon usesV2; original/new matchedpairs share runner andFULL_AND_PIECEWISE. On/off comparisons therefore include a runner difference.

Throughput on repeatedpangram input can reflect nearly100% MTPacceptance; do not generalize to realistic workload quality.
