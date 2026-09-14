# Native route sorting: clean microbenchmark qualification

Experimental; **not qualified for rollout**. Live prefill improved, but pooled
eight-stream decode measurements did not establish non-regression. Sorting
remains OFF and the previous production image is retained. See the live
qualification notes in this directory for all results.
This changes only the order in which independent native routes execute.
Weights, activation P4 planes, projection splits, arithmetic, and final
ordered routing reduction remain unchanged. Results are scattered back to
the original route slots. Bitwise equivalence is demonstrated for the cases
below; this is not a proof of universal equivalence or serving nonregression.

## Measurement

Actual layer-3 TP-rank-3 expert weights on GPU1, RTX PRO 6000 Blackwell Max-Q.
The serving container was fully stopped and GPU1 explicitly reserved for this
clean repeat. No earlier run overlapping server restoration is included.
Each arm uses identical synthetic activations and routing weights. Mixed
routing samples top-8 uniformly over 256 experts; all-hot/all-cold controls
repeat eight corresponding experts. Tiny-hot controls use 256 hot routes
and grouped cold experts for all remaining routes.

Six alternating-order pairs measure synchronized whole-call wall-clock time.
The table selects the upper-middle sample of each six-sample series. Includes
routing, counts transferred to CPU, sorting, gathers, both projections,
temporary cold reconstruction, scatter, and final route reduction. Excludes
TP communication. These are complete mixed-layer microbenchmarks, not serving
throughput. Raw samples are in [qualification_micro.json](qualification_micro.json).

| Routes | Tokens | Compact baseline (ms) | Sorted (ms) | Ratio |
| --- | ---: | ---: | ---: | ---: |
| mixed | 2048 | 18.136 | 16.658 | 1.089× |
| mixed | 4096 | 30.396 | 26.297 | 1.156× |
| all hot | 2048 | 31.126 | 30.165 | 1.032× |
| all hot | 4096 | 62.288 | 59.991 | 1.038× |
| all cold | 2048 | 3.478 | 3.480 | 0.999× |
| all cold | 4096 | 6.227 | 6.231 | 0.999× |
| tiny hot | 2048 | 3.934 | 3.938 | 0.999× |
| tiny hot | 4096 | 6.716 | 6.716 | 1.000× |

All eight outputs matched using BF16 `int16` views, including signed-zero bits.
The independent mixed-expert reconstruction oracle measured relative L2
0.01524%, unchanged from the existing helper. This reference error belongs
to the existing arithmetic; the sorted-versus-original output difference was
exactly zero. Guarded all-cold and tiny-hot controls differed in timing by
at most 0.12%, within measurement noise.

## Opt-in control

The existing grouped-prefill eligibility remains at least 2048 input tokens,
with its original one-GiB routed-buffer limit and CUDA-capture exclusion.
Within compact native processing, sort only split segments with at least
512 remaining routes. Original tiny-tail split segmentation is preserved.
The guard uses the existing route-index shape, without another GPU count or
host synchronization.

- `VLLM_ARVQ_SORT_NATIVE_PREFILL=0` or unset: original compact order.
- `VLLM_ARVQ_SORT_NATIVE_PREFILL=1`: sorted order, without filesystem checks.
- `VLLM_ARVQ_SORT_NATIVE_PREFILL=toggle`: diagnostic marker
  `/dev/shm/vllm_arvq_sort_native_prefill_on` selects sorted order.

Keep the marker state fixed throughout the entire request on every TP rank.
Sorting requires the existing grouped and compact paths to be enabled.
No context limit, request limit, cache configuration, or memory budget is
reduced. The sort adds temporary route-index/key storage; it adds no weight
cache. The helper/dispatch suite passed 23 CPU tests; two CUDA-only decoder
tests were skipped during that CPU check.
