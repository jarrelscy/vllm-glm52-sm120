# Compact native routes after grouped cold prefill

This prototype removes already-grouped cold rows from the remaining native
P4 pack/GEMV launches. It retains hot experts and low-count cold experts on the
native path, scatters their results back to their original route positions,
and preserves the ordered FP32 routing reduction. It introduces no weight
conversion or change to grouped expert arithmetic.

## Complete mixed-layer microbenchmark

Actual layer-3 weights, TP rank-3 shard, RTX PRO 6000 Blackwell Max-Q GPU1.
Routing is uniform random top-8 over 256 experts; input activations and routing
weights are synthetic and identical for both implementations. The serving
container was stopped during this microbenchmark. Times include sorting,
CPU count synchronization, gather/scatter, temporary weight reconstruction,
all native/grouped math, and final routing reduction. TP communication is
excluded. These are whole-layer microbenchmarks, not serving throughput.

| Input tokens | Current grouped helper | With native compaction | Speedup |
| --- | ---: | ---: | ---: |
| 1024 | 15.24 ms | 13.95 ms | 1.092× |
| 2048 | 23.10 ms | 18.74 ms | 1.233× |
| 4096 | 39.17 ms | 30.57 ms | 1.281× |

Each row uses six paired whole-call wall-clock measurements with alternating
execution order and CUDA synchronization. The table selects the upper middle
sample of each six-sample series; all samples are retained in
[compact_micro.json](compact_micro.json). The 1024-token result is exploratory:
the production grouped-prefill threshold remains 2048.

Outputs were bitwise identical to the current grouped helper at all three
sizes. A separate mixed expert oracle reconstructed original encoded weights
and activation planes independently; relative output L2 error was 0.01524%,
matching the existing grouped helper. The prototype's local files are
`prefill_experiment/compact_micro.py` and `compact_helper.py`; production
integration preserves tiny final chunks' original split count and retains
both-negative routes so their defined zero outputs are initialized.

A follow-up using the installed production helper with its flag OFF/ON
confirmed 4096-token latency of 38.496 → 29.923 ms (1.2865×). Outputs
were bitwise identical; the independent oracle again measured 0.01524% L2.
It used six alternating pairs, with the same upper-middle summary convention.
Its random routes differ from the three-size sweep because only the
4096-token case was generated; both arms share identical inputs. See
[compact_production_micro.json](compact_production_micro.json).

## Diagnostic production integration

Within eligible eager grouped prefill only:

- `VLLM_ARVQ_COMPACT_PREFILL=0` or unset retains the previous route schedule.
- `VLLM_ARVQ_COMPACT_PREFILL=1` enables compaction without filesystem checks.
- `VLLM_ARVQ_COMPACT_PREFILL=toggle` checks marker-file presence at
  `/dev/shm/vllm_arvq_compact_prefill_on` for same-boot A/B diagnostics.

Keep the marker state unchanged throughout an entire request on every TP rank.
Compaction does not change the grouped-prefill eligibility threshold, scratch
budget, or capture exclusion. All-hot, all-cold, low-count, mixed, zero-route,
ordered-reduction, and tiny-tail split cases are covered by CPU tests. The
combined helper/dispatch suite passed 19 CPU tests; two CUDA decoder tests were
not rerun during that CPU-only integration check.

Same-boot serving A/B and decode-regression measurements are pending. The
microbenchmark improvement does not yet establish an end-to-end speedup.
