# GLM-5.3 ARVQ MTP acceptance repair

**Status:** The acceptance defect is repaired and the fast image is selected in
Docker Compose with MTP3. The complete requested definition of done is **not
met**: the fast profile does not pass every temperature-zero spec-on/spec-off
identity fixture. Do not label it fully lossless-certified.

The V2 draft loader rebound the MTP sparse indexer to the target model's
`topk_indices_buffer`, but did not rebind the attention backend's reference.
The backend implementation is a plain object, absent from `named_modules()`.
Thus the indexer wrote selected indices into one tensor while attention read
its old tensor. Rebinding `module.impl.topk_indices_buffer` alongside the
module references restores the producer/consumer invariant before compilation
and CUDA graph capture. No per-step work is added.

This is a runtime loader bug, not a checkpoint-scale repair. The MTP `eh_proj`
is BF16 without an activation input scale. MTP expert input-scale controls,
BF16 activation execution, and reference attention arithmetic did not restore
acceptance. Rejection tracing found zero false greedy rejections; the drafts
were actually wrong. Reference attention still read the stale index buffer.

The historical PCIe image's target did not expose the buffer as an attribute,
so the faulty sharing block was inactive. Commit `b1380cf74` exposed it while
adding pipeline-stage index handoff, activating the latent loader defect even
under TP4. The loader loop originates in `32f34d393`. Both September 16 fixes12
and September 15 invariant images have the exposed attribute, explaining why
switching between those images did not resolve acceptance. Neither the sparse
selector overflow fix nor the CUDA property-cache ABI fix needs reverting.

## Measured acceptance

Same synthetic requests, temperature zero, TP4/DCP4, MTP3. Before/after
Prometheus deltas were isolated to one successful request and matched output
token counts. These are single measured requests, not a workload-wide estimate.

| Workload | Before emitted/step | Fixed emitted/step | Before tok/s | Fixed tok/s |
|---|---:|---:|---:|---:|
| Raw repeated pangram, 512 output | 2.065 | 4.000 | 73.3 | 137.3 |
| Model-card pangram, 512 output | 2.462 | 4.000 | 87.5 | 138.5 |
| Counting, chat | 2.437 | 3.906 | 88.6 | 140.9 |
| Refrigerator prose, chat, 1024 output | 1.693 | 2.642 | 62.7 | 97.7 |
| Merge code, chat, 1024 output | 1.683 | 2.909 | 61.6 | 106.3 |

Throughput above includes request/prefill overhead; decode-only pangram speed
is approximately 148 tok/s.

Both pangram runs accepted 128/128 drafts at each of the three positions.
Fixed server decode time was 27.0–27.1 ms per draft step. Output token IDs for
both pangrams matched the broken deployment. Chat prose/code include reasoning
and reached their 1024-token limits; separate completed-answer gates are used
for coherence and code correctness. Those chat output trajectories are not
identical to the broken deployment and are not an identity claim.

## Validation artifacts

Host directory:
`/home/jarrelscy/glm52/dcp-groundtruth/task44/mtp-acceptance-fix-20260916`

- `baseline.json`, `buffer-fix.json`: full synthetic requests, responses and
  counter deltas.
- `runtime-fixed-rank0.json`: all four MTP index-buffer references, including
  the attention implementation, have the same data pointer.
- `gates-mtp-fixed.json`: completed prose/code/counting and a needle at depth
  60% in a 33,553-token prompt. Correct needle: MANGO-7429-LANTERN.
- CPU loader regression: 2 parametrized cases passed, including a plain-object
  attention implementation that must observe writes from the rebound indexer.
- Actual V2 GPU rejection kernel: greedy MTP3, stochastic temperatures 0.6 and
  1.0, and invalid-draft rejection passed. Used the existing rejection tests
  with vocabulary 256 (2,560 trials per sampling test) to fit alongside serving.
- The serving-default greedy-proposal stochastic-target GPU test passed at
  temperatures 0.6 and 1.0: 32,768 trials each, testing all four emitted positions
  with more than 1,000 observations per position. This exercises the actual V2
  rejection kernel with `draft_logits=None`.
- Ruff check, Ruff format check, and git diff whitespace check passed.

## Strict lossless-gate limitation

With the same V2 runner, TP4/DCP4 and target fast-path configuration but MTP
disabled, raw pangram, model-card pangram and counting token IDs match exactly.
Free-form refrigerator prose first diverges at zero-based output token 12;
merge code first diverges at token 11 (declaration ordering). Both remain
coherent/correct, but **the complete bitwise spec-on/spec-off gate did not pass**.
Do not present sampler correctness or three matching fixtures as proof of
universal target distribution/bitwise invariance. The loader patch does not
change target parameters or arithmetic and does not bypass rejection sampling.
Shape-dependent target arithmetic remains a separate suspected cause, consistent
with the earlier invariant-runtime audit; this experiment alone does not locate
its first differing operator.

The first spec-off long-context attempt failed during LMCache KV restoration
with a CUDA illegal address after reusing an MTP-on disk cache with a different
layer layout. The isolated-cache rerun passed. Compose now separates namespaces
by `PARALLEL`: `file:///lmcache/disk/glm53-mtp-buffer-${PARALLEL:-tp4-1m-mtp}`.
No existing cache files were deleted. Cache-layout compatibility across
speculative modes remains unverified.

The isolated-cache spec-off rerun passed the 33,553-token needle and matched
its output token IDs exactly. Five of six gate fixtures now match MTP-on,
including the merge code; free prose still differs. Crucially, the two spec-off
runs themselves differ on prose and code despite identical short prompts,
weights, seed and temperature. These prompts are shorter than an LMCache disk
chunk, so restoring different disk prefixes cannot explain those short-answer
differences. This establishes restart instability without speculation; it does
not establish a numerical error bound. `strict-identity-fast.json` records the
second spec-off comparison. The full strict identity requirement remains open.

Additional controlled comparisons did not close the gate:

| Configuration | Exact fixtures out of six | Remaining divergence |
|---|---:|---|
| Fast repair, isolated spec-off cache | 5 | Prose |
| Eager compiler, CUDA graphs | 5 | Prose |
| Minimum graph size four | 4 | Prose, model-card pangram |
| Uniform quantized arithmetic, native attention | 5 | Prose |
| Above plus selective deterministic ATen controls | 4 | Prose, code |

A further selective invariant-RMSNorm control matches raw pangram, code and
counting, but prose still differs (first difference token 19). Its step time
is about 31 ms. It is also excluded from the clean repair.

Each row compares MTP-on/off token IDs at temperature zero. All completed
33,553-token needle tests passed. Selective ATen controls were an experimental
image only and are not included in the fast repair. The full conservative
invariant configuration restores acceptance but costs about 63 ms/step; its
matched spec-off comparison passed all five short fixtures exactly (both
pangrams, prose, code, counting). The invariant MTP-on needle previously passed;
the matching invariant spec-off needle has not been run. This is evidence that
the stricter arithmetic configuration can resolve the observed short-fixture
drift, but it fails the requested throughput target.

Further controls narrowed the tradeoff without producing a configuration that
meets all requested gates and speed targets:

- Uniform quantized arithmetic, selective ATen controls, invariant normalization,
  native attention, and fixed NCCL settings (`Simple`, one channel, fixed
  all-reduce tree) passed all four matched short fixtures tested: raw pangram,
  prose, code, and counting. Step time was approximately 38 ms. This profile did
  not receive a matched six-fixture/long-needle gate.
- Switching that profile to NCCL `LL` measured 32.7 ms/step and matched the
  `Simple` spec-off outputs on those four fixtures. This is a cross-protocol
  observation, not a matched LL on/off certification.
- Removing the ATen overrides from the LL profile gave about 30.7 ms/step but
  failed prose identity again. The other five fixtures, including the needle,
  matched. Cold 33,553-token needle requests took 137–139 seconds, versus about
  19.6 seconds on the fast repair.
- Explicit invariant BF16 linear kernels did not change the normalization-only
  control's outputs or its prose mismatch; they increased step time to 51 ms.
- Invariant attention without the conservative collective settings still
  failed prose identity and cost about 45 ms/step.
- A fast-kernel control using PCIe one-shot all-reduce, DCP all-to-all, and ATen
  controls measured about 44 ms/step and 65.8 seconds for the needle. Its MTP-on
  tests completed; a matching spec-off test was not run because it missed the
  speed target. It was not promoted.

These results implicate target execution invariance, but do not locate or bound
its first differing operator. They do not establish that every discrepancy is
harmless rounding. Experimental changes are retained only as investigation
artifacts/images; the selected fast image contains just the loader repair on
top of the user's fixes-1-2 image.

## Selected deployment and remaining work

`/home/jarrelscy/homeassistant/docker-compose.glm53-arvq.yaml` selects
`glm53-arvq-sm120:fixes12-mtp-buffer-20260916`, with `NUM_SPEC=3` and the original
fast arithmetic configuration. Only the MTP buffer repair and cache-namespace
separation are selected. `switch.sh` was not used. Other services were not
restarted, and the vision-build checkout remains on `b12x-pcie-graft`.

The clean-image final deployment is healthy and checked by `final-fast-gates.json`
and `final-fast-benchmark.json` in the artifact directory. The in-container loader
SHA256 matches the committed source exactly:
`2abd51e4aeb82fcaccc6e1cacf871ff661601abb90a7c4238b4f94bacb3116f4`.

Final warmed single-stream measurements (same benchmark request bodies):

| Workload | Emitted/step | End-to-end tok/s | Decode ms/step |
|---|---:|---:|---:|
| Raw pangram | 4.000 | 136.9 | 27.07 |
| Model-card pangram | 4.000 | 138.2 | 27.15 |
| Counting | 3.906 | 141.4 | 26.67 |
| Prose, 1024 output | 2.874 | 106.0 | 26.85 |
| Code, 1024 output | 3.045 | 111.2 | 27.06 |

All measured request deltas contain exactly one successful request and the
expected generated-token count. Both pangrams accepted 128/128 at each position.
The raw output is periodic. The completed-answer gate has coherent two-paragraph
prose, exact counting, six passing list-merge cases, and the exact needle answer
at 33,553 prompt tokens (19.58 seconds).

The final restart's token-ID comparison against isolated-cache spec-off matches
four of six fixtures: both pangrams, counting, and the needle. Prose diverges at
output token 19 and code at token 11. Both answers remain coherent/correct, but
this is explicitly a failed strict identity gate (`strict-identity-final-fast.json`).
The earlier five-of-six result must not be substituted for this final comparison.
 No 89-task benchmark was
started. The 134K-token needle script was prepared but not run; the measured
long-context gate is 33,553 tokens.

The acceptance defect itself is resolved. The remaining requirement is a fast
configuration with matched target execution between spec-on and spec-off,
followed by the complete identity, stochastic-sampler, and semantic gates.
Existing sampler tests and passing pangrams are not a substitute for that work.

Local commits: `d663817f4` (loader repair and regression), `8ad26d757`
(default greedy-proposal stochastic-target test). Nothing was pushed.

## Reproducible image

Build from this checkout:

```sh
docker build -f docker/Dockerfile.glm53-mtp-buffer-fix \
  -t glm53-arvq-sm120:fixes12-mtp-buffer-20260916 .
```

The base is `glm53-arvq-sm120:fixes-1-2-only-20260916`, image ID
`sha256:61a4fdc085eff99276cc5516b8d151cf5e1bb56e4c8c008712cb5782407d49c7`.
Only the corrected Python loader is copied. No diagnostic module, experimental
backend, scale adjustment, or modified entrypoint is included. Target kernels,
weights, rejection sampler, and three-token speculation are unchanged.
