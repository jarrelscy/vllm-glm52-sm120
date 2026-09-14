# Restored-prefill input and sampling correction

The captured failing batch has four scheduled tokens per request. The newly
restored requests have computed 8191 of 8192 prompt tokens. Their first input
was overwritten with prompt token 8190, instead of preserving token 8191.
The actual captured token IDs prove this for all three newly admitted requests;
see `captured_causal_evidence.json`. The lone request schedules one token and
avoids the overwrite. Independent cache-byte checks passed before forwarding.

The combine correction preserves the prepared final prompt token when the extra
scheduled positions are speculative padding, while still writing valid zero
candidate tokens into those positions. It retains the existing FULL graph shape.

Padded new-prefill positions are not real draft proposals. The sampler marks
these requests using existing CPU prefill and draft-count metadata. They emit
exactly one token from the first target distribution. Greedy sampling uses its
argmax; probabilistic sampling uses target-only recovery without subtracting
stale draft logits. Ordinary decode passes no mask and compiles the flag away.

The final actual GPU combine kernel passed all eight fixtures, including stale
padding, shuffled state indices, partial prefill, ongoing decode and graph replay.
The sampler passed 18 combinations of temperatures 0, 0.7 and 1, random/NaN/absent
draft logits, and block verification on/off. New-prefill outputs match an
independent one-logit target-only reference; ordinary request outputs match the
original sampler. All cases passed CUDA graph replay. These are kernel tests;
full serving replay results are recorded below.

All three baseline modules were copied from the stopped serving container and
hash-verified against the source used to create the candidates.

## Uniform decode graph eligibility

The input and sampler corrections alone did not fix concurrent restoration.
Captured inputs were correct, but new requests' first target logits already
differed under FULL execution. In a controlled same-boot comparison, selecting
PIECEWISE for the same padded admission restored every reference token;
returning to FULL reproduced the failure. The exact stale captured tensor has
not been identified. This isolates an execution-path problem, not failed KV
transfer: independent resident-cache byte comparisons passed.

The production correction excludes any prefilling request from uniform decode
graph eligibility before DP dispatch and input/attention preparation. Ordinary
decode and dummy graph capture retain their existing descriptors. Fifteen CPU
tests cover slot reuse, resumed requests, all-new batches and dispatch ordering.

With all diagnostics disabled, four restored 8192-token requests each reproduced
all 512 reference output tokens, at 328.55 aggregate steady decode tokens/s.
Eight requests repeated the same four prompts twice: all 4096 output tokens
matched, at 427.40 aggregate tokens/s. One, three and five concurrent requests
also matched 128 reference tokens each. These are fixed replay workloads,
not universal throughput estimates. `production_replay.json` retains comparisons,
cache-hit checks, common-window throughput and speculative acceptance.

Prompt extensions of two and four tokens matched serial references. Extensions
of one and eight tokens did not all match across batch sizes. These comparisons
cross pre-existing projection geometry/dispatch boundaries; for example M8
uses native P4 while M32 exceeds the configured native limit and uses BF16 GEMM.
This is a possible explanation, not a proven cause for every difference. The
replay evidence does not establish universal batch-invariant output or claim
that every cache-related discrepancy is fixed.

Scoped helper/test hooks passed. Checking the touched model runner also reports
inherited optional-member mypy errors and an existing torch.cuda.synchronize
lint violation; those unrelated lines were not changed semantically.
