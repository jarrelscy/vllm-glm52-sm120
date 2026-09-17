# Independent proposal randomness for probabilistic MTP

The Model Runner V2 probabilistic draft sampler reused the target sampler's
Gumbel noise at the same token position. After rejecting a proposal, verification
sampled `max(p-q, 0)` with that same noise. Conditioning the proposal noise on
rejection biases the replacement distribution, despite a correct p/q test.

The fix gives probabilistic proposals a separate Gumbel stream through
`is_drafting=True` and a position salt of `1 << 30`. Target and residual sampling
keep their existing stream. Greedy drafts and temperature-zero argmax behavior
are unchanged. This adapts the independent-draft-noise approach already present
in [upstream gumbel.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py)
as inspected on 2026-09-17, without importing upstream's other sampler changes.

Changed runtime files:

- `vllm/v1/worker/gpu/sample/gumbel.py`
- `vllm/v1/worker/gpu/spec_decode/speculator.py`

`tests/v1/worker/test_gpu_spec_decode_distribution.py` calls the real draft
sampling method and Triton rejection kernels as one composition. It checks
temperature 0.7 and 1.0, one and three draft steps, fp32/fp64 Gumbel arithmetic,
non-contiguous request-state mappings, identical distributions, temperature zero,
and one-hot greedy proposals.

On the deployed SM120 GPU, the unpatched image fails the first distribution test
with a 17.57-sigma discrepancy. All 11 cases pass with the patch. Pre-commit and
ruff checks also pass. This establishes a regression fix for this sampling
defect; it is not a full checkpoint or inference-kernel correctness audit.

Build with `docker/Dockerfile.glm53-mtp-rng-fix` using
`glm53-arvq-sm120:fixes12-mtp-buffer-20260916` as the base. The resulting deployment
tag is `glm53-arvq-sm120:fixes12-mtp-buffer-rng-20260917`. This adds no tensor
allocation or separate GPU launch; the position salt is applied inside the
existing Gumbel kernel.

The earlier Claude Code trace had already degenerated under greedy drafting
before probabilistic drafting was enabled. This bug does not explain that
original onset. The active Terminus-2 run using the unpatched probabilistic
sampler was stopped and retained as invalidated, to be rerun on the fixed image.
