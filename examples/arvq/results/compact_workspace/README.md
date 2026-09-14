# Compact attention workspace and MTP head sharing

The candidate retains TP4 + DCP4, three MTP drafts, eight sequence slots,
1,048,576 maximum positions, utilization 0.94, LMCache and the existing weights.
`VLLM_SM120_COMPACT_WORKSPACE=1` changes scratch allocation only. It skips the
unused BF16 prefill buffer at fewer than 32 local heads and bounds indexer
gather storage by the maximum admitted contexts, including draft lookahead.
Attention scheduling and arithmetic are unchanged. The multimodal V2 MTP
loader also now aliases the identical target output head through its language
model wrapper. The draft's separate normalization remains intact.

This machine selects `FLASHINFER_MLA_SPARSE_SM120`, so the FlashMLA BF16
buffer guard does not contribute to these measured savings. The gain comes
from about 4.125 GiB less indexer scratch and 0.443 GiB from sharing the MTP
head. Graph memory remains approximately 1.02–1.03 GiB per GPU.

Startup measurements on four 96 GiB SM120 GPUs:

| Measurement | Original profile | Candidate |
| --- | ---: | ---: |
| Available KV memory per GPU | 13.69 GiB | 18.26 GiB |
| Shared KV blocks | 4,196 | 5,596 |
| Shared KV token capacity | 1,074,176 | 1,432,576 |

The gain is exactly 1,400 blocks × 3,502,592 bytes = 4,903,628,800 bytes
(4.567 GiB) per GPU. With unchanged remaining runtime overhead, optional 8+8
weights would consume another 1,983,360,000 bytes per GPU, leaving
**1,287,424–1,287,680 shared KV tokens**. The range reflects the unreported
sub-block remainder. This exceeds the full 1,048,576 request limit. The loaded
checkpoint is still 8+7; these are capacity projections for 8+8, not a measured
8+8 serving run or a completed million-token request.

The candidate remains **opt-in only**. It did not pass the strict concurrent
decode no-regression gate, so the default launch profile is unchanged.

The first cold-prefill comparison measured 871.955 → 871.028 tokens/s at 2K,
967.258 → 965.273 at 4K, and 954.873 → 957.079 at 8K. All requests had zero
GPU-prefix and external-cache hits. Eight independent concurrent 8K inputs
(65,536 total input tokens) completed without an allocation assertion or OOM.

Single-stream decode measured 144.952 → 145.042 tokens/s. Pooling all five
measured runs per arm across two launches gives:

| Aggregate steady decode | Original | Candidate | Change |
| --- | ---: | ---: | ---: |
| Four streams | 327.014 | 321.816 | −1.59% |
| Eight streams | 414.430 | 410.819 | −0.87% |

The second comparison alone was −0.56% at four streams and +0.20% at eight.
There is substantial variation between launches; these results do not
establish a causal kernel slowdown. They also do not justify describing the
candidate as a proven speed-neutral default. Both arms used the same **8+7**
checkpoint; this experiment does not compare 8+8 inference speed.

CPU validation covers 28 allocation-bound cases, two multimodal head-sharing
cases, and five host-overlay checks. The overlay preserves unrelated host
query-splitting code, rejects unexpected source shapes, and is idempotent.

To test with a freshly built branch image, set
`VLLM_SM120_COMPACT_WORKSPACE=1` before startup. It defaults to `0` in the
library and is not enabled in either published deployment profile. The MTP
head-sharing correction applies whenever the updated source is loaded.

On this host the base compose file binds an older indexer over the image.
The tested override therefore uses a generated copy of that host file:

```bash
/home/jarrelscy/.venv/bin/python \
  examples/arvq/host/prepare_compact_workspace_overlay.py \
  /home/jarrelscy/glm52/vllm-vision-build/vllm/model_executor/layers/sparse_attn_indexer.py \
  /home/jarrelscy/homeassistant/.runtime/glm53-arvq/sparse_attn_indexer.py
```

Bind the generated file to
`/opt/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:ro` in the
experimental override. The original host file stays untouched. Do not enable
the flag against an old image that lacks `workspace_limits.py`.
