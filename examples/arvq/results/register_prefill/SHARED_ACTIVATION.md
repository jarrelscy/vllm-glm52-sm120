# Eight-row shared activation staging

`VLLM_ARVQ_SHARED_HOT_ACTIVATION=1` selects the qualified unswizzled kernel for
existing wide-hot prefill calls when N is divisible by 128. It is off by default.
Existing whole-batch eligibility and hot-route thresholds are unchanged. Shapes
outside the new row divisibility requirement retain the previous register kernel.

Eight row warps cooperatively stage the same activation tile once per K group.
The 256-thread CTA uses 2,304 bytes of shared memory and 80 registers per thread.
Weight loads, MMA order, FP32 accumulation, output scaling and reduction are
unchanged. Existing seven-int register descriptors and activation packing are
reused. Legacy wide and register launch exports remain available; the new export
is `wide_launch_shared_activation` in the same library.

The frozen isolated GPU micro compared nine complete MLP cases against the
previous register8 implementation. All output bits, peak allocated memory and
peak reserved memory matched. Mixed T2048 improved 1.033× and T4096 1.052×;
all-hot improved 1.168× and 1.137×. These are isolated layer measurements, not
full-model throughput claims. All-cold and ineligible-tail controls were neutral.
Raw measurements are in `shared_activation_micro.json`.

Production packaging preserves the four qualified CUDA function bodies exactly
apart from symbol renaming and formatting, recorded in
`shared_activation_integration.json`. CPU compilation produced all six legacy
and new C exports. Thirteen CPU tests passed for staging coverage, odd groups,
output-tile coverage, unchanged descriptor reuse, opt-in dispatch and fallback.
No additional GPU run was performed during packaging.

The later bank-swizzled variant was slower and is not included. This implementation
uses the qualified unswizzled eight-row variant only.
