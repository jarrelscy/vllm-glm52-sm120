# Native FP4 conversion in cold weight reconstruction

The CUDA decoders use native packed FP4-to-half2 conversion and half2 addition
for the two codebooks. FP4 sums are exactly representable in FP16. Each result
is converted to FP32 before the original block-scale multiply, global-scale
multiply and final FP16/BF16 rounding. Both ARVQ layouts and existing library
APIs remain unchanged. No expanded codebook or additional scratch is used.

The repository sources match the qualified microbenchmark source after removing
comments and whitespace; `source_equivalence.json` records hashes. The existing
two-library build structure is retained. No binary artifacts are committed.

## Validation

The isolated GPU qualification checked both formats, gate/down shapes and
FP16/BF16 output: all eight weight comparisons passed exact output bits.
Four fused-gather cases also matched both baseline weights and independently
gathered activation rows. The new regression fixture covers all 256 signed
FP4 sum pairs, including signed zeros, and compares integer views of output
bits through eager execution and CUDA graph replay.

All nine full layer-3 pipeline cases at 2,048, 4,096 and 2,177 tokens passed
bitwise output comparisons and unchanged allocated peak memory. Both arms
retained the same shared-eight hot kernel, cold activation fusion, route sum,
weights and routing. Only decoder libraries differed.

## Measured scope

These are isolated per-layer microbenchmarks, not whole-model throughput.
Five alternating rounds measured mixed cases:

| Tokens | Baseline | Native conversion | Speedup |
| --- | --- | --- | --- |
| 2,048 | 10.050 ms | 9.464 ms | 1.062x |
| 4,096 | 13.846 ms | 12.767 ms | 1.085x |
| 2,177 | 18.182 ms | 17.230 ms | 1.055x |

The all-cold 4,096 case improved 10.490 to 9.407 ms. The all-hot control never
executes this decoder; its initial 3.3% timing loss disappeared on an eight-round
repeat, which measured 18.649 versus 18.521 ms. That repeat confirmed the mixed
case at 14.217 versus 13.065 ms. Both repeat cases remained bitwise and peak-equal.
Raw observations, including variability, are retained in the JSON files.

CPU compilation of both production libraries passed with every legacy export
present. Scoped pre-commit hooks passed. The expanded CUDA regression tests
were prepared after the isolated GPU campaign and require a later GPU lease;
they must not be reported as already executed against the rebuilt libraries.
