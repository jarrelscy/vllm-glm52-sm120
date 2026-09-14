# Two P4 activations per native FP4 MMA

Dense o_proj pairing is implemented behind the static, default-OFF
`VLLM_NVFP4_P4_PAIRED=1` flag. See [measured dense results](PAIRED_P4_RESULTS.md).
Routed expert pairing below remains an unimplemented proposal.

## Available columns

The kernel uses `m16n8k64` MMA. With four residual activation planes, its
B-operand load enables columns `q < 4`, where `q = lane / 4`. Columns 4–7
receive zero. For lane coordinates `q = lane / 4`, `c = lane % 4`, the
accumulators represent:

| Register | Output row | Output column |
| --- | ---: | ---: |
| `d0` | `q` | `2*c` |
| `d1` | `q` | `2*c+1` |
| `d2` | `q+8` | `2*c` |
| `d3` | `q+8` | `2*c+1` |

A pair could place one slot's four planes in columns 0–3 and the other slot's
four planes in columns 4–7. Load the activation slot using `q / 4` and its
plane using `q % 4`. Both slots must use identical weights: the same hot/cold
kind and local expert index. Matching token positions alone is insufficient.

The weight operand, codebooks, block scales and serialized formats stay the
same. Each slot retains all four planes with coefficients
`1, 1/16, 1/256, 1/4096`. Each K64 iteration still issues two MMAs for cold
ARVQ or one MMA for hot NVFP4, now producing two slots' contributions.

## Output and routing

For each row accumulator, use exponents `-8*(c%2)` and `-8*(c%2)-4`, then
sum only across `shfl_xor(..., 1)`. Lane groups `c=0,1` reconstruct the first
slot; groups `c=2,3` reconstruct the second. Store from `c=0` and `c=2` to
the respective original slot indices.

The partial layout remains `[original_slot, N, split]`. Existing split
reduction, per-expert global scales, routing weights and output combination
can remain unchanged. An unmatched slot fills the second four columns with
zero and does not store a second output.

A GPU pairing pass could produce a fixed-capacity `partner[slots]` array.
Keep `grid.z=slots`: each pair's leader computes both outputs; its follower
returns before loading the shared codebook. This avoids host readback and
data-dependent launch sizes during CUDA graph replay. Every slot must have
exactly one writer. Invalid routes must retain the existing zero-output
behavior.

A bounded first routing implementation could match experts within each warp
of 32 routed slots, pairing consecutive occurrences. With top-8 routing,
that covers four adjacent token positions. It may miss cross-warp matches;
it makes no assumption that all draft tokens choose the same experts.
The same pairing map can serve gate/up and down because their routing is
shared, even though their activation values differ.

## First experiment

Dense attention `o_proj` is the simpler first microbenchmark: all token rows
share one weight matrix, so consecutive tokens can pair without a routing
pass. Validate both slots independently against the existing P4 kernel,
including odd token counts, global scales and split reduction. Compare
complete activation packing, paired projection and output conversion with
rotating weights.

Only then test routed experts, including unique experts, repeated experts,
mixed hot/cold routes and unmatched slots. Measure the actual pairing rate,
pairing-pass cost, inactive-CTA cost and occupancy. A single token's top-8
experts normally cannot pair with each other. Potential savings therefore
depend on routing reuse in MTP or concurrent requests; fewer weight loads
and MMAs do not by themselves establish lower latency.
