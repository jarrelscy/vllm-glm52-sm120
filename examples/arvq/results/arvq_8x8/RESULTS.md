# ARVQ 8+7 versus 8+8

8+8 is modestly faster in these isolated native-expert microbenchmarks, while
using more memory. It is not an unconditional improvement or a measured serving
throughput gain. The running checkpoint remains 8+7 and unrotated.

## Complete mixed-expert MLP

Actual GLM layer 3 TP3 weights; synthetic independent activations, six cold and
two hot routes per token. The table shows rotating-weight median microseconds:

| Token positions | Routing | 8+7 | Matched 8+8 | Full-256 synthetic 8+8 |
| ---: | --- | ---: | ---: | ---: |
| 1 | Reused | 40.47 | 40.52 | 40.45 |
| 4 | Reused | 80.95 | 76.82 | 76.11 |
| 4 | Unique | 92.51 | 89.42 | 89.32 |
| 8 | Reused | 130.53 | 121.84 | 122.81 |
| 8 | Unique | 186.02 | 180.04 | 180.93 |
| 32 | Reused | 453.79 | 416.65 | 420.63 |
| 32 | Unique | 729.91 | 695.66 | 698.47 |

The full-256 M4 examples correspond to about 3.6–6.4% higher **MLP microbenchmark**
throughput, not whole-model throughput. M1 mixed routes are effectively flat.
All-cold M4 rotating cases were 82.22→76.35 µs with reused experts and
84.50→79.19 µs with unique experts. Warm and isolated cold-projection results,
including every timing sample, are in [results.json](results.json).

Both arms use identical splits: gate/up 16 for at most 32 routes, otherwise 8;
down 2. Timing includes activation expansion, four-plane packing, both native
projections, FP16 SwiGLU, and the original FP32 weighted combine. No TP
communication is included. The 48 rotating banks span real 195-cold/61-hot expert
pools larger than L2. Reused/unique routing are controlled patterns, not measured
production collision frequencies. Each variant receives identical warmups and
eight timing rounds with alternating order; absolute clocks can vary between
cases, so compare paired results within each case.

## Arithmetic and implementation

Matched 8+8 retains the original residual indices below 128. Across all 14 cases,
raw FP32 gate/up and down projections and the complete pipeline matched 8+7
bitwise using integer views. The same checks pass against the new production
8+8 export. Independent full-256 projection oracles have relative L2 errors
2.5e-7 and 1.1e-7. These synthetic upper codewords establish arithmetic
correctness and timing, **not model accuracy or a fitted larger codebook**.

The 16-bit index is aligned and never crosses a 32-bit word boundary. Static
SASS inspection of the projection function found four fewer global load
instructions and removed four `SHF.R.W.U32` funnel shifts. Both variants retain
the same three FP4 MMA instructions across the cold/hot branches (two cold, one
hot), and eight shared-table loads. The 8+8 kernel uses 56 registers versus 55,
and 2,048 versus 1,536 bytes of shared codebook storage. These observations
support cheaper index extraction as an explanation; they do not establish a
universal compute-versus-bandwidth bottleneck. See
[sass_static_counts.json](sass_static_counts.json).

The production ABI preserves the existing 8+7 exports and adds
`hybrid_launch_8x8`, `arvq_dequant_8x8`, and `arvq_dequant_fp16_8x8`. The new packed
layout has 64 words per tile and 512 codebook words; it does not require a guard
word. Legacy 8+7 retains 60 words, 384 codebook words, and its readable guard.
Production tests cover both FP16/BF16 decoders, signed-zero bit patterns,
full-256 indices, mixed hot/cold routes, changed-activation graph replay, and
legacy bitwise behavior. Python hybrid/prefill tests passed 52 cases on CUDA;
the subsequent guard-only validation passed CPU tests without changing kernels.

## Memory cost

Including per-128-weight FP8 scales, the format increases from 1.9375 to 2.0625
bits per weight before shared dictionaries and global scales. This exceeds the
previous 2-bit cold budget. Across 13,450 cold experts, the wider index payload
adds exactly 7,933,132,800 bytes to the checkpoint, plus 76,800 bytes for larger
shared codebooks. Replicated dictionaries bring GPU growth to 1,983,360,000
bytes per TP rank (1.84715 GiB), or 7.38859 GiB across the box. Cold storage grows
about 6.45%; whole-checkpoint size grows about 2.74%. This is a capacity tradeoff,
not a default recommendation. The original checkpoint was not modified.

The [standalone reproducer](../../format_experiment/README.md), raw timing and
validation JSON files, and compile information accompany these results.
