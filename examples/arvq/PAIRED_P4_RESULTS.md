# Paired P4 dense projection

The optional dense o_proj kernel places two token positions in the eight native
FP4 MMA columns, retaining four activation planes each. It is enabled only by
`VLLM_NVFP4_P4_PAIRED=1`; the source default is OFF. This flag is static for the
server process: changing it requires restart and CUDA graph recapture.
M1 keeps the original kernel. Routed expert pairing remains unimplemented.

## Microbenchmark

Actual GLM layer 3 o_proj, TP shard N=6144/K=4096, on one SM120 GPU. The complete
path includes BF16 input conversion, four-plane packing, projection, split/global
reduction and BF16 output conversion. CUDA graphs rotate 21 independent weight
copies totaling 297,271,380 bytes, exceeding L2. Results are medians of five timing
batches. No communication or server overhead is included.

| Positions | Existing dispatch µs | Paired µs | Paired split |
| ---: | ---: | ---: | ---: |
| 1 | 17.73 | 17.73 | Original kernel |
| 2 | 23.90 | 17.83 | 8 |
| 4 | 24.82 | 18.47 | 4 |
| 8 | 39.82 | 25.39 | 2 |
| 16 | 45.75 | 36.55 | 2 |

The M16 control includes fresh GPU weight dequantization and BF16 GEMM; the
other controls use the existing native kernel. Retuning the unpaired M2 split
alone reduces its latency to 18.46 µs. M4/M8 retain gains against the retuned
unpaired controls. M1–8 numbers use the initial paired prototype; M16 uses the
specialized production dense kernel. See [raw measurements](results/paired_p4_micro.json).

At 78 o_proj calls, additive measured savings imply about 0.50 ms at M4 and
1.13 ms at M8 per target-model invocation. These are optimistic contributions,
not measured serving throughput. Other dense layers, attention, MoE and
communication remain unchanged.

The independent encoded-weight/P4 activation oracle agrees within relative L2
5.6e-7. The underlying NVFP4 weight approximation remains 9.51% relative L2;
pairing does not requantize the weights. CUDA tests also cover odd M3/5/7/9/15,
M16, partial final output CTAs, and graph replay with changed activations. M17
still follows the configured fallback. At M16, moving from BF16 temporary-weight
GEMM to native P4 changes arithmetic; serving acceptance must be measured.

## Reproduction and configuration

Build `arvq/build.sh` and `arvq/build_dense.sh` in the serving image, then run:

```bash
python examples/arvq/bench_paired_dense.py --model /path/to/original-checkpoint \
  --tokens 4 --output baseline.json
python examples/arvq/bench_paired_dense.py --model /path/to/original-checkpoint \
  --tokens 4 --paired --output paired.json
```

`--kernel-dir` can select an installed directory containing `hybrid.so` and
`dense.so`. Use an isolated GPU and repeat with M1/2/4/8/16. The source checkpoint
must contain the original BF16 layer 3 o_proj tensor. Neither the benchmark nor
the serving option changes checkpoint files.

For the measured serving candidate, set `VLLM_NVFP4_P4_MAX_TOKENS=16` together
with the paired flag. The library's default maximum remains 4. Paired dispatch
uses split 8 for M2, 4 for M3–4, and 2 for M5–16. Above the configured native
maximum, existing temporary BF16 dequantization/GEMM remains in use. CUDA graph
capture sizes must cover the token-position counts being compared.
