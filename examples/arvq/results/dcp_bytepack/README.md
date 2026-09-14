# Exact 6/7-byte DCP candidate exchange

Optional and default OFF: set `VLLM_EXPERIMENT_DCP_BYTEPACK=1`. This preserves
all 2,048 candidates per rank and their raw FP32 score bits, replacing the
8-byte wire record with 6 or 7 bytes. The selector reads compressed IDs directly;
its canonical comparison, radix selection and ordering methods are unchanged.
There is one collective and no adaptive certificate or normal-path CPU sync.

Eligibility is C1 prefill, TP/DCP 4, interleave 1, canonical raw selection,
SM120, 128–4,096 query rows, and global context 4,096–524,288. Width is chosen
from global metadata uniformly across ranks: 6 bytes through 131,072 context,
otherwise 7. The local-ID sentinel is 65,535; valid IDs stop at 65,534. The
7-byte representation stores global IDs with sentinel 0xffffff. Other gates
include FP32 logits, contiguous int32 row starts, no CUDA capture, and the
qualified NCCL 4-channel / 1 MiB configuration. Decode, multiple-prefill batches,
unsupported layouts and QUERY_SPLIT use the original path. The repository hook
conservatively treats any nonzero QUERY_SPLIT setting, including toggle mode,
as a reason to retain the original path; it does not replace the host feature.

`VLLM_EXPERIMENT_DCP_BYTEPACK_SHADOW=1` is diagnostic only: the first call for
each layer/width also executes the original full exchange, compares IDs across
all ranks, and returns original IDs. It allocates extra memory and must be off
for performance measurements. A failed comparison disables the candidate.

## Evidence

- Local GPU tests cover rows 129/496/1008/2177/4032, both formats, raw score
  bits including nonfinite values, sentinel/maximum IDs, ties, empty rows,
  canonical output, and graph replay. Extra selector allocation: zero bytes.
- Distributed actual-wrapper tests cover those rows at 128K/512K contexts with
  balanced and tied scores: all 20 cases on all four ranks pass normal and
  shadow output comparisons; every measured allocation peak decreases.
- Three-round timings alternate original/candidate and report maximum-rank
  wall time, including packing, exchange and selection. They exclude logits
  computation and whole-model work. Width 6 generally improves 1.20–1.28×;
  width 7 generally improves 1.09–1.13×. An initial rows-496/width-7 tie case
  has two elevated samples. Those remain in `results/`; a separate seven-round
  repeat improves every sample (mean 0.76632 → 0.70115 ms). No samples were
  removed or row exclusions introduced.
- Actual-model shadow comparison passes both widths across 22 layers and all
  four ranks. See `actual_model_shadow.json`. This proves those observed IDs,
  not universal performance or a whole-model speedup.

`source_equivalence.json` records qualified and repository hashes and checks
AST equivalence excluding imports. Changes are SPDX headers, formatting, import
ordering, and the repository Triton import convention; arithmetic is unchanged.
`test_dcp_bytepack.py` exercises default-off/config/capture boundaries, rank-uniform
width decisions, wire bit patterns, and unchanged canonical selector methods.
Run without CUDA using the repository's Python environment, for example:

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python tests/v1/attention/test_dcp_bytepack.py
```

The source checkpoint, quantization, KV budget, context limit and concurrency
limit are unchanged. Actual serving throughput qualification is separate.
