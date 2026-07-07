# GLM-5.2 hybrid — reproducible benchmarking

`bench.py` reproduces every cell of the config matrix below. It measures **decode
tok/s** the correct way (non-streamed, prefix-cache-warmed, true
`usage.completion_tokens`) and **prefill tok/s**.

> ⚠️ **Do not measure speculative decode by counting streamed SSE chunks.** Under
> MTP/DSpark, vLLM bundles multiple accepted tokens into one SSE delta, so chunk
> counting undercounts throughput by ~the acceptance factor (we saw 104 deltas =
> 300 real tokens). `bench.py` uses non-streamed responses to avoid this.

## 1. Launch a server config

All configs serve an OpenAI-compatible API on `:8001`. Pick a weight variant via
`MODEL_DIR` (`…/glm52-models/{1m,500k,250k}`) and a topology via `PARALLEL`
(docker entrypoint) or the flags in `serve_mtp.sh`.

| col | config | flags (in addition to `--kv-cache-dtype fp8_ds_mla --enforce-eager`) |
|-----|--------|----------------------------------------------------------------------|
| A | pp4-base       | `--pipeline-parallel-size 4` |
| B | pp4-mtp        | `--pipeline-parallel-size 4  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'` |
| C | tp2pp2-base    | `--tensor-parallel-size 2 --pipeline-parallel-size 2` |
| D | tp2pp2-mtp     | `--tensor-parallel-size 2 --pipeline-parallel-size 2 --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'` |
| E | tp4-base       | `--tensor-parallel-size 4` |
| F | tp4-dspark     | `--tensor-parallel-size 4 --speculative-config '{"model":"RedHatAI/GLM-5.2-speculator.dspark","method":"dspark","num_speculative_tokens":5}'` |

Common env for all: `VLLM_DISABLE_FP8_W8A16=1` (v2-only, bit-exact; set `=0` for
the optional v4 fp8 path, +6–8% base), `--gpu-memory-utilization 0.95`,
`--max-num-batched-tokens 2048 --no-enable-flashinfer-autotune`. PP configs use
`VLLM_PP_LAYER_PARTITION=21,19,19,19` (or `20,20,20,18` to reach ~1M with MTP).
Speculative decode must run **enforce-eager** (CUDA graphs hurt spec here).

Or via the docker entrypoint modes: `PARALLEL=pp4-1m|pp4-mtp|tp2pp2|tp4-dspark|pp4-tpdraft`.

## 2. Run the benchmark

```bash
export VLLM_API_KEY=...            # only if you launched with --api-key
python bench.py                    # 3 workloads, short context
python bench.py --workload code
python bench.py --ctx 200000 --docfile book.txt   # long-context (use a public-domain book)
```

### Exact prompts (baked into `bench.py`)
- **count** (EASY, low-entropy → high acceptance): `Write out the integers from 1 to 500, one per line. Output only the numbers.`
- **code** (medium): `Implement in Python, as separate functions each with a full docstring and inline comments: bubble_sort, insertion_sort, selection_sort, merge_sort, and quick_sort. Output only the code.`
- **tcp** (COMPLEX, high-entropy): `Write a detailed 400-word explanation of how TCP congestion control works.`

All runs use `temperature=0`, `enable_thinking=false`. Acceptance length is read
from the server's `Speculative metrics` log lines (not in the OpenAI response).

## 3. Reference matrix (4× RTX PRO 6000 Blackwell, SM120, non-streamed)

Decode tok/s: base = one content-independent number; spec = `[easy/code/complex]`.
Ceiling = KV-pool tokens that fit.

| variant | A pp4-base | B pp4-mtp | C tp2pp2-base | D tp2pp2-mtp | E tp4-base | F tp4-dspark |
|---------|-----------|-----------|---------------|--------------|-----------|--------------|
| **1m**   | 1.27M / 18.7 | 999K / 29/24/24 | 752K / 21 | 580K / 40/36/30 | 371K / 15.2 | 247K / 66/53/28 |
| **500k** | 659K / 18.7  | 296K / 28/26/22 | 380K / 21 | 219K / 39/37/31 | 176K / 15.5 | 105K / 64/60/31 |
| **250k** | 327K / 18.7  | ~31K / 29/27/23 | 160K / 21 | ~35K / 40/37/31 | 83K / 15.5 | ~38K / 65/59/33 |

Prefill ~1.2–1.6K tok/s across configs. NVFP4/AQLM expert split: 1m 29/71, 500k 48/52, 250k 57/43.

**Takeaways:** decode speed is set by the *config*, not the weight variant (all three
decode identically per-config; they differ only in context ceiling — use `1m`).
`tp2pp2-mtp` is the best speed/context balance; `tp4-dspark` is fastest but
short-context; `pp4-base`/`pp4-mtp` reach the longest context. MTP beats base on
every workload (1.26–1.57× at ns=2). Full analysis in `MTP_PROGRESS.md`.
