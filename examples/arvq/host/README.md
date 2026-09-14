# Existing host switch integration

The live host's `/home/jarrelscy/homeassistant/switch.sh` now selects the measured
ARVQ + dense P4 + MTP profile for `glm5.3-hybrid-1m`. Existing aliases
`glm-5.3`, `glm5.3-vision`, and `glm5.3-1m` resolve to it, as do the new explicit
aliases `glm5.3-arvq` and `glm5.3-arvq-mtp`. The plain `glm5.3` alias continues
to select the separate Flash model.

```bash
cd /home/jarrelscy/homeassistant
./switch.sh glm5.3-arvq
```

This host directory is not a Git checkout. `switch-arvq.patch` tracks the exact
local script change here; `docker-compose.glm53-arvq.yaml` is the companion
deployment override. The script loads that override after the existing
`docker-compose.yaml`, validates the model shards and compose configuration,
then follows its usual stop/start/readiness flow.

To reproduce on a host with the matching base switch and compose deployment,
copy the override beside `switch.sh` and apply the patch there with `patch -p1`.
The override preserves the measured bind mounts and cache directories, builds
`Dockerfile.arvq` from `/home/jarrelscy/glm52/vllm-arvq-serving`, and selects
`glm53-arvq-sm120:paired-compact`. Adjust these host paths for another machine.

Settings: TP4 + DCP4, MTP three drafts, paired dense NVFP4 P4 through sixteen positions,
CUDA graph sizes `[1,2,4,8,16,32]`, eight request slots, 4096 batched tokens,
**1048576 total tokens per request**, utilization 0.94, and LMCache enabled.
The measured PCIe communication policy is retained. The shared GPU KV capacity
is **1073920 tokens**, not that capacity per request. LMCache uses the dedicated
ARVQ directory, CPU24 GiB and disk100 GiB per worker, the DCP-aware V3 connector,
256-token chunks, and stable hashing. Grouped cold prefill is enabled with `VLLM_ARVQ_GROUPED_PREFILL=1`
for eligible batches of at least 2048 tokens, including startup memory profiling.
Native route compaction is enabled with `VLLM_ARVQ_COMPACT_PREFILL=1`; dense
pairing uses `VLLM_NVFP4_P4_PAIRED=1`. Both library defaults remain OFF.
`PARALLEL=tp4-1m` can explicitly select the no-MTP mode.

To select four request slots with matching graph coverage:

```bash
MAX_NUM_SEQS=4 ARVQ_CAPTURE_SIZES='[1,2,4,8,16]' ./switch.sh glm5.3-arvq
```

The default command selects eight. `MAX_NUM_SEQS` is an admission limit, not a
reservation of the full context window for every slot. With eight slots and
LMCache enabled, measured total short-prompt MTP throughput was 136.2/277.3/344.9
tokens/s at one/four/eight simultaneous streams, with all drafts accepted.
See [full-context/concurrency validation](../results/context_concurrency/NOTES.md).
No full-million-token input request was run in this configuration test.

The full local checkpoint remains unrotated. Rotation is retired from production
and active tuning; `rotation-prototype-v1` retains historical experiments only.

Grouped prefill plus compaction reduced cold 4096-token input latency from 7.192
to 4.165 seconds and 8192-token latency from 15.595 to 8.571 seconds across rounds.
The same-boot compaction gain alone was 12.6–13.5% in throughput. Paired dense
execution reached total short-request MTP throughput of 136.1/200.5/274.1 tokens/s
at one/two/four streams, with all drafts accepted. See
[implementation and measurement limits](../PREFILL.md).

Validation: shell syntax passes; all four checked direct/alias invocations
resolve and validate the override before reaching the stop phase; an incomplete
ARVQ-named cold shard is rejected. Initial integration matched the merged image,
complete command, critical environment, and every bind mount against the measured
M8+MTP container. The current paired/compact override passes compose validation,
and the live server confirms paired mode, native maximum 16, grouped prefill,
and compaction ON. The final full-context launch uses fixed production mode `1` for both prefill
options, eight slots, LMCache, and no diagnostic overrides.
The published patch passes a reverse-application check against the updated
host script. These checks do not invoke the script's global stop loop.

LMCache disk persistence was verified after the final restart: an exact 8192-token
prompt plus 32 output tokens fell from 8.284 to 0.528 seconds, with all four ranks
restoring, zero GPU-prefix hits, and identical completion text.
[Proof and scope](../results/lmcache/README.md).
