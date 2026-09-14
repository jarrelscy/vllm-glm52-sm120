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
`glm53-arvq-sm120:dense-p4-m8`. Adjust these host paths for another machine.

Settings: TP4 + DCP4, MTP three drafts, dense NVFP4 P4 through eight positions,
CUDA graph sizes `[1,2,4,8,16]`, four requests, 4096 batched tokens, 950000 context
limit, utilization 0.96, LMCache disabled, and the measured PCIe communication
policy. `PARALLEL=tp4-1m` can explicitly select the no-MTP mode.

The full local checkpoint remains unrotated. The H128 weight samples published
under `rotation-prototype-v1` are separate tuning artifacts. Do not substitute
them for full model shards.

Validation: shell syntax passes; all four checked direct/alias invocations
resolve and validate the override before reaching the stop phase; an incomplete
ARVQ-named cold shard is rejected; the merged image, complete command, critical
environment, and every bind mount match the measured running M8+MTP container.
The published patch passes a reverse-application check against the updated
host script. These checks do not invoke the script's global stop loop.
