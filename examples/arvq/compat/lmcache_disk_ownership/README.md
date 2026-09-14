# LMCache persisted disk ownership

The qualified LMCache image rebuilds its disk index using only the model-name
prefix. In a shared directory, every TP worker therefore adopts other ranks'
files and charges them against its own disk limit. A TP1 crash attempted to
remove an `@4@2@` key concurrently with another worker's eviction.

The patch filters startup and lazy restoration by model, world size, and global
worker ID before touching foreign sidecars. Explicit removal rejects foreign
keys; missing data-file removal finishes index retirement without crashing.
Other unlink errors propagate. Cache format, directory, enablement, and per-rank
100 GB limits remain unchanged. No existing cache files are migrated or deleted
by installation. Two engines sharing the same model/world/rank namespace still
need separate cache directories.

`install.py` accepts only the exact qualified original SHA256, or the already
patched SHA256. It patches a temporary sibling and verifies the complete output
before atomic replacement. Unknown sources fail closed. The source's original
Apache license header is preserved. `Dockerfile.arvq` invokes this installer.

The base source was verified with a CPU-only container invocation against
`glm53-arvq-sm120:lossless-prefill-candidate`. Original SHA256:
`f123cfab752fe503589d8a8896ac78b5b6f4f64c565a45755958d0eb3bfd8c7f`.
Patched SHA256:
`5bf49e600b2feb52fd78d9654ad9a5fedcb3a9d49e2aa47ef619f5bab3123b87`.

Seven CPU filesystem regressions use real LMCache key parsing, schema-2 sidecars,
and backend methods, bypassing worker initialization. They cover mixed ranks,
world sizes and models, lazy restore, repeated scans, foreign orphan preservation,
missing-file eviction, foreign removal refusal, permission errors, and legacy
metadata-free explicit keys. Run in an installed image:

```bash
CUDA_VISIBLE_DEVICES= /opt/vllm/.venv/bin/python \
  /opt/arvq-compat/lmcache_disk_ownership/test_disk_ownership.py
```

For a temporary candidate file, set `LMCACHE_DISK_TEST_TARGET` to its path.
The tests create and remove only their own temporary fixtures.

For the TP4/DCP4 profile, also set
`LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3` on the next boot. MLA's normal default
queries only rank zero because its KV is normally replicated; DCP shards require
all four replies. The existing client already takes their minimum. This prevents
old asymmetric entries from being advertised solely because rank zero still has
them. `test_lookup_config.py` passes two real-dependency CPU tests for environment
parsing, all-four-server selection, and minimum-prefix behavior. Socket creation
is mocked; no running server is contacted. The existing failed-load block
reporting remains a fallback for entries lost after lookup.
