# Compact workspace and MTP head alias CPU review

Review date: 2026-09-14. No repository files were edited, staged or committed. No inference or GPU workload was launched.

## Result

No blocking issue found for publication as an opt-in allocation change plus a target-head lookup correctness fix. Existing attention scheduling, query split policy, selected keys, arithmetic, context and sequence limits are unchanged by these diffs.

- `workspace_limits.py` defaults off. Enabled indexer reservation is min(original allocation, max_num_seqs × (max_model_len + speculative_tokens + 1)). Each request's legal context plus lookahead is bounded; DCP local token counts cannot exceed corresponding global counts. The runtime assertion catches a metadata gather exceeding that allocation instead of permitting an undersized slice.
- FlashMLA's BF16 prefill reservation is skipped only when its existing metadata dispatch uses the mixed-batch path (num_heads < 32), with fp8_ds_mla and the compact flag enabled. The threshold is the same as the actual branch selector. This is not the active FlashInfer SM120 backend's measured saving.
- MTP changes only the head lookup from the outer multimodal wrapper to its already-unwrapped language model. Existing `_should_share` policy remains intact, as does the draft head norm. GLM MTP loader explicitly remaps the same checkpoint `lm_head.weight` into `model.layers.<spec>.shared_head.head.weight` (glm4_moe_mtp.py around 271), supporting aliasing the target's identically sharded head. Models requiring a distinct head retain it through the existing policy.
- Host overlay helper changes three bounded anchors and checks their uniqueness, complete prior application and Python syntax. It preserves unrelated QUERY_SPLIT source and is idempotent on its own output. It deliberately rejects altered/partial applications rather than silently overwriting foreign tuning.

## Verification

CPU-only tests, CUDA_VISIBLE_DEVICES empty:

- 28 allocation-limit tests passed using host venv and `--confcutdir=tests/v1/attention`.
- 5 host overlay tests passed using host venv and `--confcutdir=examples/arvq/host`.
- 2 actual MTP loader sharing tests passed in the lab venv, with the unchanged repository test copied into an isolated lab test directory. The host venv could not collect this test because psutil was missing; this was an environment failure, not a test failure. The lab-installed utils.py hash exactly matches the repository and active serving image, so the successful tests exercise the reviewed source.

Read-only SHA256 comparison confirms all three files match the active image exactly:

- workspace_limits.py: `7530ba5d65924d9e353942f204ac2295686e51179ffecf12efd5d12d2efa6ace`
- flashmla_sparse.py: `b379bc863f2db174ba7c966618721937d20b0d9bf6f1a01a232bb358bee691f7`
- eagle/utils.py: `abede19518d8721b64c819773dd55ebcc81b4d170fbf1619750f5fc479ef4933`

The host indexer is intentionally a separate overlay with foreign QUERY_SPLIT support; do not replace it wholesale with the repository's indexer merely to make hashes match.

## Historical result versus active deployment

The existing `examples/arvq/results/compact_workspace/NOTES.md` explicitly concerns the historical 8+7 checkpoint. That candidate met its memory goal but failed the strict rollout gate: pooled C4 decode was 1.59% lower, and the then-default profile was restored. Keep this historical result unchanged and do not describe it as an 8+8 benchmark or an unconditional speed win.

The current read-only environment identifies `GLM-5.3-Vision-NVFP4-ARVQ-hybrid-initial-8x8`, compact flag 1. Current boot logs report 16.36 GiB KV memory and 1,283,584 shared cache tokens. These are separate active 8+8 observations, not a reinterpretation of the historical 8+7 trial. This review does not establish a new throughput gain or full-model cross-boot bitwise guarantee.
