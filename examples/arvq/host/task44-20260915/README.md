# Task 44 serving fix

The host patches select the locally built image
`glm53-arvq-sm120:accuracy-task44-20260915` for both `./switch.sh Glm-5.3`
and `./switch.sh glm-5.3-arvq`. The existing checkpoint, 1,048,576-token
context, eight request slots, MTP3, vision and LMCache settings are retained.

The image fixes native ragged indexer weight packing and causal bounds.
Production SM120 MTP3 uses flattening and bypasses this branch. The fix is
not presented as a demonstrated remedy for the task's over-exploration.
See `tools/task44/README.md` for numerical evidence and its limits.

This is a local image, not a published registry artifact. `image.json` records
its identity and local rebuild context. Full serving revalidation is recorded
in the investigation report after the sequential controls finish.
