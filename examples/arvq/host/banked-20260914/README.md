# Banked local GLM-5.3 launch — 2026-09-14

`./switch.sh glm-5.3` selects the banked local image and initial-fit, unrotated
8+8 checkpoint. It retains 1,048,576 context, eight request slots, TP4/DCP4/MTP3,
vision, residual-activation P4 kernels and LMCache. Source-built experiments and
sampler diagnostics are disabled. The PV checkpoint remains staged separately.

The image owns the previously mounted code overlays. Data/cache mounts remain;
no serving Python file is mounted from a moving worktree. The switch profile uses
`--no-build` and checks that the saved image exists before stopping the server.
`bank_manifest.json` records the image ID and tested configuration relationship.
The image is local to this machine; this directory does not publish a registry
image or claim that the tag can be pulled elsewhere.

Local rebuild snapshot, Dockerfile and 451 per-file overlay hashes are preserved
at `/home/jarrelscy/homeassistant/cold-format-lab/banked_20260914/`. The image
starts from `glm53-arvq-sm120:prefill-runtime-counts-candidate`; kernel code is
published on `arvq-hybrid-sm120` through commit fa7e6019b.

Measured before this packaging-only change: 8K prefill 2099 tokens/s,128K 1963,
256K 1801,512K 1665; C1 decode 141–142, C4~322, C8~407–414. One C8 run omitted a token
near position 494; later repeated and observed runs matched. That discrepancy
remains unresolved. The 3000/150 targets are not achieved; no new indexer-budget
or tie-selection prototype is enabled. See results/direct_output for evidence.

Optimization is paused at the user's request. Preserve the staged prototypes
and both checkpoints; do not delete interim weights or silently adopt PV.
