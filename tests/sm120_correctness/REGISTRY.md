# Optimization idea → gating tests (the REGISTRY)

Every planned/roadmapped optimization of the GLM-5.2 SM120 serving path
maps to the tests that MUST pass before it is promoted. **The FINAL GATE
(row 0) applies to EVERY row — no exceptions.**

| # | Optimization idea | Gating tests (in order of cost) | Gate |
|---|---|---|---|
| **0** | **ANY prefill or decode change (mandatory promotion gate)** | `tier4_canary/test_final_gate_64k.py` (64K multi-depth synthesis: teacher-forced NLL chunk envelope + depth + answer + finish), `tier4_canary/test_1m_capability.py`, `tier4_canary/test_reasoning_canary.py` | NLL in golden band, all 256-tok chunks in envelope, answer/serial/FINAL:95 correct, depth in [0.6·min, 1.5·max], finish=stop; max_model_len & KV pool ≥ 950K; canary depths ≥60% ref + correct. **FAIL ⇒ STOP THE LINE** |
| 1 | DSMEM / cluster-resident codebook variant of the AQLM gather | `tier1_kernel/test_kernel_variant_equivalence.py` (register flag in `kernel_variants.json`), `test_hybrid_gemv_bitexact.py`, `test_reduction_order.py` (CPU), then rows 0 + acceptance | bit-exact (maxdiff==0) vs shipped kernel |
| 2 | w2 lane-occupancy repack (4 rows/warp, "order-preserving" reduction) | `tier1_kernel/test_reduction_order.py` (the tree is executable + non-vacuity checks prove the golden pins lane/K partitioning), `test_hybrid_gemv_bitexact.py` (esp. `w2_real` [6144×512] + partial-lane-tail `w2_small` K=192), `test_kernel_variant_equivalence.py` | bit-exact; "order-preserving" claim is checked mechanically, not by review |
| 3 | q_b_proj replication removing the DCP q all_gather | `tier2_dist/test_qgather_replication.py` (single-GPU math + torchrun all_gather variant), `tier3_server/test_teacher_forced_logits.py`, row 0 | GEMM-reorder elementwise bound; logits envelope |
| 4 | One-shot P2P allreduce replacing NCCL for small TP messages | `tier2_dist/test_allreduce_equivalence.py` (register impl in `kernel_variants.json`; real message sizes [1/4/8/16, 6144] bf16/fp16), `tier3_server/test_mtp_acceptance_envelope.py`, row 0 | |diff| ≤ (N−1)·eps·Σ|xᵢ| of exact sum; deterministic if declared; acceptance in envelope |
| 5 | L2 residency policy / `__ldcs` hints | `tier1_kernel/test_kernel_variant_equivalence.py` (built-in `aqlm_cb_l1` variant already exercises the cache-hint machinery), `test_hybrid_gemv_bitexact.py` | bit-exact (load-path changes must not touch numerics) |
| 6 | Per-layer MoE megakernel (w13→silu→w2→combine) | `tier1_kernel/test_hybrid_gemv_bitexact.py` + `test_kernel_variant_equivalence.py` per stage; NOTE: the fused chain crosses the silu/topk-weighted-sum boundary — register with an explicit `compare` hook that A/Bs against the shipped `_apply_gemv` composition; `tier3_server/test_mtp_acceptance_envelope.py`; row 0 | per-stage bit-exact; end-to-end chain bit-exact vs shipped composition on identical inputs |
| 7 | Prefill chunk-size changes + prefill dequant rewrites + inductor fusion passes | `tier1_kernel/test_dequant_prefill.py` (bit-exact vs shipped dequant), `tier3_server/test_teacher_forced_logits.py`, `tier3_server/test_needle_depth.py` (32K/130K/749K), row 0 (the 64K teacher-forced chunk envelope is the strongest prefill gate) | dequant maxdiff==0; NLL/needle/final-gate envelopes |
| 8 | DCP backend a2a vs ag_rs (and any combine rewrite) | `tier2_dist/test_dcp_combine.py` (pack lossless, combine vs CPU ref, degenerate/empty-shard rows, torchrun a2a==ag_rs), `tier3_server/test_needle_depth.py`, `test_mtp_acceptance_envelope.py`, row 0 | fp-reorder tolerance vs the algebraic reference; finite on empty shards |
| 9 | Any cudagraph capture change | `tier3_server/test_graph_capture_parity.py` (two-phase eager-vs-graphed probes), `test_mtp_acceptance_envelope.py`, row 0 | answer parity + depth ratio ∈ [0.6, 1.67] + finish=stop; acceptance in envelope |
| — | Attention backend / KV-cache dtype routing (the SM100 root cause) | `tier3_server/test_backend_routing.py` (CPU priority pins + source tripwires + live-server log check), `tier1_kernel/test_ds_mla_kv_write.py` (656-byte layout, arbitrary-fp32 non-pow2 scales, roundtrip, NaN isolation) | exact priority list; ds_mla guard present; byte-exact KV writes |

## Rules for optimizer agents

1. **Append your flag to `kernel_variants.json`** (schema in
   `common/variant_registry.py`) — `tier1_kernel/test_registry_cpu.py`
   fails if a new `#ifdef`/`VLLM_*` env knob exists in the source without
   a registry entry, and `test_kernel_variant_equivalence.py` then A/Bs
   your variant automatically.
2. Run tiers in cost order: `cpu` → `kernel` → `unit` → `server` → `canary`.
   A cheap-tier failure makes the expensive tiers pointless.
3. Server-level gates are ENVELOPES, not byte equality — byte-identical
   greedy output is unachievable on this multi-GPU server (NCCL
   nondeterminism). Kernel-level tests are bit-exact and MUST stay so.
4. Never re-baseline a golden to make a candidate pass. Goldens are
   captured from the KNOWN-GOOD shipped config only (`make_goldens.sh`).
5. A canary or final-gate failure is a STOP-THE-LINE event: revert,
   bisect, and only then continue optimizing.
