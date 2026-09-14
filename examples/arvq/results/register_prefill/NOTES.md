# Register reuse in hot prefill

Four token pairs per warp reuse each loaded weight fragment while preserving
four activation planes, independent FP32 accumulators and original K/split
reduction order. This replaces the earlier one-pair-per-warp wide kernel only
inside its existing opt-in shape/routing gate. Decode remains unchanged.

Both two-pair and four-pair prototypes passed nine complete real layer3 8+8
cases each, including mixed/hot/cold routes and a noneligible token count.
Four-pair mixed timing improved 13.591→13.184 ms at2048 tokens and
18.450→17.523 ms at4096 tokens. Incremental allocation was unchanged.
The preferred kernel uses56 registers/thread,128 threads/CTA and no spills.

With raw-KV gather and fused cold gathering/route reduction, the live four-pair
stack measured1798/1801 cold prefill tokens/s at8K and1713 at128K, with141
single-stream decode tokens/s. These are whole-stack measurements, not isolated
attribution to this kernel. CPU integration tests passed30 cases; the build
preserves the original wide entrypoint and adds wide_launch_register.

The rebuilt production binary also passed all nine complete real-layer bitwise
comparisons against the qualified lab binary. This was a parity-only run with
one timing sample per variant; it does not establish a new speed comparison.
