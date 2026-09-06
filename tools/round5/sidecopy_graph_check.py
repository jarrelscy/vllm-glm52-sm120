# SPDX-License-Identifier: Apache-2.0
"""VLLM_GLM_QAG_SIDECOPY capture-replicating check (single GPU).

Verifies, inside a captured CUDA graph replaying the exact op pattern the
lever inserts (side stream waits an event standing in for the collective's
completion, runs the epilogue copy, records an event; compute stream runs
the conversion-sized independent work, then fences and consumes):

  1. capture succeeds with the cross-stream copy + event fence;
  2. replay output is byte-identical to the sequential baseline graph
     (copy after the independent work on one stream);
  3. replay-time delta = how much of the copy the overlap hides.

The real collective wait is exercised only at the live gate (needs 4 ranks);
this proxy proves the graph-capture mechanics and the copy's byte identity.

RESULT (2026-09-06, RTX PRO 6000, prod container, min over 200x5 replays):
  sequential 1234.9 us, overlapped 1339.7 us -> saving -104.9 us/step.
MEASURED NEGATIVE IN-GRAPH: the two cross-stream event edges per layer cost
more than the 2-3us epilogue copy they hide (~1.3us/edge at replay). The
q-AG side-copy lever was therefore NOT shipped (its envs/mla wiring was
reverted); this harness is kept as the do-not-retry record. Consistent with
the round-3/4 lesson: eager overlap wins do not transfer into FULL graphs.
"""

import sys

import torch

WS = 4
T = 4  # verify-pass tokens (batch=1, MTP ns=3)
HL = 16  # per-rank mqa heads before AG
DQ = 576  # kv_lora_rank + rope
NUM_LAYERS = 78


def build(sequential: bool):
    ag_out = torch.randn(WS * T, HL, DQ, device="cuda", dtype=torch.bfloat16)
    # conversion-sized independent work (~tens of us): a small matmul chain
    a = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    consumed = torch.empty(T, WS * HL, DQ, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()

    def one_layer():
        if sequential:
            c = a @ b  # independent work
            q = (
                ag_out.reshape(WS, T, HL, DQ)
                .movedim(0, 1)
                .reshape(T, WS * HL, DQ)
            )  # epilogue COPY on compute stream, after the work
        else:
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream())  # "collective complete"
            q = torch.empty(T, WS * HL, DQ, device="cuda", dtype=torch.bfloat16)
            ev = torch.cuda.Event()
            with torch.cuda.stream(side):
                side_s = torch.cuda.current_stream()
                side_s.wait_event(done)
                src = ag_out.reshape(WS, T, HL, DQ).movedim(0, 1)
                q.view(T, WS, HL, DQ).copy_(src)
                ev.record(side_s)
            c = a @ b  # independent work overlaps the side copy
            torch.cuda.current_stream().wait_event(ev)
        consumed.copy_(q)  # stand-in consumer
        return c

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_layer()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(NUM_LAYERS):
            one_layer()
    return g, ag_out, consumed


def time_graph(g, replays=200, inner=5):
    best = float("inf")
    for _ in range(replays):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        for _ in range(inner):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / inner)
    return best


def main():
    torch.cuda.set_device("cuda:0")
    torch.manual_seed(3)
    g_seq, in_seq, out_seq = build(sequential=True)
    g_ovl, in_ovl, out_ovl = build(sequential=False)
    in_ovl.copy_(in_seq)
    g_seq.replay()
    torch.cuda.synchronize()
    ref = out_seq.clone()
    g_ovl.replay()
    torch.cuda.synchronize()
    same = torch.equal(ref.view(torch.uint8), out_ovl.view(torch.uint8))
    print(f"[{'PASS' if same else 'FAIL'}] sidecopy graph: consumer bytes identical")
    if not same:
        sys.exit(1)
    t_seq = time_graph(g_seq)
    t_ovl = time_graph(g_ovl)
    print(
        f"sidecopy x{NUM_LAYERS}: sequential {t_seq*1e3:.1f} us  "
        f"overlapped {t_ovl*1e3:.1f} us  saving {1e3*(t_seq-t_ovl):.1f} us/step"
    )


if __name__ == "__main__":
    main()
