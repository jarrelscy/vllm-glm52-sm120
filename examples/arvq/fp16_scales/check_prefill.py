# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes as C
import json
from pathlib import Path

import numpy as np
import torch

root = Path("/opt/vllm/vllm/model_executor/layers/quantization/arvq")
base = C.CDLL("/work/baseline-prefill.so")
new = C.CDLL(str(root / "prefill.so"))
fused = C.CDLL(str(root / "decode_gather.so"))
ptr = lambda t: C.c_void_p(t.data_ptr())
stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
rng = np.random.default_rng(43)
N, K = 64, 256
pairs = rng.integers(0, 65536, (N, K // 8), dtype=np.uint16)
pairs[:, 0] = 0
pairs[:, -1] = 65535
b = rng.integers(0, 2**32, 512, dtype=np.uint32)
f = (
    pairs.reshape(N // 16, 2, 8, K // 64, 2, 4)
    .transpose(0, 3, 4, 1, 2, 5)
    .copy()
    .reshape(N // 16, K // 64, 128)
)
p = f[..., ::2].astype(np.uint32) | (f[..., 1::2].astype(np.uint32) << 16)
cw = torch.from_numpy(p).cuda()
cb = torch.from_numpy(b).cuda()
s8 = torch.randint(0, 127, (N // 16, K // 128, 16), device="cuda", dtype=torch.uint8)
s16 = s8.view(torch.float8_e4m3fn).half()
res = []
for dtype, symbol in [
    (torch.float16, "arvq_dequant_fp16_8x8"),
    (torch.bfloat16, "arvq_dequant_8x8"),
]:
    outs = [torch.empty(N, K, device="cuda", dtype=dtype) for _ in range(2)]
    for lib, sc, out in zip([base, new], [s8, s16], outs):
        fn = getattr(lib, symbol)
        fn.argtypes = [C.c_void_p] * 3 + [
            C.c_float,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_void_p,
        ]
        assert fn(ptr(cw), ptr(cb), ptr(sc), 1.0, ptr(out), N, K, stream) == 0
    assert torch.equal(*outs)
    res.append({"path": symbol, "matched_scales_bitexact": True})
s16 = (s16 * 0.973).half()
out = torch.empty(N, K, device="cuda", dtype=torch.float16)
x = torch.randn(5, K, device="cuda", dtype=torch.bfloat16)
routes = torch.tensor([9, 0, 5], device="cuda")
rows = torch.empty(3, K, device="cuda", dtype=torch.float16)
fn = fused.arvq_dequant_gather_fp16_8
fn.argtypes = (
    [C.c_void_p] * 3
    + [C.c_float, C.c_void_p, C.c_int, C.c_int, C.c_void_p]
    + [C.c_void_p] * 3
    + [C.c_int] * 2
)
assert (
    fn(
        ptr(cw),
        ptr(cb),
        ptr(s16),
        1.0,
        ptr(out),
        N,
        K,
        stream,
        ptr(x),
        ptr(routes),
        ptr(rows),
        3,
        2,
    )
    == 0
)
levels = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)
books = levels[(b[:, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15]
weights = (books[pairs & 255] + books[256 + (pairs >> 8)]).reshape(N, K)
sc = s16.cpu().float().numpy().transpose(0, 2, 1).reshape(N, K // 128)
expected = torch.from_numpy(weights * np.repeat(sc, 128, axis=1)).half().cuda()
assert torch.equal(out, expected)
assert torch.equal(rows, x[routes // 2].half())
res.append(
    {
        "path": "fused_gather",
        "non_fp8_scales_cpu_oracle_bitexact": True,
        "gather_bitexact": True,
    }
)
print(json.dumps(res, indent=2))
