# SPDX-License-Identifier: Apache-2.0
"""TIER 4 — the 1M-context capability gate.

Every decode/prefill idea must KEEP the 1M window: a candidate config
must still initialize at ~1M max_model_len with a KV pool that actually
holds ~1M tokens.  Gates against the RUNNING server:

  * /v1/models reports max_model_len >= GLM_MIN_MODEL_LEN (default 950000)
  * the serving container's boot log shows a GPU KV cache sized
    >= GLM_MIN_KV_TOKENS (default 950000) tokens.

Boot the candidate with MAXLEN=950000 (or the default 1M profile) —
this test then verifies the capability was not silently forfeited
(e.g. by a memory-hungrier kernel, a bigger workspace, or a draft-model
KV reservation).
"""

import os
import pathlib
import re
import subprocess
import sys

_SUITE = pathlib.Path(__file__).resolve().parents[1]
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

from common.pytest_shim import module_main, pytest  # noqa: E402

MIN_MODEL_LEN = int(os.environ.get("GLM_MIN_MODEL_LEN", "950000"))
MIN_KV_TOKENS = int(os.environ.get("GLM_MIN_KV_TOKENS", "950000"))


def _server_container():
    name = os.environ.get("GLM_CONTAINER")
    if name:
        return name
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--format", "{{.Names}}"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    for cand in out.strip().splitlines():
        if cand.startswith("glm"):
            return cand
    return None


def test_max_model_len_keeps_1m():
    from common.server_client import max_model_len, server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    mml = max_model_len()
    if mml is None:
        pytest.skip("/v1/models does not report max_model_len")
    assert mml >= MIN_MODEL_LEN, \
        (f"max_model_len={mml} < {MIN_MODEL_LEN}: the candidate config "
         "forfeited the 1M window. If this boot intentionally used a "
         "short MAXLEN, re-run the gate against a 1M boot.")
    print(f"[1m] max_model_len={mml} >= {MIN_MODEL_LEN}")


def test_kv_pool_holds_1m_tokens():
    from common.server_client import server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    name = _server_container()
    if name is None:
        pytest.skip("cannot find serving container (set GLM_CONTAINER)")
    log = subprocess.check_output(["docker", "logs", name], text=True,
                                  stderr=subprocess.STDOUT, timeout=120)
    # vLLM logs e.g. "GPU KV cache size: 1,434,368 tokens"
    sizes = [int(m.replace(",", "")) for m in
             re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log)]
    if not sizes:
        pytest.skip("no 'GPU KV cache size' line in the container log "
                    "(rotated away? re-run right after boot, or set "
                    "GLM_CONTAINER to the booting container)")
    kv = max(sizes)
    assert kv >= MIN_KV_TOKENS, \
        (f"GPU KV cache holds only {kv} tokens (< {MIN_KV_TOKENS}): the "
         "candidate eats too much VRAM to keep the 1M window. "
         "DO NOT PROMOTE without either recovering the memory or an "
         "explicit decision to shrink the context window.")
    print(f"[1m] KV pool {kv} tokens >= {MIN_KV_TOKENS}")


if __name__ == "__main__":
    module_main(globals())
