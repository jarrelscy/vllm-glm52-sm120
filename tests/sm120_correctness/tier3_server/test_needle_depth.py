# SPDX-License-Identifier: Apache-2.0
"""TIER 3 — needle-in-a-haystack coherence-at-depth gate.

Guards every decode/prefill/DCP/cudagraph change (ideas 1-9): a needle
placed mid-context must be retrieved verbatim at 32K and 130K (and 749K
when the server's max_model_len allows and GLM_SM120_LONG=1 — the 749K
prefill is expensive).

This is the primary "the KV cache / attention still works at depth"
gate; the SM100-class silent corruption fails it immediately at the
depths where the broken kernel engages.
"""

import os
import pathlib
import sys

_SUITE = pathlib.Path(__file__).resolve().parents[1]
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

from common.pytest_shim import module_main, pytest  # noqa: E402

DEPTHS = (32_000, 130_000)
LONG_DEPTH = 749_000


def _needle(depth: int):
    from common.server_client import needle_probe, server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    ok, ans, ptoks = needle_probe(depth)
    print(f"[needle @{depth}] prompt_tokens={ptoks} answer={ans[:80]!r}")
    assert ok, \
        (f"NEEDLE FAIL at ~{depth} tokens (prompt_tokens={ptoks}): "
         f"answer {ans[:200]!r} — coherence at depth is broken. "
         "STOP: this is the silent-corruption signature.")


def test_needle_32k():
    _needle(DEPTHS[0])


def test_needle_130k():
    _needle(DEPTHS[1])


def test_needle_749k():
    from common.server_client import max_model_len, server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    if os.environ.get("GLM_SM120_LONG") != "1":
        pytest.skip("749K needle costs a ~750K prefill; set GLM_SM120_LONG=1")
    mml = max_model_len()
    if mml is not None and mml < LONG_DEPTH + 5000:
        pytest.skip(f"server max_model_len={mml} < {LONG_DEPTH}")
    _needle(LONG_DEPTH)


if __name__ == "__main__":
    module_main(globals())
