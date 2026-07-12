# SPDX-License-Identifier: Apache-2.0
"""TIER 3 — attention backend ROUTING invariants (the SM100 root cause).

The SM100 bug was a ROUTING regression: fp8_ds_mla landed on a broken
kernel and the model silently truncated deep reasoning — no crash, no
error.  These tests pin the SM120 routing table so that class of change
can never land silently:

  CPU part (no server, needs importable vllm — runs in the container):
    * _get_backend_priorities(use_mla=True, major==12) is EXACTLY
      [TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120] — nothing reorders or
      inserts a backend ahead of the SM120 sparse path.
    * the enum resolves to FlashInferMLASparseSM120Backend, which only
      supports compute capability 12.
    * SOURCE TRIPWIRE: the impl's hard guard
      `kv_cache_dtype != "fp8_ds_mla" -> raise` is still present, so a
      plain-e4m3 cache can never silently reach a kernel that assumes
      the 656-byte ds_mla layout (it must fail LOUDLY at init).

  Server part (live server, GLM_SM120_SERVER_TESTS=1):
    * the running container's log shows FLASHINFER_MLA_SPARSE_SM120 was
      selected and the KV cache dtype is fp8_ds_mla.
"""

import pathlib
import re
import subprocess
import sys

_SUITE = pathlib.Path(__file__).resolve().parents[1]
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

from common.pytest_shim import module_main, pytest  # noqa: E402

REPO_ROOT = _SUITE.parents[1]


# ---------------------------------------------------------------------------
# CPU part
# ---------------------------------------------------------------------------


def test_sm120_priority_list():
    vllm_cuda = pytest.importorskip(
        "vllm.platforms.cuda",
        reason="needs importable vllm (glm52-sm120 container)")
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    got = vllm_cuda._get_backend_priorities(
        use_mla=True,
        device_capability=DeviceCapability(major=12, minor=0),
        num_heads=96,
        kv_cache_dtype="fp8_ds_mla",
    )
    assert got == [AttentionBackendEnum.TRITON_MLA,
                   AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120], \
        (f"SM120 MLA backend priorities changed: {got!r}. This is the "
         "routing table whose regression caused the SM100 silent-collapse "
         "bug — any change here must be deliberate and re-validated with "
         "the full server tier + canary.")

    # and independent of kv dtype hints (the sparse path is selected by
    # supports_combination, not by silently swapping priorities)
    got2 = vllm_cuda._get_backend_priorities(
        use_mla=True,
        device_capability=DeviceCapability(major=12, minor=0),
        num_heads=96,
        kv_cache_dtype="fp8_e4m3",
    )
    assert got2 == got


def test_sm120_backend_class_contract():
    pytest.importorskip("vllm")
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cls = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
    assert cls.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert cls.supports_compute_capability(DeviceCapability(12, 0))
    assert not cls.supports_compute_capability(DeviceCapability(10, 0)), \
        "SM120 sparse backend must not be eligible on SM100"
    assert "fp8_ds_mla" in cls.supported_kv_cache_dtypes


def test_impl_hard_fails_on_non_ds_mla_source_tripwire():
    """The e4m3-on-ds-mla-kernel guard must stay: a non-packed cache
    dtype reaching the SM120 sparse impl must raise at init, never run.

    Static tripwire (works without constructing the impl): the guard
    block exists verbatim in the source.  If you refactor it, keep an
    equivalent hard failure and update this pattern.
    """
    src = (REPO_ROOT / "vllm" / "v1" / "attention" / "backends" / "mla" /
           "flashinfer_mla_sparse_sm120.py").read_text()
    pat = re.compile(
        r"if\s+self\.kv_cache_dtype\s*!=\s*[\"']fp8_ds_mla[\"']\s*:"
        r"\s*\n\s*raise\s+NotImplementedError", re.S)
    assert pat.search(src), \
        ("flashinfer_mla_sparse_sm120.py no longer hard-fails on "
         "kv_cache_dtype != 'fp8_ds_mla'. That guard is the last line of "
         "defense against the SM100 bug class (wrong cache layout fed to "
         "the sparse kernel => silent garbage attention). Restore it or "
         "an equivalent loud failure.")


def test_cuda_routing_block_source_tripwire():
    """cuda.py must keep an explicit major==12 branch (not fall through
    to the generic list that puts non-SM120 sparse backends in play)."""
    src = (REPO_ROOT / "vllm" / "platforms" / "cuda.py").read_text()
    assert re.search(r"major\s*==\s*12", src), \
        "cuda.py lost its explicit SM120 (major==12) routing branch"


# ---------------------------------------------------------------------------
# Server part (double-gated by conftest: GLM_SM120_SERVER_TESTS=1)
# ---------------------------------------------------------------------------


def _server_container() -> str | None:
    import os
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


def _server_log(name: str) -> str:
    return subprocess.check_output(
        ["docker", "logs", name], text=True, stderr=subprocess.STDOUT,
        timeout=120)


def test_server_selected_sm120_sparse_backend():
    from common.server_client import server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    name = _server_container()
    if name is None:
        pytest.skip("cannot find the serving container (set GLM_CONTAINER)")
    log = _server_log(name)
    assert "FLASHINFER_MLA_SPARSE_SM120" in log, \
        ("server log never mentions FLASHINFER_MLA_SPARSE_SM120 — the "
         "SM120 sparse MLA backend was NOT selected. THIS IS THE SM100 "
         "BUG CLASS. Do not ship.")
    assert "fp8_ds_mla" in log, \
        "server log never mentions fp8_ds_mla — KV cache dtype changed"
    bad = re.findall(r"Using (FLASHMLA_SPARSE|FLASHINFER_MLA_SPARSE)\b(?!_SM120)",
                     log)
    assert not bad, f"a non-SM120 sparse backend was selected: {bad}"


if __name__ == "__main__":
    module_main(globals())
