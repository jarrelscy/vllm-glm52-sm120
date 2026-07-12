# SPDX-License-Identifier: Apache-2.0
"""TIER 1 (CPU-ONLY) — kernel-variant registry schema + shipped-source
flag inventory.

Runs everywhere (stdlib + json).  Two jobs:
  * the registry parses and every entry is well-formed (an optimizer
    agent appending a malformed entry fails HERE, loudly, before their
    variant silently escapes A/B coverage);
  * every env/compile flag present in the shipped kernel source has a
    registry entry — new `#ifdef`/getenv knobs cannot be added without
    registering them for equivalence testing.
"""

import pathlib
import re
import sys

_SUITE = pathlib.Path(__file__).resolve().parents[1]
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

from common.pytest_shim import module_main  # noqa: E402
from common.variant_registry import REGISTRY_PATH, load_registry  # noqa: E402

CU_SOURCE = _SUITE.parents[1] / "csrc" / "quantization" / "aqlm_moe" / \
    "aqlm_moe_v2.cu"
HYBRID_PY = _SUITE.parents[1] / "vllm" / "model_executor" / "layers" / \
    "quantization" / "nvfp4_aqlm_hybrid.py"

# Compile-time knobs of the shipped source that are BUILD-config, not
# kernel variants (documented exclusions).
KNOWN_NON_VARIANT_DEFINES: set[str] = set()
# Runtime env knobs of the wrapper that do not change kernel numerics.
KNOWN_NON_VARIANT_ENVS = {"VLLM_HYBRID_EXPERT_STATS"}


def test_registry_schema():
    variants = load_registry()
    assert len(variants) >= 3, \
        f"registry {REGISTRY_PATH} lost its built-in variants"


def test_all_source_flags_registered():
    variants = load_registry()
    registered_defines = set()
    for v in variants:
        for c in v.get("cflags", []):
            m = re.match(r"-D([A-Za-z_0-9]+)", c)
            if m:
                registered_defines.add(m.group(1))

    src = CU_SOURCE.read_text()
    defines = set(re.findall(r"#ifndef\s+([A-Z_0-9]+)\n#define", src)) | \
        set(re.findall(r"#if\s+!?([A-Z_0-9]+)\b", src))
    defines -= {"defined"} | KNOWN_NON_VARIANT_DEFINES
    unregistered = {d for d in defines
                    if d not in registered_defines}
    assert not unregistered, \
        (f"compile-time kernel knobs without a kernel_variants.json "
         f"entry: {sorted(unregistered)} — register them so the A/B "
         "equivalence test covers every build variant")


def test_all_env_flags_registered():
    variants = load_registry()
    registered_envs = set()
    for v in variants:
        registered_envs.update(v.get("env", {}))

    src = HYBRID_PY.read_text()
    envs = set(re.findall(r"os\.environ\.get\(\s*[\"'](VLLM_[A-Z_0-9]+)",
                          src))
    envs -= KNOWN_NON_VARIANT_ENVS
    unregistered = envs - registered_envs
    assert not unregistered, \
        (f"runtime env kernel knobs without a kernel_variants.json entry: "
         f"{sorted(unregistered)} — register them (kind=python_env with a "
         "compare hook) so they are A/B gated")


if __name__ == "__main__":
    module_main(globals())
