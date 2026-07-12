# SPDX-License-Identifier: Apache-2.0
"""Opt-in gate for the SM120 correctness suite.

This suite must NEVER run as part of a plain ``pytest`` invocation in the
repo (CI, upstream merges, developer habit).  Every test below this
directory is skipped unless ``GLM_SM120_TESTS=1`` is set in the
environment.  The intended entry point is ``run_all.sh``, which sets the
variable itself.

Server-priced tiers (tier3_server, tier4_canary) are double-gated on
``GLM_SM120_SERVER_TESTS=1`` because they consume GPU-hours on the live
4-GPU server; ``run_all.sh`` only sets that for ``--tier server|canary``.

All gating is self-contained in this directory: no top-level pytest/CI
config is touched, so upstream merges can never accidentally enable it.
"""

import os
import random
import sys

import pytest

# Make `from common import ...` work from any invocation directory; the
# tier directories are intentionally not packages.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OPT_IN_ENV = "GLM_SM120_TESTS"
SERVER_ENV = "GLM_SM120_SERVER_TESTS"
DIST_ENV = "GLM_SM120_DIST_TESTS"

_SERVER_DIRS = ("tier3_server", "tier4_canary")
_DIST_DIRS = ("tier2_dist",)


def pytest_configure(config):
    # Register markers locally so nothing warns and users can select with
    # `-m sm120_correctness` once opted in.
    config.addinivalue_line(
        "markers",
        "sm120_correctness: opt-in GLM-5.2 SM120 correctness/regression suite",
    )
    config.addinivalue_line(
        "markers", "sm120_tier(name): tier of the SM120 suite "
        "(cpu, kernel, dist, server, canary)")


def _tier_of(item) -> str:
    p = str(item.fspath)
    if any(d in p for d in _SERVER_DIRS):
        return "canary" if "tier4_canary" in p else "server"
    if any(d in p for d in _DIST_DIRS):
        return "dist"
    if "tier1_kernel" in p:
        return "kernel"
    return "cpu"


def pytest_collection_modifyitems(config, items):
    opted_in = os.environ.get(OPT_IN_ENV) == "1"
    server_ok = os.environ.get(SERVER_ENV) == "1"

    skip_all = pytest.mark.skip(
        reason=f"opt-in SM120 correctness suite; set {OPT_IN_ENV}=1 "
        "(intended entry point: tests/sm120_correctness/run_all.sh)")
    skip_server = pytest.mark.skip(
        reason="server/canary tier costs GPU-hours on the live server; "
        f"double-gated on {SERVER_ENV}=1 (run_all.sh --tier server|canary)")

    here = os.path.dirname(os.path.abspath(__file__))
    for item in items:
        if not str(item.fspath).startswith(here):
            continue
        tier = _tier_of(item)
        item.add_marker(pytest.mark.sm120_correctness)
        item.add_marker(pytest.mark.sm120_tier(tier))
        if not opted_in:
            item.add_marker(skip_all)
        elif tier in ("server", "canary") and not server_ok:
            item.add_marker(skip_server)


@pytest.fixture(autouse=True)
def _seeded():
    """Every test starts from the same seeds (self-contained + seeded)."""
    random.seed(0)
    try:
        import numpy as np
        np.random.seed(0)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(0)
    except ImportError:
        pass
    yield
