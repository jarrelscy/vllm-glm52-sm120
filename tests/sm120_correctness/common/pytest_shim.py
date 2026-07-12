# SPDX-License-Identifier: Apache-2.0
"""pytest, or a minimal stdlib shim of it.

The server/canary tiers must run on a bare host (no pytest, no numpy,
no torch — only stdlib + the network).  Test modules in those tiers do

    from common.pytest_shim import pytest

and gain: ``pytest.skip``, ``pytest.importorskip``, ``pytest.fail``,
``pytest.mark.*`` (no-ops) — enough for dual-mode files that are both
pytest-collectible (inside the container) and directly executable
(``python3 test_x.py`` on the host).  ``run_module()`` is the tiny
runner used in ``__main__`` mode: runs every ``test_*`` callable,
prints one PASS/FAIL/SKIP line each, exits non-zero on failure.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import traceback

try:
    import pytest  # type: ignore

    _HAVE_PYTEST = True
    try:
        SkipException = pytest.skip.Exception  # pytest's Skipped
    except AttributeError:  # very old pytest
        SkipException = Exception
except ImportError:
    _HAVE_PYTEST = False

    class SkipException(Exception):
        pass

    class _Mark:
        def __getattr__(self, _name):
            def deco(*a, **kw):
                if len(a) == 1 and callable(a[0]) and not kw:
                    return a[0]
                return lambda f: f
            return deco

    class _PytestShim:
        mark = _Mark()

        @staticmethod
        def skip(reason: str = ""):
            raise SkipException(reason)

        @staticmethod
        def fail(reason: str = ""):
            raise AssertionError(reason)

        @staticmethod
        def importorskip(name: str, reason: str | None = None):
            try:
                return importlib.import_module(name)
            except ImportError:
                raise SkipException(reason or f"requires {name}")

        @staticmethod
        def fixture(*a, **kw):
            if len(a) == 1 and callable(a[0]):
                return a[0]
            return lambda f: f

        @staticmethod
        def main(args=None):
            raise RuntimeError("real pytest not installed")

    pytest = _PytestShim()  # type: ignore


def run_module(mod_globals: dict) -> int:
    """Mini runner for __main__ mode: junit-ish one line per test."""
    name = mod_globals.get("__file__", "<module>")
    tests = [(k, v) for k, v in sorted(mod_globals.items())
             if k.startswith("test_") and callable(v)]
    failed = skipped = passed = 0
    for tname, fn in tests:
        if inspect.signature(fn).parameters:
            print(f"SKIP {tname}: requires pytest fixtures/parametrize "
                  "(run inside the container with pytest)")
            skipped += 1
            continue
        try:
            fn()
            print(f"PASS {tname}")
            passed += 1
        except SkipException as e:
            print(f"SKIP {tname}: {e}")
            skipped += 1
        except BaseException as e:  # noqa: BLE001
            if e.__class__.__name__ in ("Skipped", "SkipException"):
                print(f"SKIP {tname}: {e}")
                skipped += 1
                continue
            failed += 1
            print(f"FAIL {tname}: {e.__class__.__name__}: {e}")
            traceback.print_exc()
    print(f"== {name}: {passed} passed, {failed} failed, {skipped} skipped ==")
    return 1 if failed else 0


def module_main(mod_globals: dict):
    sys.exit(run_module(mod_globals))
