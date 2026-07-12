# SPDX-License-Identifier: Apache-2.0
"""DECODE-K variant compare hooks (kernel_variants.json kind=python_env).

The env flags GLM_MOE_DEDUP / GLM_MOE_LANE_ROWS are read by the JIT
extension PER LAUNCH (aqlm_moe_v2.cu launch_hybrid), so a single loaded
extension can be A/B'ed in-process: the harness sets the variant's env
before calling the hook; the hook computes the reference with the flags
popped (V2 kernel path) and the test output with them restored.
"""

import os

from common import kernels as K

_FLAGS = ("GLM_MOE_DEDUP", "GLM_MOE_LANE_ROWS")


def _ab(tc, gpu):
    ext = K.load_ext()
    saved = {k: os.environ.pop(k, None) for k in _FLAGS}
    try:
        ref = K.run_hybrid_gemv(ext, tc)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    got = K.run_hybrid_gemv(ext, tc)
    return ref, got


def moe_dedup(tc, gpu):
    return _ab(tc, gpu)


def moe_lane_rows(tc, gpu):
    return _ab(tc, gpu)


def moe_dedup_lane_rows(tc, gpu):
    return _ab(tc, gpu)
