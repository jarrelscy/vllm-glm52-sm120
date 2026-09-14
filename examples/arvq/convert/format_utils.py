# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 dictionary encoding shared by checkpoint conversion."""

import torch
from fit_codebooks import LEVELS


def pack_cb(fit):
    c = torch.cat([fit["c0"], fit["c1"]]).cuda()
    levels = torch.tensor(LEVELS, device="cuda")
    n = (c[:, :, None] - levels).abs().argmin(-1)
    assert torch.equal(levels[n], c)
    return (
        (n.to(torch.int64) << (torch.arange(8, device="cuda") * 4))
        .sum(-1)
        .to(torch.int32)
    )
