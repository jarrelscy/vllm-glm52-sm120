# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 + AQLM hybrid quantization support for the Inkling MoE.

Checkpoint variant: ``jarrelscy/Inkling-512k-NVFP4-AQLM-hybrid``. Every
routed-MoE layer (2-65) splits its 256 experts into a small "hot" group
(kept at higher precision: NVFP4, or plain bf16 for layer 2 which has no
NVFP4 base) and a "cold" group compressed with AQLM (additive quantization,
multi-codebook, group size 8 along the input dim). See ``moe.py`` /
``hybrid_moe.py`` for the runtime side; this module only parses the
checkpoint's quantization config.

Unlike ``InklingNvfp4Config`` (whose exclude-list lives directly in
``config.json``'s ``quantization_config`` block), the per-layer hot/cold
split (``aqlm_layer_books``) lives ONLY in the separate
``hf_quant_config.json`` file, under the ``aqlm_hybrid`` key.
``config.json``'s own ``quantization_config`` is deliberately minimal (see
``build_hybrid.py:make_hybrid_config``/``write_side_files``) and does not
carry it. vLLM's HF-config loader only loads ``hf_quant_config.json`` as a
fallback when ``config.json`` has no ``quantization_config`` at all
(``transformers_utils/config.py``), so it never reaches
``hf_config.quantization_config`` for this checkpoint -- this module fetches
it directly via ``get_hf_file_to_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridLayerInfo:
    """Per-layer hot/cold split, parsed from ``aqlm_layer_books[str(layer)]``."""

    n_hot: int
    n_cold: int
    packed: bool
    hot_format: str  # "nvfp4" or "bf16"

    def __post_init__(self) -> None:
        assert self.hot_format in ("nvfp4", "bf16")
        assert self.packed == (self.hot_format == "nvfp4")


class InklingAqlmHybridConfig:
    """Checkpoint-wide NVFP4+AQLM hybrid descriptor.

    Holds the per-layer hot/cold expert counts and the AQLM book layout
    (entry counts / code dtypes, identical across layers) needed to build
    the hybrid MoE weight tensors and quant method.
    """

    QUANT_METHOD = "inkling_nvfp4_aqlm_hybrid"

    def __init__(
        self,
        group_size: int,
        w13_book_entries: list[int],
        w2_book_entries: list[int],
        w13_code_dtypes: list[str],
        w2_code_dtypes: list[str],
        layer_books: dict[int, HybridLayerInfo],
    ) -> None:
        assert group_size == 8, "Inkling AQLM hybrid only supports group_size=8"
        assert len(w13_book_entries) == len(w13_code_dtypes)
        assert len(w2_book_entries) == len(w2_code_dtypes)
        self.group_size = group_size
        self.w13_book_entries = w13_book_entries
        self.w2_book_entries = w2_book_entries
        self.w13_code_dtypes = w13_code_dtypes
        self.w2_code_dtypes = w2_code_dtypes
        self.layer_books = layer_books

    @staticmethod
    def _parse_aqlm_hybrid_block(block: dict) -> "InklingAqlmHybridConfig":
        layer_books = {
            int(layer_str): HybridLayerInfo(
                n_hot=int(v["n_hot"]),
                n_cold=int(v["n_cold"]),
                packed=bool(v["packed"]),
                hot_format=str(v["hot_format"]),
            )
            for layer_str, v in block["aqlm_layer_books"].items()
        }
        return InklingAqlmHybridConfig(
            group_size=int(block.get("group_size", 8)),
            w13_book_entries=list(block["w13_book_entries"]),
            w2_book_entries=list(block["w2_book_entries"]),
            w13_code_dtypes=list(block["w13_code_dtypes"]),
            w2_code_dtypes=list(block["w2_code_dtypes"]),
            layer_books=layer_books,
        )

    @classmethod
    def from_model(
        cls, model: str, revision: str | None
    ) -> "InklingAqlmHybridConfig | None":
        """Detect + parse the hybrid config from the model repo/path.

        Returns ``None`` if this is not a hybrid checkpoint (no
        ``hf_quant_config.json``, or no ``aqlm_hybrid`` block in it) -- the
        normal case for plain-NVFP4 or bf16 Inkling checkpoints.
        """
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        hq = get_hf_file_to_dict(
            "hf_quant_config.json", model, revision or "main"
        )
        if hq is None:
            return None
        block = hq.get("aqlm_hybrid")
        if block is None:
            return None
        if block.get("quant_method") != cls.QUANT_METHOD:
            return None
        return cls._parse_aqlm_hybrid_block(block)

    def is_hybrid(self, layer_id: int) -> bool:
        return layer_id in self.layer_books

    def layer_info(self, layer_id: int) -> HybridLayerInfo:
        return self.layer_books[layer_id]
