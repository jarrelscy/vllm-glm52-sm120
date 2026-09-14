# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare host overlays while retaining unrelated local indexer features."""

import argparse
import ast
from pathlib import Path

from prepare_compact_workspace_overlay import prepare as prepare_compact

ENV_ANCHOR = '    "VLLM_GLM_DCP_AG_RAW_TOPK": lambda: bool('
ENV_ENTRY = """    "VLLM_GLM_RAW_KV_GATHER": lambda: bool(
        int(os.getenv("VLLM_GLM_RAW_KV_GATHER", "0"))
    ),
"""
SIGNATURE = "    row_starts: torch.Tensor | None = None,\n) -> None:"
CALL = "                row_starts=chunk.cu_seqlen_ks,\n"
BODY_ANCHOR = "    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's"


def _apply(source: str, replacements: tuple, marker: str) -> str:
    """Reject partial patches before applying any text replacement."""
    if marker in source:
        restored = source
        for original, replacement in replacements:
            if restored.count(replacement) != 1:
                raise ValueError("Partial or altered serving overlay")
            restored = restored.replace(replacement, original, 1)
        if marker in restored:
            raise ValueError("Unexpected serving overlay marker")
        # Also verify original anchors on the recovered input.
        _apply(restored, replacements, marker)
        compile(source, "serving_overlay", "exec")
        return source
    for original, _ in replacements:
        if source.count(original) != 1:
            raise ValueError(f"Expected one host anchor: {original}")
    for original, replacement in replacements:
        source = source.replace(original, replacement, 1)
    compile(source, "serving_overlay", "exec")
    return source


def indexer_replacements(repository_source: str) -> tuple:
    """Use the reviewed repository hook rather than maintain a second body."""
    tree = ast.parse(repository_source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_merge_dcp_topk_global"
    )
    hook = next(
        node
        for node in function.body
        if isinstance(node, ast.If)
        and "VLLM_EXPERIMENT_DCP_BYTEPACK" in ast.unparse(node.test)
    )
    lines = repository_source.splitlines(keepends=True)
    body = "".join(lines[hook.lineno - 1 : hook.end_lineno]) + "\n"
    signature = SIGNATURE.replace(
        ") -> None:",
        "    bytepack_c1_prefill: bool = False,\n"
        "    bytepack_context: int = 0,\n"
        "    bytepack_layer: str | None = None,\n) -> None:",
    )
    call = (
        CALL
        + """                bytepack_c1_prefill=(
                    attn_metadata_narrowed.num_prefills == 1
                    and attn_metadata_narrowed.num_decodes == 0
                ),
                bytepack_context=attn_metadata_narrowed.max_seq_len,
                bytepack_layer=k_cache_prefix,
"""
    )
    replacements = (
        (SIGNATURE, signature),
        (BODY_ANCHOR, body + BODY_ANCHOR),
        (CALL, call),
    )
    # Repository changes must be reviewed rather than silently drifting.
    for _, replacement in replacements:
        if repository_source.count(replacement) != 1:
            raise ValueError("Repository bytepack hook changed")
    return replacements


def prepare_indexer(source: str, repository_source: str) -> str:
    return _apply(
        prepare_compact(source),
        indexer_replacements(repository_source),
        "bytepack_",
    )


def prepare_envs(source: str) -> str:
    return _apply(
        source, ((ENV_ANCHOR, ENV_ENTRY + ENV_ANCHOR),), "VLLM_GLM_RAW_KV_GATHER"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexer-source", type=Path, required=True)
    parser.add_argument("--envs-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).parents[3])
    args = parser.parse_args()
    repository_source = (
        args.repository / "vllm/model_executor/layers/sparse_attn_indexer.py"
    ).read_text()
    # Validate both before writing either output; input files are never edited.
    outputs = {
        "sparse_attn_indexer.py": prepare_indexer(
            args.indexer_source.read_text(), repository_source
        ),
        "envs.py": prepare_envs(args.envs_source.read_text()),
    }
    if any(
        (args.output_dir / name).resolve()
        in (args.indexer_source.resolve(), args.envs_source.resolve())
        for name in outputs
    ):
        raise ValueError("Output directory would overwrite a source file")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        target = args.output_dir / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(content)
        temporary.replace(target)


if __name__ == "__main__":
    main()
