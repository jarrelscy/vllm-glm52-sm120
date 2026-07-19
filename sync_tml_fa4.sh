#!/usr/bin/env bash
# Vendor-sync the tml-fa4 kernel from the canonical fork into this tree's
# vllm/third_party/tml_fa4, rewriting imports flash_attn.cute ->
# vllm.third_party.tml_fa4. This mirrors EXACTLY what
# cmake/external_projects/tml_fa4.cmake's install(CODE ...) step does, so a
# plain `pip install -e .` build produces the same result. Run this after
# editing the fork so a dev-loop bind-mount picks up fork changes without a
# full rebuild.
#
#   Usage: ./sync_tml_fa4.sh [FORK_SRC]   (default: $TML_FA4_SRC_DIR or the fork)
set -euo pipefail
SRC="${1:-${TML_FA4_SRC_DIR:-/home/jarrelscy/tml-fa4-fork}}/flash_attn/cute"
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vllm/third_party/tml_fa4"
[ -d "$SRC" ] || { echo "src not found: $SRC" >&2; exit 1; }
mkdir -p "$DST"
n=0
while IFS= read -r -d '' f; do
  rel="${f#"$SRC"/}"
  mkdir -p "$DST/$(dirname "$rel")"
  sed 's/flash_attn\.cute/vllm.third_party.tml_fa4/g' "$f" > "$DST/$rel"
  n=$((n+1))
done < <(find "$SRC" -name '*.py' -print0)
echo "synced $n files: $SRC -> $DST"
