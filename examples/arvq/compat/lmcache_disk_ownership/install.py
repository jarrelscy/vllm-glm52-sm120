# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply the reviewed LMCache patch only to its exact qualified source."""

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

ORIGINAL = "f123cfab752fe503589d8a8896ac78b5b6f4f64c565a45755958d0eb3bfd8c7f"
PATCHED = "5bf49e600b2feb52fd78d9654ad9a5fedcb3a9d49e2aa47ef619f5bab3123b87"
DEFAULT = (
    "/opt/vllm/.venv/lib/python3.12/site-packages/"
    "lmcache/v1/storage_backend/local_disk_backend.py"
)


def install(target: Path) -> str:
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest == PATCHED:
        return "already patched"
    if digest != ORIGINAL:
        raise ValueError(f"Unqualified LMCache source {target}: SHA256 {digest}")
    # Patch a sibling copy; reject errors before atomically replacing the source.
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        candidate = Path(directory) / target.name
        shutil.copy2(target, candidate)
        subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                str(candidate),
                str(Path(__file__).with_name("local_disk_backend.patch")),
            ],
            check=True,
            capture_output=True,
        )
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != PATCHED:
            raise ValueError("Patched source differs from qualified bytes")
        candidate.replace(target)
    return "patched"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path(DEFAULT))
    print(install(parser.parse_args().target))
