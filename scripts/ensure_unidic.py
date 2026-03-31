"""Ensure Japanese UniDic resources are installed when available.

The ``unidic`` package ships without the actual dictionary data and requires
``python -m unidic download`` after installation. Japanese synthesis fails at
runtime when that step is skipped, so startup scripts call this helper to
perform the one-time download automatically.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_unidic() -> None:
    """Download UniDic data if the full ``unidic`` package is installed."""
    if _has_module("unidic_lite"):
        print("unidic-lite is installed; no extra dictionary download is required.")
        return

    if not _has_module("unidic"):
        print("UniDic is not installed; skipping Japanese dictionary bootstrap.")
        return

    import unidic

    dicdir = Path(getattr(unidic, "DICDIR", Path(unidic.__file__).resolve().parent / "dicdir"))
    mecabrc = dicdir / "mecabrc"
    if mecabrc.exists():
        return

    print(f"UniDic data is missing at {mecabrc}. Downloading...")
    subprocess.run([sys.executable, "-m", "unidic", "download"], check=True)

    if not mecabrc.exists():
        raise RuntimeError(
            f"UniDic download completed but {mecabrc} is still missing."
        )


if __name__ == "__main__":
    ensure_unidic()

