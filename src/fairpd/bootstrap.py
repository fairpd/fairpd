"""Make the third-party code importable and point it at the data root.

FSW (https://github.com/jyparkkr/FSW) is a flat collection of top-level
packages (``algorithms``, ``datasets``, ``metrics``, ``trainers``,
``backbones``, ``configs``) that expects to be run from its own directory, and
it imports CL-Gym (``cl_gym``). Both live under ``third_party/``; this module
puts them on ``sys.path``. It must be imported before anything from FSW or
CL-Gym, because CL-Gym reads ``CL_DATA_DIR`` at import time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party"
FSW_DIR = Path(os.environ.get("FSW_DIR", THIRD_PARTY / "FSW"))
CLGYM_DIR = Path(os.environ.get("CLGYM_DIR", THIRD_PARTY / "cl-gym"))


def default_data_dir() -> Path:
    """Data root: ``CL_DATA_DIR`` if set, else ``$GLOBALSCRATCH/cl_data`` when that
    scratch exists and is writable (it does not on every cluster), else ``~/data/cl_data``."""
    if os.environ.get("CL_DATA_DIR"):
        return Path(os.environ["CL_DATA_DIR"])
    scratch = os.environ.get("GLOBALSCRATCH")
    if scratch and os.path.isdir(scratch) and os.access(scratch, os.W_OK):
        return Path(scratch) / "cl_data"
    return Path.home() / "data" / "cl_data"


def setup(data_dir: str | os.PathLike | None = None) -> Path:
    """Put FSW and CL-Gym on ``sys.path`` and export ``CL_DATA_DIR``; return the data root."""
    root = Path(data_dir) if data_dir else default_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["CL_DATA_DIR"] = str(root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for p in (CLGYM_DIR, FSW_DIR):
        if not p.is_dir():
            raise FileNotFoundError(f"third-party directory missing: {p}")
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return root
