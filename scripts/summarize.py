"""Aggregate run sidecars over seeds: mean and standard deviation per configuration.

    python scripts/summarize.py out/runs --prefix p4-

A configuration is a run name without its ``-s<seed>`` suffix.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np


def load(root: Path, prefix: str):
    groups = collections.defaultdict(list)
    for f in sorted(root.glob(f"{prefix}*/metrics.json")):
        m = json.loads(f.read_text())
        groups[re.sub(r"-s\d+$", "", f.parent.name)].append(m)
    return groups


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", nargs="?", default="out/runs")
    p.add_argument("--prefix", default="")
    args = p.parse_args(argv)
    for name, runs in load(Path(args.root), args.prefix).items():
        met = runs[0]["metric"]
        other = {"EO": "DP", "DP": "EO", "EER": None}[met]
        acc = np.array([r["accuracy"] for r in runs])
        dis = np.array([r[met] for r in runs])
        s = f"{name:34s} n={len(runs)} acc={acc.mean():.3f}±{acc.std():.3f} {met}={dis.mean():.3f}±{dis.std():.3f}"
        if other:
            s += f" ({other}={np.mean([r[other] for r in runs]):.3f})"
        s += f" t={np.mean([r['wall_seconds'] for r in runs]):.0f}s"
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
