"""Apply the selection rule to the validation runs; write ``campaigns/selected.json``.

The rule (PLAN.md §4, fixed before the selection runs): for each dataset and
target measure, among the configurations whose mean validation accuracy over
the selection seeds is at least replay's minus ``--tolerance`` (1 point), take
the one with the lowest mean validation disparity; replay itself is a
candidate. Ties go to the higher accuracy.

    python scripts/select_config.py out/runs
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"mn": "MNIST", "fm": "FashionMNIST", "bm": "BiasedMNIST", "dr": "Drug", "bi": "Bios"}


def campaign_args(campaign: Path) -> dict:
    """Run name -> its run_cil.py arguments, without seed and split."""
    out = {}
    for line in campaign.read_text().splitlines():
        if not line.strip():
            continue
        name, *rest = shlex.split(line)
        keep, skip = [], False
        for tok in rest:
            if skip:
                skip = False
                continue
            if tok in ("--seed", "--split"):
                skip = True
                continue
            keep.append(tok)
        out[name] = keep
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default="out/runs")
    p.add_argument("--campaign", default=str(ROOT / "campaigns" / "select.txt"))
    p.add_argument("--prefix", default="sel-", help="run-name prefix of the selection runs ('bio-sel-' for BiasBios)")
    p.add_argument("--dataset-code", default=None, help="dataset code when the prefix carries none (BiasBios: bi)")
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--seeds", type=int, default=3, help="seeds a configuration needs to be considered")
    p.add_argument("--out", default=str(ROOT / "campaigns" / "selected.json"))
    p.add_argument("--group-balanced", nargs="*", default=["BiasedMNIST"],
                   help="datasets whose test split has a group composition fixed by construction (Biased "
                        "MNIST: half of each class in other colours); there the rule's accuracy is the "
                        "group-balanced validation accuracy, the validation estimate of test accuracy")
    args = p.parse_args(argv)

    cargs = campaign_args(Path(args.campaign))
    runs: dict[tuple, dict] = {}
    for f in sorted(Path(args.root).glob(f"{args.prefix}*/metrics.json")):
        m = json.loads(f.read_text())
        name = f.parent.name
        if args.dataset_code:
            mm = re.match(re.escape(args.prefix) + r"(eer|eo|dp)-(.+)-s(\d+)$", name)
            if not mm:
                continue
            key, cfg = (args.dataset_code, mm.group(1).upper()), mm.group(2)
        else:
            mm = re.match(re.escape(args.prefix) + r"(\w+)-(eer|eo|dp)-(.+)-s(\d+)$", name)
            if not mm:
                continue
            key, cfg = (mm.group(1), mm.group(2).upper()), mm.group(3)
        runs.setdefault(key, {}).setdefault(cfg, []).append((name, m))
    # a replay run (eta = 0) measures EO and DP alike: where one measure has no replay run
    # of its own and both share the learning rate and replay weight (BiasBios), it is shared
    for (ds, met) in list(runs):
        other = (ds, "DP" if met == "EO" else "EO")
        if "replay" not in runs[(ds, met)] and other in runs and "replay" in runs[other]:
            runs[(ds, met)]["replay"] = runs[other]["replay"]

    selected = {}
    for (ds, met), configs in sorted(runs.items()):
        stats = {}
        acc_key = "accuracy_group_balanced" if DATASETS[ds] in args.group_balanced else "accuracy"
        for cfg, rs in configs.items():
            if len(rs) < args.seeds:
                continue
            if any(acc_key not in m for _, m in rs):
                continue
            acc = float(np.mean([m[acc_key] for _, m in rs]))
            dis = float(np.mean([m[met] for _, m in rs]))
            stats[cfg] = (acc, dis, rs[0][0])
        if "replay" not in stats:
            print(f"{ds}/{met}: replay incomplete, skipped")
            continue
        ref_acc, ref_dis, _ = stats["replay"]
        cands = {c: s for c, s in stats.items() if s[0] >= ref_acc - args.tolerance}
        best = min(cands, key=lambda c: (round(cands[c][1], 6), -cands[c][0]))
        acc, dis, example = stats[best]
        dataset = DATASETS[ds]
        selected[f"{dataset}/{met}"] = {
            "config": best, "args": cargs.get(example, []), "accuracy_used": acc_key,
            "val_accuracy": acc, "val_disparity": dis,
            "replay_val_accuracy": ref_acc, "replay_val_disparity": ref_dis,
            "n_configs": len(stats), "n_candidates": len(cands),
        }
        print(f"{dataset}/{met}: {best}  val acc {acc:.3f} (replay {ref_acc:.3f})  "
              f"{met} {dis:.3f} (replay {ref_dis:.3f})  [{len(cands)}/{len(stats)} within tolerance]")
        for c in sorted(stats, key=lambda c: stats[c][1]):
            mark = "*" if c == best else (" " if c in cands else "x")
            print(f"   {mark} {c:22s} acc {stats[c][0]:.3f}  {met} {stats[c][1]:.3f}")
    if Path(args.out).exists():
        previous = json.loads(Path(args.out).read_text())
        previous.update(selected)
        selected = previous
    Path(args.out).write_text(json.dumps(selected, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
