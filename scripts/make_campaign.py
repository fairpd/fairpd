"""Write campaign files (``campaigns/<name>.txt``) for ``scripts/campaign.py``.

Campaigns are generated here and committed with the code, so that every
machine that runs a campaign runs the same list. Run names are unique across
clusters (each name carries the campaign's prefix).

    python scripts/make_campaign.py pilot3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# FSW's selected learning rate and replay weight per dataset and measure
# (App. tables of arXiv:2410.01324v3, the comments under each table).
FSW_SETTINGS = {
    ("MNIST", "EER"): dict(lr=0.001, tau=10.0, alpha=0.001, lmbd=1.0),
    ("FashionMNIST", "EER"): dict(lr=0.001, tau=10.0, alpha=0.001, lmbd=0.1),
    ("BiasedMNIST", "EO"): dict(lr=0.001, tau=5.0, alpha=0.0005, lmbd=0.5),
    ("BiasedMNIST", "DP"): dict(lr=0.001, tau=10.0, alpha=0.01, lmbd=0.5),
    ("Drug", "EO"): dict(lr=0.01, tau=1.0, alpha=0.01, lmbd=0.1),
    ("Drug", "DP"): dict(lr=0.001, tau=1.0, alpha=0.002, lmbd=0.1),
    ("Bios", "EO"): dict(lr=2e-5, tau=2.0, alpha=0.001, lmbd=1.0),
    ("Bios", "DP"): dict(lr=2e-5, tau=2.0, alpha=0.001, lmbd=1.0),
}
SHORT = {"MNIST": "mn", "FashionMNIST": "fm", "BiasedMNIST": "bm", "Drug": "dr", "Bios": "bi"}


def fmt(x: float) -> str:
    return f"{x:g}"


def line(name: str, dataset: str, metric: str, method: str, seed: int, split: str = "test", **kw) -> str:
    s = FSW_SETTINGS[(dataset, metric)]
    args = [f"--dataset {dataset}", f"--metric {metric}", f"--method {method}", f"--seed {seed}",
            f"--split {split}", f"--lr {fmt(kw.pop('lr', s['lr']))}", f"--tau {fmt(kw.pop('tau', s['tau']))}"]
    if method == "FSW":
        args += [f"--alpha {fmt(kw.pop('alpha', s['alpha']))}", f"--lambda {fmt(kw.pop('lmbd', s['lmbd']))}"]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        args.append(flag if v is True else f"{flag} {v}")
    return f"{name} " + " ".join(args)


def pilot3():
    lines = []
    for seed in (0, 1):
        for ds, met in (("BiasedMNIST", "EO"), ("BiasedMNIST", "DP"), ("Drug", "EO"), ("Drug", "DP")):
            p = f"p3-{SHORT[ds]}-{met.lower()}"
            lines.append(line(f"{p}-replay-s{seed}", ds, met, "pd", seed, pd_eta=0))
            lines.append(line(f"{p}-bal-s{seed}", ds, met, "pd", seed, pd_eta=0, pd_base="balanced"))
            for eta in (0.1, 1, 10):
                lines.append(line(f"{p}-pd{fmt(eta)}-s{seed}", ds, met, "pd", seed, pd_eta=eta))
            if (ds, met) == ("BiasedMNIST", "EO"):
                for eta in (0.1, 1):
                    lines.append(line(f"{p}-balpd{fmt(eta)}-s{seed}", ds, met, "pd", seed, pd_eta=eta,
                                      pd_base="balanced"))
                lines.append(line(f"{p}-fsw-s{seed}", ds, met, "FSW", seed))
        for ds in ("MNIST", "FashionMNIST"):
            p = f"p3-{SHORT[ds]}-eer"
            lines.append(line(f"{p}-replay-s{seed}", ds, "EER", "pd", seed, pd_eta=0))
            lines.append(line(f"{p}-replay1ep-s{seed}", ds, "EER", "pd", seed, pd_eta=0, epochs=1))
            lines.append(line(f"{p}-fsw-s{seed}", ds, "EER", "FSW", seed))
    return lines


def pilot4():
    lines = []
    for seed in (0, 1):
        for ds in ("MNIST", "FashionMNIST"):
            p = f"p4-{SHORT[ds]}-eer"
            for eta in (0.1, 1):
                for k in (1, 10):
                    lines.append(line(f"{p}-pd{fmt(eta)}-brake{k}-s{seed}", ds, "EER", "pd", seed, pd_eta=eta,
                                      pd_estimator="probe", pd_brake=k))
        for ds in ("BiasedMNIST", "Drug"):
            p = f"p4-{SHORT[ds]}-dp"
            for eta in (0.1, 1, 10):
                lines.append(line(f"{p}-rate{fmt(eta)}-s{seed}", ds, "DP", "pd", seed, pd_eta=eta, pd_rate=True))
            lines.append(line(f"{p}-balrate1-s{seed}", ds, "DP", "pd", seed, pd_eta=1, pd_rate=True,
                              pd_base="balanced"))
        p = "p4-dr-eo"
        for eta in (0.01, 0.03):
            lines.append(line(f"{p}-pd{fmt(eta)}-s{seed}", "Drug", "EO", "pd", seed, pd_eta=eta))
        lines.append(line(f"{p}-balpd0.1-s{seed}", "Drug", "EO", "pd", seed, pd_eta=0.1, pd_base="balanced"))
    return lines


# --- hyperparameter selection, on the validation split (PLAN.md §4) -----------
SELECT_SEEDS = (0, 1, 2)
GRID = {
    "EER": [dict(pd_estimator="probe", pd_brake=k, pd_eta=e) for k in (1, 3) for e in (0.3, 1, 3)],
    "EO": [dict(pd_base=b, pd_eta=e) for b in ("mean", "balanced") for e in (0.03, 0.1, 0.3, 1, 3, 10)],
    "DP": [dict(pd_rate=True, pd_eta=e) for e in (0.01, 0.03, 0.1, 0.3)],
}


def config_name(cfg: dict) -> str:
    parts = []
    if cfg.get("pd_estimator") == "probe":
        parts.append(f"brake{fmt(cfg['pd_brake'])}")
    if cfg.get("pd_base") == "balanced":
        parts.append("bal")
    if cfg.get("pd_rate"):
        parts.append("rate")
    parts.append(f"eta{fmt(cfg['pd_eta'])}")
    return "-".join(parts)


def select():
    lines = []
    pairs = [("MNIST", "EER"), ("FashionMNIST", "EER"), ("BiasedMNIST", "EO"), ("BiasedMNIST", "DP"),
             ("Drug", "EO"), ("Drug", "DP")]
    for seed in SELECT_SEEDS:
        for ds, met in pairs:
            p = f"sel-{SHORT[ds]}-{met.lower()}"
            lines.append(line(f"{p}-replay-s{seed}", ds, met, "pd", seed, split="val", pd_eta=0))
            for cfg in GRID[met]:
                lines.append(line(f"{p}-{config_name(cfg)}-s{seed}", ds, met, "pd", seed, split="val", **cfg))
    return lines


# --- final runs, on the test split (PLAN.md §4) --------------------------------
FINAL_SEEDS = (0, 1, 2, 3, 4)


def selected_cfg(key: str) -> dict:
    """The configuration selected on validation (campaigns/selected.json), as keyword arguments."""
    import json
    sel = json.loads((ROOT / "campaigns" / "selected.json").read_text())[key]
    cfg, toks, i = {}, sel["args"], 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--pd-"):
            name = t[2:].replace("-", "_")
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                cfg[name] = toks[i + 1]
                i += 2
                continue
            cfg[name] = True
        i += 1
    cfg.setdefault("pd_eta", 0)
    return cfg


def final():
    lines = []
    for seed in FINAL_SEEDS:
        # replay with equal weights, with FSW's learning rate and replay weight for each
        # benchmark and measure (they differ between EO and DP)
        for ds, met in (("MNIST", "EER"), ("FashionMNIST", "EER"), ("BiasedMNIST", "EO"), ("BiasedMNIST", "DP"),
                        ("Drug", "EO"), ("Drug", "DP")):
            lines.append(line(f"fin-{SHORT[ds]}-{met.lower()}-replay-s{seed}", ds, met, "pd", seed, pd_eta=0))
        # FairPD, the selected configuration per benchmark and measure
        for ds, met in (("MNIST", "EER"), ("FashionMNIST", "EER"), ("BiasedMNIST", "EO"), ("BiasedMNIST", "DP"),
                        ("Drug", "EO"), ("Drug", "DP")):
            cfg = selected_cfg(f"{ds}/{met}")
            if float(cfg["pd_eta"]) == 0:
                continue                      # selection kept replay: the replay runs are FairPD's runs
            lines.append(line(f"fin-{SHORT[ds]}-{met.lower()}-fairpd-s{seed}", ds, met, "pd", seed, **cfg))
        # ablations
        for ds in ("BiasedMNIST", "Drug"):
            cfg = selected_cfg(f"{ds}/EO")
            lines.append(line(f"fin-{SHORT[ds]}-eo-balance-s{seed}", ds, "EO", "pd", seed, pd_eta=0,
                              pd_base="balanced"))
            if float(cfg["pd_eta"]) > 0:
                other = "mean" if cfg.get("pd_base") == "balanced" else "balanced"
                alt = dict(cfg, pd_base=other)
                lines.append(line(f"fin-{SHORT[ds]}-eo-fairpd{other}-s{seed}", ds, "EO", "pd", seed, **alt))
        for ds in ("MNIST", "FashionMNIST"):
            cfg = selected_cfg(f"{ds}/EER")
            nobrake = {k: v for k, v in cfg.items() if k != "pd_brake"}
            lines.append(line(f"fin-{SHORT[ds]}-eer-nobrake-s{seed}", ds, "EER", "pd", seed, **nobrake))
            # where FSW's class-level gains come from
            lines.append(line(f"fin-{SHORT[ds]}-eer-replay1ep-s{seed}", ds, "EER", "pd", seed, pd_eta=0, epochs=1))
            lines.append(line(f"fin-{SHORT[ds]}-eer-fsw-s{seed}", ds, "EER", "FSW", seed))
        if seed < 3:
            lines.append(line(f"fin-bm-eo-fsw-s{seed}", "BiasedMNIST", "EO", "FSW", seed))
            # sensitivity to the dual step on Biased MNIST
            cfg = selected_cfg("BiasedMNIST/EO")
            for eta in (0.03, 0.1, 0.3, 1, 3, 10):
                lines.append(line(f"fin-bm-eo-eta{fmt(eta)}-s{seed}", "BiasedMNIST", "EO", "pd", seed,
                                  **dict(cfg, pd_eta=eta)))
    return lines


def timing():
    """Wall-clock comparison: run one after another on one node (parallel 1)."""
    lines = []
    for ds, met in (("MNIST", "EER"), ("BiasedMNIST", "EO")):
        lines.append(line(f"tim-{SHORT[ds]}-replay", ds, met, "pd", 0, pd_eta=0))
        lines.append(line(f"tim-{SHORT[ds]}-fairpd", ds, met, "pd", 0, **selected_cfg(f"{ds}/{met}")))
        lines.append(line(f"tim-{SHORT[ds]}-fsw", ds, met, "FSW", 0))
    return lines


# --- BiasBios (BERT), on A100s with TF32: one selection seed, as the runs are costly
BIOS_GRID = {
    "EO": [dict(pd_base=b, pd_eta=e) for b in ("mean", "balanced") for e in (0.1, 0.3, 1)],
    "DP": [dict(pd_rate=True, pd_eta=e) for e in (0.01, 0.03, 0.1)],
}


def bios_select():
    lines = [line("bio-sel-eo-replay-s0", "Bios", "EO", "pd", 0, split="val", pd_eta=0, tf32=True)]
    for met, grid in BIOS_GRID.items():
        for cfg in grid:
            lines.append(line(f"bio-sel-{met.lower()}-{config_name(cfg)}-s0", "Bios", met, "pd", 0, split="val",
                              tf32=True, **cfg))
    return lines


def bios_replay():
    """Replay's final BiasBios runs: they do not depend on the selection."""
    return [line(f"bio-eo-replay-s{seed}", "Bios", "EO", "pd", seed, pd_eta=0, tf32=True) for seed in FINAL_SEEDS]


def bios_final():
    lines = []
    for seed in FINAL_SEEDS:
        for met in ("EO", "DP"):
            cfg = selected_cfg(f"Bios/{met}")
            if float(cfg["pd_eta"]) > 0:
                lines.append(line(f"bio-{met.lower()}-fairpd-s{seed}", "Bios", met, "pd", seed, tf32=True, **cfg))
    return lines


def bios_tradeoff():
    """The BiasBios configuration with the lowest validation EO, whatever its accuracy (the rule kept replay)."""
    cfg = dict(pd_base="balanced", pd_eta=1)
    return [line(f"bio-eo-lowesteo-s{seed}", "Bios", "EO", "pd", seed, tf32=True, **cfg) for seed in FINAL_SEEDS]


CAMPAIGNS = {"pilot3": pilot3, "pilot4": pilot4, "select": select, "final": final, "timing": timing,
             "bios_tradeoff": bios_tradeoff,
             "bios_select": bios_select, "bios_replay": bios_replay, "bios_final": bios_final}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", choices=sorted(CAMPAIGNS))
    args = p.parse_args(argv)
    lines = CAMPAIGNS[args.name]()
    out = ROOT / "campaigns" / f"{args.name}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
