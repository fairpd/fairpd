"""Build the manuscript's tables, figures and number macros from the run sidecars.

    python scripts/build_tables.py --runs out/runs --out redaction/discover-informatics

Reads the pulled sidecars of the final campaign (``fin-*``, ``bios-*``,
``tim-*``), the selection (``campaigns/selected.json``) and the baselines'
published results (``fairpd.published``, parsed from FSW's LaTeX source), and
writes ``tab_*.tex``, ``numbers.tex`` and ``fig_*.pdf`` side by side in the
output folder, with no subfolders (submission systems flatten the upload). No
number in the manuscript is typed by hand. ``RUNS_DIR`` overrides ``--runs``
(to rebuild from the shipped runs or from a rerun).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairpd.published import load as load_published  # noqa: E402

NAMES = {"MNIST": "MNIST", "FashionMNIST": "FMNIST", "BiasedMNIST": "Biased MNIST", "Drug": "DRUG",
         "Bios": "BiasBios"}
SHORT = {"MNIST": "mn", "FashionMNIST": "fm", "BiasedMNIST": "bm", "Drug": "dr", "Bios": "bi"}
BASELINES = ["Joint", "Fine-tune", "iCaRL", "WA", "CLAD", "GSS", "OCS", "FaIRL", "FSW"]


# ----------------------------------------------------------------------------- data
def load_runs(root: Path) -> dict:
    """run configuration (name without -s<seed>) -> list of metrics dicts."""
    groups = defaultdict(list)
    for f in sorted(root.glob("*/metrics.json")):
        groups[re.sub(r"-s\d+$", "", f.parent.name)].append(json.loads(f.read_text()))
    return groups


def stat(runs, key):
    v = np.array([r[key] for r in runs], dtype=float)
    return float(v.mean()), float(v.std()), len(v)


def f3(x: float) -> str:
    """Three decimals, with the leading zero (journal style, ISO 80000-1)."""
    return f"{x:.3f}"


NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def count(n: int) -> str:
    """A count in running text, spelled out up to ten as the manuscript's prose does."""
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


def cell(mean, std, bold=False):
    s = f"{f3(mean)}\\tiny{{$\\pm${f3(std)}}}"
    return f"\\textbf{{{s}}}" if bold else s


# --------------------------------------------------------------------------- tables
def table_compare(pub, runs, pairs, label, caption, ours):
    """Accuracy and disparity per benchmark: published baselines, our replay, FairPD."""
    rows = []
    for m in BASELINES:
        vals = []
        for ds, meas in pairs:
            p = pub.get((m, ds, meas))
            vals.append(None if p is None else (p["acc"], p["disp"]))
        rows.append((m if m != "FSW" else "FSW \\cite{park2026fsw}", vals, "pub"))
    for label_row, prefix_fn in ours:
        vals = []
        for ds, meas in pairs:
            key = prefix_fn(ds, meas)
            if key in runs and len(runs[key]) > 0:
                a, sa, _ = stat(runs[key], "accuracy")
                d, sd, _ = stat(runs[key], meas)
                vals.append(((a, sa), (d, sd)))
            else:
                vals.append(None)
        rows.append((label_row, vals, "ours"))
    # best (lowest) disparity among continual methods (not Joint/Fine-tune), per column
    best = []
    for j in range(len(pairs)):
        cands = [r[1][j][1][0] for r in rows if r[0] not in ("Joint", "Fine-tune") and r[1][j] is not None]
        best.append(min(cands) if cands else None)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{NAMES[ds]}}}" for ds, _ in pairs)
    sub = " & ".join(f"Acc. & {meas}" for _, meas in pairs)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{l" + "cc" * len(pairs) + "}", "\\toprule",
             f"Method & {head} \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * j}-{3 + 2 * j}}}" for j in range(len(pairs))),
             f" & {sub} \\\\", "\\midrule"]
    for i, (name, vals, kind) in enumerate(rows):
        if kind == "ours" and rows[i - 1][2] == "pub":
            lines.append("\\midrule")
        cells = []
        for j, v in enumerate(vals):
            if v is None:
                cells += ["n/r", "n/r"]
                continue
            (a, sa), (d, sd) = v
            cells += [cell(a, sa), cell(d, sd)]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if name == "Fine-tune":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_rows(runs, header, rows, label, caption, cols):
    """A table of our runs: rows (name, run key, {measure: column}) over columns [(title, key, measure)]."""
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             header, "\\midrule"]
    for name, keys in rows:
        cells = []
        for (title, metric), key in zip(cols, keys):
            if key is None or key not in runs:
                cells.append("n/r")
                continue
            m, s, _ = stat(runs[key], metric)
            cells.append(cell(m, s))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_runs = ROOT / "results" / "runs" if (ROOT / "results" / "runs").is_dir() else ROOT / "out" / "runs"
    default_out = ROOT / "redaction" / "discover-informatics"
    if not default_out.is_dir():
        default_out = ROOT / "paper"
    p.add_argument("--runs", default=os.environ.get("RUNS_DIR", str(default_runs)),
                   help="run sidecars (default: the shipped results/runs; RUNS_DIR for a rerun's out/runs)")
    p.add_argument("--out", default=str(default_out))
    args = p.parse_args(argv)
    runs = load_runs(Path(args.runs))
    pub = load_published()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} configurations from {args.runs}")
    selected = json.loads((ROOT / "campaigns" / "selected.json").read_text())
    nums = {}

    def ours_key(kind):
        def fn(ds, meas):
            # BiasBios: one replay run per seed measures EO and DP alike (same learning rate and tau)
            replay = f"fin-{SHORT[ds]}-{meas.lower()}-replay" if ds != "Bios" else "bio-eo-replay"
            if kind == "replay":
                return replay
            sel = selected.get(f"{ds}/{meas}")
            if sel is not None and sel["config"] == "replay":
                return replay
            return f"fin-{SHORT[ds]}-{meas.lower()}-fairpd" if ds != "Bios" else f"bio-{meas.lower()}-fairpd"
        return fn

    def kept_replay(pairs):
        names = [NAMES[d] for d, q in pairs if selected.get(f"{d}/{q}", {}).get("config") == "replay"]
        if not names:
            return ""
        return (". Where the selection retained no constraint (" + ", ".join(names)
                + "), the row of \\method{} repeats the replay runs")

    ours = [("Replay, equal weights", ours_key("replay")), ("\\method{}", ours_key("fairpd"))]
    tables = {
        "eer": table_compare(pub, runs, [("MNIST", "EER"), ("FashionMNIST", "EER")], "tab:eer",
                             "Average accuracy and EER disparity on the benchmarks whose groups are the classes "
                             "(mean $\\pm$ standard deviation over five seeds). The upper rows are the results "
                             "reported in \\cite{park2026fsw}; the lower rows are our runs", ours),
        "eo": table_compare(pub, runs, [("BiasedMNIST", "EO"), ("Drug", "EO"), ("Bios", "EO")], "tab:eo",
                            "Average accuracy and equalized-odds disparity (EO) on the benchmarks with a sensitive "
                            "attribute (mean $\\pm$ standard deviation over five seeds). The upper rows are the "
                            "results reported in \\cite{park2026fsw} (n/r: not reported there, the run exceeding "
                            "five days); the lower rows are our runs"
                            + kept_replay([("BiasedMNIST", "EO"), ("Drug", "EO"), ("Bios", "EO")]), ours),
        "dp": table_compare(pub, runs, [("BiasedMNIST", "DP"), ("Drug", "DP"), ("Bios", "DP")], "tab:dp",
                            "Average accuracy and demographic-parity disparity (DP), as in Table~\\ref{tab:eo}"
                            + kept_replay([("BiasedMNIST", "DP"), ("Drug", "DP"), ("Bios", "DP")]), ours),
    }
    # ablation: what the multipliers add, on the EO benchmarks and the class-level ones
    abl_rows, abl_cols = [], [("Acc.", "accuracy"), ("EO", "EO"), ("Acc.", "accuracy"), ("EO", "EO")]
    selb = selected.get("BiasedMNIST/EO", {}).get("config", "")
    seld = selected.get("Drug/EO", {}).get("config", "")
    bm_alt = "fin-bm-eo-fairpdmean" if selb.startswith("bal") else "fin-bm-eo-fairpdbalanced"
    dr_alt = "fin-dr-eo-fairpdbalanced" if not seld.startswith("bal") else "fin-dr-eo-fairpdmean"
    abl_rows = [
        ("Replay, equal weights ($\\eta = 0$)", ["fin-bm-eo-replay"] * 2 + ["fin-dr-eo-replay"] * 2),
        ("Group-balanced weights ($\\eta = 0$)", ["fin-bm-eo-balance"] * 2 + ["fin-dr-eo-balance"] * 2),
        ("\\method{}, replay's base weights",
         [("fin-bm-eo-fairpd" if not selb.startswith("bal") else bm_alt)] * 2
         + [("fin-dr-eo-fairpd" if not seld.startswith("bal") else dr_alt)] * 2),
        ("\\method{}, balanced base weights",
         [("fin-bm-eo-fairpd" if selb.startswith("bal") else bm_alt)] * 2
         + [("fin-dr-eo-fairpd" if seld.startswith("bal") else dr_alt)] * 2),
    ]
    header = ("& \\multicolumn{2}{c}{Biased MNIST} & \\multicolumn{2}{c}{DRUG} \\\\ "
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5} & Acc. & EO & Acc. & EO \\\\")
    tables["ablation_eo"] = table_rows(
        runs, "Configuration " + header, abl_rows, "tab:ablation-eo",
        "Ablation on the equalized-odds benchmarks (test split, five seeds): static group balancing against "
        "the multipliers, with either base weights. The configuration selected on validation is "
        f"{'balanced' if selb.startswith('bal') else 'replay'}'s base on Biased MNIST and "
        f"{'balanced' if seld.startswith('bal') else 'replay'}'s base on DRUG",
        abl_cols)
    eer_rows = [
        ("Replay, equal weights, 5 epochs per task", ["fin-mn-eer-replay"] * 2 + ["fin-fm-eer-replay"] * 2),
        ("Replay, equal weights, 1 epoch per task", ["fin-mn-eer-replay1ep"] * 2 + ["fin-fm-eer-replay1ep"] * 2),
        ("\\fsw{}, our runs (HiGHS)", ["fin-mn-eer-fsw"] * 2 + ["fin-fm-eer-fsw"] * 2),
        ("\\method{}, multipliers on replay weights", ["fin-mn-eer-nobrake"] * 2 + ["fin-fm-eer-nobrake"] * 2),
        ("\\method{}, multipliers as a brake", ["fin-mn-eer-fairpd"] * 2 + ["fin-fm-eer-fairpd"] * 2),
    ]
    tables["ablation_eer"] = table_rows(
        runs, "Configuration & \\multicolumn{2}{c}{MNIST} & \\multicolumn{2}{c}{FMNIST} \\\\ "
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5} & Acc. & EER & Acc. & EER \\\\",
        eer_rows, "tab:ablation-eer",
        "The class-level benchmarks (test split, five seeds): replay trained for five or for one epoch per task, "
        "our runs of \\fsw{}, and \\method{} with its multipliers acting on the replayed classes' weights or "
        "as a brake on the current task",
        [("Acc.", "accuracy"), ("EER", "EER"), ("Acc.", "accuracy"), ("EER", "EER")])
    for name, tex in tables.items():
        (out / f"tab_{name}.tex").write_text(tex)

    # ---- settings: FSW's learning rate and replay weight, FairPD's selected configuration
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_campaign import FSW_SETTINGS
    pretty = {"replay": "none ($\\eta = 0$)"}

    def cfg_text(c):
        if c in pretty:
            return pretty[c]
        parts = []
        for tok in c.split("-"):
            if tok.startswith("brake"):
                parts.append(f"brake, $\\kappa = {tok[5:]}$, probe")
            elif tok == "bal":
                parts.append("balanced base")
            elif tok == "rate":
                parts.append("rates")
            elif tok.startswith("eta"):
                parts.append(f"$\\eta = {tok[3:]}$")
        return ", ".join(parts)

    rows = ["\\begin{table}[t]", "\\centering",
            "\\caption{Learning rate and replay weight $\\tau$ (those \\fsw{} selected \\cite{park2026fsw}), and "
            "the configuration of \\method{} selected on the validation split}\\label{tab:settings}",
            "\\footnotesize", "\\begin{tabular}{llccl}", "\\toprule",
            "Benchmark & Measure & Learning rate & $\\tau$ & \\method{} \\\\", "\\midrule"]
    for (ds, meas), s in FSW_SETTINGS.items():
        sel = selected.get(f"{ds}/{meas}")
        conf = cfg_text(sel["config"]) if sel else "n/r"
        lr = f"{s['lr']:g}".replace("e-05", "\\times 10^{-5}")
        rows.append(f"{NAMES[ds]} & {meas} & ${lr}$ & {s['tau']:g} & {conf} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tab_settings.tex").write_text("\n".join(rows) + "\n")

    # ---- FSW's kept samples (fsw_weights.json of our FSW runs)
    kept = defaultdict(list)          # dataset -> list over runs of [(task, epoch, kept fraction, per group)]
    for f in sorted(Path(args.runs).glob("fin-*-fsw-s*/fsw_weights.json")):
        ds = {"mn": "MNIST", "fm": "FashionMNIST", "bm": "BiasedMNIST"}[f.parent.name.split("-")[1]]
        rows_w = json.loads(f.read_text())
        kept[ds].append(rows_w)
    later = []
    for ds, rs in kept.items():
        for r in rs:
            for row in r:
                if row["epoch"] >= 2:
                    n = sum(g["n"] for g in row["groups"].values())
                    k = sum(g["kept"] for g in row["groups"].values())
                    if ds in ("MNIST", "FashionMNIST"):
                        later.append(k / n)
    if later:
        nums["NumFSWkeptRange"] = f"{100 * min(later):.0f} to {100 * max(later):.0f}\\%"
        nums["NumFSWkeptMedian"] = f"{100 * float(np.median(later)):.0f}\\%"
    if kept:
        plt = setup_matplotlib()
        fig, axes = plt.subplots(1, len(kept), figsize=(6.6, 1.9), sharey=True)
        axes = np.atleast_1d(axes)
        markers = {"0": "o", "1": "s", "all": "^"}
        for ax, (ds, rs) in zip(axes, sorted(kept.items(), key=lambda kv: list(NAMES).index(kv[0]))):
            # mean over seeds of the kept fraction, per (task, epoch); per group for Biased MNIST
            per = defaultdict(list)
            for r in rs:
                for row in r:
                    if ds == "BiasedMNIST":
                        for g, v in row["groups"].items():
                            per[(row["task"], row["epoch"], g)].append(v["kept"] / max(v["n"], 1))
                    else:
                        n = sum(v["n"] for v in row["groups"].values())
                        k = sum(v["kept"] for v in row["groups"].values())
                        per[(row["task"], row["epoch"], "all")].append(k / n)
            groups = sorted({k[2] for k in per})
            for g in groups:
                keys = sorted(k for k in per if k[2] == g)
                x = np.arange(len(keys))
                y = [100 * np.mean(per[k]) for k in keys]
                label = {"all": "all samples", "0": "own colour", "1": "other colour"}[g]
                ax.plot(x, y, marker=markers.get(g, "o"), ms=2.5, lw=0.9, label=label,
                        color={"all": "#333333", "0": "#1b6ca8", "1": "#d1495b"}[g],
                        linestyle={"all": "-", "0": "-", "1": "--"}[g])
                tasks = [k[0] for k in keys]
                starts = [i for i in range(len(tasks)) if i == 0 or tasks[i] != tasks[i - 1]]
            for s in starts[1:]:
                ax.axvline(s - 0.5, color="0.8", lw=0.6, zorder=0)
            ax.set_xticks([s for s in starts])
            ax.set_xticklabels([f"T{tasks[s]}" for s in starts])
            ax.set_title(NAMES[ds])
            ax.set_ylim(0, 105)
        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        axes[0].set_ylabel("current samples kept (%)")
        fig.supxlabel("task (ticks) and its epochs", fontsize=8, y=0.1)
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(out / "fig_fsw_kept.pdf", metadata={"CreationDate": None})
        plt.close(fig)

    # ---- sensitivity to the dual step on Biased MNIST: EO and accuracy against eta
    sens = sorted((float(k.rsplit("eta", 1)[1]), runs[k]) for k in runs if k.startswith("fin-bm-eo-eta"))
    if sens:
        plt = setup_matplotlib()
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1))
        etas = [e for e, _ in sens]
        refs = [("replay (ours)", runs.get("fin-bm-eo-replay"), "--", "#444444"),
                ("static group balance (ours)", runs.get("fin-bm-eo-balance"), ":", "#3a7d44")]
        q = pub.get(("FSW", "BiasedMNIST", "EO"))
        j = pub.get(("Joint", "BiasedMNIST", "EO"))
        for ax, metric, scale, ylabel in ((axes[0], "EO", 1.0, "EO disparity"),
                                          (axes[1], "accuracy", 100.0, "average accuracy (%)")):
            m = [scale * stat(r, metric)[0] for _, r in sens]
            sd = [scale * stat(r, metric)[1] for _, r in sens]
            ax.errorbar(etas, m, yerr=sd, fmt="-o", color="#1b6ca8", ms=3, lw=0.9, capsize=2, label="FairPD")
            for name, r, ls, col in refs:
                if r:
                    ax.axhline(scale * stat(r, metric)[0], ls=ls, color=col, lw=0.9, label=name)
            key = "disp" if metric == "EO" else "acc"
            if q:
                ax.axhline(scale * q[key][0], ls="-.", color="#d1495b", lw=0.9, label="FSW (reported)")
            if j:
                ax.axhline(scale * j[key][0], ls=(0, (1, 3)), color="#000000", lw=0.9, label="joint (reported)")
            ax.set_xscale("log")
            ax.set_xlabel(r"dual step $\eta$")
            ax.set_ylabel(ylabel)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=6.5)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(out / "fig_sensitivity_bm.pdf", metadata={"CreationDate": None})
        plt.close(fig)

    # ---- share of the current task's weight on the other-colour group (FairPD, Biased MNIST)
    diag = sorted(Path(args.runs).glob("fin-bm-eo-fairpd-s*/pd_diagnostics.json"))
    if diag:
        plt = setup_matplotlib()
        share_runs = []
        for f in diag:
            ep = json.loads(f.read_text())["epochs"]
            sh = []
            for e in ep:
                t, mu = e["task"], e["mu"]
                cur = [2 * (t - 1), 2 * (t - 1) + 1]
                # base weights of the current batch: half of it per class, 5% of a class other-colour
                w_other = sum(0.5 * 0.05 + mu[2 * y + 1] for y in cur)
                w_own = sum(0.5 * 0.95 + mu[2 * y] for y in cur)
                sh.append(100 * w_other / (w_other + w_own))
            share_runs.append(sh)
        n = min(len(r) for r in share_runs)
        arr = np.array([r[:n] for r in share_runs])
        tasks = [e["task"] for e in json.loads(diag[0].read_text())["epochs"]][:n]
        fig, ax = plt.subplots(figsize=(3.3, 2.0))
        x = np.arange(n)
        ax.plot(x, arr.mean(axis=0), "-", color="#d1495b", lw=1.0, label="FairPD (mean of seeds)")
        ax.fill_between(x, arr.min(axis=0), arr.max(axis=0), color="#d1495b", alpha=0.2, lw=0,
                        label="range over seeds")
        ax.axhline(5, ls="--", color="#444444", lw=0.9, label="replay (5%)")
        for i in range(1, n):
            if tasks[i] != tasks[i - 1]:
                ax.axvline(i - 0.5, color="0.8", lw=0.6, zorder=0)
        ax.set_ylim(0, 100)
        ax.set_xlabel("epoch (tasks separated by grey lines)")
        ax.set_ylabel("other-colour share (%)")
        ax.legend(frameon=False, fontsize=6, loc="center right")
        fig.tight_layout()
        fig.savefig(out / "fig_multipliers_bm.pdf", metadata={"CreationDate": None})
        plt.close(fig)
        nums["NumBmShareMedian"] = f"{float(np.median(arr[:, 1:])):.0f}\\%"

    # ---- cost: wall-clock of one run each, one after another on one node (tim-*)
    tim = {k: v[0] for k, v in runs.items() if k.startswith("tim-")}
    if tim:
        rows = ["\\begin{table}[t]", "\\centering",
                "\\caption{Wall-clock time of one run (seed 0), the runs executed one after another on the same "
                "processor with four threads; for \\fsw{}, the part spent computing gradients and solving its "
                "linear programs}\\label{tab:cost}", "\\footnotesize", "\\begin{tabular}{lrrrr}", "\\toprule",
                "Benchmark & Replay (s) & \\method{} (s) & \\fsw{} (s) & of which gradients and LP (s) \\\\",
                "\\midrule"]
        for ds in ("MNIST", "BiasedMNIST"):
            r = tim.get(f"tim-{SHORT[ds]}-replay")
            q = tim.get(f"tim-{SHORT[ds]}-fairpd")
            w = tim.get(f"tim-{SHORT[ds]}-fsw")
            if not (r and q and w):
                continue
            rows.append(f"{NAMES[ds]} & {r['wall_seconds']:.0f} & {q['wall_seconds']:.0f} & "
                        f"{w['wall_seconds']:.0f} & {w['grad_seconds'] + w['lp_seconds']:.0f} \\\\")
            nums[f"NumCost{SHORT[ds].upper()}Ratio"] = f"{w['wall_seconds'] / q['wall_seconds']:.1f}"
        rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (out / "tab_cost.tex").write_text("\n".join(rows) + "\n")

    # ---- numbers used in the text
    def mean_of(key, metric):
        return stat(runs[key], metric)[0] if key in runs else float("nan")

    for ds, meas in (("BiasedMNIST", "EO"), ("Drug", "EO"), ("MNIST", "EER"), ("FashionMNIST", "EER")):
        k = ours_key("fairpd")(ds, meas)
        r = ours_key("replay")(ds, meas)
        tag = SHORT[ds].capitalize() + meas.capitalize()
        nums[f"Num{tag}FairpdAcc"] = f3(mean_of(k, "accuracy"))
        nums[f"Num{tag}FairpdDisp"] = f3(mean_of(k, meas))
        nums[f"Num{tag}ReplayAcc"] = f3(mean_of(r, "accuracy"))
        nums[f"Num{tag}ReplayDisp"] = f3(mean_of(r, meas))
        q = pub.get(("FSW", ds, meas))
        if q:
            nums[f"Num{tag}FswAcc"] = f3(q["acc"][0])
            nums[f"Num{tag}FswDisp"] = f3(q["disp"][0])
    for ds in ("MNIST", "FashionMNIST"):
        tag = SHORT[ds].capitalize()
        for kind, mname in (("replay1ep", "ReplayOneep"), ("fsw", "Fsw"), ("nobrake", "Nobrake")):
            k = f"fin-{SHORT[ds]}-eer-{kind}"
            nums[f"Num{tag}{mname}Acc"] = f3(mean_of(k, "accuracy"))
            nums[f"Num{tag}{mname}Eer"] = f3(mean_of(k, "EER"))
    # ablation, sensitivity, DP and published values cited in the text
    def put(name, key, metric):
        if key in runs:
            nums[name] = f3(stat(runs[key], metric)[0])

    for tag, key in (("BmEoBalance", "fin-bm-eo-balance"), ("DrEoBalance", "fin-dr-eo-balance"),
                     ("BmEoFairpdMean", "fin-bm-eo-fairpdmean"), ("DrEoFairpdBal", "fin-dr-eo-fairpdbalanced"),
                     ("BmEoFswOurs", "fin-bm-eo-fsw")):
        put(f"Num{tag}Acc", key, "accuracy")
        put(f"Num{tag}Disp", key, "EO")
    for tag, key in (("BmDpFairpd", "fin-bm-dp-fairpd"), ("DrDpFairpd", "fin-dr-dp-fairpd"),
                     ("DrDpReplay", "fin-dr-dp-replay"), ("BmDpReplay", "fin-bm-dp-replay")):
        if key in runs:
            nums[f"Num{tag}Std"] = f3(stat(runs[key], "DP")[1])
    put("NumBmDpFairpdAcc", "fin-bm-dp-fairpd", "accuracy")
    put("NumBmDpFairpdDisp", "fin-bm-dp-fairpd", "DP")
    put("NumDrDpFairpdAcc", "fin-dr-dp-fairpd", "accuracy")
    put("NumDrDpFairpdDisp", "fin-dr-dp-fairpd", "DP")
    for d, tag in (("BiasedMNIST", "Bm"), ("Drug", "Dr"), ("Bios", "Bi")):
        sel = selected.get(f"{d}/DP")
        if sel and sel["config"] != "replay":
            nums[f"Num{tag}DpEta"] = sel["config"].rsplit("eta", 1)[1]
    put("NumDrDpReplayAcc", "fin-dr-dp-replay", "accuracy")
    put("NumDrDpReplayDisp", "fin-dr-dp-replay", "DP")
    put("NumDrDpReplayEo", "fin-dr-dp-replay", "EO")
    put("NumBmDpReplayAcc", "fin-bm-dp-replay", "accuracy")
    put("NumBmDpReplayDisp", "fin-bm-dp-replay", "DP")
    put("NumBmEoFairpdDp", ours_key("fairpd")("BiasedMNIST", "EO"), "DP")
    put("NumDrEoReplayDp", "fin-dr-eo-replay", "DP")
    put("NumBmEoReplayDp", "fin-bm-eo-replay", "DP")
    # the largest change FSW makes to EO or DP relative to equal weights, in its own ablation
    deltas = [abs(pub[("FSW", ds, q)]["disp"][0] - pub[("Replay (W/o FSW)", ds, q)]["disp"][0])
              for ds in ("BiasedMNIST", "Drug", "Bios") for q in ("EO", "DP")
              if ("FSW", ds, q) in pub and ("Replay (W/o FSW)", ds, q) in pub]
    if deltas:
        nums["NumFswAblationMaxDelta"] = f3(max(deltas))
    put("NumDrEoFairpdDp", ours_key("fairpd")("Drug", "EO"), "DP")
    if sens:
        eo = [stat(r, "EO")[0] for _, r in sens]
        acc = [stat(r, "accuracy")[0] for _, r in sens]
        nums["NumBmSensEoMin"], nums["NumBmSensEoMax"] = f3(min(eo)), f3(max(eo))
        nums["NumBmSensAccMin"], nums["NumBmSensAccMax"] = f3(min(acc)), f3(max(acc))
    for m, ds, meas, tag in (("FSW", "BiasedMNIST", "DP", "FswBmDp"), ("FSW", "Drug", "DP", "FswDrDp"),
                             ("Joint", "BiasedMNIST", "EO", "JointBmEo"), ("FaIRL", "BiasedMNIST", "EO", "FairlBmEo"),
                             ("FaIRL", "Drug", "EO", "FairlDrEo"), ("Replay (W/o FSW)", "BiasedMNIST", "EO", "WoBmEo"),
                             ("Replay (W/o FSW)", "MNIST", "EER", "WoMnEer"),
                             ("Replay (W/o FSW)", "FashionMNIST", "EER", "WoFmEer"),
                             ("Replay (W/o FSW)", "Drug", "EO", "WoDrEo"),
                             ("WA", "BiasedMNIST", "EO", "WaBmEo"), ("Joint", "Drug", "EO", "JointDrEo")):
        q = pub.get((m, ds, meas))
        if q:
            nums[f"Num{tag}Acc"], nums[f"Num{tag}Disp"] = f3(q["acc"][0]), f3(q["disp"][0])
    for a, b, tag in (("fin-bm-eo-replay", ours_key("fairpd")("BiasedMNIST", "EO"), "BmEoCutReplay"),
                      ("fin-dr-eo-replay", ours_key("fairpd")("Drug", "EO"), "DrEoCutReplay")):
        if a in runs and b in runs:
            ra, rb = stat(runs[a], "EO")[0], stat(runs[b], "EO")[0]
            nums[f"Num{tag}"] = f"{100 * (ra - rb) / ra:.0f}\\%"
    q = pub.get(("FSW", "BiasedMNIST", "EO"))
    k = ours_key("fairpd")("BiasedMNIST", "EO")
    if q and k in runs:
        nums["NumBmEoCutFsw"] = f"{100 * (q['disp'][0] - stat(runs[k], 'EO')[0]) / q['disp'][0]:.0f}\\%"
    # FSW's kept fractions on Biased MNIST by group, epochs >= 2
    own, other = [], []
    for r in kept.get("BiasedMNIST", []):
        for row in r:
            if row["epoch"] >= 2:
                g = row["groups"]
                own.append(g["0"]["kept"] / max(g["0"]["n"], 1))
                other.append(g["1"]["kept"] / max(g["1"]["n"], 1))
    if own:
        nums["NumFSWkeptOwnRange"] = f"{100 * min(own):.0f} to {100 * max(own):.0f}\\%"
        nums["NumFSWkeptOtherRange"] = f"{100 * min(other):.0f} to {100 * max(other):.0f}\\%"
    # the brake: last task's accuracy after the last task, per seed (MNIST)
    last = []
    for f in sorted(Path(args.runs).glob("fin-mn-eer-fairpd-s*/metrics.json")):
        m = json.loads(f.read_text())
        last.append(m["accuracy_matrix"][-1][-1])
    last_r = [json.loads(f.read_text())["accuracy_matrix"][-1][-1]
              for f in sorted(Path(args.runs).glob("fin-mn-eer-replay-s*/metrics.json"))]
    if last_r:
        nums["NumMnReplayLastMin"], nums["NumMnReplayLastMax"] = f3(min(last_r)), f3(max(last_r))
    if last:
        nums["NumMnBrakeLastMin"] = f3(min(last))
        nums["NumMnBrakeLastMax"] = f3(max(last))
        nums["NumMnBrakeStarved"] = count(sum(1 for v in last if v < 0.7))
        eers = [json.loads(f.read_text())["EER"] for f in sorted(Path(args.runs).glob("fin-mn-eer-fairpd-s*/metrics.json"))]
        worst = int(np.argmin(last))
        nums["NumMnBrakeStarvedSeedEer"] = f3(eers[worst])
        nums["NumMnBrakeEerMax"] = f3(max(eers))
        nums["NumMnBrakeSeeds"] = count(len(last))
    for name, r in tim.items():
        tag = name.replace("tim-", "").replace("-", " ").title().replace(" ", "")
        nums[f"NumTime{tag}"] = f"{r['wall_seconds']:.0f}"
        if name.endswith("-fsw"):
            nums[f"NumTime{tag}Grad"] = f"{r['grad_seconds']:.0f}"
            nums[f"NumTime{tag}Lp"] = f"{r['lp_seconds']:.1f}"
    # BiasBios: test runs of replay (= FairPD under the rule), the selection on validation, the trade-off
    put("NumBiReplayAcc", "bio-eo-replay", "accuracy")
    put("NumBiReplayEo", "bio-eo-replay", "EO")
    put("NumBiReplayDp", "bio-eo-replay", "DP")
    val = {k: v[0] for k, v in runs.items() if k.startswith("bio-sel-")}
    if "bio-sel-eo-replay" in val:
        r = val["bio-sel-eo-replay"]
        nums["NumBiValReplayAcc"], nums["NumBiValReplayEo"], nums["NumBiValReplayDp"] = (
            f3(r["accuracy"]), f3(r["EO"]), f3(r["DP"]))
        eo = [v for k, v in val.items() if k.startswith("bio-sel-eo-") and k != "bio-sel-eo-replay"]
        dp = [v for k, v in val.items() if k.startswith("bio-sel-dp-")]
        if eo:
            nums["NumBiValEoMin"], nums["NumBiValEoMax"] = f3(min(v["EO"] for v in eo)), f3(max(v["EO"] for v in eo))
            nums["NumBiValEoAccMin"] = f3(min(v["accuracy"] for v in eo))
            nums["NumBiValEoAccMax"] = f3(max(v["accuracy"] for v in eo))
        if dp:
            nums["NumBiValDpMin"], nums["NumBiValDpMax"] = f3(min(v["DP"] for v in dp)), f3(max(v["DP"] for v in dp))
            nums["NumBiValDpAccMin"] = f3(min(v["accuracy"] for v in dp))
            nums["NumBiValDpAccMax"] = f3(max(v["accuracy"] for v in dp))
    put("NumBiTradeAcc", "bio-eo-lowesteo", "accuracy")
    put("NumBiTradeEo", "bio-eo-lowesteo", "EO")
    put("NumBiTradeDp", "bio-eo-lowesteo", "DP")
    if "bio-eo-lowesteo" in runs:
        nums["NumBiTradeSeeds"] = count(len(runs["bio-eo-lowesteo"]))
        if "bio-eo-replay" in runs:
            loss = 100 * (stat(runs["bio-eo-replay"], "accuracy")[0] - stat(runs["bio-eo-lowesteo"], "accuracy")[0])
            nums["NumBiTradeAccLoss"] = f"{loss:.0f}"
    for m, q, tag in (("FSW", "EO", "FswBiEo"), ("FSW", "DP", "FswBiDp"), ("Replay (W/o FSW)", "EO", "WoBiEo"),
                      ("FaIRL", "EO", "FairlBiEo"), ("Joint", "EO", "JointBiEo")):
        v = pub.get((m, "Bios", q))
        if v:
            nums[f"Num{tag}Acc"], nums[f"Num{tag}Disp"] = f3(v["acc"][0]), f3(v["disp"][0])
    bad = [k for k in nums if not re.fullmatch(r"[A-Za-z]+", k)]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    lines = ["% generated by scripts/build_tables.py; do not edit"]
    for k, v in sorted(nums.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(tables) + 2} tables, {len(nums)} numbers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
