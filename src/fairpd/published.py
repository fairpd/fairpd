"""The baselines' published results, read from FSW's LaTeX source.

Park, Kim and Whang, "Fair Class-Incremental Learning using Sample Weighting",
KDD 2026 (arXiv:2410.01324v3): the appendix tables ``tbl:performance_eer_disp``
(MNIST, FMNIST), ``tbl:performance_eo_disp`` and ``tbl:performance_dp_disp``
(Biased MNIST, DRUG, BiasBios), each "mean +- std" over five seeds, and the
ablation tables ``tbl:performance_ablation_{eer,eo,dp}`` (FSW against equal
weights). Parsing the source rather than retyping keeps every published number
traceable to its table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "sources" / "arXiv-2410.01324v3" / "appendix.tex"
JSON = ROOT / "results" / "published_fsw.json"   # the same numbers, written by save() from the source

COLUMNS = {
    "tbl:performance_eer_disp": [("MNIST", "EER"), ("FashionMNIST", "EER")],
    "tbl:performance_eo_disp": [("BiasedMNIST", "EO"), ("Drug", "EO"), ("Bios", "EO")],
    "tbl:performance_dp_disp": [("BiasedMNIST", "DP"), ("Drug", "DP"), ("Bios", "DP")],
    "tbl:performance_ablation_eer": [("MNIST", "EER"), ("FashionMNIST", "EER")],
    "tbl:performance_ablation_eo": [("BiasedMNIST", "EO"), ("Drug", "EO"), ("Bios", "EO")],
    "tbl:performance_ablation_dp": [("BiasedMNIST", "DP"), ("Drug", "DP"), ("Bios", "DP")],
}
METHOD_NAMES = {
    "Joint Training": "Joint", "Fine Tuning": "Fine-tune", "iCaRL": "iCaRL", "WA": "WA", "CLAD": "CLAD",
    "GSS": "GSS", "OCS": "OCS", "FaIRL": "FaIRL", "\\method{}": "FSW", "W/o \\method{}": "Replay (W/o FSW)",
}
CELL = re.compile(r"(\.\d+)\s*\\tiny\{\$\\pm\$\s*\}?(\.\d+)")


def _table_body(text: str, label: str) -> str:
    end = text.index(f"\\label{{{label}}}")
    start = text.rindex("\\begin{table", 0, end)
    return text[start:end]


def load(source: Path = SOURCE) -> dict:
    """{(method, dataset, measure): {"acc": (mean, std), "disp": (mean, std)}}; missing cells are absent.

    Parses FSW's LaTeX source when it is present, else reads ``results/published_fsw.json``.
    """
    if not Path(source).exists():
        rows = json.loads(JSON.read_text())["results"]
        return {(r["method"], r["dataset"], r["measure"]): {"acc": tuple(r["acc"]), "disp": tuple(r["disp"]),
                                                             "table": r["table"]} for r in rows}
    text = source.read_text()
    out = {}
    for label, cols in COLUMNS.items():
        body = _table_body(text, label)
        for row in body.split("\\\\"):
            row = row.strip()
            m = re.search(r"\{(\\bf\s*)?([^{}]*(?:\{\})?[^{}]*)\}\s*&", row)
            if not m:
                continue
            raw = m.group(2).strip()
            name = METHOD_NAMES.get(raw)
            if name is None:
                continue
            cells = row[m.end() - 1:].split("&")[1:]
            values = []
            for c in cells:
                mm = CELL.search(c.replace("\\textbf{", "").replace("\\underline{", ""))
                values.append((float(mm.group(1)), float(mm.group(2))) if mm else None)
            for k, (ds, meas) in enumerate(cols):
                acc, disp = values[2 * k], values[2 * k + 1]
                if acc is None or disp is None:
                    continue
                key = (name, ds, meas)
                if "ablation" in label and name == "FSW":
                    continue          # the main tables hold FSW's numbers; the ablation repeats them
                out[key] = {"acc": acc, "disp": disp, "table": label}
    return out


def save(path: Path = JSON) -> None:
    rows = [{"method": m, "dataset": d, "measure": q, "acc": list(v["acc"]), "disp": list(v["disp"]),
             "table": v["table"]} for (m, d, q), v in sorted(load(SOURCE).items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source": "Park, Kim and Whang, Fair Class-Incremental Learning using Sample Weighting, KDD 2026, "
                  "doi:10.1145/3770854.3780167; arXiv:2410.01324v3, appendix tables (mean, standard deviation "
                  "over five seeds)",
        "results": rows}, indent=1))


if __name__ == "__main__":
    save()
    for k, v in sorted(load().items()):
        print(k, v)
