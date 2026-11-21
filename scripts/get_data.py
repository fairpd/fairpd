"""Download the five benchmarks into the data root, and build FSW's caches.

Run once per machine, before any run (on a cluster, where the jobs can read the data root):

    .venv/bin/python scripts/get_data.py --datasets MNIST FashionMNIST BiasedMNIST Drug Bios

The data root is ``CL_DATA_DIR``, else ``$GLOBALSCRATCH/cl_data``, else
``~/data/cl_data`` (``fairpd.bootstrap``). Sources:

* MNIST, Fashion-MNIST: torchvision's mirrors (Biased MNIST is MNIST coloured
  at load time by FSW's code, with the run's seed);
* DRUG: UCI Machine Learning Repository, dataset 373 ("Drug consumption
  (quantified)"), through ``ucimlrepo``, cached by FSW's loader as
  ``drug.pickle``;
* BiasBios: the INLP release of De-Arteaga et al.'s biographies
  (https://storage.googleapis.com/ai2i/nullspace/biasbios/{train,dev,test}.pickle),
  tokenised once by FSW's loader with ``bert-base-uncased`` (128 tokens) into
  ``bios/*.pickle.embeddings/``; this needs the tokenizer in the HF cache
  (``scripts/prefetch.py --model bert-base-uncased --tokenizer-only``).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairpd import bootstrap  # noqa: E402

BIOS_URL = "https://storage.googleapis.com/ai2i/nullspace/biasbios/{}.pickle"
BIOS_SIZES = {"train": 455284434, "dev": 66595445, "test": 182106720}   # bytes, 2026-11-21


def fetch(url: str, dest: Path, size: int | None = None) -> None:
    if dest.exists() and (size is None or dest.stat().st_size == size):
        print(f"present: {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    if size is not None and tmp.stat().st_size != size:
        raise RuntimeError(f"{url}: got {tmp.stat().st_size} bytes, expected {size}")
    tmp.rename(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["MNIST", "FashionMNIST", "Drug"],
                   choices=["MNIST", "FashionMNIST", "BiasedMNIST", "Drug", "Bios"])
    p.add_argument("--data-dir", default=None)
    args = p.parse_args(argv)
    root = bootstrap.setup(args.data_dir)
    print(f"data root: {root}")

    import torchvision

    if "MNIST" in args.datasets or "BiasedMNIST" in args.datasets:
        for train in (True, False):
            torchvision.datasets.MNIST(str(root), train=train, download=True)
    if "FashionMNIST" in args.datasets:
        for train in (True, False):
            torchvision.datasets.FashionMNIST(str(root), train=train, download=True)
    if "Drug" in args.datasets:
        from datasets import Drug
        Drug(num_tasks=3, per_task_memory_examples=64, per_task_examples=2_000_000)
        print(f"DRUG cached: {root / 'drug.pickle'} sha256={sha256(root / 'drug.pickle')}")
    if "Bios" in args.datasets:
        bios = root / "bios"
        bios.mkdir(exist_ok=True)
        for split, size in BIOS_SIZES.items():
            fetch(BIOS_URL.format(split), bios / f"{split}.pickle", size)
        from datasets import Bios
        Bios(num_tasks=5, per_task_memory_examples=320, per_task_examples=10_000_000, max_length=128)
        print("BiasBios tokenised and cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
