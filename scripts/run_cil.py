"""Entry point for one class-incremental run: see ``fairpd.runner``.

    python scripts/run_cil.py --dataset MNIST --metric EER --method pd \
        --lr 0.001 --tau 10 --pd-eta 0.05 --seed 0 --run-dir out/runs/<name>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairpd.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
