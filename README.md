# FairPD: primal-dual fairness constraints for class-incremental learning

FairPD trains a class-incremental learner with replay under group-fairness
constraints. Each task minimises the replay objective (the mean loss on the
current data plus `tau` times the mean loss on a small memory of past samples)
subject to one constraint per group, whose Lagrange multiplier is updated by
dual ascent on the measured violation and acts on the training step:

* **equalized odds**: the loss of each (class, sensitive attribute) group may
  not exceed its class's mean by more than `eps`; the multipliers are added to
  the groups' weights over the current and the replayed data, so an
  under-served group gains weight without any sample being discarded;
* **demographic parity**: the groups' soft prediction rates of each class are
  constrained, and the multipliers penalise the rate of the group where it is
  too high;
* **class level** (no sensitive attribute): no past class may fall behind the
  average loss, measured on a held-out probe taken from the memory budget, and
  the multipliers slow the current task down.

It runs inside the released code of FSW (Park, Kim and Whang, "Fair
Class-Incremental Learning using Sample Weighting", KDD 2026), with FSW's
benchmarks (MNIST, Fashion-MNIST, Biased MNIST, DRUG, BiasBios), models,
memory, training settings and measures. The repository also contains what the
analysis of FSW needs: its linear program solved with HiGHS instead of CPLEX
(`src/fairpd/lp_highs.py`) and a record of the samples it keeps.

## Installation

With [uv](https://docs.astral.sh/uv/) and Python 3.12:

```bash
uv sync --extra dev            # torch 2.6.0 (CUDA 12.4 build on Linux), numpy, scipy, transformers, ...
bash scripts/get_fsw.sh        # FSW's code at the commit used, into third_party/FSW (it has no licence)
uv run python -m pytest tests  # CPU only, a few seconds
```

## Usage

The data are downloaded into `CL_DATA_DIR` (default `~/data/cl_data`):

```bash
uv run python scripts/get_data.py --datasets MNIST BiasedMNIST
```

Replay on MNIST, one epoch per task, on the CPU in about half a minute:

```bash
uv run python scripts/run_cil.py --dataset MNIST --metric EER --method pd \
    --lr 0.001 --tau 10 --pd-eta 0 --epochs 1 --cpu --run-dir out/runs/example
```

It prints, the time depending on the machine,

```
{"accuracy": 0.9187038069292196, "wall_seconds": 12.881033897399902, "EER": 0.031475313175367906}
```

FairPD with the configuration selected for equalized odds on Biased MNIST
(15 epochs per task, a few minutes on four CPU threads):

```bash
uv run python scripts/run_cil.py --dataset BiasedMNIST --metric EO --method pd \
    --lr 0.001 --tau 5 --pd-base balanced --pd-eta 0.3 --cpu --run-dir out/runs/bm-fairpd
```

`--method FSW --alpha <a> --lambda <l>` runs FSW, `--method vanilla` FSW's own
code with equal weights, and `--split val` evaluates on a validation split held
out from the training data. Each run writes `config.json`, `env.json` and
`metrics.json` into its `--run-dir`.

## Tables and figures

`results/runs/` holds the outputs of every run the paper reports (their
`config.json`, `metrics.json`, reduced `env.json`, and the multipliers or FSW's
weights where recorded), `results/archive/sel-v1/` the first application of
the selection rule on Biased MNIST, and `results/published_fsw.json` the
baselines' results reported by FSW. From them,

```bash
uv run python scripts/build_tables.py          # writes paper/
```

rebuilds:

| content | file |
|---|---|
| accuracy and equalized-odds disparity against the published baselines | `paper/tab_eo.tex` |
| accuracy and demographic-parity disparity | `paper/tab_dp.tex` |
| accuracy and class-level (EER) disparity | `paper/tab_eer.tex` |
| ablation on the equalized-odds benchmarks | `paper/tab_ablation_eo.tex` |
| class-level benchmarks: replay with 5 or 1 epochs, FSW, FairPD variants | `paper/tab_ablation_eer.tex` |
| learning rates, replay weights and selected configurations | `paper/tab_settings.tex` |
| wall-clock time of replay, FairPD and FSW (quoted in the text) | `paper/tab_cost.tex` |
| every number quoted in the text | `paper/numbers.tex` |
| sensitivity to the dual step on Biased MNIST | `paper/fig_sensitivity_bm.pdf` |
| share of the current task's weight on the other-colour group | `paper/fig_multipliers_bm.pdf` |
| current samples kept by FSW in each epoch | `paper/fig_fsw_kept.pdf` |

## Rerunning the experiments

The campaigns are lists of runs in `campaigns/`, written by
`scripts/make_campaign.py`:

1. `select.txt`: the grid of FairPD configurations on the validation split
   (seeds 0 to 2) for MNIST, Fashion-MNIST, Biased MNIST and DRUG;
   `scripts/select_config.py` applies the selection rule and writes
   `campaigns/selected.json` (BiasBios: `bios_select.txt`, one seed, then
   `select_config.py --prefix bio-sel- --dataset-code bi --seeds 1`);
2. `final.txt`: the selected configurations, replay, the ablations, the
   sensitivity study and FSW, on the test split with seeds 0 to 4
   (BiasBios: `bios_replay.txt`; `bios_final.txt` is empty because the
   selection kept replay there, and `bios_tradeoff.txt` runs the configuration
   with the lowest validation equalized-odds disparity);
3. `timing.txt`: one run each of replay, FairPD and FSW, to be run one after
   another on one machine.

`scripts/campaign.py` runs a campaign, several runs at a time, skipping runs
whose `metrics.json` exists, locally or in any batch job:

```bash
uv run python scripts/campaign.py campaigns/final.txt --parallel 8 --threads 4
# split over several jobs: --shard 0/2, --shard 1/2; on a Slurm cluster, e.g.
# sbatch -c 32 --mem=64G --wrap "uv run python scripts/campaign.py campaigns/final.txt --parallel 8 --threads 4 --shard 0/2"
RUNS_DIR=out/runs uv run python scripts/build_tables.py --out out/paper   # tables from the rerun
```

The BiasBios runs pass `--tf32`, for which an Ampere or newer GPU is needed to
matter. The runs of the perceptron benchmarks took 9 hours in all, four threads each, from 7 s to 7 min per run, on AMD EPYC 9454P processors (part of the selection runs on an EPYC 7551P); the BiasBios runs took 21 GPU-hours, one NVIDIA A100 40 GB per run, with TF32 matrix products.

## Data

* MNIST (LeCun et al.) and Fashion-MNIST (Xiao et al., MIT licence) are
  downloaded through torchvision; Biased MNIST is built from MNIST by FSW's
  code at load time, with the run's seed.
* DRUG: "Drug consumption (quantified)", UCI Machine Learning Repository,
  doi:10.24432/C5TC7S (Fehrman et al.), through `ucimlrepo`; see its UCI page
  for its terms.
* BiasBios: the biographies of De-Arteaga et al. (https://github.com/microsoft/biosbias,
  MIT licence for the code), in the release of Ravfogel et al.
  (https://github.com/shauli-ravfogel/nullspace_projection), downloaded from
  the files that repository points to.

## Licence

MIT (`LICENSE`), for the code of this repository; `third_party/` keeps its own
licences.
