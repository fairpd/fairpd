"""Run one class-incremental experiment in FSW's protocol and write its sidecars.

Mirrors ``third_party/FSW/run.py`` and ``configs.make_params``: the same
benchmarks, backbones, trainers and metric collectors, built the same way, so
that a run of this script and a run of FSW's own script differ only in the
algorithm. Additions:

* ``--method pd``: the primal-dual method (``fairpd.algorithms``);
* ``--method FSW`` solves FSW's linear program with HiGHS (``fairpd.lp_highs``),
  since CPLEX is not available; ``--method vanilla`` is FSW's own code with
  uniform weights (FSW's "W/o FSW");
* ``--split val`` evaluates on a validation split held out from the training
  data (for hyperparameter selection); ``--split test`` on the test split;
* ``--run-dir``: the run's directory, with ``config.json``, ``env.json`` and
  ``metrics.json``. A run whose ``metrics.json`` exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fairpd import bootstrap

DATASETS = ["MNIST", "FashionMNIST", "BiasedMNIST", "Drug", "Bios"]
FAIRNESS_DATASETS = ["BiasedMNIST", "Drug", "Bios"]

# FSW's per-dataset settings (scripts/*.sh, configs.make_params, paper App. "Models and Hyperparameters")
DATASET_DEFAULTS = {
    "MNIST":        dict(model="MLP",  num_tasks=5, per_task_classes=2, epochs=5,  memory_per_task=64),
    "FashionMNIST": dict(model="MLP",  num_tasks=5, per_task_classes=2, epochs=5,  memory_per_task=64),
    "BiasedMNIST":  dict(model="MLP",  num_tasks=5, per_task_classes=2, epochs=15, memory_per_task=128),
    "Drug":         dict(model="MLP",  num_tasks=3, per_task_classes=2, epochs=25, memory_per_task=64),
    "Bios":         dict(model="bert", num_tasks=5, per_task_classes=5, epochs=10, memory_per_task=320),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--method", required=True, choices=["vanilla", "FSW", "pd"])
    p.add_argument("--metric", required=True, choices=["EER", "EO", "DP"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-dir", default=None, help="data root (default: $CL_DATA_DIR, $GLOBALSCRATCH/cl_data, ~/data/cl_data)")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=None, help="epochs per task (default: FSW's per dataset)")
    p.add_argument("--memory-per-task", type=int, default=None, help="replay samples per task (default: FSW's)")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--tau", type=float, required=True, help="weight of the replay loss")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--token-max-length", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tf32", action="store_true",
                   help="allow TF32 matrix multiplications on Ampere GPUs (used for BiasBios on A100s)")
    # FSW
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--lambda", dest="lmbd", type=float, default=0.0)
    # primal-dual
    p.add_argument("--pd-eta", type=float, default=0.0, help="dual step size (0: plain replay)")
    p.add_argument("--pd-eps", type=float, default=0.0, help="constraint tolerance")
    p.add_argument("--pd-ema", type=float, default=0.9)
    p.add_argument("--pd-estimator", choices=["batch", "probe"], default="batch")
    p.add_argument("--pd-probe-per-group", type=int, default=8)
    p.add_argument("--pd-probe-every", type=int, default=20)
    p.add_argument("--pd-no-normalize", action="store_true")
    p.add_argument("--pd-base", choices=["mean", "balanced"], default="mean",
                   help="base weights: the plain batch mean (FSW's objective) or group-balanced within each class")
    p.add_argument("--pd-brake", type=float, default=0.0,
                   help="class-level variant: the old classes' multipliers scale the current batch by 1/(1+k*sum mu)")
    p.add_argument("--pd-rate", action="store_true",
                   help="DP: constrain the soft prediction rates P(yhat=y|z) instead of the loss proxy")
    p.add_argument("--pd-scope", choices=["old", "all"], default="old",
                   help="classes under cross-class (EER) constraints: past tasks only, or all seen")
    p.add_argument("--pd-probe-extra", action="store_true",
                   help="draw the probe outside the replay memory (extra budget) instead of from it")
    p.add_argument("--eval-every-epoch", action="store_true",
                   help="evaluate after every epoch, as FSW's scripts do (the reported metrics read only "
                        "the last epoch of each task, so by default evaluation runs at task ends only)")
    p.add_argument("--force", action="store_true", help="rerun even if metrics.json exists")
    return p.parse_args(argv)


def git_commit() -> dict:
    stamp = bootstrap.REPO_ROOT / ".git_commit"
    if stamp.exists():
        return dict(line.split("=", 1) for line in stamp.read_text().split() if "=" in line)
    try:
        sha = subprocess.check_output(["git", "-C", str(bootstrap.REPO_ROOT), "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(bootstrap.REPO_ROOT), "status", "--porcelain"],
                                             stderr=subprocess.DEVNULL).decode().strip())
        return {"commit": sha, "dirty": str(dirty).lower()}
    except Exception:
        return {"commit": "unknown"}


def env_info(torch) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "cpu": platform.processor(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def make_validation_split(benchmark, args):
    """Hold out a stratified fraction of each task's training data (never of its memory) as the evaluation set."""
    rng = np.random.RandomState(10_000 + args.seed)
    for task in range(1, benchmark.num_tasks + 1):
        ds = benchmark.trains[task]
        seq = np.asarray(benchmark.seq_indices_train[task])
        mem = set(np.asarray(benchmark.memory_indices_train[task]).tolist())
        targ = np.asarray(ds.targets)[seq]
        sens = np.asarray(ds.sensitives)[seq] if hasattr(ds, "sensitives") else np.zeros_like(targ)
        if args.dataset == "BiasedMNIST":
            sens = (sens != targ).astype(int)
        val = []
        for key in sorted(set(zip(targ.tolist(), np.asarray(sens).tolist()))):
            mask = (targ == key[0]) & (np.asarray(sens) == key[1])
            cand = [int(i) for i in seq[mask] if int(i) not in mem]
            k = int(round(args.val_fraction * len(cand)))
            if k > 0:
                val.extend(rng.choice(cand, size=k, replace=False).tolist())
        val_set = set(val)
        benchmark.seq_indices_train[task] = [int(i) for i in seq if int(i) not in val_set]
        benchmark.tests[task] = ds                      # evaluate on the held-out training samples
        benchmark.seq_indices_test[task] = sorted(val)


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if (run_dir / "metrics.json").exists() and not args.force:
        print(f"skip: {run_dir}/metrics.json exists")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = bootstrap.setup(args.data_dir)

    import torch
    import cl_gym  # noqa: F401  (FSW expects it importable)

    d = DATASET_DEFAULTS[args.dataset]
    epochs = args.epochs or d["epochs"]
    mem_per_task = args.memory_per_task or d["memory_per_task"]
    if args.dataset not in FAIRNESS_DATASETS and args.metric != "EER":
        raise ValueError("MNIST and FashionMNIST are evaluated with EER")
    if args.dataset in FAIRNESS_DATASETS and args.metric == "EER":
        raise ValueError("the fairness datasets are evaluated with EO or DP")

    # --- FSW's run.py: seeding and threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(args.threads)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    # --- FSW's make_params, restricted to the keys its code reads
    params = {
        "dataset": args.dataset, "model": d["model"], "seed": args.seed,
        "method": {"vanilla": "vanilla", "FSW": "FSW", "pd": "pd"}[args.method],
        "num_tasks": d["num_tasks"], "epochs_per_task": epochs,
        "per_task_examples": 1_000_000 * d["per_task_classes"],
        "per_task_memory_examples": mem_per_task,
        "batch_size_train": args.batch_size, "batch_size_memory": args.batch_size,
        "batch_size_validation": 256, "token_max_length": args.token_max_length,
        "random_class_idx": False, "tau": args.tau, "optimizer": "sgd",
        "learning_rate": args.lr, "momentum": 0.9, "learning_rate_decay": 1.0,
        "metric": args.metric, "fairness_agg": "mean",
        "alpha": args.alpha if args.method == "FSW" else 0.0,
        "lambda": args.lmbd, "lambda_old": 0.0,
        "all_layer_gradient": False, "optim_version": "v0", "round": False,
        "cuda": 0, "verbose": 1, "post_processing": False, "pp_eps": 0.0,
        "fairloss": "CE", "device": device,
        "criterion": torch.nn.CrossEntropyLoss(),
        "learning_rate_decay_epoch": [30],      # FSW: epochs <= 50 -> decay at epoch 30, i.e. never here
        "output_dir": str(run_dir / "fsw_out"),
        "save_model": False,
        "num_dataloader_workers": args.threads,
        # primal-dual
        "pd_eta": args.pd_eta, "pd_eps": args.pd_eps, "pd_ema": args.pd_ema,
        "pd_estimator": args.pd_estimator, "pd_probe_per_group": args.pd_probe_per_group,
        "pd_probe_every": args.pd_probe_every, "pd_normalize": not args.pd_no_normalize,
        "pd_probe_from_memory": not args.pd_probe_extra, "pd_scope": args.pd_scope,
        "pd_base": args.pd_base, "pd_brake": args.pd_brake, "pd_rate": args.pd_rate,
    }
    Path(params["output_dir"]).mkdir(parents=True, exist_ok=True)
    for sub in ("metrics", "plots"):
        Path(params["output_dir"], sub).mkdir(parents=True, exist_ok=True)

    config = {k: v for k, v in params.items() if k not in ("device", "criterion")}
    config.update({"split": args.split, "val_fraction": args.val_fraction if args.split == "val" else 0.0,
                   "data_root": "CL_DATA_DIR", "git": git_commit(), "argv": sys.argv[1:],
                   "eval_every_epoch": bool(args.eval_every_epoch), "tf32": bool(args.tf32)})
    (run_dir / "config.json").write_text(json.dumps(config, indent=1, default=str))

    # --- benchmark (FSW's run.py)
    from datasets import MNIST, FashionMNIST, BiasedMNIST, Drug, Bios
    common = dict(num_tasks=params["num_tasks"], per_task_memory_examples=params["per_task_memory_examples"],
                  per_task_examples=params["per_task_examples"], joint=False, random_class_idx=False)
    if args.dataset == "MNIST":
        benchmark, input_dim = MNIST(**common), (28, 28)
    elif args.dataset == "FashionMNIST":
        benchmark, input_dim = FashionMNIST(**common), (28, 28)
    elif args.dataset == "BiasedMNIST":
        benchmark, input_dim = BiasedMNIST(**common), (3, 28, 28)
    elif args.dataset == "Drug":
        benchmark, input_dim = Drug(**common), (12)
    else:
        benchmark, input_dim = Bios(max_length=params["token_max_length"], **common), None
    if args.split == "val":
        make_validation_split(benchmark, args)
    class_idx = benchmark.class_idx
    num_classes = len(class_idx)

    if params["model"] == "MLP":
        from backbones import MLP2Layers2
        backbone = MLP2Layers2(input_dim=input_dim, hidden_dim_1=256, hidden_dim_2=256,
                               output_dim=num_classes, class_idx=class_idx, config=params).to(device)
    else:
        from backbones import BertClassifier
        backbone = BertClassifier(num_classes, benchmark.bert_config["model"], params).to(device)

    if args.dataset in FAIRNESS_DATASETS:
        from metrics import FairMetricCollector as MetricCollector
        from trainers.fair_trainer import FairContinualTrainer2 as ContinualTrainer
    else:
        from metrics import MetricCollector2 as MetricCollector
        from trainers.imbalance_trainer import ImbalanceContinualTrainer1 as ContinualTrainer

    if args.method == "pd":
        from fairpd.algorithms import PrimalDualAttribute, PrimalDualClass
        Algorithm = PrimalDualAttribute if args.dataset in FAIRNESS_DATASETS else PrimalDualClass
    elif args.dataset in FAIRNESS_DATASETS:
        from algorithms.sensitive import Heuristic3 as Algorithm
    else:
        from algorithms.imbalance import Heuristic2 as Algorithm

    algorithm = Algorithm(backbone, benchmark, params, requires_memory=True)
    if args.method == "FSW":
        from fairpd.lp_highs import absolute_and_nonabsolute_minsum_lp_highs
        params["solver"] = absolute_and_nonabsolute_minsum_lp_highs
        conv = "converter_LP_absolute_additional" + ("" if args.metric == "EER" else "_" + args.metric)
        params["converter"] = getattr(algorithm, conv)

    if not args.eval_every_epoch:
        base_collector = MetricCollector

        class MetricCollector(base_collector):   # noqa: N801  evaluate at the last epoch of each task
            def on_after_training_epoch(self, trainer):
                if trainer.current_epoch % self.epochs_per_task == 0:
                    super().on_after_training_epoch(trainer)

            def plot_metrics(self, logger=None):  # the per-epoch curves are not collected
                return None

    collector = MetricCollector(num_tasks=params["num_tasks"], eval_interval="epoch",
                                epochs_per_task=params["epochs_per_task"])
    trainer = ContinualTrainer(algorithm, params, callbacks=[collector])
    for name, meter in collector.meters.items():
        if name in ("std", "EER", "EO", "DP"):
            meter.agg = np.mean

    t0 = time.time()
    trainer.run()
    wall = time.time() - t0

    meters = collector.meters
    acc_overall = [float(x) for x in meters["accuracy"].compute_overall()]
    out = {
        "dataset": args.dataset, "method": args.method, "metric": args.metric, "seed": args.seed,
        "split": args.split,
        "accuracy": float(np.mean(acc_overall)),
        "accuracy_per_task": acc_overall,
        "accuracy_matrix": np.asarray(meters["accuracy"].get_data()).tolist(),
        "wall_seconds": wall,
    }
    if "accuracy_s0" in meters and "accuracy_s1" in meters:
        # per-group accuracies (mean over classes of the accuracy within z = 0 / z = 1),
        # averaged over tasks like the accuracy; their mean is the group-balanced accuracy
        try:
            s0 = [float(x) for x in meters["accuracy_s0"].compute_overall()]
            s1 = [float(x) for x in meters["accuracy_s1"].compute_overall()]
            out["accuracy_s0"], out["accuracy_s1"] = float(np.mean(s0)), float(np.mean(s1))
            out["accuracy_group_balanced"] = float(np.mean([(a + b) / 2 for a, b in zip(s0, s1)]))
        except Exception as exc:
            out["accuracy_group_error"] = repr(exc)
    for name in ("EER", "EO", "DP", "std"):
        if name in meters:
            try:
                vals = [float(x) for x in meters[name].compute_overall()]
                out[name] = float(np.mean(vals))
                out[f"{name}_per_task"] = vals
            except Exception as exc:  # a meter the trainer does not fill for this dataset
                out[f"{name}_error"] = repr(exc)
    for key, store in (("lp_seconds", getattr(algorithm, "optim_time", {})),
                       ("grad_seconds", getattr(algorithm, "grad_calculation_time", {}))):
        out[key] = float(sum(sum(v.values()) for v in store.values())) if store else 0.0
    if args.method == "pd":
        algorithm.dump_diagnostics(run_dir / "pd_diagnostics.json")
    if args.method == "FSW":
        # FSW's weights per task, epoch and group (its own record, algorithm.weight_all):
        # how many current samples the LP keeps (weight > 0.05, FSW's drop threshold)
        # and how many weights are fractional.
        summary = []
        for t, epochs in enumerate(getattr(algorithm, "weight_all", []), start=1):
            for e, by_group in enumerate(epochs, start=1):
                row = {"task": t, "epoch": e, "groups": {}}
                for g, w in by_group.items():
                    w = np.asarray(w, dtype=float)
                    row["groups"][str(g)] = {
                        "n": int(w.size), "kept": int((w > 0.05).sum()),
                        "fractional": int(((w > 1e-6) & (w < 1 - 1e-6)).sum()),
                        "mean_weight": float(w.mean()) if w.size else 0.0,
                    }
                summary.append(row)
        (run_dir / "fsw_weights.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "env.json").write_text(json.dumps(env_info(torch), indent=1))
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("accuracy", "wall_seconds") if k in out}
                     | {m: out[m] for m in ("EER", "EO", "DP") if m in out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
