"""Primal-dual group weighting for fair class-incremental learning.

The method plugs into FSW's framework (``third_party/FSW``) as an algorithm
class, so that data splits, backbone, replay memory, trainer and metrics are
FSW's own. Training on task ``l`` solves, by stochastic primal-dual updates,

    min_theta  L_cur(theta) + tau * L_mem(theta)
    s.t.       c_g(theta) <= eps   for every group g seen so far,

where L_cur is the mean loss on the current task, L_mem the mean loss on the
replay memory (FSW's objective, recovered exactly when the dual step is 0), and
c_g a loss-disparity surrogate of the target fairness measure:

* EER (groups are classes):         c_y     = l_y - mean_{y'} l_{y'}
* EO  (groups are class x attribute): c_{y,z} = l_{y,z} - mean_{z'} l_{y,z'}
* DP  (FSW's sufficient condition):   c_{y,z} = r_{y,z} l_{y,z} - mean_{z'} r_{y,z'} l_{y,z'},
                                      r_{y,z} = m_{y,z} / m_{*,z}

The reference term (the mean) is held fixed in the primal step, so a group's
multiplier only ever raises that group's weight: no group's loss is ascended.
The multipliers mu_g >= 0 are updated by projected dual ascent on running
estimates of c_g, and enter the primal step as additive group weights over the
current batch and the replay batch together.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from algorithms.imbalance import Heuristic2
from algorithms.sensitive import Heuristic3


class PrimalDualMixin:
    """Shared implementation for the class-level and the sensitive-attribute variants."""

    group_by_attribute: bool = True   # False: groups are classes (EER)

    # ------------------------------------------------------------------ setup
    def _pd_init(self):
        p = self.params
        self.pd_eps = float(p.get("pd_eps", 0.0))
        self.pd_eta = float(p.get("pd_eta", 0.0))            # dual step size; 0 recovers plain replay
        self.pd_ema = float(p.get("pd_ema", 0.9))            # smoothing of the group-loss estimates
        self.pd_normalize = bool(p.get("pd_normalize", True))
        self.pd_estimator = p.get("pd_estimator", "batch")    # 'batch' (prequential) or 'probe' (held out)
        self.pd_probe_per_group = int(p.get("pd_probe_per_group", 8))
        self.pd_probe_every = int(p.get("pd_probe_every", 20))
        self.pd_probe_from_memory = bool(p.get("pd_probe_from_memory", True))
        self.pd_scope = p.get("pd_scope", "old")             # cross-class constraints: 'old' or 'all' classes
        self.pd_base = p.get("pd_base", "mean")              # 'mean' (FSW's objective) or 'balanced' groups
        self.pd_brake = float(p.get("pd_brake", 0.0))         # class-level: multipliers scale the current batch down
        self.pd_rate = bool(p.get("pd_rate", False))          # DP: constraints on soft prediction rates
        self.pd_metric = p.get("metric", "EER")
        self.pd_mu_max = float(p.get("pd_mu_max", 1e6))
        num_classes = len(self.benchmark.class_idx)
        self.pd_num_z = 2 if self.group_by_attribute else 1
        self.pd_num_groups = num_classes * self.pd_num_z
        self.mu = torch.zeros(self.pd_num_groups)
        self.est = torch.zeros(self.pd_num_groups)
        self.est_seen = torch.zeros(self.pd_num_groups, dtype=torch.bool)
        self.pd_step = 0
        self.probe_indices = {}          # task -> list of indices into benchmark.trains[task]
        self.pd_log = []                 # per-epoch diagnostics
        # DP ratios r_{y,z} = m_{y,z} / m_{*,z} (FSW's sufficient condition), from training counts
        self.pd_ratio = torch.ones(self.pd_num_groups)
        self.rate_mean = torch.zeros(self.pd_num_groups, num_classes)   # E[softmax | y', z]
        self.rate_seen = torch.zeros(self.pd_num_groups, dtype=torch.bool)
        self.rate_mu = torch.zeros(num_classes, self.pd_num_z)          # multipliers of the rate constraints
        if self.group_by_attribute and self.pd_metric == "DP":
            m = getattr(self.benchmark, "m_dict", None)
            if m is None:
                raise ValueError("DP needs benchmark.m_dict (training counts per class and attribute)")
            for y in range(num_classes):
                for z in range(2):
                    tot = float(np.sum(m[z][:num_classes]))
                    self.pd_ratio[y * 2 + z] = m[z][y] / tot if tot > 0 else 0.0

    # ---------------------------------------------------------------- groups
    def _attribute(self, targ, sensitive_label):
        if not self.group_by_attribute:
            return torch.zeros_like(targ)
        if self.params["dataset"] == "BiasedMNIST":
            return torch.ne(targ, sensitive_label).long()      # FSW's convention: 1 = bias-conflicting
        return sensitive_label.long()

    def _group_ids(self, targ, sensitive_label):
        z = self._attribute(targ, sensitive_label)
        return targ.long() * self.pd_num_z + z

    def _seen_classes(self, task_id):
        return int(task_id) * self.benchmark.num_classes_per_split

    # ------------------------------------------------------------ constraints
    def _constraints(self, num_seen_classes):
        """Return (c, defined) for every group: c_g and whether it can be evaluated."""
        G, Z = self.pd_num_groups, self.pd_num_z
        c = torch.zeros(G)
        defined = torch.zeros(G, dtype=torch.bool)
        k = num_seen_classes * Z
        est, seen = self.est[:k], self.est_seen[:k]
        if not self.group_by_attribute:
            # Cross-class constraints. With scope 'old' (default) only the classes of
            # past tasks are constrained, against the mean over all seen classes: the
            # constraint is "no old class falls behind", i.e. no unfair forgetting.
            # The current classes start each task with a large loss (they have never
            # been predicted), which would otherwise wind their multipliers up.
            if int(seen.sum()) >= 2:
                ref = est[seen].mean()
                mask = seen.clone()
                if self.pd_scope == "old":
                    n_old = num_seen_classes - self.benchmark.num_classes_per_split
                    mask[max(n_old, 0):] = False
                c[:k] = torch.where(mask, est - ref, torch.zeros_like(est))
                defined[:k] = mask
            return c, defined
        vals = est * self.pd_ratio[:k] if self.pd_metric == "DP" else est
        vals = vals.view(num_seen_classes, Z)
        ok = seen.view(num_seen_classes, Z).all(dim=1, keepdim=True).expand(-1, Z)
        dev = vals - vals.mean(dim=1, keepdim=True)
        c[:k] = torch.where(ok, dev, torch.zeros_like(dev)).reshape(-1)
        defined[:k] = ok.reshape(-1)
        return c, defined

    def _group_sums(self, gids, losses):
        G = self.pd_num_groups
        sums = torch.bincount(gids, weights=losses, minlength=G)
        counts = torch.bincount(gids, minlength=G).to(sums.dtype)
        return sums, counts

    def _update_estimates(self, gids, losses):
        sums, counts = self._group_sums(gids, losses.to(torch.float64))
        have = counts > 0
        vals = (sums / counts.clamp(min=1)).to(self.est.dtype)
        b = self.pd_ema
        new = torch.where(self.est_seen, b * self.est + (1 - b) * vals, vals)
        self.est = torch.where(have, new, self.est)
        self.est_seen = self.est_seen | have

    def _dual_step(self, task_id):
        if self.pd_eta <= 0 or self.pd_rate:
            return          # with rate constraints (DP), the rate multipliers are the only ones
        c, defined = self._constraints(self._seen_classes(task_id))
        upd = torch.clamp(self.mu + self.pd_eta * (c - self.pd_eps), min=0.0, max=self.pd_mu_max)
        self.mu = torch.where(defined, upd, self.mu)

    # ------------------------------------------------------------------ probe
    def _groups_of(self, task_id, indices):
        ds = self.benchmark.trains[task_id]
        idx = np.asarray(indices, dtype=np.int64)
        targ = torch.as_tensor(np.asarray(ds.targets)[idx]).long()
        if self.group_by_attribute:
            sens = torch.as_tensor(np.asarray(ds.sensitives)[idx]).long()
        else:
            sens = targ.clone()
        return self._group_ids(targ, sens).numpy()

    def _choose_probe(self, task_id):
        """Hold out pd_probe_per_group samples per group of this task, never trained on.

        With ``pd_probe_from_memory`` (default) the probe is taken out of the
        task's replay-memory allocation, so that replay plus probe use FSW's
        memory budget; otherwise it is drawn from the rest of the task's data.
        """
        seq = np.asarray(self.benchmark.seq_indices_train[task_id])
        mem = np.asarray(self.benchmark.memory_indices_train.get(task_id, []), dtype=np.int64)
        rng = np.random.RandomState(int(self.params["seed"]) * 1000 + int(task_id))
        chosen = []
        if self.pd_probe_from_memory:
            gids = self._groups_of(task_id, mem)
            for g in np.unique(gids):
                cand = mem[gids == g]
                k = min(self.pd_probe_per_group, len(cand) // 2)
                if k > 0:
                    chosen.extend(rng.choice(cand, size=k, replace=False).tolist())
            chosen_set = set(int(i) for i in chosen)
            self.benchmark.memory_indices_train[task_id] = [int(i) for i in mem if int(i) not in chosen_set]
        else:
            mem_set = set(mem.tolist())
            gids = self._groups_of(task_id, seq)
            for g in np.unique(gids):
                cand = [int(i) for i in seq[gids == g] if int(i) not in mem_set]
                k = min(self.pd_probe_per_group, len(cand) // 4)
                if k > 0:
                    chosen.extend(rng.choice(cand, size=k, replace=False).tolist())
            chosen_set = set(int(i) for i in chosen)
        self.benchmark.seq_indices_train[task_id] = [int(i) for i in seq if int(i) not in chosen_set]
        self.probe_indices[task_id] = sorted(int(i) for i in chosen)

    @torch.no_grad()
    def _probe_estimates(self, task_id):
        device = self.params["device"]
        was_training = self.backbone.training
        self.backbone.eval()
        crit = torch.nn.CrossEntropyLoss(reduction="none")
        sums = torch.zeros(self.pd_num_groups)
        counts = torch.zeros(self.pd_num_groups)
        for t in range(1, int(task_id) + 1):
            idx = self.probe_indices.get(t, [])
            if not idx:
                continue
            loader = DataLoader(Subset(self.benchmark.trains[t], idx), batch_size=256, shuffle=False)
            for items in loader:
                inp, targ, task_ids, _, _, sens, *_ = items
                if isinstance(inp, list):
                    inp = [x.to(device) for x in inp]
                else:
                    inp = inp.to(device)
                out = self.backbone(inp, task_ids.to(device))
                loss = crit(out, targ.to(device)).cpu()
                gids = self._group_ids(targ, sens)
                sums.index_add_(0, gids, loss)
                counts.index_add_(0, gids, torch.ones_like(loss))
        if was_training:
            self.backbone.train()
        have = counts > 0
        vals = sums / counts.clamp(min=1)
        b = self.pd_ema
        new = torch.where(self.est_seen, b * self.est + (1 - b) * vals, vals)
        self.est = torch.where(have, new, self.est)
        self.est_seen = self.est_seen | have

    # -------------------------------------------------------------- training
    def before_training_task(self):
        super().before_training_task()
        if self.pd_estimator == "probe":
            self._choose_probe(self.current_task)

    def prepare_train_loader(self, task_id, epoch=0):
        num_workers = self.params.get("num_dataloader_workers", torch.get_num_threads())
        return self.benchmark.load(task_id, self.params["batch_size_train"],
                                   num_workers=num_workers, pin_memory=True)[0]

    def _memory_batch(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)
        inp, targ, task_ids, _, _, sens, *_ = batch
        device = self.params["device"]
        if isinstance(inp, list):
            inp = [x.to(device) for x in inp]
        else:
            inp = inp.to(device)
        return inp, targ.to(device), task_ids.to(device), sens.to(device)

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        device = self.params["device"]
        task_id = int(task_ids[0])
        if sensitive_label is None:
            sensitive_label = targ
        optimizer.zero_grad()
        crit = torch.nn.CrossEntropyLoss(reduction="none")
        out = self.backbone(inp, task_ids)
        loss_cur = crit(out, targ)
        gid_cur = self._group_ids(targ, sensitive_label).cpu()

        use_mem = task_id > 1 and self.params["tau"] and self.episodic_memory_loader is not None
        if use_mem:
            inp_m, targ_m, tid_m, sens_m = self._memory_batch()
            out_m = self.backbone(inp_m, tid_m)
            loss_mem = crit(out_m, targ_m)
            gid_mem = self._group_ids(targ_m, sens_m).cpu()

        # running estimates and dual step (before the primal step)
        if self.pd_eta > 0:
            if self.pd_estimator == "batch":
                self._update_estimates(gid_cur, loss_cur.detach().cpu())
                if use_mem:
                    self._update_estimates(gid_mem, loss_mem.detach().cpu())
            elif self.pd_step % self.pd_probe_every == 0:
                self._probe_estimates(task_id)
            self._dual_step(task_id)
        self.pd_step += 1

        # group weights: base (plain replay) + multipliers, optionally renormalised.
        # A group's weight multiplies its batch-mean loss; groups absent from the
        # batch get no weight in this step.
        extra_scale = self.pd_ratio if self.pd_metric == "DP" else torch.ones(self.pd_num_groups)

        def group_weights(gids, base_scale):
            counts = torch.bincount(gids, minlength=self.pd_num_groups).float()
            present = counts > 0
            base = base_scale * counts / max(len(gids), 1)
            if self.pd_base == "balanced" and self.pd_num_z > 1:
                # each class keeps its batch share, split equally over its groups present
                per_class = counts.view(-1, self.pd_num_z)
                n_present = (per_class > 0).sum(dim=1, keepdim=True).clamp(min=1).float()
                class_mass = base_scale * per_class.sum(dim=1, keepdim=True) / max(len(gids), 1)
                base = torch.where(per_class > 0, class_mass / n_present, torch.zeros_like(per_class)).reshape(-1)
            w = torch.where(present, base + self.mu * extra_scale, torch.zeros_like(base))
            return counts, base, w

        n_c, b_c, w_c = group_weights(gid_cur, 1.0)
        base_tot, w_tot = b_c.sum(), w_c.sum()
        if use_mem:
            n_m, b_m, w_m = group_weights(gid_mem, float(self.params["tau"]))
            base_tot, w_tot = base_tot + b_m.sum(), w_tot + w_m.sum()
        scale = (base_tot / w_tot) if (self.pd_normalize and w_tot > 0) else torch.tensor(1.0)

        def per_sample(gids, counts, w):
            lut = scale * w / counts.clamp(min=1)
            return lut[gids].to(device=device, dtype=loss_cur.dtype)

        coef_cur = per_sample(gid_cur, n_c, w_c)
        if self.pd_brake > 0 and not self.group_by_attribute and task_id > 1:
            # Forgetting constraints act through the current task's weight: replayed
            # samples are memorised, so raising their weight barely moves the old
            # classes; scaling down the current batch's gradient does.
            n_old = (task_id - 1) * self.benchmark.num_classes_per_split
            brake = 1.0 / (1.0 + self.pd_brake * float(self.mu[:n_old].sum()))
            coef_cur = coef_cur * brake
        total = (coef_cur * loss_cur).sum()
        if use_mem:
            total = total + (per_sample(gid_mem, n_m, w_m) * loss_mem).sum()
        if self.pd_rate:
            total = total + self._rate_penalty(task_id, out, gid_cur, out_m if use_mem else None,
                                               gid_mem if use_mem else None)
        total.backward()
        optimizer.step()

    # ------------------------------------------------------- DP rate constraints
    def _rate_prior(self, num_seen):
        """P(y' | z) over the seen classes, from the training counts m_{y',z}."""
        m = self.benchmark.m_dict
        p = torch.zeros(self.pd_num_groups)
        for z in range(self.pd_num_z):
            tot = float(sum(m[z][y] for y in range(num_seen)))
            for y in range(num_seen):
                p[y * self.pd_num_z + z] = m[z][y] / tot if tot > 0 else 0.0
        return p

    def _rate_step(self, task_id, probs_list, gids_list):
        """Update the per-group mean softmax vectors and the rate multipliers."""
        num_seen = self._seen_classes(task_id)
        G, Z = self.pd_num_groups, self.pd_num_z
        C = self.rate_mean.shape[1]
        for probs, gids in zip(probs_list, gids_list):
            sums = torch.zeros(G, C).index_add_(0, gids, probs)
            counts = torch.bincount(gids, minlength=G).float()
            have = counts > 0
            vals = sums / counts.clamp(min=1).unsqueeze(1)
            b = self.pd_ema
            new = torch.where(self.rate_seen.unsqueeze(1), b * self.rate_mean + (1 - b) * vals, vals)
            self.rate_mean = torch.where(have.unsqueeze(1), new, self.rate_mean)
            self.rate_seen = self.rate_seen | have
        prior = self._rate_prior(num_seen)
        k = num_seen * Z
        if not bool(self.rate_seen[:k].all()):
            return
        # rho[y, z] = sum_{y'} P(y'|z) E[softmax_y | y', z]
        weighted = (prior[:k].unsqueeze(1) * self.rate_mean[:k]).view(num_seen, Z, C)
        rho = weighted.sum(dim=0)[:, :num_seen].T                      # (num_seen classes y, Z)
        dev = rho - rho.mean(dim=1, keepdim=True)
        mu = self.rate_mu[:num_seen]
        self.rate_mu[:num_seen] = torch.clamp(mu + self.pd_eta * (dev - self.pd_eps), min=0.0, max=self.pd_mu_max)

    def _rate_penalty(self, task_id, out_cur, gid_cur, out_mem, gid_mem):
        """sum_{y,z} mu_{y,z} * rho_hat_{y,z}: lowers class y's soft rate where it is too high."""
        num_seen = self._seen_classes(task_id)
        outs = [out_cur] + ([out_mem] if out_mem is not None else [])
        gids = [gid_cur] + ([gid_mem] if gid_mem is not None else [])
        probs = [torch.softmax(o, dim=1) for o in outs]
        if self.pd_eta > 0:
            self._rate_step(task_id, [p.detach().cpu() for p in probs], gids)
        prior = self._rate_prior(num_seen).to(out_cur.device)
        pen = out_cur.new_zeros(())
        mu = self.rate_mu.to(out_cur.device)                           # (classes, Z)
        for p, g in zip(probs, gids):
            g = g.to(out_cur.device)
            counts = torch.bincount(g, minlength=self.pd_num_groups).float().clamp(min=1)
            z = g % self.pd_num_z
            # sample i in group (y', z): weight P(y'|z)/n_(y',z), times sum_y mu_{y,z} softmax_y(x_i)
            w = prior[g] / counts[g]
            pen = pen + (w * (p * mu[:, z].T).sum(dim=1)).sum()
        return pen

    def training_epoch_end(self):
        self.pd_log.append({
            "task": int(self.current_task),
            "step": int(self.pd_step),
            "mu": [round(float(v), 6) for v in self.mu.tolist()],
            "est": [round(float(v), 6) for v in self.est.tolist()],
        })
        return super().training_epoch_end()

    def dump_diagnostics(self, path):
        with open(path, "w") as f:
            json.dump({"epochs": self.pd_log,
                       "probe_sizes": {int(k): len(v) for k, v in self.probe_indices.items()}}, f)


class PrimalDualClass(PrimalDualMixin, Heuristic2):
    """Groups are classes; the target measure is FSW's EER disparity (MNIST, FMNIST)."""

    group_by_attribute = False

    def __init__(self, backbone, benchmark, params, **kwargs):
        super().__init__(backbone, benchmark, params, **kwargs)
        self._pd_init()


class PrimalDualAttribute(PrimalDualMixin, Heuristic3):
    """Groups are (class, sensitive attribute); the target measure is EO or DP."""

    group_by_attribute = True

    def __init__(self, backbone, benchmark, params, **kwargs):
        super().__init__(backbone, benchmark, params, **kwargs)
        self._pd_init()
