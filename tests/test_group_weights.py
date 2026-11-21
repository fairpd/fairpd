"""The primal-dual step: with a zero dual step it is FSW's plain replay objective."""

import types

import pytest
import torch

from fairpd import bootstrap

bootstrap.setup()

from fairpd.algorithms import PrimalDualMixin  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self, d=5, k=4):
        super().__init__()
        self.lin = torch.nn.Linear(d, k)

    def forward(self, x, task_ids=None):
        return self.lin(x)


def make_algo(eta, metric="EO", by_attr=True, tau=2.0):
    algo = PrimalDualMixin.__new__(PrimalDualMixin)
    algo.group_by_attribute = by_attr
    algo.params = {"pd_eta": eta, "pd_eps": 0.0, "pd_ema": 0.0, "pd_normalize": True,
                   "pd_estimator": "batch", "metric": metric, "dataset": "Drug", "tau": tau,
                   "device": torch.device("cpu"), "seed": 0}
    algo.benchmark = types.SimpleNamespace(class_idx=list(range(2)), num_classes_per_split=1,
                                           m_dict={0: [10, 10], 1: [10, 10]})
    algo.backbone = Tiny(k=2)
    algo._pd_init()
    return algo


def batches(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(16, 5, generator=g)
    y = torch.ones(16, dtype=torch.long)               # current task: class 1
    s = (torch.arange(16) % 4 == 0).long()             # 25 % minority
    xm = torch.randn(8, 5, generator=g)
    ym = torch.zeros(8, dtype=torch.long)              # memory: class 0
    sm = (torch.arange(8) % 2).long()
    return x, y, s, xm, ym, sm


def run_step(algo, x, y, s, xm, ym, sm):
    opt = torch.optim.SGD(algo.backbone.parameters(), lr=0.1)
    mem = iter([(xm, ym, torch.full((8,), 1), torch.arange(8), torch.ones(8), sm)])
    algo.episodic_memory_loader = [None]
    algo.episodic_memory_iter = mem
    before = [p.detach().clone() for p in algo.backbone.parameters()]
    algo.training_step(torch.full((16,), 2), x, y, opt, None, sensitive_label=s)
    return [p.detach() - b for p, b in zip(algo.backbone.parameters(), before)]


def reference_step(algo_state, x, y, xm, ym, tau):
    model = Tiny(k=2)
    model.load_state_dict(algo_state)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ce = torch.nn.CrossEntropyLoss()
    loss = ce(model(x), y) + tau * ce(model(xm), ym)
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    opt.step()
    return [p.detach() - b for p, b in zip(model.parameters(), before)]


def test_zero_dual_step_is_plain_replay():
    algo = make_algo(eta=0.0)
    state = {k: v.clone() for k, v in algo.backbone.state_dict().items()}
    x, y, s, xm, ym, sm = batches()
    got = run_step(algo, x, y, s, xm, ym, sm)
    want = reference_step(state, x, y, xm, ym, tau=2.0)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-6)


def test_dual_variables_raise_the_worse_group_only():
    algo = make_algo(eta=1.0)
    # the minority group of class 1 has the higher loss estimate
    algo.est[:] = torch.tensor([0.5, 0.5, 0.2, 1.0])
    algo.est_seen[:] = True
    algo._dual_step(task_id=2)
    mu = algo.mu.tolist()
    assert mu[3] > 0 and mu[2] == 0          # (y=1, z=1) raised, (y=1, z=0) not
    assert mu[0] == 0 and mu[1] == 0         # class 0 balanced: no constraint active


def test_dp_ratios_from_counts():
    algo = make_algo(eta=1.0, metric="DP")
    # m_{y,z} / m_{*,z} with equal counts: 1/2 each
    assert torch.allclose(algo.pd_ratio, torch.full((4,), 0.5))


@pytest.mark.parametrize("by_attr", [True, False])
def test_group_ids(by_attr):
    algo = make_algo(eta=0.0, by_attr=by_attr)
    y = torch.tensor([0, 1, 1])
    s = torch.tensor([1, 0, 1])
    gid = algo._group_ids(y, s).tolist()
    assert gid == ([1, 2, 3] if by_attr else [0, 1, 1])


def test_balanced_base_splits_class_mass_equally():
    algo = make_algo(eta=0.0)
    algo.pd_base = "balanced"
    x, y, s, xm, ym, sm = batches()
    # one step with the balanced base equals a step on the loss where each
    # class's two groups get half of the class's batch share
    state = {k: v.clone() for k, v in algo.backbone.state_dict().items()}
    got = run_step(algo, x, y, s, xm, ym, sm)
    model = Tiny(k=2)
    model.load_state_dict(state)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    lc, lm = ce(model(x), y), ce(model(xm), ym)
    cur = 0.5 * lc[s == 0].mean() + 0.5 * lc[s == 1].mean()
    mem = 2.0 * (0.5 * lm[sm == 0].mean() + 0.5 * lm[sm == 1].mean())
    (cur + mem).backward()
    before = [p.detach().clone() for p in model.parameters()]
    opt.step()
    want = [p.detach() - b for p, b in zip(model.parameters(), before)]
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-6)


def test_rate_mode_uses_rate_multipliers_only():
    algo = make_algo(eta=1.0, metric="DP")
    algo.pd_rate = True
    algo.est[:] = torch.tensor([0.5, 0.5, 0.2, 1.0])
    algo.est_seen[:] = True
    algo._dual_step(task_id=2)
    assert torch.all(algo.mu == 0)           # the loss-proxy multipliers stay at zero
