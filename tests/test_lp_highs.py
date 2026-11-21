import itertools

import numpy as np

from fairpd.lp_highs import absolute_and_nonabsolute_minsum_lp_highs


def objective(A, b, C, d, x):
    return np.abs(b - A @ x).sum() + (d - C @ x).sum()


def test_matches_brute_force_on_vertices():
    # A tiny instance: the LP optimum is attained at a vertex of the feasible set;
    # compare against the best feasible 0/1 vector and check the LP is no worse.
    rng = np.random.RandomState(0)
    n, m = 2, 7
    A = rng.randn(n, m)
    b = rng.randn(n)
    C = rng.rand(n, m) * 0.1
    d = np.full(n, 10.0)          # d - C x >= 0 holds for every x in [0,1]^m
    x = absolute_and_nonabsolute_minsum_lp_highs(A, b, C, d)
    assert x.shape == (m,)
    assert np.all(x >= -1e-9) and np.all(x <= 1 + 1e-9)
    best01 = min(objective(A, b, C, d, np.array(v, float)) for v in itertools.product([0, 1], repeat=m))
    assert objective(A, b, C, d, x) <= best01 + 1e-7


def test_basic_solution_has_few_fractional_weights():
    # 2n equality rows: a basic optimum has at most 2n fractional x_i (the
    # structural fact the paper's analysis of FSW rests on).
    rng = np.random.RandomState(1)
    n, m = 3, 400
    A = rng.randn(n, m) / m
    b = rng.randn(n) * 0.1
    C = np.abs(rng.randn(n, m)) / m
    d = np.full(n, 1.0)
    x = absolute_and_nonabsolute_minsum_lp_highs(A, b, C, d)
    frac = np.sum((x > 1e-7) & (x < 1 - 1e-7))
    assert frac <= 2 * n
