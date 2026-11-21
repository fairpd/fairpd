"""FSW's linear program, solved with HiGHS instead of CPLEX.

FSW solves, once per epoch,

    min_{x in [0,1]^m}  sum_i |b_i - A_i x|  +  sum_i yy_i
    s.t.                yy_i = d_i - C_i x,  yy_i >= 0,

with ``absolute_and_nonabsolute_minsum_LP_solver_v1`` (CPLEX, through docplex),
which needs a CPLEX licence for more than 1000 variables. This is the same
program, written for ``scipy.optimize.linprog`` (HiGHS), with the variables
ordered as [x (m), y- (n), y+ (n), yy (n)] and the equality constraints

    A x - y- + y+ = b,     C x + yy = d.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def absolute_and_nonabsolute_minsum_lp_highs(A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray,
                                             binary: bool = False) -> np.ndarray:
    if binary:
        raise NotImplementedError("the binary (MILP) variant is not used by FSW")
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    n, m = A.shape
    if C.shape != (n, m) or b.shape[0] != n or d.shape[0] != n:
        raise ValueError(f"shape mismatch: A{A.shape} b{b.shape} C{C.shape} d{d.shape}")
    eye = sparse.identity(n, format="csr")
    zero = sparse.csr_matrix((n, n))
    a_eq = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(A), -eye, eye, zero]),
        sparse.hstack([sparse.csr_matrix(C), zero, zero, eye]),
    ]).tocsr()
    b_eq = np.concatenate([b, d])
    c = np.concatenate([np.zeros(m), np.ones(3 * n)])
    bounds = [(0.0, 1.0)] * m + [(0.0, None)] * (3 * n)
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"HiGHS failed: {res.message}")
    return res.x[:m]
