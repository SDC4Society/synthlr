"""
Experiment 1 (Section 4.1): plug-in vs adjusted logistic regression on
DMS/QMS synthetic data, sweeping n and m for one epsilon.

Usage (from code root):
    uv run python scripts/run_experiment1.py                 # eps in {0.5, 1.0, 1.1, 4}
    uv run python scripts/run_experiment1.py --eps 0.5 1.0   # a subset
    uv run python scripts/run_experiment1.py --eps 1.0 --trials 2   # smoke test

Both estimators maximize a weighted log-likelihood over the J cells,
    sum_j w_j * l_j(beta),   l_j(beta) = y_j log p_j + (1-y_j) log(1-p_j),
with
    plug-in :  w = arrow_m / m                                   (eq. loss_m_ip)
    adjusted:  w = (n+gamma)/n * (arrow_m/m - gamma/(n+gamma) * 1/J)   (eq. def_U)
where n is the size of the ORIGINAL data and gamma = J * gamma_cell.

History: the implementation released with the first revision (v1.0 of this repository) computed
the correction term of the adjusted objective incorrectly (it was identically
zero) and used the synthetic size m in place of n, so the adjusted objective
degenerated to  m/(m+gamma) * cross-entropy.  Fixed 2026-09-24.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synthlr import sampler
from synthlr.privacy import gamma_min_DMS, gamma_min_QMS, gamma_min_QMS_exact


def gamma_cell_QMS(epsilon, m):
    """Per-cell dummy size for QMS, following the paper's convention: the exact
    solution of inequality (1.10) for epsilon <= 1, and the closed-form value of
    Corollary 2.4 (simplified_QMS_gammamin) for epsilon > 1.  The condition tables
    (gen_tables.py) use the same convention."""
    return gamma_min_QMS_exact(epsilon, m) if epsilon <= 1 else gamma_min_QMS(epsilon, m)


# ---------------------------------------------------------------------------
# Cells.  Cell index j = sum_k x_k 2^k + y 2^d  (same as transformer() before).
# ---------------------------------------------------------------------------

def cell_features(d):
    """Return Xc (J x (d+1), first column = intercept) and yc (J,)."""
    J = 2 ** (d + 1)
    j = np.arange(J)
    yc = (j // (2 ** d)).astype(float)
    jm = j % (2 ** d)
    bits = ((jm[:, None] >> np.arange(d)[None, :]) & 1).astype(float)
    Xc = np.hstack([np.ones((J, 1)), bits])
    return Xc, yc


def data_to_freq(X, y):
    n, d = X.shape
    J = 2 ** (d + 1)
    j = X.astype(int) @ (2 ** np.arange(d)) + y.astype(int) * 2 ** d
    return np.bincount(j, minlength=J).astype(float)


def logistic(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ---------------------------------------------------------------------------
# Objective (negative, for minimize) and weights
# ---------------------------------------------------------------------------

def neg_obj_and_grad(beta, Xc, yc, w):
    """-(sum_j w_j l_j(beta)) and its gradient."""
    z = Xc @ beta
    p = logistic(z)
    ll = yc * np.log(np.clip(p, 1e-15, 1)) + (1 - yc) * np.log(np.clip(1 - p, 1e-15, 1))
    return -(w * ll).sum(), -(Xc.T @ (w * (yc - p)))


def plugin_weights(arrow_m, m):
    return arrow_m / m


def adjusted_weights(arrow_m, m, n, gamma):
    """arrow_U of eq. (def_U); n = original data size, gamma = sum of dummies."""
    J = len(arrow_m)
    return (n + gamma) / n * (arrow_m / m - (gamma / (n + gamma)) / J)


def beta_radius(d):
    """Radius of Beta in eq. (beta_ristriction): log(0.99/0.01) / ||X||_2 with
    ||X||_2 = sqrt(1+d) for binary explanatory variables plus the intercept.
    For d = 9 this is about 1.45.  (Before 2026-09-24 the code used 4.6, i.e.
    log(99) itself, which is the radius of a much larger ball.)"""
    return math.log(0.99 / 0.01) / math.sqrt(1 + d)


def fit(Xc, yc, w, constraint, d, stats=None, radius=None):
    """Maximize sum_j w_j l_j(beta) over Beta with SLSQP from beta = 0.

    When the weights have negative entries (adjusted estimator with large gamma)
    the objective is not concave and the maximizer sits on the boundary of Beta.
    SLSQP then occasionally stops with an infeasible iterate; in that case the
    iterate is projected onto Beta and the optimization is restarted from there,
    and the feasible candidate with the better objective is returned.  Feasible
    outputs are returned as they are, whether or not SLSQP reports success.
    """
    def solve(x0, maxiter):
        return minimize(neg_obj_and_grad, x0, args=(Xc, yc, w), jac=True, method="SLSQP",
                        constraints=[constraint], options={"maxiter": maxiter, "ftol": 1e-9})

    R = beta_radius(d) if radius is None else radius

    def feasible(x):
        return np.linalg.norm(x) <= R * (1 + 1e-6)

    res = solve(np.zeros(d + 1), 200)
    x = res.x
    if stats is not None:
        stats["n_fit"] += 1
        stats["n_fail"] += int(not res.success)
        stats["max_nit"] = max(stats["max_nit"], int(res.nit))
    if not feasible(x):
        if stats is not None:
            stats["n_infeasible"] = stats.get("n_infeasible", 0) + 1
        x_proj = x / np.linalg.norm(x) * R * (1 - 1e-6)
        cands = [x_proj]
        res2 = solve(x_proj, 1000)
        if feasible(res2.x):
            cands.append(res2.x)
        x = min(cands, key=lambda b: neg_obj_and_grad(b, Xc, yc, w)[0])
    return x


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(epsilon, save_dir, num_trials=100, seed=1,
                   nums_records=(1000, 10000, 100000, 1000000), d=9,
                   m_factors=(0.1, 1, 10, 100), tag=None):
    np.random.seed(seed)          # beta*, X, y, DMS and QMS all draw from np.random

    J = 2 ** (d + 1)
    sizes_m = [int(J * f) for f in m_factors]
    Xc, yc = cell_features(d)

    shape = (len(sizes_m), len(nums_records), num_trials)
    res = {k: np.zeros(shape) for k in
           ("plugin_QMS", "plugin_DMS", "unbiased_QMS", "unbiased_DMS")}
    stats = {k: dict(n_fit=0, n_fail=0, max_nit=0) for k in res}

    # Beta = {beta : ||beta||_2 <= R}, written as ||beta||^2 <= R^2 so that the
    # constraint is smooth at the starting point 0 (SLSQP otherwise reports
    # "positive directional derivative" on some adjusted fits).
    R = beta_radius(d)
    constraint = NonlinearConstraint(lambda b: b @ b, 0, R ** 2, jac=lambda b: 2 * b)

    print(f"epsilon = {epsilon}, d = {d}, J = {J}, trials = {num_trials}, radius of Beta = {R:.4f}")
    print(f"m = {sizes_m}")
    for m in sizes_m:
        print(f"  m={m:7d}: gamma_cell QMS={gamma_cell_QMS(epsilon, m):.4f}"
              f"  DMS={gamma_min_DMS(epsilon, m):.4f}")

    t0 = time.time()
    for i, n in enumerate(nums_records):
        print(f"\nn = {n}")
        for t in range(num_trials):
            if t % 10 == 0:
                print(f"  trial {t}  ({time.time() - t0:.0f}s)", flush=True)

            beta = (np.random.rand(d + 1) - 0.5) * 2
            beta *= math.sqrt(4) / (d + 1)

            X = (np.random.rand(n, d) > 0.5).astype(float)
            prob = logistic(X @ beta[1:] + beta[0])
            y = np.random.binomial(1, prob).astype(float)
            arrow_n = data_to_freq(X, y)

            for j, m in enumerate(sizes_m):
                # ---- QMS ----
                g_cell = gamma_cell_QMS(epsilon, m)
                gamma = g_cell * J
                arrow_m = np.asarray(sampler.QMS(m, arrow_n + g_cell), dtype=float)

                b = fit(Xc, yc, plugin_weights(arrow_m, m), constraint, d, stats["plugin_QMS"])
                res["plugin_QMS"][j, i, t] = ((b - beta) ** 2).sum()
                b = fit(Xc, yc, adjusted_weights(arrow_m, m, n, gamma), constraint, d, stats["unbiased_QMS"])
                res["unbiased_QMS"][j, i, t] = ((b - beta) ** 2).sum()

                # ---- DMS ----
                g_cell = gamma_min_DMS(epsilon, m)
                gamma = g_cell * J
                theta = np.random.dirichlet(arrow_n + g_cell)
                arrow_m = np.random.multinomial(m, theta).astype(float)

                b = fit(Xc, yc, plugin_weights(arrow_m, m), constraint, d, stats["plugin_DMS"])
                res["plugin_DMS"][j, i, t] = ((b - beta) ** 2).sum()
                b = fit(Xc, yc, adjusted_weights(arrow_m, m, n, gamma), constraint, d, stats["unbiased_DMS"])
                res["unbiased_DMS"][j, i, t] = ((b - beta) ** 2).sum()

    os.makedirs(save_dir, exist_ok=True)
    eps_tag = tag if tag is not None else str(epsilon).replace(".", "")  # 0.5 -> "05", 4 -> "4"
    for k, a in res.items():
        np.save(os.path.join(save_dir, f"res_{k}_eps{eps_tag}.npy"), a)
    meta = dict(epsilon=epsilon, d=d, J=J, nums_records=list(nums_records), sizes_m=sizes_m,
                num_trials=num_trials, seed=seed, beta_radius=R, optimizer_stats=stats,
                elapsed_sec=round(time.time() - t0), date=time.strftime("%Y-%m-%d"))
    with open(os.path.join(save_dir, f"meta_eps{eps_tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {save_dir} with tag eps{eps_tag}  ({meta['elapsed_sec']}s)")
    print("optimizer:", json.dumps(stats))
    print("\nmean squared error, rows m=", sizes_m, " cols n=", list(nums_records))
    for k, a in res.items():
        print(k)
        print(np.array2string(a.mean(axis=2), precision=4, suppress_small=True))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, nargs="+", default=[0.5, 1.0, 1.1, 4],
                    help="privacy budgets to run (default: all four of the paper)")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                                                 "journal", "arrays", "experiment1"))
    args = ap.parse_args()
    for eps in args.eps:
        eps = int(eps) if float(eps).is_integer() and eps >= 2 else eps   # 4.0 -> tag "4"
        run_experiment(eps, args.out, num_trials=args.trials, seed=args.seed)
