"""
Experiment: domain size J x signal strength  (theory-consistent).

Two reviewer points are answered together:
  * "When is the adjusted estimator superior to the plug-in one?"
  * dependence on J (the theory gives the adjusted estimator a ~sqrt(J) larger
    usable-m window).

Bias-variance picture being tested
----------------------------------
  plug-in MSE   ~ ( gamma/(n+gamma) )^2 * (population non-uniformity)   [bias]
  adjusted MSE  ~ ( (n+gamma)/n )^2 * variance                          [variance]
At fixed r=m/n, gamma/(n+gamma) grows with J, so the plug-in bias grows with J,
while the adjusted variance inflation ((n+gamma)/n)^2 also grows with J. Which
estimator wins is therefore a tradeoff that depends on BOTH J and the signal
strength (how far the population is from uniform). We sweep signal strength
(max log-odds) and J to map the transition:
  - weak signal  -> plug-in is ~unbiased; debiasing only adds variance -> plug-in wins/ties
  - strong signal + large J -> plug-in bias dominates -> adjusted wins
This reconciles with the paper's existing single-J (weak, dense beta*) experiment
where the two estimators look nearly identical.

Theory-consistency
------------------
The operative restriction behind Beta / C=5 sqrt(J) is fitted probs in
[0.01,0.99], i.e. max_x |beta.x| <= log(0.99/0.01). This cap is J-independent
(the ||beta||_2 ball in the paper is a conservative Cauchy-Schwarz encoding that
shrinks with d only because ||X||_2 = sqrt(1+d)). We therefore use binary x with
a SPARSE beta* whose signal (max log-odds) is set explicitly and is J-independent,
and we maximise over the operative Beta imposed as the linear constraint
-L <= [1,x].beta <= L over all distinct cells.

The objective is evaluated in the frequency domain (sum over J cells), so cost
is O(J) per evaluation, independent of m.

Run:  uv run python scripts/experiment_vary_J.py
"""

import math
import os
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from synthlr.privacy import gamma_min_DMS


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
EPSILON          = 1.1
DS               = [4, 6, 8, 10]            # J = 2^{d+1} = 32,128,512,2048
RATIOS           = [0.003, 0.01, 0.03]      # m/n
SIGNAL_MAXLOGITS = [0.5, 1.5, 3.0, 4.5]     # max_x|beta.x| levels (weak -> strong)
N                = 20_000_000
NUM_TRIALS       = 100
SEED             = 1
K_ACTIVE         = 3                         # active coords in beta*
L_LOGIT          = math.log(0.99 / 0.01)     # operative cap on |beta.x| (4.595)
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "journal",
                        "arrays", "experiment_vary_J")


def cell_features(d):
    J = 2 ** (d + 1)
    j = np.arange(J)
    y = (j // (2 ** d)).astype(float)
    jm = j % (2 ** d)
    bits = ((jm[:, None] >> np.arange(d)[None, :]) & 1).astype(float)
    Xc = np.hstack([np.ones((J, 1)), bits])
    return Xc, y


def make_beta_star(d, rng, maxlogit):
    """Sparse beta*: K_ACTIVE random coords at +-(maxlogit/K_ACTIVE).
    max_x|beta.x| = maxlogit, J-independent."""
    k = min(K_ACTIVE, d)
    mag = maxlogit / k
    beta = np.zeros(d + 1)
    coords = rng.choice(np.arange(1, d + 1), size=k, replace=False)
    beta[coords] = rng.choice([-1.0, 1.0], size=k) * mag
    return beta


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def neg_obj_and_grad(beta, Xc, yc, w):
    z = Xc @ beta
    p = logistic(z)
    ll = yc * np.log(np.clip(p, 1e-15, 1)) + (1 - yc) * np.log(np.clip(1 - p, 1e-15, 1))
    return -(w * ll).sum(), -(Xc.T @ (w * (yc - p)))


def fit(Xc, yc, w, constraint, d):
    return minimize(neg_obj_and_grad, np.zeros(d + 1), args=(Xc, yc, w),
                    jac=True, method="SLSQP", constraints=[constraint],
                    options={"maxiter": 200, "ftol": 1e-9}).x


def run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    assert max(SIGNAL_MAXLOGITS) <= L_LOGIT + 1e-9, "signal exceeds [0.01,0.99] cap"

    shape = (len(SIGNAL_MAXLOGITS), len(RATIOS), len(DS), NUM_TRIALS)
    res = {k: np.zeros(shape) for k in ("plugin", "unbiased")}

    for di, d in enumerate(DS):
        J = 2 ** (d + 1)
        Xc, yc = cell_features(d)
        constraint = LinearConstraint(Xc[:2 ** d], -L_LOGIT, L_LOGIT)
        print(f"\n=== d={d}, J={J} ===")
        for t in range(NUM_TRIALS):
            if t % 25 == 0:
                print(f"  trial {t}")
            for si, sg in enumerate(SIGNAL_MAXLOGITS):
                beta = make_beta_star(d, rng, sg)
                pp = logistic(Xc @ beta)
                p = np.where(yc == 1, pp, 1 - pp)
                p = p / p.sum()
                arrow_n = rng.multinomial(N, p).astype(float)
                for ri, r in enumerate(RATIOS):
                    m = max(int(round(r * N)), J + 1)
                    gcell = gamma_min_DMS(EPSILON, m)
                    gamma = gcell * J
                    theta = rng.dirichlet(arrow_n + gcell)
                    arrow_m = rng.multinomial(m, theta).astype(float)

                    bp = fit(Xc, yc, arrow_m / m, constraint, d)
                    res["plugin"][si, ri, di, t] = ((bp - beta) ** 2).sum() / (d + 1)

                    w_unb = (N + gamma) / N * (arrow_m / m - (gamma / (N + gamma)) / J)
                    bu = fit(Xc, yc, w_unb, constraint, d)
                    res["unbiased"][si, ri, di, t] = ((bu - beta) ** 2).sum() / (d + 1)

    tag = f"eps{str(EPSILON).replace('.', '')}_n{N:.0e}".replace("+", "")
    np.save(os.path.join(SAVE_DIR, f"res_plugin_DMS_{tag}.npy"), res["plugin"])
    np.save(os.path.join(SAVE_DIR, f"res_unbiased_DMS_{tag}.npy"), res["unbiased"])
    np.save(os.path.join(SAVE_DIR, f"meta_{tag}.npy"),
            dict(epsilon=EPSILON, ds=DS, ratios=RATIOS, signal_maxlogits=SIGNAL_MAXLOGITS,
                 n=N, num_trials=NUM_TRIALS, k_active=K_ACTIVE, seed=SEED),
            allow_pickle=True)

    print("\n===== SUMMARY: mean per-coordinate MSE, winner (plg/adj) =====")
    for si, sg in enumerate(SIGNAL_MAXLOGITS):
        print(f"\n#### signal max-logodds = {sg} ####")
        for ri, r in enumerate(RATIOS):
            print(f"  m/n={r}:  " + "   ".join(
                f"J{2**(d+1)}:{res['plugin'][si,ri,di].mean():.3f}/"
                f"{res['unbiased'][si,ri,di].mean():.3f}"
                f"({'adj' if res['unbiased'][si,ri,di].mean()<res['plugin'][si,ri,di].mean() else 'plg'})"
                for di, d in enumerate(DS)))
    print(f"\nSaved to {SAVE_DIR} (tag sig_{tag}); format plug/adj per J.")


if __name__ == "__main__":
    run()
