"""
Experiment 1 (Table 2 and Figures 2-4): per-coordinate squared error of the
plug-in and adjusted logistic-regression estimators on DMS/QMS synthetic data,
sweeping n in {1e3, 1e4, 1e5, 1e6} and m in {J/10, J, 10J, 100J} with J = 1024,
over 100 trials.

Usage:
    uv run python scripts/run_experiment1.py            # eps in {0.5, 1.0, 1.1, 4}
    uv run python scripts/run_experiment1.py 0.5 1.0    # only the given eps values

The random seed is fixed (np.random.seed(1)), so the results are reproducible.
The QMS dummy size uses the exact solution (gamma_min_QMS_exact) in the
high-privacy regime epsilon <= 1, and the simplified-corollary approximation
(gamma_min_QMS) for epsilon > 1, matching the conventions used for the
corresponding results in the paper. Output .npy arrays are written to
../journal/arrays/experiment1/ with the tag eps{05,10,11,4}.
"""

import math
import os

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from synthlr import sampler
from synthlr.privacy import gamma_min_DMS, gamma_min_QMS, gamma_min_QMS_exact


# ---------------------------------------------------------------------------
# Helper functions (same as QMS.ipynb)
# ---------------------------------------------------------------------------

def transformer(x, y):
    d = len(x)
    coe = 2 ** np.arange(d)
    j = np.dot(coe, x) + y * 2**d
    return int(j)


def inv_transformer(j, d):
    x = np.zeros(d)
    y = j // (2 ** d)
    j = j % (2 ** d)
    d -= 1
    while d >= 0:
        x[d] = j // (2 ** d)
        j = j % (2 ** d)
        d -= 1
    return (x, y)


def freq_to_data(freq_vec):
    J = len(freq_vec)
    n = int(np.sum(freq_vec))
    d = int(math.log(J, 2)) - 1

    X = np.zeros([n, d])
    y = np.zeros([n])

    i = 0
    for j in range(J):
        j_freq = int(freq_vec[j])
        if j_freq == 0:
            continue
        x_j, y_j = inv_transformer(j, d)
        X[i: i + j_freq] += x_j
        y[i: i + j_freq] += y_j
        i += j_freq

    assert i == n, i
    return (X, y)


def data_to_freq(X, y):
    n, d = X.shape
    J = 2 ** (d + 1)
    freq = np.zeros(J)
    for i in range(n):
        j = transformer(X[i], y[i])
        freq[j] += 1
    return freq


def logistic(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy(beta, X, y):
    z = X @ beta[1:] + beta[0]
    hat_ps = logistic(z)
    return -(y * np.log(hat_ps) + (1 - y) * np.log(1 - hat_ps)).mean()


def cross_entropy_grad(beta, X, y):
    n, d = X.shape
    z = X @ beta[1:] + beta[0]
    hat_ps = logistic(z)
    grad = np.zeros(d + 1)
    grad[0] = (hat_ps - y).mean()
    grad[1:] = X.T @ (hat_ps - y) / n
    return grad


def _obj_term_grad(beta, j, d):
    x, _ = inv_transformer(j, d)
    z = np.dot(beta[1:], x) + beta[0]
    grad = np.zeros(d + 1)
    grad[0] = logistic(z) - 1 + logistic(-z) - 0
    grad[1:] = (logistic(z) - 1 + logistic(-z) - 0) * x
    return grad


def unbiased_objective_function(beta, X, y, gamma):
    n, d = X.shape
    J = 2 ** (d + 1)
    reduction_term = np.array([_obj_term_grad(beta, j, d)
                                for j in range(int(2**d))]).sum(axis=0) / J

    zs = beta[0] + np.dot(X, beta[1:])
    hat_ps = logistic(zs)
    return -(n / (n + gamma) * (y * np.log(hat_ps) + (1 - y) * np.log(1 - hat_ps)).mean()
             - gamma / (n + gamma) * (-reduction_term[0]))


def unbiased_objective_function_grad(beta, X, y, gamma):
    n, d = X.shape
    J = 2 ** (d + 1)
    reduction_term_grad = -np.array([_obj_term_grad(beta, j, d)
                                      for j in range(int(2**d))]).sum(axis=0) / J
    return n / (n + gamma) * cross_entropy_grad(beta, X, y) - gamma / (n + gamma) * reduction_term_grad


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(epsilon, save_dir):
    np.random.seed(1)

    nums_records = [1000, 10000, 100000, 1000000]
    d = 9
    J = 2 ** (d + 1)
    sizes_m = [J // 10, J, J * 10, J * 100]

    num_trials = 100
    res_plugin_QMS   = np.zeros([len(sizes_m), len(nums_records), num_trials])
    res_plugin_DMS   = np.zeros([len(sizes_m), len(nums_records), num_trials])
    res_unbiased_QMS = np.zeros([len(sizes_m), len(nums_records), num_trials])
    res_unbiased_DMS = np.zeros([len(sizes_m), len(nums_records), num_trials])

    constraint = NonlinearConstraint(lambda beta: np.linalg.norm(beta, 2), 0, 4.6)
    method = 'SLSQP'

    # QMS dummy size: exact for the high-privacy regime epsilon <= 1, and the
    # simplified-corollary approximation for epsilon > 1, matching the paper.
    qms_gamma = gamma_min_QMS_exact if epsilon <= 1 else gamma_min_QMS

    print(f"epsilon = {epsilon}, J = {J}")
    print(f"QMS gamma at m=J: {qms_gamma(epsilon, J):.4f} "
          f"({'exact' if epsilon <= 1 else 'approx'})")
    print(f"gamma_min_DMS at m=J:       {gamma_min_DMS(epsilon, J):.4f}")

    for i, n in enumerate(nums_records):
        print(f"\nn = {n}")
        for t in range(num_trials):
            if t % 10 == 0:
                print(f"  trial {t}")

            beta = (np.random.rand(d + 1) - 0.5) * 2
            beta *= math.sqrt(4) / (d + 1)

            X = np.zeros([n, d])
            X[np.random.rand(n, d) > 0.5] = 1

            z = np.dot(X, beta[1:]) + beta[0]
            prob = logistic(z)
            y = np.random.binomial(1, prob)

            arrow_n = data_to_freq(X, y)

            for j, m in enumerate(sizes_m):
                # QMS
                dummies_QMS = np.ones(J) * qms_gamma(epsilon, m)
                arrow_m_QMS = sampler.QMS(m, arrow_n + dummies_QMS)
                syn_X_QMS, syn_y_QMS = freq_to_data(arrow_m_QMS)

                beta_ini = np.zeros(d + 1)
                hat_beta_plugin = minimize(
                    cross_entropy, beta_ini, args=(syn_X_QMS, syn_y_QMS),
                    constraints=[constraint], jac=cross_entropy_grad, method=method
                ).x
                res_plugin_QMS[j, i, t] = ((hat_beta_plugin - beta) ** 2).sum()

                gamma_QMS = dummies_QMS.sum()
                hat_beta_unb = minimize(
                    unbiased_objective_function, beta_ini,
                    args=(syn_X_QMS, syn_y_QMS, gamma_QMS),
                    constraints=[constraint], jac=unbiased_objective_function_grad,
                    method=method
                ).x
                res_unbiased_QMS[j, i, t] = ((hat_beta_unb - beta) ** 2).sum()

                # DMS
                dummies_DMS = np.ones(J) * gamma_min_DMS(epsilon, m)
                p = np.random.default_rng().dirichlet(arrow_n + dummies_DMS)
                arrow_m_DMS = np.random.default_rng().multinomial(m, p, size=1)[0]
                syn_X_DMS, syn_y_DMS = freq_to_data(arrow_m_DMS)

                beta_ini = np.zeros(d + 1)
                hat_beta_plugin = minimize(
                    cross_entropy, beta_ini, args=(syn_X_DMS, syn_y_DMS),
                    constraints=[constraint], jac=cross_entropy_grad, method=method
                ).x
                res_plugin_DMS[j, i, t] = ((hat_beta_plugin - beta) ** 2).sum()

                gamma_DMS = dummies_DMS.sum()
                hat_beta_unb = minimize(
                    unbiased_objective_function, beta_ini,
                    args=(syn_X_DMS, syn_y_DMS, gamma_DMS),
                    constraints=[constraint], jac=unbiased_objective_function_grad,
                    method=method
                ).x
                res_unbiased_DMS[j, i, t] = ((hat_beta_unb - beta) ** 2).sum()

    os.makedirs(save_dir, exist_ok=True)
    eps_tag = str(epsilon).replace(".", "")  # 0.5 -> "05"

    np.save(os.path.join(save_dir, f"res_plugin_QMS_eps{eps_tag}.npy"),   res_plugin_QMS)
    np.save(os.path.join(save_dir, f"res_plugin_DMS_eps{eps_tag}.npy"),   res_plugin_DMS)
    np.save(os.path.join(save_dir, f"res_unbiased_QMS_eps{eps_tag}.npy"), res_unbiased_QMS)
    np.save(os.path.join(save_dir, f"res_unbiased_DMS_eps{eps_tag}.npy"), res_unbiased_DMS)

    print(f"\nSaved to {save_dir} with tag eps{eps_tag}")


if __name__ == "__main__":
    import sys
    save_dir = os.path.join(os.path.dirname(__file__), "..", "journal", "arrays", "experiment1")
    epsilons = [float(a) for a in sys.argv[1:]] or [0.5, 1.0, 1.1, 4.0]
    for eps in epsilons:
        run_experiment(epsilon=eps, save_dir=save_dir)
