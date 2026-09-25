"""
Tests for scripts/run_experiment1.py (Section 4.1 experiment) and the
QMS sampler fix of 2026-09-24.

Run from the code root:  uv run pytest tests/test_experiment1.py -q
"""
import collections
import itertools
import math
import os
import sys

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import gammaln

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_experiment1 import (cell_features, data_to_freq, neg_obj_and_grad,
                             plugin_weights, adjusted_weights, logistic, fit,
                             beta_radius)
from synthlr import sampler
from synthlr.phi import phi_QMS
from scipy.optimize import NonlinearConstraint


def _records(freq, d):
    """Expand a frequency vector into records (X, y), cell order = index order."""
    Xc, yc = cell_features(d)
    rep = freq.astype(int)
    return np.repeat(Xc[:, 1:], rep, axis=0), np.repeat(yc, rep)


def test_data_to_freq_matches_cell_index():
    d = 4
    rng = np.random.default_rng(0)
    X = (rng.random((500, d)) > 0.5).astype(float)
    y = (rng.random(500) > 0.5).astype(float)
    freq = data_to_freq(X, y)
    assert freq.sum() == 500
    Xr, yr = _records(freq, d)
    # multiset equality of records
    a = sorted(map(tuple, np.column_stack([X, y])))
    b = sorted(map(tuple, np.column_stack([Xr, yr])))
    assert a == b


def test_plugin_objective_equals_record_cross_entropy():
    d = 5
    rng = np.random.default_rng(1)
    Xc, yc = cell_features(d)
    freq = rng.integers(0, 20, size=2 ** (d + 1)).astype(float)
    m = freq.sum()
    beta = rng.normal(size=d + 1)
    f, g = neg_obj_and_grad(beta, Xc, yc, plugin_weights(freq, m))
    Xr, yr = _records(freq, d)
    p = logistic(Xr @ beta[1:] + beta[0])
    ce = -(yr * np.log(p) + (1 - yr) * np.log(1 - p)).mean()
    assert f == pytest.approx(ce, rel=1e-12)


def test_gradient_matches_finite_difference():
    d = 5
    rng = np.random.default_rng(2)
    Xc, yc = cell_features(d)
    freq = rng.integers(0, 20, size=2 ** (d + 1)).astype(float)
    w = adjusted_weights(freq, freq.sum(), n=1000.0, gamma=3000.0)
    for _ in range(5):
        beta = rng.normal(size=d + 1)
        err = check_grad(lambda b: neg_obj_and_grad(b, Xc, yc, w)[0],
                         lambda b: neg_obj_and_grad(b, Xc, yc, w)[1], beta)
        assert err < 1e-5


def test_beta_radius_matches_paper():
    # eq. (beta_ristriction): log(0.99/0.01) / sqrt(1+d)
    assert beta_radius(9) == pytest.approx(math.log(99) / math.sqrt(10))
    assert beta_radius(9) == pytest.approx(1.4531, abs=1e-4)


def test_fit_output_is_in_beta():
    d = 5
    rng = np.random.default_rng(6)
    Xc, yc = cell_features(d)
    R = beta_radius(d)
    constraint = NonlinearConstraint(lambda b: b @ b, 0, R ** 2, jac=lambda b: 2 * b)
    for _ in range(20):
        freq = rng.integers(0, 5, size=2 ** (d + 1)).astype(float)
        w = adjusted_weights(freq, freq.sum(), n=100.0, gamma=5000.0)   # heavy dummies: boundary case
        x = fit(Xc, yc, w, constraint, d)
        assert np.linalg.norm(x) <= R * (1 + 1e-6)


def test_adjusted_weights_reduce_to_plugin_when_gamma_zero():
    freq = np.array([3.0, 5.0, 2.0, 0.0])
    np.testing.assert_allclose(adjusted_weights(freq, 10, n=100, gamma=0.0),
                               plugin_weights(freq, 10))


def test_adjusted_weights_are_unbiased_for_arrow_n_over_n():
    """E[arrow_U | arrow_n] = arrow_n / n  (Lemma U_unbiased), checked on DMS."""
    rng = np.random.default_rng(3)
    J, n, m = 8, 200, 50
    arrow_n = rng.multinomial(n, rng.dirichlet(np.ones(J) * 2)).astype(float)
    g_cell = 40.0            # gamma_dot = 320 >> n: the regime where the old code degenerated
    gamma = g_cell * J
    R = 20000
    acc = np.zeros(J)
    for _ in range(R):
        theta = rng.dirichlet(arrow_n + g_cell)
        arrow_m = rng.multinomial(m, theta).astype(float)
        acc += adjusted_weights(arrow_m, m, n, gamma)
    est = acc / R
    # standard error of each coordinate is about (n+gamma)/n * sqrt(phi p(1-p)/m / R)
    assert np.abs(est - arrow_n / n).max() < 0.02
    assert est.sum() == pytest.approx(1.0, abs=1e-9)


def test_adjusted_does_not_degenerate_and_beats_plugin_under_heavy_dummies():
    """The old code returned beta_hat = 0 (error = ||beta*||^2) whenever gamma >> m.
    With n large and gamma/(n+gamma) large, the adjusted fit should be far from 0
    and much closer to beta* than the plug-in fit."""
    np.random.seed(0)
    d, n = 6, 200000
    J = 2 ** (d + 1)
    Xc, yc = cell_features(d)
    beta = np.array([0.3, 0.8, -0.6, 0.5, -0.4, 0.2, 0.7]) * 0.7    # ||beta|| = 1.03 < radius 1.74
    X = (np.random.rand(n, d) > 0.5).astype(float)
    y = np.random.binomial(1, logistic(X @ beta[1:] + beta[0])).astype(float)
    arrow_n = data_to_freq(X, y)
    m = 50000
    g_cell = 3000.0                       # gamma_dot = 384000 ~ 2n: gamma/(n+gamma) ~ 0.66
    gamma = g_cell * J
    R = beta_radius(d)
    constraint = NonlinearConstraint(lambda b: b @ b, 0, R ** 2, jac=lambda b: 2 * b)
    err_plg, err_adj = [], []
    for _ in range(5):
        theta = np.random.dirichlet(arrow_n + g_cell)
        arrow_m = np.random.multinomial(m, theta).astype(float)
        bp = fit(Xc, yc, plugin_weights(arrow_m, m), constraint, d)
        ba = fit(Xc, yc, adjusted_weights(arrow_m, m, n, gamma), constraint, d)
        err_plg.append(((bp - beta) ** 2).sum())
        err_adj.append(((ba - beta) ** 2).sum())
    b2 = (beta ** 2).sum()
    assert np.mean(err_adj) < 0.2 * b2          # not the degenerate beta_hat = 0
    assert np.mean(err_adj) < 0.5 * np.mean(err_plg)   # plug-in is shrunk toward 0 by the dummies


def test_old_objective_was_degenerate():
    """Documents the bug fixed on 2026-09-24 (v1.0 of this repository): the adjusted objective equalled m/(m+gamma) * cross-entropy because the
    correction term was identically zero and n was taken from the synthetic data."""
    def old_obj_term_grad(beta, x):
        z = np.dot(beta[1:], x) + beta[0]
        g = np.zeros(len(beta))
        g[0] = logistic(z) - 1 + logistic(-z) - 0
        g[1:] = (logistic(z) - 1 + logistic(-z) - 0) * x
        return g

    def old_unbiased_objective(beta, X, y, gamma):
        n, d = X.shape                      # n here is the synthetic size m
        J = 2 ** (d + 1)
        Xc, _ = cell_features(d)
        reduction_term = np.array([old_obj_term_grad(beta, Xc[j, 1:]) for j in range(2 ** d)]).sum(axis=0) / J
        p = logistic(beta[0] + X @ beta[1:])
        return -(n / (n + gamma) * (y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
                 - gamma / (n + gamma) * (-reduction_term[0]))

    d = 4
    rng = np.random.default_rng(4)
    freq = rng.integers(0, 10, size=2 ** (d + 1)).astype(float)
    Xr, yr = _records(freq, d)
    m = len(yr)
    gamma = 500.0
    beta = rng.normal(size=d + 1)
    p = logistic(Xr @ beta[1:] + beta[0])
    ce = -(yr * np.log(p) + (1 - yr) * np.log(1 - p)).mean()
    assert old_unbiased_objective(beta, Xr, yr, gamma) == pytest.approx(m / (m + gamma) * ce, rel=1e-12)


# ---------------------------------------------------------------------------
# QMS sampler: beta must be 1/A_sub at every split
# ---------------------------------------------------------------------------

def _exact_qms_pmf(m, a):
    A = sum(a)
    out = {}
    for ys in itertools.product(range(m + 1), repeat=len(a)):
        if sum(ys) != m:
            continue
        lp = gammaln(m + 1) - sum(gammaln(v + 1) for v in ys) - math.log(A) - (m - 1) * math.log(A + m)
        lp += sum(math.log(ac) + (yc - 1) * math.log(ac + yc) for ac, yc in zip(a, ys))
        out[ys] = math.exp(lp)
    return out


def test_recursive_qms_matches_exact_distribution():
    a = [3.3, 0.3, 1.3, 0.3, 2.0, 0.3, 0.3, 0.3]     # n_j + gamma_cell, J = 8
    m, N = 4, 60000
    P = _exact_qms_pmf(m, a)
    assert sum(P.values()) == pytest.approx(1.0, abs=1e-9)
    np.random.seed(0)
    A = sum(a)
    probs = np.array(a) / A
    C = collections.Counter(tuple(int(v) for v in sampler._recursive_QMS(m, probs, 1 / A))
                            for _ in range(N))
    tv = 0.5 * sum(abs(C.get(k, 0) / N - p) for k, p in P.items())
    assert tv < 0.03          # the pre-fix sampler gave about 0.13 here; sampling noise is about 0.02
    # per-cell variance m pi (1-pi) phi
    phi = phi_QMS(m, A)
    ys = np.array(list(C.keys()))
    w = np.array(list(C.values())) / N
    mu = w @ ys
    var = w @ (ys - mu) ** 2
    np.testing.assert_allclose(var, m * probs * (1 - probs) * phi, rtol=0.15)


def test_split_qms_uses_same_beta_convention():
    """_split_QMS + _recursive_QMS on the pieces must give the same marginal variance."""
    a = np.array([5.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    m, N = 6, 30000
    A = a.sum()
    probs = a / A
    np.random.seed(1)
    out = np.zeros((N, len(a)))
    for r in range(N):
        ms, ps = sampler._split_QMS(m, probs, 1 / A, 2)
        out[r] = np.concatenate([sampler._recursive_QMS(mi, pi, 1 / A) for mi, pi in zip(ms, ps)])
    phi = phi_QMS(m, A)
    np.testing.assert_allclose(out.mean(axis=0), m * probs, atol=0.03)
    np.testing.assert_allclose(out.var(axis=0), m * probs * (1 - probs) * phi, rtol=0.2)
