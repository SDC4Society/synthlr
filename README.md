# Reproducibility code for "Is the outcome of logistic regression valid on differentially private synthetic data?"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21023052.svg)](https://doi.org/10.5281/zenodo.21023052)

Authors: Hajime Ono and Nobuaki Hoshino.
Journal of Privacy and Confidentiality (submission 1029).

This repository contains the code and pre-computed result arrays needed to
reproduce all numerical figures and tables in the paper. The experiments use
**synthetic data generated entirely within the code** (no external or
confidential data are required).

The samplers studied are the Dirichlet–Multinomial Sampler (DMS) and the
Quasi-Multinomial Sampler (QMS); the estimators are the plug-in estimator and
the adjusted estimator.

## Contents

```
synthlr/                  core package: samplers and privacy/variance functions
  sampler.py              DMS / QMS samplers
  privacy.py              minimum dummy size (gamma) for DMS and QMS
  phi.py                  variance-inflation factor phi for QMS
scripts/
  run_experiment1.py      generates the Experiment-1 arrays (Table 2, Figures 2-4)
  experiment_vary_J.py    generates the domain-size sweep arrays (Figure 5)
  gen_figs_combined.py    Figures 2-4 from the Experiment-1 arrays
  gen_fig_vary_J.py       Figure 5 from the vary-J arrays
  gen_tables.py           mean-error sub-tables and condition tables (all epsilon)
notebooks/
  phi_approx.ipynb        appendix figures on the phi approximation
journal/arrays/           pre-computed result arrays used in the paper
tests/                    unit tests for synthlr (pytest)
pyproject.toml, uv.lock   environment specification
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

This creates a `.venv` and installs all dependencies. Run any script with:

```bash
uv run python scripts/<script>.py
```

Unit tests:

```bash
uv run pytest
```

## Reproducing the figures and tables

All figure/table scripts read the pre-computed arrays in `journal/arrays/` and
write into `figs/` or `tables/` (created automatically). Run from the
repository root.

| Paper item | Command | Output |
|---|---|---|
| Figure 2 (epsilon = 1.1 and 4) | `uv run python scripts/gen_figs_combined.py` | `figs/exploring_m/combined_eps11.pdf`, `combined_eps4.pdf` |
| Figure 3 (epsilon = 0.5) | (same command) | `figs/exploring_m/combined_eps05.pdf` |
| Figure 4 (epsilon = 1.0) | (same command) | `figs/exploring_m/combined_eps10.pdf` |
| Figure 5 (domain size J) | `uv run python scripts/gen_fig_vary_J.py` | `figs/plg_vs_adj/vary_J_compare_eps11.pdf` |
| Appendix phi-approximation figures | run `notebooks/phi_approx.ipynb` | `figs/phi_approx/*.eps` |
| Mean-error sub-tables and condition tables (epsilon = 0.5, 1.0, 1.1, 4) | `uv run python scripts/gen_tables.py` | `tables/mean_subtables.tex`, `tables/cond_tables.tex` |

`gen_tables.py` re-derives the analytic quantities with a scipy-free
implementation (exact QMS dummy size for epsilon <= 1, the simplified-corollary
approximation for epsilon > 1), generates the cell highlighting automatically,
and asserts agreement with the published epsilon = 1.1 table before writing.

## Regenerating the result arrays

The arrays in `journal/arrays/` are the ones used to produce the paper. They can
be regenerated from scratch; the random seed is fixed, so a re-run on the same
NumPy/BLAS stack reproduces them.

```bash
# Experiment 1 (Table 2, Figures 2-4): epsilon in {0.5, 1.0, 1.1, 4}
uv run python scripts/run_experiment1.py
# or a subset, e.g.:
uv run python scripts/run_experiment1.py --eps 0.5 1.0

# Domain-size sweep (Figure 5). Uses n = 2e7 and is computationally heavy;
# a workstation is recommended.
uv run python scripts/experiment_vary_J.py
```

Notes:
- `run_experiment1.py` selects the QMS dummy size by regime to match the paper:
  the exact solution (`gamma_min_QMS_exact`) for epsilon <= 1, and the
  simplified-corollary approximation (`gamma_min_QMS`) for epsilon > 1. DMS uses
  `gamma_min_DMS` throughout.
- `experiment_vary_J.py` evaluates the objective in the frequency domain, so its
  cost is independent of the synthetic-data size m.
- `run_experiment1.py` also evaluates both estimators in the frequency domain:
  the plug-in and adjusted estimators maximize `sum_j w_j l_j(beta)` over the
  J cells with `w = arrow_m/m` and `w = (n+gamma)/n (arrow_m/m - gamma/(n+gamma)/J)`
  (eq. 3.2 of the paper), respectively. The QMS sampler (`_recursive_QMS`,
  `_split_QMS`) is exercised against the exact quasi-multinomial distribution
  in `tests/test_experiment1.py`.
- The QMS sampler in the earlier release used the overdispersion parameter of
  the whole table at every level of the recursive split; it now uses the
  parameter of each subtree (`beta/(p1+p2)`), which is what the quasi-multinomial
  distribution requires. The earlier release also computed the adjusted
  objective incorrectly (its correction term was identically zero and n was
  taken from the synthetic data). Both are fixed in this version and the arrays
  in `journal/arrays/experiment1/` were regenerated; see the change log below.

## Change log

- v1.1 (2026-09-25, DOI [10.5281/zenodo.22952800](https://doi.org/10.5281/zenodo.22952800)): fixed the adjusted objective in `run_experiment1.py` and the
  per-subtree overdispersion in `synthlr/sampler.py`; the estimators are now
  maximized over the set Beta of the paper (radius log(99)/sqrt(1+d), about
  1.45 for d = 9) instead of the ball of radius 4.6 used in v1.0; regenerated
  the Experiment-1 arrays; added `tests/test_experiment1.py`. The domain-size sweep
  (`experiment_vary_J.py`, Figure 5) was not affected and its arrays are unchanged.
- v1.0 (2026-06-29, DOI [10.5281/zenodo.21023053](https://doi.org/10.5281/zenodo.21023053)): initial release with the first revision of the paper.

## License

MIT License; see `LICENSE`.

This work was supported by JST K Program Grant Number JPMJKP24U5, Japan.
