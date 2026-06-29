"""Figure: domain size J vs per-coordinate MSE, plug-in vs adjusted (DMS, eps=1.1,
strong sparse signal, n=2e7, 100 trials). One panel per ratio r=m/n. Answers the
reviewer's "when is the adjusted estimator superior" by showing the adjusted
estimator's sqrt(J) advantage: at fixed r the plug-in bias grows with J while the
adjusted estimator stays accurate, the gap widening with J."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "journal/arrays/experiment_vary_J"
OUT = os.environ.get("FIGS_OUT_VARYJ", "figs/plg_vs_adj")
os.makedirs(OUT, exist_ok=True)
tag = "eps11_n2e07"
meta = np.load(f"{ARR}/meta_{tag}.npy", allow_pickle=True).item()
DS = meta["ds"]; RAT = meta["ratios"]
J = [2 ** (d + 1) for d in DS]
pl = np.load(f"{ARR}/res_plugin_DMS_{tag}.npy")     # [ratio, J, trials]
ub = np.load(f"{ARR}/res_unbiased_DMS_{tag}.npy")

fig, axes = plt.subplots(1, len(RAT), figsize=(11, 3.6), sharey=True)
for ax, ri in zip(axes, range(len(RAT))):
    pm, ps = pl[ri].mean(1), pl[ri].std(1)
    um, us = ub[ri].mean(1), ub[ri].std(1)
    ax.errorbar(J, pm, yerr=ps, marker="o", ms=5, capsize=3, color="C3",
                label="Plg")
    ax.errorbar(J, um, yerr=us, marker="s", ms=5, capsize=3, color="C0",
                ls="--", label="Adj")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(J); ax.set_xticklabels([str(j) for j in J])
    ax.set_title(rf"$m/n={RAT[ri]:g}$")
    ax.set_xlabel("J"); ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=9)
axes[0].set_ylabel("Per-coordinate MSE")
fig.suptitle(rf"DMS, $\epsilon=1.1$ (strong signal)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(OUT, "vary_J_compare_eps11.pdf")
fig.savefig(path, bbox_inches="tight"); plt.close(fig)
print("wrote", path)
