"""Generate uniform, uncropped 2x2 combined figures for eps in {0.5,1.0,1.1,4}
from the experiment1 (100-trial) arrays. Saves PDFs into figs/exploring_m.

Layout: sized for full-width display (one eps per row in the paper). A single
shared legend (m) sits on top instead of one legend per panel, and the y-axis
limits are FIXED and common across all eps so the four figures are directly
comparable; the x-axis is set explicitly because the sharex autoscale over-pads
the log axis on the left."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = os.environ.get("ARR", "journal/arrays/experiment1")
OUT = os.environ.get("FIGS_OUT", "figs/exploring_m")
os.makedirs(OUT, exist_ok=True)

nums_records = [1000, 10000, 100000, 1000000]
m_labels = ["J/10", "J", "10J", "100J"]
panels = [("plugin", "DMS", "Plg, DMS"), ("plugin", "QMS", "Plg, QMS"),
          ("unbiased", "DMS", "Adj, DMS"), ("unbiased", "QMS", "Adj, QMS")]

YLIM = (1e-3, 1e1)   # shared across ALL eps so the panels are directly comparable;
                     # a fit on the boundary of Beta (radius 1.45) has error at most
                     # (1.45 + ||beta*||)^2 < 7.5, so the top of 10 covers every case
XLIM = (7e2, 1.4e6)  # tight around the data n=1e3..1e6 (sharex autoscale over-pads on log)
floor = 1e-3         # lower clip so std bars stay positive on the log axis


def make_fig(tag, eps_label, fname):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.0), sharex=True, sharey=True)
    handles = labels = None
    for ax, (est, syn, title) in zip(axes.flat, panels):
        a = np.load(f"{ARR}/res_{est}_{syn}_eps{tag}.npy")  # [m,n,trials]
        mean = a.mean(axis=2); std = a.std(axis=2)
        for mi, mlab in enumerate(m_labels):
            lo = np.clip(mean[mi] - std[mi], floor, None)
            yerr = np.vstack([mean[mi] - lo, std[mi]])
            ax.errorbar(nums_records, mean[mi], yerr=yerr, marker="o",
                        markersize=5, capsize=3, linewidth=1.4, label=mlab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, which="both", alpha=0.2)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    fig.supxlabel("n", fontsize=15)
    fig.supylabel("Error", fontsize=15)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.85])  # leave room for the top legend
    fig.legend(handles, labels, title="m", loc="upper center", ncol=4,
               fontsize=13, title_fontsize=13, frameon=True,
               bbox_to_anchor=(0.5, 1.0))
    path = os.path.join(OUT, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    for tag, lab in [("05", "0.5"), ("10", "1.0"), ("11", "1.1"), ("4", "4")]:
        make_fig(tag, lab, f"combined_eps{tag}.pdf")
