"""Remembered vs forgotten in IED vs no-IED trials, encoding & retrieval.

Stacked proportion bars (remembered/forgotten) for IED vs no-IED trials, one panel
per phase, with trial counts, rem:forg ratio, and a chi-square test of the
IED x memory association annotated.

Counts come from ied_memory_ratio_by_iedstatus.reconstruct (raw old-item totals)
minus the authoritative IED counts. See that module for the labeling logic.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ied_memory_ratio_by_iedstatus as core

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "Helvetica"

REM_COLOR = "#FFD43B"   # yellow
FORG_COLOR = "#9C7AC9"  # purple
OUT_DIR = Path(__file__).resolve().parent.parent / "OUTPUTS" / "updated_IED_figures"


def compute():
    ied_e = pd.read_csv(core.IED_ENC)
    ied_r = pd.read_csv(core.IED_RET)
    enc_pts = sorted(ied_e["Patient"].unique())
    ret_pts = sorted(ied_r[ied_r["MemoryOutcome"].isin(["remembered", "forgotten"])]["Patient"].unique())

    enc = core.reconstruct("phase1", enc_pts)
    ret = core.reconstruct("phase3", ret_pts)
    e_rem, e_forg = core.counts(enc)
    r_rem, r_forg = core.counts(ret)
    ie_rem, ie_forg = core.ied_counts(core.IED_ENC)
    ir_rem, ir_forg = core.ied_counts(core.IED_RET)

    return {
        "Encoding": {
            "IED": (ie_rem, ie_forg),
            "no-IED": (e_rem - ie_rem, e_forg - ie_forg),
        },
        "Retrieval": {
            "IED": (ir_rem, ir_forg),
            "no-IED": (r_rem - ir_rem, r_forg - ir_forg),
        },
    }


def draw_panel(ax, data, title):
    groups = ["IED", "no-IED"]
    x = np.arange(len(groups))
    for i, g in enumerate(groups):
        rem, forg = data[g]
        n = rem + forg
        p_rem, p_forg = rem / n, forg / n
        ax.bar(i, p_rem, width=0.6, color=REM_COLOR, edgecolor="black", linewidth=1.0, zorder=2)
        ax.bar(i, p_forg, bottom=p_rem, width=0.6, color=FORG_COLOR, edgecolor="black", linewidth=1.0, zorder=2)
        # just the percentages inside each segment
        ax.text(i, p_rem / 2, f"{p_rem:.0%}", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(i, p_rem + p_forg / 2, f"{p_forg:.0%}", ha="center", va="center", fontsize=13)

    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["with IEDs", "no IEDs"], fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, 1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = compute()

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 5.8), sharey=True)
    for ax, phase in zip(axes, ["Encoding", "Retrieval"]):
        draw_panel(ax, data[phase], phase)
    axes[0].set_ylabel("Proportion of target trials", fontsize=15, fontweight="bold")
    for ax in axes:
        plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=REM_COLOR, ec="black"),
               plt.Rectangle((0, 0), 1, 1, fc=FORG_COLOR, ec="black")]
    fig.legend(handles, ["Remembered", "Forgotten"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04),
               prop={"weight": "bold", "size": 13})
    fig.suptitle("Remembered vs forgotten in IED vs no-IED target trials",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()

    png = OUT_DIR / "ied_memory_ratio_ied_vs_noied.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"Saved:\n  {png}")


if __name__ == "__main__":
    main()
