#!/usr/bin/env python
"""Combined two-subplot figure for the CA-PRC and EC-HPC coherence
two-band slow-gamma_c x Stim interactions (Hypothesis 1c). 10-bin overlay.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
COEF_DIR = SCRIPT_DIR / "outputs" / "h1c_imbalanced_retrieval_mlm"
INPUT_CSV = SCRIPT_DIR / "outputs" / "csvs" / "combined_retrieval_coherence_all_mlmr_input.csv"

THETA_LO, THETA_HI = 4.88, 7.81
SG_LO, SG_HI = 30.27, 54.69

PANELS = [
    {
        "pair": "CA_PRC",
        "label": "CA-PRC",
        "focus_label": "CA-PRC slow gamma coherence",
        "interaction_p": 0.003,
    },
    {
        "pair": "EC_HPC",
        "label": "EC-SUB",
        "focus_label": "EC-SUB slow gamma coherence",
        "interaction_p": 0.008,
    },
]

COND_COLOR = {"nostim": "#1f77b4", "stim": "#d62728"}
COND_LABEL = {"nostim": "No stim", "stim": "Stim"}


def load_pair_data(pair):
    df = pd.read_csv(INPUT_CSV)
    df = df[(df["yes_or_no"].isin(["yes", "no"])) & (df["trial_type"] != "new")]
    df = df[df["Region"] == pair].copy()
    df["Accuracy"] = (df["yes_or_no"] == "yes").astype(int)
    df["StimCond"] = np.where(df["trial_type"] == "nostim", "nostim", "stim")
    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in freq_cols])
    sg_cols = [c for c, f in zip(freq_cols, freqs) if SG_LO <= f <= SG_HI]
    df["slow_gamma"] = df[sg_cols].mean(axis=1)
    df["slow_gamma_c"] = df["slow_gamma"] - df["slow_gamma"].mean()
    return df


def plot_panel(ax, panel):
    pair = panel["pair"]
    df = load_pair_data(pair)
    pred = pd.read_csv(COEF_DIR / f"coherence_twoband_{pair}_predictions.csv")

    for cond in ["nostim", "stim"]:
        sub = pred[pred["stim"] == cond]
        ax.fill_between(sub["band"], sub["p_lo"], sub["p_hi"],
                        color=COND_COLOR[cond], alpha=0.18)
        ax.plot(sub["band"], sub["p_hat"],
                color=COND_COLOR[cond], lw=2.2, label=COND_LABEL[cond])

    for cond in ["nostim", "stim"]:
        sub = df[df["StimCond"] == cond]
        bins = pd.qcut(sub["slow_gamma_c"], 10, duplicates="drop")
        agg = sub.groupby(bins, observed=True).agg(
            x=("slow_gamma_c", "mean"),
            acc=("Accuracy", "mean"),
            n=("Accuracy", "size"),
        )
        agg["se"] = np.sqrt(agg["acc"] * (1 - agg["acc"]) / agg["n"])
        ax.errorbar(agg["x"], agg["acc"], yerr=agg["se"],
                    fmt="o", color=COND_COLOR[cond], alpha=0.8,
                    ms=6, capsize=3, lw=1.0)

    ax.axhline(0.5, ls=":", color="0.5", lw=0.8)
    ax.axvline(0.0, ls=":", color="0.5", lw=0.8)
    ax.set_xlabel(f"{panel['focus_label']} (centered)",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("P(remembered)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{panel['label']} slow gamma coherence x Prior Stimulation\n"
        f"(p = {panel['interaction_p']:.3f}, "
        f"n trials = {len(df)}, N subjects = {df['Patient'].nunique()})",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.25)


def main():
    plt.rcParams.update({"font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax, panel in zip(axes, PANELS):
        plot_panel(ax, panel)
    fig.text(
        0.5, 0.02,
        "Lines: GLMM-predicted P(remembered) at the mean level of theta "
        "coherence; shading: model-based 95% CI from vcov(glmer). "
        "Points: data binned into 10 quantiles (deciles) within stim "
        "condition; error bars show binomial SE of the bin mean.",
        ha="center", va="bottom", fontsize=9, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = COEF_DIR / "CAPRC_ECHPC_slowgamma_x_stim.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
