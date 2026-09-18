#!/usr/bin/env python
"""Plot the GLMM slow-gamma_c x Stim interaction slope for the EC-HPC
coherence two-band model (Hypothesis 1c). Four figure variants saved:

  v1: all trial-level points (raw 0/1 jittered)
  v2: per-subject mean accuracy points (one dot per patient per stim cond)
  v3: GLMM-predicted line with model-based 95% CI shading (no overlay)
  v4: 10-bin quantile overlay (mean +/- SE per bin) on top of GLMM line + CI
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

PAIR = "EC_HPC"
PAIR_LABEL = "EC-SUB"
FOCUS_BAND = "slow_gamma_c"
FOCUS_LABEL = "EC-SUB slow gamma coherence"
INTERACTION_P = 0.008


def load_pair_data():
    df = pd.read_csv(INPUT_CSV)
    df = df[(df["yes_or_no"].isin(["yes", "no"])) & (df["trial_type"] != "new")]
    df = df[df["Region"] == PAIR].copy()
    df["Accuracy"] = (df["yes_or_no"] == "yes").astype(int)
    df["StimCond"] = np.where(df["trial_type"] == "nostim", "nostim", "stim")

    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in freq_cols])
    theta_cols = [c for c, f in zip(freq_cols, freqs) if THETA_LO <= f <= THETA_HI]
    sg_cols = [c for c, f in zip(freq_cols, freqs) if SG_LO <= f <= SG_HI]
    df["theta"] = df[theta_cols].mean(axis=1)
    df["slow_gamma"] = df[sg_cols].mean(axis=1)
    df["theta_c"] = df["theta"] - df["theta"].mean()
    df["slow_gamma_c"] = df["slow_gamma"] - df["slow_gamma"].mean()
    return df


COND_COLOR = {"nostim": "#1f77b4", "stim": "#d62728"}
COND_LABEL = {"nostim": "No stim", "stim": "Stim"}


def overlay_predictions(ax):
    pred = pd.read_csv(COEF_DIR / f"coherence_twoband_{PAIR}_predictions.csv")
    for cond in ["nostim", "stim"]:
        sub = pred[pred["stim"] == cond]
        ax.fill_between(
            sub["band"], sub["p_lo"], sub["p_hi"],
            color=COND_COLOR[cond], alpha=0.18,
        )
        ax.plot(sub["band"], sub["p_hat"],
                color=COND_COLOR[cond], lw=2.2, label=COND_LABEL[cond])


def style_ax(ax, n_trials, n_subjects):
    ax.axhline(0.5, ls=":", color="0.5", lw=0.8)
    ax.axvline(0.0, ls=":", color="0.5", lw=0.8)
    ax.set_xlabel(f"{FOCUS_LABEL} (centered)", fontsize=11, fontweight="bold")
    ax.set_ylabel("P(remembered)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{PAIR_LABEL} slow gamma coherence x Prior Stimulation\n"
        f"(p = {INTERACTION_P:.3f}, n trials = {n_trials}, "
        f"N subjects = {n_subjects})",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.25)


def plot_version(version, df, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    overlay_predictions(ax)

    if version == "v1":
        for cond in ["nostim", "stim"]:
            sub = df[df["StimCond"] == cond]
            jitter = (np.random.RandomState(0).rand(len(sub)) - 0.5) * 0.06
            ax.scatter(
                sub[FOCUS_BAND], sub["Accuracy"] + jitter,
                c=COND_COLOR[cond], s=10, alpha=0.25,
                edgecolors="none",
            )

    elif version == "v2":
        ps = pd.read_csv(COEF_DIR / f"coherence_twoband_{PAIR}_per_subject.csv")
        for cond in ["nostim", "stim"]:
            sub = ps[ps["StimCond"] == cond]
            ax.scatter(
                sub[FOCUS_BAND], sub["Accuracy"],
                c=COND_COLOR[cond], s=np.clip(sub["n_trials"] * 1.0, 30, 220),
                alpha=0.78, edgecolors="white", linewidths=0.6,
            )

    elif version == "v3":
        pass

    elif version == "v4":
        for cond in ["nostim", "stim"]:
            sub = df[df["StimCond"] == cond]
            if sub.empty:
                continue
            bins = pd.qcut(sub[FOCUS_BAND], 10, duplicates="drop")
            agg = sub.groupby(bins, observed=True).agg(
                x=(FOCUS_BAND, "mean"),
                acc=("Accuracy", "mean"),
                n=("Accuracy", "size"),
            )
            agg["se"] = np.sqrt(agg["acc"] * (1 - agg["acc"]) / agg["n"])
            ax.errorbar(
                agg["x"], agg["acc"], yerr=agg["se"],
                fmt="o", color=COND_COLOR[cond], alpha=0.8,
                ms=6, capsize=3, lw=1.0,
            )

    n_subjects = int(df["Patient"].nunique())
    n_trials = len(df)
    style_ax(ax, n_trials, n_subjects)

    captions = {
        "v1": ("Lines: GLMM-predicted P(remembered) at the mean level of "
               "theta coherence; shading: model-based 95% CI. "
               "Dots: every trial (jittered around 0/1)."),
        "v2": ("Lines: GLMM-predicted P(remembered) at the mean level of "
               "theta coherence; shading: model-based 95% CI. "
               "Dots: per-subject means (one per stim condition; "
               "dot size proportional to that subject's trial count)."),
        "v3": ("Lines: GLMM-predicted P(remembered) at the mean level of "
               "theta coherence; shading: model-based 95% CI from "
               "vcov(glmer)."),
        "v4": ("Lines: GLMM-predicted P(remembered) at the mean level of "
               "theta coherence; shading: model-based 95% CI. "
               "Points: data binned into 10 quantiles (deciles) within stim "
               "condition; error bars show binomial SE of the bin mean."),
    }
    fig.text(
        0.5, 0.02, captions[version],
        ha="center", va="bottom", fontsize=9, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {version}: {out_path}")


def main():
    plt.rcParams.update({"font.family": "serif"})
    df = load_pair_data()
    print(f"{PAIR_LABEL} trials: {len(df)}, subjects: {df['Patient'].nunique()}")

    base = COEF_DIR
    plot_version("v1", df, base / f"{PAIR}_slowgamma_x_stim_v1_all_trials.png")
    plot_version("v2", df, base / f"{PAIR}_slowgamma_x_stim_v2_per_subject.png")
    plot_version("v3", df, base / f"{PAIR}_slowgamma_x_stim_v3_model_band.png")
    plot_version("v4", df, base / f"{PAIR}_slowgamma_x_stim_v4_10bins.png")


if __name__ == "__main__":
    main()
