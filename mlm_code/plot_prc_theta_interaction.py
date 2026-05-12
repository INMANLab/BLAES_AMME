#!/usr/bin/env python
"""Single-panel PRC theta x prior BLA stimulation interaction figure.

Source GLMM = the per-region single-band model used by the
h1abc_full_retrieval_mlm pipeline (which produced the FDR-corrected BLAMTL
PDF report):

    Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)

with raw (uncentered) theta band power as band_c. Predictions are written by
save_h1abc_theta_glmm_predictions.R into power_BLAMTL_GLMM/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REPORT_DIR = REPO_ROOT / "OUTPUTS" / "retrieval_memory_reports"
GLMM_DIR = REPORT_DIR / "stats" / "h1abc_full_retrieval_mlm" / "power_BLAMTL_GLMM"
FIG_DIR = REPORT_DIR / "figures" / "BLAMTL_power_GLMM_theta_interactions"
FIG_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV = REPO_ROOT / "OUTPUTS" / "csvs" / "combined_retrieval_power_all_mlmr_input.csv"

THETA_LO, THETA_HI = 4.88, 7.81
HPC_SUB = ("HPC", "CA", "DG")

COND_COLOR = {"nostim": "#1f77b4", "stim": "#d62728"}
COND_LABEL = {"nostim": "No stim", "stim": "Stim"}


def _add_allhpc_rows(df, freq_cols):
    sub = df[df["Region"].isin(HPC_SUB)].copy()
    if sub.empty:
        return df
    sub["trial_id"] = sub.groupby(["Patient", "Region"]).cumcount()
    grp_keys = ["Patient", "trial_id", "trial_type", "yes_or_no"]
    agg = (sub.groupby(grp_keys, as_index=False)[list(freq_cols)]
              .mean(numeric_only=True))
    agg["Region"] = "ALLHPC"
    agg = agg.drop(columns=["trial_id"])
    return pd.concat([df, agg], ignore_index=True)


def load_region_data(region):
    df = pd.read_csv(INPUT_CSV)
    df = df[(df["yes_or_no"].isin(["yes", "no"])) & (df["trial_type"] != "new")]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"

    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    if region == "ALLHPC":
        df = _add_allhpc_rows(df, freq_cols)

    df = df[df["Region"] == region].copy()
    df["Accuracy"] = (df["yes_or_no"] == "yes").astype(int)
    df["StimCond"] = np.where(df["trial_type"] == "nostim", "nostim", "stim")

    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in freq_cols])
    theta_cols = [c for c, f in zip(freq_cols, freqs) if THETA_LO <= f <= THETA_HI]
    df["theta"] = df[theta_cols].mean(axis=1)
    return df


def main():
    plt.rcParams.update({"font.family": "serif"})

    df_prc = load_region_data("PRC")
    pred = pd.read_csv(GLMM_DIR / "power_PRC_theta_predictions.csv")

    p_raw = 0.004
    q_fdr = 0.014
    n_trials = len(df_prc)
    n_subjects = int(df_prc["Patient"].nunique())

    fig, ax = plt.subplots(1, 1, figsize=(7, 6.2))

    for cond in ["nostim", "stim"]:
        sub = pred[pred["stim"] == cond]
        ax.fill_between(
            sub["band"], sub["p_lo"], sub["p_hi"],
            color=COND_COLOR[cond], alpha=0.18,
        )
        ax.plot(sub["band"], sub["p_hat"],
                color=COND_COLOR[cond], lw=2.2, label=COND_LABEL[cond])

    for cond in ["nostim", "stim"]:
        sub = df_prc[df_prc["StimCond"] == cond]
        if sub.empty:
            continue
        bins = pd.qcut(sub["theta"], 10, duplicates="drop")
        agg = sub.groupby(bins, observed=True).agg(
            x=("theta", "mean"),
            acc=("Accuracy", "mean"),
            n=("Accuracy", "size"),
        )
        agg["se"] = np.sqrt(agg["acc"] * (1 - agg["acc"]) / agg["n"])
        ax.errorbar(
            agg["x"], agg["acc"], yerr=agg["se"],
            fmt="o", color=COND_COLOR[cond], alpha=0.8,
            ms=6, capsize=3, lw=1.0,
        )

    ax.axhline(0.5, ls=":", color="0.5", lw=0.8)
    ax.axvline(0.0, ls=":", color="0.5", lw=0.8)
    ax.set_xlabel("PRC theta power", fontsize=12, fontweight="bold")
    ax.set_ylabel("P(remembered)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"PRC theta power x Prior Stimulation\n"
        f"(p = {p_raw:.3f}, q_FDR = {q_fdr:.3f}, "
        f"n trials = {n_trials}, N subjects = {n_subjects})",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=14)
    ax.grid(alpha=0.25)
    ax.text(0.0, 0.97, "*", fontsize=28, fontweight="bold",
            ha="center", va="center", color="black")

    fig.text(
        0.5, 0.02,
        "Lines: GLMM-predicted P(remembered); shading: model-based 95% CI. "
        "Points: data binned into 10 quantiles (deciles) within stim condition; "
        "error bars show binomial SE of the bin mean.",
        ha="center", va="bottom", fontsize=9, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = FIG_DIR / "PRC_theta_interaction_slope.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"PRC trials: {n_trials}, subjects: {n_subjects}")


if __name__ == "__main__":
    main()
