#!/usr/bin/env python
"""2x2 figure: slow gamma power x prior BLA stimulation interaction for each
region in the BLAMTL family (ALLHPC, BLA, EC, PRC). Same v4 style as the
theta version: GLMM line + model-based 95% CI shading + 10-bin overlay.

Stats source = the per-region single-band GLMM from run_h1abc_full_retrieval.R
(which produced the FDR-corrected BLAMTL PDF). No region survives FDR within
the BLAMTL slow gamma family (best q_FDR = 0.096 for EC and PRC), so no *
annotations.
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
FIG_DIR = REPORT_DIR / "figures" / "BLAMTL_power_GLMM_slow_gamma_interactions"
FIG_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV = REPO_ROOT / "OUTPUTS" / "csvs" / "combined_retrieval_power_all_mlmr_input.csv"

SG_LO, SG_HI = 30.27, 54.69
HPC_SUB = ("HPC", "CA", "DG")

COND_COLOR = {"nostim": "#1f77b4", "stim": "#d62728"}
COND_LABEL = {"nostim": "No stim", "stim": "Stim"}

# Order user requested: ALLHPC, BLA, EC, PRC. p_raw and q_FDR pulled from
# fdr_power_BLAMTL_GLMM_retrieval/power_BLAMTL_GLMM_slow_gamma_band_c_x_StimCondstim.csv
PANELS = [
    ("ALLHPC", 0.918, 0.918, False),
    ("BLA",    0.647, 0.862, False),
    ("EC",     0.048, 0.096, False),
    ("PRC",    0.044, 0.096, False),
]


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
    sg_cols = [c for c, f in zip(freq_cols, freqs) if SG_LO <= f <= SG_HI]
    df["slow_gamma"] = df[sg_cols].mean(axis=1)
    return df


def draw_panel(ax, region, p_raw, q_fdr, mark_star):
    df = load_region_data(region)
    pred = pd.read_csv(GLMM_DIR / f"power_{region}_slow_gamma_predictions.csv")

    n_trials = len(df)
    n_subjects = int(df["Patient"].nunique())

    for cond in ["nostim", "stim"]:
        sub = pred[pred["stim"] == cond]
        ax.fill_between(sub["band"], sub["p_lo"], sub["p_hi"],
                        color=COND_COLOR[cond], alpha=0.18)
        ax.plot(sub["band"], sub["p_hat"],
                color=COND_COLOR[cond], lw=2.2, label=COND_LABEL[cond])

    for cond in ["nostim", "stim"]:
        sub = df[df["StimCond"] == cond]
        if sub.empty:
            continue
        bins = pd.qcut(sub["slow_gamma"], 10, duplicates="drop")
        agg = sub.groupby(bins, observed=True).agg(
            x=("slow_gamma", "mean"),
            acc=("Accuracy", "mean"),
            n=("Accuracy", "size"),
        )
        agg["se"] = np.sqrt(agg["acc"] * (1 - agg["acc"]) / agg["n"])
        ax.errorbar(agg["x"], agg["acc"], yerr=agg["se"],
                    fmt="o", color=COND_COLOR[cond], alpha=0.8,
                    ms=6, capsize=3, lw=1.0)

    ax.axhline(0.5, ls=":", color="0.5", lw=0.8)
    ax.axvline(0.0, ls=":", color="0.5", lw=0.8)
    ax.set_xlabel(f"{region} slow gamma power", fontsize=11, fontweight="bold")
    ax.set_ylabel("P(remembered)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{region} slow gamma power x Prior Stimulation\n"
        f"(p = {p_raw:.3f}, q_FDR = {q_fdr:.3f}, "
        f"n trials = {n_trials}, N subjects = {n_subjects})",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=12)
    ax.grid(alpha=0.25)

    if mark_star:
        ax.text(0.0, 0.97, "*", fontsize=28, fontweight="bold",
                ha="center", va="center", color="black")


def main():
    plt.rcParams.update({"font.family": "serif"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for ax, (region, p_raw, q_fdr, star) in zip(axes, PANELS):
        draw_panel(ax, region, p_raw, q_fdr, star)

    fig.text(
        0.5, 0.015,
        "Lines: GLMM-predicted P(remembered); shading: model-based 95% CI. "
        "Points: data binned into 10 quantiles (deciles) within stim condition; "
        "error bars show binomial SE of the bin mean. No region survives FDR "
        "within the BLAMTL slow gamma family.",
        ha="center", va="bottom", fontsize=9, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = FIG_DIR / "BLAMTL_slow_gamma_interactions_2x2.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
