#!/usr/bin/env python
"""3x1 figures for BLAMTL coherence x prior BLA stimulation interactions:
one figure for theta, one for slow gamma. Pairs: BLA-ALLHPC, BLA-EC, BLA-PRC.
Same v4 style as the power versions: GLMM line + model-based 95% CI shading
+ 10-bin overlay.

Stats source = the per-pair single-band GLMM from run_h1abc_full_retrieval.R
(which produced the FDR-corrected BLAMTL coherence PDF). No pair survives
FDR within either band family.
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
GLMM_DIR = REPORT_DIR / "stats" / "h1abc_full_retrieval_mlm" / "coherence_BLAMTL_GLMM"
INPUT_CSV = REPO_ROOT / "OUTPUTS" / "csvs" / "combined_retrieval_coherence_all_mlmr_input.csv"

THETA_LO, THETA_HI = 4.88, 7.81
SG_LO, SG_HI = 30.27, 54.69
ALLHPC_SOURCES = ("BLA_HPC", "BLA_CA", "BLA_DG")

COND_COLOR = {"nostim": "#1f77b4", "stim": "#d62728"}
COND_LABEL = {"nostim": "No stim", "stim": "Stim"}

# Order requested: BLA-ALLHPC, BLA-EC, BLA-PRC. p_raw / q_FDR pulled from
# fdr_coherence_BLAMTL_GLMM_retrieval/coherence_BLAMTL_GLMM_*.csv
THETA_PANELS = [
    ("BLA_ALLHPC", 0.030, 0.089, False),
    ("BLA_EC",     0.433, 0.433, False),
    ("BLA_PRC",    0.289, 0.433, False),
]
SLOW_GAMMA_PANELS = [
    ("BLA_ALLHPC", 0.520, 0.780, False),
    ("BLA_EC",     0.828, 0.828, False),
    ("BLA_PRC",    0.209, 0.628, False),
]


def _add_allhpc_pair_rows(df, freq_cols):
    sub = df[df["Region"].isin(ALLHPC_SOURCES)].copy()
    if sub.empty:
        return df
    sub["trial_id"] = sub.groupby(["Patient", "Region"]).cumcount()
    grp_keys = ["Patient", "trial_id", "trial_type", "yes_or_no"]
    agg = (sub.groupby(grp_keys, as_index=False)[list(freq_cols)]
              .mean(numeric_only=True))
    agg["Region"] = "BLA_ALLHPC"
    agg = agg.drop(columns=["trial_id"])
    return pd.concat([df, agg], ignore_index=True)


def load_pair_data(pair, band_name):
    df = pd.read_csv(INPUT_CSV)
    df = df[(df["yes_or_no"].isin(["yes", "no"])) & (df["trial_type"] != "new")]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"

    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    if pair == "BLA_ALLHPC":
        df = _add_allhpc_pair_rows(df, freq_cols)

    df = df[df["Region"] == pair].copy()
    df["Accuracy"] = (df["yes_or_no"] == "yes").astype(int)
    df["StimCond"] = np.where(df["trial_type"] == "nostim", "nostim", "stim")

    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in freq_cols])
    if band_name == "theta":
        cols = [c for c, f in zip(freq_cols, freqs) if THETA_LO <= f <= THETA_HI]
    else:
        cols = [c for c, f in zip(freq_cols, freqs) if SG_LO <= f <= SG_HI]
    df[band_name] = df[cols].mean(axis=1)
    return df


def draw_panel(ax, pair, band_name, band_label, p_raw, q_fdr, mark_star):
    df = load_pair_data(pair, band_name)
    pred = pd.read_csv(GLMM_DIR / f"coherence_{pair}_{band_name}_predictions.csv")

    n_trials = len(df)
    n_subjects = int(df["Patient"].nunique())
    pair_pretty = pair.replace("_", "-")

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
        bins = pd.qcut(sub[band_name], 10, duplicates="drop")
        agg = sub.groupby(bins, observed=True).agg(
            x=(band_name, "mean"),
            acc=("Accuracy", "mean"),
            n=("Accuracy", "size"),
        )
        agg["se"] = np.sqrt(agg["acc"] * (1 - agg["acc"]) / agg["n"])
        ax.errorbar(agg["x"], agg["acc"], yerr=agg["se"],
                    fmt="o", color=COND_COLOR[cond], alpha=0.8,
                    ms=6, capsize=3, lw=1.0)

    ax.axhline(0.5, ls=":", color="0.5", lw=0.8)
    ax.axvline(0.0, ls=":", color="0.5", lw=0.8)
    ax.set_xlabel(f"{pair_pretty} {band_label} coherence",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("P(remembered)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{pair_pretty} {band_label} coherence x Prior Stimulation\n"
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


def render(panels, band_name, band_label, fig_dir, family_caption):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    for ax, (pair, p_raw, q_fdr, star) in zip(axes, panels):
        draw_panel(ax, pair, band_name, band_label, p_raw, q_fdr, star)
    fig.text(
        0.5, 0.015,
        "Lines: GLMM-predicted P(remembered); shading: model-based 95% CI. "
        "Points: data binned into 10 quantiles (deciles) within stim condition; "
        f"error bars show binomial SE of the bin mean. {family_caption}",
        ha="center", va="bottom", fontsize=9, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = fig_dir / f"BLAMTL_coherence_{band_name}_interactions_3x1.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    plt.rcParams.update({"font.family": "serif"})

    theta_dir = REPORT_DIR / "figures" / "BLAMTL_coherence_GLMM_theta_interactions"
    sg_dir    = REPORT_DIR / "figures" / "BLAMTL_coherence_GLMM_slow_gamma_interactions"
    theta_dir.mkdir(parents=True, exist_ok=True)
    sg_dir.mkdir(parents=True, exist_ok=True)

    render(THETA_PANELS, "theta", "theta", theta_dir,
           "No pair survives FDR within the BLAMTL coherence theta family.")
    render(SLOW_GAMMA_PANELS, "slow_gamma", "slow gamma", sg_dir,
           "No pair survives FDR within the BLAMTL coherence slow gamma family.")


if __name__ == "__main__":
    main()
