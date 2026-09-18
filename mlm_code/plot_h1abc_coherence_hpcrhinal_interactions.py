#!/usr/bin/env python
"""HPCrhinal + HippSubRhinal coherence x prior BLA stimulation interaction
figures. Same v4 style: GLMM line + model-based 95% CI shading + 10-bin
overlay. Stats source = the per-pair single-band GLMM from
run_h1abc_full_retrieval.R.

Layouts (per user):
  HPCrhinal theta       : 2x1 (ALLHPC-EC, ALLHPC-PRC)
  HPCrhinal slow gamma  : 2x1 (ALLHPC-EC +, ALLHPC-PRC *)
  HippSubRhinal theta   : 2x3 grid showing all 5 valid pairs (one blank cell)
  HippSubRhinal slow_g  : 1x2 showing only the 2 FDR survivors (EC-HPC,
                          CA-PRC), each starred
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
INPUT_CSV = REPO_ROOT / "OUTPUTS" / "csvs" / "combined_retrieval_coherence_all_mlmr_input.csv"

THETA_LO, THETA_HI = 4.88, 7.81
SG_LO, SG_HI = 30.27, 54.69

ALLHPC_PAIR_SOURCES = {
    "ALLHPC_EC":  ("EC_HPC",  "CA_EC",  "DG_EC"),
    "ALLHPC_PRC": ("HPC_PRC", "CA_PRC", "DG_PRC"),
}

COND_COLOR = {"nostim": "#1f77b4", "stim": "#d62728"}
COND_LABEL = {"nostim": "No stim", "stim": "Stim"}


def pretty_label(name):
    # ALLHPC = macro hippocampus -> "HPC"; HPC region = subiculum -> "SUB"
    parts = []
    for p in str(name).split("_"):
        if p == "ALLHPC":
            parts.append("HPC")
        elif p == "HPC":
            parts.append("SUB")
        else:
            parts.append(p)
    return "-".join(parts)


# Panel definitions: (pair, p_raw, q_FDR, mark_star)
# All q-values pulled from the latest FDR family CSVs:
# fdr_coherence_HPCrhinal_GLMM_retrieval/coherence_HPCrhinal_GLMM_*.csv
# fdr_coherence_HippSubRhinal_GLMM_retrieval/coherence_HippSubRhinal_GLMM_*.csv
# (HPCrhinal family now includes EC_PRC; HippSubRhinal excludes EC_PRC because
# it's owned by the HPCrhinal scope.)
HPCRHINAL_THETA = [
    ("ALLHPC_EC",  0.327, 0.523, False),
    ("ALLHPC_PRC", 0.435, 0.523, False),
    ("EC_PRC",     0.523, 0.523, False),
]
HPCRHINAL_SLOW_GAMMA = [
    ("ALLHPC_EC",  0.064, 0.096, False),
    ("ALLHPC_PRC", 0.017, 0.052, False),
    ("EC_PRC",     0.595, 0.595, False),
]
HIPPSUBRHINAL_THETA = [
    ("CA_EC",   0.229, 0.573, False),
    ("EC_HPC",  0.501, 0.793, False),
    ("CA_PRC",  0.634, 0.793, False),
    ("DG_PRC",  0.014, 0.068, False),
    ("HPC_PRC", 0.935, 0.935, False),
]
HIPPSUBRHINAL_SLOW_GAMMA_ALL = [
    ("CA_EC",   0.650, 0.650, False),
    ("EC_HPC",  0.008, 0.020, True),
    ("CA_PRC",  0.003, 0.013, True),
    ("DG_PRC",  0.208, 0.260, False),
    ("HPC_PRC", 0.148, 0.247, False),
]
HIPPSUBRHINAL_SLOW_GAMMA_SIG = [
    ("EC_HPC", 0.008, 0.020, True),
    ("CA_PRC", 0.003, 0.013, True),
]


def _maybe_build_allhpc(df, pair, freq_cols):
    if pair not in ALLHPC_PAIR_SOURCES:
        return df
    sources = ALLHPC_PAIR_SOURCES[pair]
    sub = df[df["Region"].isin(sources)].copy()
    if sub.empty:
        return df
    sub["trial_id"] = sub.groupby(["Patient", "Region"]).cumcount()
    grp_keys = ["Patient", "trial_id", "trial_type", "yes_or_no"]
    agg = (sub.groupby(grp_keys, as_index=False)[list(freq_cols)]
              .mean(numeric_only=True))
    agg["Region"] = pair
    agg = agg.drop(columns=["trial_id"])
    return pd.concat([df, agg], ignore_index=True)


def load_pair_data(pair, band_name):
    df = pd.read_csv(INPUT_CSV)
    df = df[(df["yes_or_no"].isin(["yes", "no"])) & (df["trial_type"] != "new")]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"

    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    df = _maybe_build_allhpc(df, pair, freq_cols)
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


def draw_panel(ax, glmm_dir, pair, band_name, band_label,
               p_raw, q_fdr, mark_star):
    df = load_pair_data(pair, band_name)
    pred = pd.read_csv(glmm_dir / f"coherence_{pair}_{band_name}_predictions.csv")

    n_trials = len(df)
    n_subjects = int(df["Patient"].nunique())
    pair_pretty = pretty_label(pair)

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


CAPTION_BASE = (
    "Lines: GLMM-predicted P(remembered); shading: model-based 95% CI. "
    "Points: data binned into 10 quantiles (deciles) within stim condition; "
    "error bars show binomial SE of the bin mean."
)


def render_grid(panels, glmm_dir, band_name, band_label,
                fig_dir, out_name, nrows, ncols, figsize, family_caption):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    for i, (pair, p_raw, q_fdr, star) in enumerate(panels):
        draw_panel(axes[i], glmm_dir, pair, band_name, band_label,
                   p_raw, q_fdr, star)
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")
    fig.text(
        0.5, 0.015,
        f"{CAPTION_BASE} {family_caption}",
        ha="center", va="bottom", fontsize=9, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = fig_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    plt.rcParams.update({"font.family": "serif"})

    stats_base = REPORT_DIR / "stats" / "h1abc_full_retrieval_mlm"
    hpcrhinal_glmm = stats_base / "coherence_HPCrhinal_GLMM"
    hipprhinal_glmm = stats_base / "coherence_HippSubRhinal_GLMM"

    fig_root = REPORT_DIR / "figures"
    hpcrhinal_dir = fig_root / "HPCrhinal_coherence_GLMM_interactions"
    hipprhinal_dir = fig_root / "HippSubRhinal_coherence_GLMM_interactions"
    hpcrhinal_dir.mkdir(parents=True, exist_ok=True)
    hipprhinal_dir.mkdir(parents=True, exist_ok=True)

    # HPCrhinal: 3x1 each band (ALLHPC-EC, ALLHPC-PRC, EC-PRC)
    render_grid(
        HPCRHINAL_THETA, hpcrhinal_glmm, "theta", "theta",
        hpcrhinal_dir, "HPCrhinal_coherence_theta_interactions_3x1.png",
        nrows=1, ncols=3, figsize=(18, 6.2),
        family_caption=("No pair survives FDR within the HPCrhinal coherence "
                        "theta family."),
    )
    render_grid(
        HPCRHINAL_SLOW_GAMMA, hpcrhinal_glmm, "slow_gamma", "slow gamma",
        hpcrhinal_dir, "HPCrhinal_coherence_slow_gamma_interactions_3x1.png",
        nrows=1, ncols=3, figsize=(18, 6.2),
        family_caption=("No pair survives FDR within the HPCrhinal coherence "
                        "slow gamma family; HPC-PRC is the closest at "
                        "q = 0.052 (+)."),
    )

    # HippSubRhinal theta: 2x3 grid (one blank cell). DG_EC was dropped from
    # the FDR family (only 2 patients).
    render_grid(
        HIPPSUBRHINAL_THETA, hipprhinal_glmm, "theta", "theta",
        hipprhinal_dir, "HippSubRhinal_coherence_theta_interactions_2x3.png",
        nrows=2, ncols=3, figsize=(18, 11),
        family_caption=("No pair survives FDR within the HippSubRhinal "
                        "coherence theta family."),
    )

    # HippSubRhinal slow gamma main figure: only the 2 FDR survivors, 1x2.
    render_grid(
        HIPPSUBRHINAL_SLOW_GAMMA_SIG, hipprhinal_glmm,
        "slow_gamma", "slow gamma",
        hipprhinal_dir,
        "HippSubRhinal_coherence_slow_gamma_FDRsig_1x2.png",
        nrows=1, ncols=2, figsize=(13, 6.2),
        family_caption=("Only the FDR-significant pairs from the HippSubRhinal "
                        "slow gamma family are shown; * marks each."),
    )

    # HippSubRhinal slow gamma supplement: all 5 valid pairs, 2x3 grid.
    render_grid(
        HIPPSUBRHINAL_SLOW_GAMMA_ALL, hipprhinal_glmm,
        "slow_gamma", "slow gamma",
        hipprhinal_dir,
        "HippSubRhinal_coherence_slow_gamma_interactions_2x3.png",
        nrows=2, ncols=3, figsize=(18, 11),
        family_caption=("EC-SUB and CA-PRC survive FDR within the HippSubRhinal "
                        "slow gamma family; * marks the FDR-significant panels."),
    )


if __name__ == "__main__":
    main()
