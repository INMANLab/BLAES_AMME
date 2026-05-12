#!/usr/bin/env python
"""PAC (slow gamma) x prior BLA stimulation interaction figures for all 4
GLMM PAC reports: BLAMTL, HPCrhinal, HippSubBLA, HippSubRhinal. Same v4 style
as the power/coherence figures.

Stats source = the per-pair single-band PAC GLMM from
run_h1abc_full_retrieval.R (engine behind the FDR-corrected PAC PDFs). PAC
slow gamma is 30-50 Hz. No PAC pair survives FDR in any scope.
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
INPUT_CSV = REPO_ROOT / "OUTPUTS" / "csvs" / "combined_retrieval_pac_all_mlmr_input.csv"

PAC_SG_LO, PAC_SG_HI = 30.0, 50.0

ALLHPC_PAIR_SOURCES = {
    "BLA_ALLHPC": ("BLA_HPC", "BLA_CA", "BLA_DG"),
    "ALLHPC_EC":  ("EC_HPC",  "CA_EC",  "DG_EC"),
    "ALLHPC_PRC": ("HPC_PRC", "CA_PRC", "DG_PRC"),
}

COND_COLOR = {"nostim": "#1f77b4", "stim": "#d62728"}
COND_LABEL = {"nostim": "No stim", "stim": "Stim"}


# (pair, p_raw, q_FDR, mark_star) per scope. Pulled from
# fdr_pac_<SCOPE>_GLMM_retrieval/*_band_c_x_StimCondstim.csv
SCOPES = {
    "BLAMTL": dict(
        glmm_subdir="pac_BLAMTL_GLMM",
        fig_subdir="BLAMTL_pac_GLMM_slow_gamma_interactions",
        out_name="BLAMTL_pac_slow_gamma_interactions_3x1.png",
        nrows=1, ncols=3, figsize=(18, 6.2),
        family_caption=("No pair survives FDR within the BLAMTL PAC slow "
                        "gamma family; closest is BLA-EC (q = 0.227)."),
        panels=[
            ("BLA_ALLHPC", 0.309, 0.348, False),
            ("BLA_EC",     0.076, 0.227, False),
            ("BLA_PRC",    0.348, 0.348, False),
        ],
    ),
    "HPCrhinal": dict(
        glmm_subdir="pac_HPCrhinal_GLMM",
        fig_subdir="HPCrhinal_pac_GLMM_slow_gamma_interactions",
        out_name="HPCrhinal_pac_slow_gamma_interactions_3x1.png",
        nrows=1, ncols=3, figsize=(18, 6.2),
        family_caption=("No pair survives FDR within the HPCrhinal PAC slow "
                        "gamma family."),
        panels=[
            ("ALLHPC_EC",  0.203, 0.499, False),
            ("ALLHPC_PRC", 0.332, 0.499, False),
            ("EC_PRC",     0.828, 0.828, False),
        ],
    ),
    "HippSubBLA": dict(
        glmm_subdir="pac_HippSubBLA_GLMM",
        fig_subdir="HippSubBLA_pac_GLMM_slow_gamma_interactions",
        out_name="HippSubBLA_pac_slow_gamma_interactions_3x1.png",
        nrows=1, ncols=3, figsize=(18, 6.2),
        family_caption=("No pair survives FDR within the HippSubBLA PAC slow "
                        "gamma family; closest is BLA-HPC (q = 0.352)."),
        panels=[
            ("BLA_CA",  0.850, 0.850, False),
            ("BLA_DG",  0.394, 0.591, False),
            ("BLA_HPC", 0.117, 0.352, False),
        ],
    ),
    "HippSubRhinal": dict(
        glmm_subdir="pac_HippSubRhinal_GLMM",
        fig_subdir="HippSubRhinal_pac_GLMM_slow_gamma_interactions",
        out_name="HippSubRhinal_pac_slow_gamma_interactions_2x2.png",
        nrows=2, ncols=2, figsize=(13, 11),
        family_caption=("No pair survives FDR within the HippSubRhinal PAC "
                        "slow gamma family. DG-EC and DG-PRC dropped "
                        "(< 4 patients). EC-PRC is owned by the HPCrhinal "
                        "scope."),
        panels=[
            ("CA_EC",   0.273, 0.735, False),
            ("EC_HPC",  0.817, 0.817, False),
            ("CA_PRC",  0.405, 0.735, False),
            ("HPC_PRC", 0.551, 0.735, False),
        ],
    ),
}


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


def load_pair_data(pair):
    df = pd.read_csv(INPUT_CSV)
    df = df[(df["yes_or_no"].isin(["yes", "no"])) & (df["trial_type"] != "new")]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"

    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    df = _maybe_build_allhpc(df, pair, freq_cols)
    df = df[df["Region"] == pair].copy()
    df["Accuracy"] = (df["yes_or_no"] == "yes").astype(int)
    df["StimCond"] = np.where(df["trial_type"] == "nostim", "nostim", "stim")

    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in freq_cols])
    sg_cols = [c for c, f in zip(freq_cols, freqs) if PAC_SG_LO <= f <= PAC_SG_HI]
    df["slow_gamma"] = df[sg_cols].mean(axis=1)
    return df


def draw_panel(ax, glmm_dir, pair, p_raw, q_fdr, mark_star):
    df = load_pair_data(pair)
    pred = pd.read_csv(glmm_dir / f"pac_{pair}_slow_gamma_predictions.csv")

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
    ax.set_xlabel(f"{pair_pretty} slow gamma PAC",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("P(remembered)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{pair_pretty} slow gamma PAC x Prior Stimulation\n"
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


def render_scope(scope_name, cfg, stats_base, fig_root):
    glmm_dir = stats_base / cfg["glmm_subdir"]
    fig_dir = fig_root / cfg["fig_subdir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(cfg["nrows"], cfg["ncols"],
                             figsize=cfg["figsize"])
    axes = np.atleast_1d(axes).flatten()
    for i, (pair, p_raw, q_fdr, star) in enumerate(cfg["panels"]):
        draw_panel(axes[i], glmm_dir, pair, p_raw, q_fdr, star)
    for j in range(len(cfg["panels"]), len(axes)):
        axes[j].axis("off")

    fig.text(
        0.5, 0.015,
        f"{CAPTION_BASE} {cfg['family_caption']}",
        ha="center", va="bottom", fontsize=9, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = fig_dir / cfg["out_name"]
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    plt.rcParams.update({"font.family": "serif"})
    stats_base = REPORT_DIR / "stats" / "h1abc_full_retrieval_mlm"
    fig_root   = REPORT_DIR / "figures"
    for scope_name, cfg in SCOPES.items():
        render_scope(scope_name, cfg, stats_base, fig_root)


if __name__ == "__main__":
    main()
