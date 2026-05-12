#!/usr/bin/env python
"""GLMM band x prior-stimulation interaction figures for the encoding-phase
H1abc HPCrhinal and HippSubRhinal scopes (power, coherence, PAC).

Mirrors the retrieval figure scripts (plot_h1abc_slow_gamma_interactions_2x2.py,
plot_h1abc_coherence_hpcrhinal_interactions.py, plot_h1abc_pac_all.py) but:
  - reads encoding combined CSVs
  - reads encoding GLMM coef CSVs (h1abc_full_encoding_mlm / *_GLMM / *_coefs.csv)
    to pull p-values directly, plus the fdr_*_encoding family CSVs for q-values
  - writes figures to OUTPUTS/encoding_memory_reports/figures/
  - includes only HPCrhinal and HippSubRhinal scopes
  - EC_PRC appears only in HPCrhinal (not double-counted in HippSubRhinal)
  - HippSubRhinal contains no ALLHPC; no BLA appears in either scope
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REPORT_DIR = REPO_ROOT / "OUTPUTS" / "encoding_memory_reports"
STATS_BASE = REPORT_DIR / "stats" / "h1abc_full_encoding_mlm"
FDR_BASE = REPORT_DIR / "stats"
FIG_ROOT = REPORT_DIR / "figures"

CSV_DIR = REPO_ROOT / "OUTPUTS" / "csvs"
INPUT_CSVS = {
    "power":     CSV_DIR / "combined_encoding_power_all_mlmr_input.csv",
    "coherence": CSV_DIR / "combined_encoding_coherence_all_mlmr_input.csv",
    "pac":       CSV_DIR / "combined_encoding_pac_all_mlmr_input.csv",
}

THETA_LO, THETA_HI = 4.88, 7.81
SG_LO, SG_HI = 30.27, 54.69
PAC_SG_LO, PAC_SG_HI = 30.0, 50.0

HPC_SUB = ("HPC", "CA", "DG")
ALLHPC_PAIR_SOURCES = {
    "ALLHPC_EC":  ("EC_HPC",  "CA_EC",  "DG_EC"),
    "ALLHPC_PRC": ("HPC_PRC", "CA_PRC", "DG_PRC"),
}

COND_COLOR = {"nostim": "#1f77b4", "stim": "#d62728"}
COND_LABEL = {"nostim": "No stim", "stim": "Stim"}

# Scope/measure panel definitions: per-band lists of units to render.
# Power: regions. Coherence/PAC: pairs. HippSubRhinal omits DG_EC (skipped at
# fit time, <4 patients) and EC_PRC (owned by HPCrhinal).
SCOPE_UNITS = {
    ("power", "HPCrhinal"):       ["ALLHPC", "EC", "PRC"],
    ("power", "HippSubRhinal"):   ["CA", "DG", "HPC", "EC", "PRC"],
    ("coherence", "HPCrhinal"):     ["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"],
    ("coherence", "HippSubRhinal"): ["CA_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"],
    ("pac", "HPCrhinal"):           ["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"],
    ("pac", "HippSubRhinal"):       ["CA_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"],
}

# Grid layout (nrows, ncols, figsize_wh) per scope, used for every band.
GRID = {
    ("power", "HPCrhinal"):       (1, 3, (18, 6.2)),
    ("power", "HippSubRhinal"):   (2, 3, (18, 11)),
    ("coherence", "HPCrhinal"):     (1, 3, (18, 6.2)),
    ("coherence", "HippSubRhinal"): (2, 3, (18, 11)),
    ("pac", "HPCrhinal"):           (1, 3, (18, 6.2)),
    ("pac", "HippSubRhinal"):       (2, 3, (18, 11)),
}

MEASURE_BANDS = {
    "power":     [("theta", "theta"), ("slow_gamma", "slow gamma")],
    "coherence": [("theta", "theta"), ("slow_gamma", "slow gamma")],
    "pac":       [("slow_gamma", "slow gamma")],
}

CAPTION_BASE = (
    "Lines: GLMM-predicted P(remembered); shading: model-based 95% CI. "
    "Points: data binned into 10 quantiles (deciles) within stim condition; "
    "error bars show binomial SE of the bin mean."
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _add_allhpc_power_rows(df, freq_cols):
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


def _add_allhpc_pair_rows(df, freq_cols, target_label):
    sources = ALLHPC_PAIR_SOURCES[target_label]
    sub = df[df["Region"].isin(sources)].copy()
    if sub.empty:
        return df
    sub["trial_id"] = sub.groupby(["Patient", "Region"]).cumcount()
    grp_keys = ["Patient", "trial_id", "trial_type", "yes_or_no"]
    agg = (sub.groupby(grp_keys, as_index=False)[list(freq_cols)]
              .mean(numeric_only=True))
    agg["Region"] = target_label
    agg = agg.drop(columns=["trial_id"])
    return pd.concat([df, agg], ignore_index=True)


def load_unit_data(measure, unit, band_name):
    df = pd.read_csv(INPUT_CSVS[measure])
    df = df[(df["yes_or_no"].isin(["yes", "no"])) & (df["trial_type"] != "new")]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"

    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    if measure == "power" and unit == "ALLHPC":
        df = _add_allhpc_power_rows(df, freq_cols)
    elif measure in ("coherence", "pac") and unit in ALLHPC_PAIR_SOURCES:
        df = _add_allhpc_pair_rows(df, freq_cols, unit)

    df = df[df["Region"] == unit].copy()
    df["Accuracy"] = (df["yes_or_no"] == "yes").astype(int)
    df["StimCond"] = np.where(df["trial_type"] == "nostim", "nostim", "stim")

    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in freq_cols])
    if measure == "pac":
        cols = [c for c, f in zip(freq_cols, freqs)
                if PAC_SG_LO <= f <= PAC_SG_HI]
    elif band_name == "theta":
        cols = [c for c, f in zip(freq_cols, freqs)
                if THETA_LO <= f <= THETA_HI]
    else:
        cols = [c for c, f in zip(freq_cols, freqs)
                if SG_LO <= f <= SG_HI]
    df[band_name] = df[cols].mean(axis=1)
    return df


def load_p_q(measure, scope, unit, band_name):
    """Pull raw p from the per-panel coefs CSV and q_FDR from the family CSV."""
    coefs_path = (STATS_BASE / f"{measure}_{scope}_GLMM" /
                  f"{measure}_{unit}_{band_name}_coefs.csv")
    if not coefs_path.exists():
        return np.nan, np.nan
    coefs = pd.read_csv(coefs_path)
    row = coefs[coefs["term"] == "band_c:StimCondstim"]
    p_raw = float(row["p.value"].iloc[0]) if not row.empty else np.nan

    fam_path = (FDR_BASE / f"fdr_{measure}_{scope}_GLMM_encoding" /
                f"{measure}_{scope}_GLMM_{band_name}_band_c_x_StimCondstim.csv")
    q_fdr = np.nan
    if fam_path.exists():
        fam = pd.read_csv(fam_path)
        frow = fam[fam["unit"] == unit]
        if not frow.empty:
            q_fdr = float(frow["q_FDR"].iloc[0])
    return p_raw, q_fdr


# ── Panel drawing ───────────────────────────────────────────────────────────

def draw_panel(ax, measure, scope, unit, band_name, band_label):
    df = load_unit_data(measure, unit, band_name)
    pred_path = (STATS_BASE / f"{measure}_{scope}_GLMM" /
                 f"{measure}_{unit}_{band_name}_predictions.csv")
    if not pred_path.exists():
        ax.text(0.5, 0.5, f"{unit}\n(no fit)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="0.4")
        ax.set_axis_off()
        return
    pred = pd.read_csv(pred_path)

    p_raw, q_fdr = load_p_q(measure, scope, unit, band_name)
    n_trials = len(df)
    n_subjects = int(df["Patient"].nunique())
    unit_pretty = unit.replace("_", "-")

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
    if measure == "power":
        xlab = f"{unit_pretty} {band_label} power"
    elif measure == "coherence":
        xlab = f"{unit_pretty} {band_label} coherence"
    else:
        xlab = f"{unit_pretty} slow gamma PAC"
    ax.set_xlabel(xlab, fontsize=11, fontweight="bold")
    ax.set_ylabel("P(remembered)", fontsize=11, fontweight="bold")

    p_txt = f"p = {p_raw:.3f}" if not np.isnan(p_raw) else "p = NA"
    q_txt = f"q_FDR = {q_fdr:.3f}" if not np.isnan(q_fdr) else "q_FDR = NA"
    mark_star = (not np.isnan(q_fdr)) and (q_fdr < 0.05)
    title_meas = "PAC" if measure == "pac" else f"{band_label} {measure}"
    ax.set_title(
        f"{unit_pretty} {title_meas} x Prior Stimulation\n"
        f"({p_txt}, {q_txt}, n trials = {n_trials}, N subjects = {n_subjects})",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=12)
    ax.grid(alpha=0.25)

    if mark_star:
        ax.text(0.0, 0.97, "*", fontsize=28, fontweight="bold",
                ha="center", va="center", color="black")


# ── Grid + figure ───────────────────────────────────────────────────────────

def render_grid(measure, scope, band_name, band_label):
    units = SCOPE_UNITS[(measure, scope)]
    nrows, ncols, figsize = GRID[(measure, scope)]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, unit in enumerate(units):
        draw_panel(axes[i], measure, scope, unit, band_name, band_label)
    for j in range(len(units), len(axes)):
        axes[j].axis("off")

    # Family-level caption: report any FDR survivors at q<.05 (or marginal at
    # q<.10 if none significant).
    family_caption = build_family_caption(measure, scope, band_name)
    fig.text(
        0.5, 0.015,
        f"{CAPTION_BASE} {family_caption}",
        ha="center", va="bottom", fontsize=9, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig_dir = FIG_ROOT / f"{scope}_{measure}_GLMM_interactions"
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = "slow_gamma" if band_name == "slow_gamma" else band_name
    if measure == "pac":
        out_name = f"{scope}_pac_slow_gamma_interactions.png"
    else:
        out_name = f"{scope}_{measure}_{suffix}_interactions.png"
    out_path = fig_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def build_family_caption(measure, scope, band_name):
    fam_path = (FDR_BASE / f"fdr_{measure}_{scope}_GLMM_encoding" /
                f"{measure}_{scope}_GLMM_{band_name}_band_c_x_StimCondstim.csv")
    if not fam_path.exists():
        return ""
    fam = pd.read_csv(fam_path).dropna(subset=["q_FDR"])
    sig = fam[fam["q_FDR"] < 0.05]
    marg = fam[(fam["q_FDR"] >= 0.05) & (fam["q_FDR"] < 0.10)]
    base = f"Encoding {scope} {measure} {band_name.replace('_', ' ')} family."
    if not sig.empty:
        names = ", ".join(sig["unit"].astype(str).str.replace("_", "-"))
        return f"{base} FDR-significant pairs (q < .05): {names} (* marks panel)."
    if not marg.empty:
        names = ", ".join(
            f"{r['unit'].replace('_','-')} (q = {r['q_FDR']:.3f})"
            for _, r in marg.iterrows()
        )
        return f"{base} No FDR-significant survivors; marginals: {names}."
    return f"{base} No pair survives FDR within this family."


def main():
    plt.rcParams.update({"font.family": "serif"})
    for measure in ["power", "coherence", "pac"]:
        for scope in ["HPCrhinal", "HippSubRhinal"]:
            for band_name, band_label in MEASURE_BANDS[measure]:
                render_grid(measure, scope, band_name, band_label)


if __name__ == "__main__":
    main()
