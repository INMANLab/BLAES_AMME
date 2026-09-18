#!/usr/bin/env python
"""Main-effect GLMM panel figures for the h1abc main-effect pipeline.

For each (phase, measure, scope, contrast, band): one grid of panels, one
panel per region/pair. Each panel shows GLMM-predicted P(remembered) as a
function of the band value (single curve, no stim split), with binned data
points overlaid. Panels with q_FDR < .05 (focal term = band_c, FDR-BH
within band x scope x measure x contrast) get a star marker.

Reads predictions saved by run_h1abc_maineffect.R and FDR family CSVs
written by build_h1abc_maineffect_report.py.

Usage:
  python plot_h1abc_maineffect_glmm.py              # render everything
  python plot_h1abc_maineffect_glmm.py encoding     # one phase
  python plot_h1abc_maineffect_glmm.py encoding coherence HippSubRhinal all
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_DIR = REPO_ROOT / "OUTPUTS" / "csvs"

THETA_LO, THETA_HI = 4.88, 7.81
SG_LO, SG_HI = 30.27, 54.69
PAC_SG_LO, PAC_SG_HI = 30.0, 50.0

HPC_SUB = ("HPC", "CA", "DG")
ALLHPC_PAIR_SOURCES = {
    "BLA_ALLHPC": ("BLA_HPC", "BLA_CA", "BLA_DG"),
    "ALLHPC_EC":  ("EC_HPC",  "CA_EC",  "DG_EC"),
    "ALLHPC_PRC": ("HPC_PRC", "CA_PRC", "DG_PRC"),
}

PHASES = ("encoding", "retrieval")
MEASURES = ("power", "coherence", "pac")
CONTRASTS = ("all", "nostim", "stim")
SCOPES = ("BLAMTL", "HPCrhinal", "HippSubBLA", "HippSubRhinal")

CONTRAST_COLOR = {
    "all":    "#4c4cb3",   # blue-violet
    "nostim": "#1f77b4",   # blue
    "stim":   "#d62728",   # red
}
CONTRAST_LABEL = {
    "all":    "All trials",
    "nostim": "Endogenous (no-stim)",
    "stim":   "Stim only",
}

SCOPE_UNITS = {
    ("power", "BLAMTL"):        ["BLA", "ALLHPC", "EC", "PRC"],
    ("power", "HPCrhinal"):     ["ALLHPC", "EC", "PRC"],
    ("power", "HippSubBLA"):    ["BLA", "CA", "DG", "HPC"],
    ("power", "HippSubRhinal"): ["CA", "DG", "HPC"],
    ("coherence", "BLAMTL"):        ["BLA_ALLHPC", "BLA_EC", "BLA_PRC"],
    ("coherence", "HPCrhinal"):     ["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"],
    ("coherence", "HippSubBLA"):    ["BLA_CA", "BLA_DG", "BLA_HPC"],
    ("coherence", "HippSubRhinal"): ["CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"],
    ("pac", "BLAMTL"):        ["BLA_ALLHPC", "BLA_EC", "BLA_PRC"],
    ("pac", "HPCrhinal"):     ["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"],
    ("pac", "HippSubBLA"):    ["BLA_CA", "BLA_DG", "BLA_HPC"],
    ("pac", "HippSubRhinal"): ["CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"],
}

# (nrows, ncols, figsize_wh) per (measure, scope).
GRID = {
    ("power", "BLAMTL"):        (1, 4, (22, 6.2)),
    ("power", "HPCrhinal"):     (1, 3, (18, 6.2)),
    ("power", "HippSubBLA"):    (1, 4, (22, 6.2)),
    ("power", "HippSubRhinal"): (1, 3, (18, 6.2)),
    ("coherence", "BLAMTL"):        (1, 3, (18, 6.2)),
    ("coherence", "HPCrhinal"):     (1, 3, (18, 6.2)),
    ("coherence", "HippSubBLA"):    (1, 3, (18, 6.2)),
    ("coherence", "HippSubRhinal"): (2, 3, (18, 11)),
    ("pac", "BLAMTL"):        (1, 3, (18, 6.2)),
    ("pac", "HPCrhinal"):     (1, 3, (18, 6.2)),
    ("pac", "HippSubBLA"):    (1, 3, (18, 6.2)),
    ("pac", "HippSubRhinal"): (2, 3, (18, 11)),
}

MEASURE_BANDS = {
    "power":     [("theta", "theta"), ("slow_gamma", "slow gamma")],
    "coherence": [("theta", "theta"), ("slow_gamma", "slow gamma")],
    "pac":       [("slow_gamma", "slow gamma")],
}

CAPTION_BASE = (
    "Line: GLMM-predicted P(remembered); shading: model-based 95% CI. "
    "Points: data binned into 10 quantiles (deciles); error bars show "
    "binomial SE of the bin mean."
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def pretty_label(name):
    parts = []
    for p in str(name).split("_"):
        if p == "ALLHPC":
            parts.append("HPC")
        elif p == "HPC":
            parts.append("SUB")
        else:
            parts.append(p)
    return "-".join(parts)


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


def input_csv(phase, measure):
    return CSV_DIR / f"combined_{phase}_{measure}_all_mlmr_input.csv"


def load_unit_data(phase, measure, unit, band_name, contrast):
    df = pd.read_csv(input_csv(phase, measure))
    df = df[(df["yes_or_no"].isin(["yes", "no"])) &
            (df["trial_type"].isin(["nostim", "stim"]))]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"

    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    if measure == "power" and unit == "ALLHPC":
        df = _add_allhpc_power_rows(df, freq_cols)
    elif measure in ("coherence", "pac") and unit in ALLHPC_PAIR_SOURCES:
        df = _add_allhpc_pair_rows(df, freq_cols, unit)

    df = df[df["Region"] == unit].copy()
    df["Accuracy"] = (df["yes_or_no"] == "yes").astype(int)
    df["StimCond"] = np.where(df["trial_type"] == "nostim", "nostim", "stim")
    if contrast == "nostim":
        df = df[df["StimCond"] == "nostim"]
    elif contrast == "stim":
        df = df[df["StimCond"] == "stim"]

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


def phase_folder(phase):
    return "retrieval_memory_reports" if phase == "retrieval" else "encoding_memory_reports"


def stats_dir(phase, measure, scope, contrast):
    return (REPO_ROOT / "OUTPUTS" / phase_folder(phase) / "stats" /
            f"h1abc_maineffect_{phase}_mlm" /
            f"{measure}_{scope}_{contrast}")


def fdr_family_path(phase, measure, scope, contrast, band_name):
    base = (REPO_ROOT / "OUTPUTS" / phase_folder(phase) / "stats" /
            f"fdr_{measure}_{scope}_GLMM_MainEffect_{contrast}_{phase}")
    return base / f"{measure}_{scope}_GLMM_MainEffect_{contrast}_{band_name}_band_c.csv"


def fig_dir(phase, scope, measure, contrast):
    d = (REPO_ROOT / "OUTPUTS" / phase_folder(phase) / "figures" /
         f"{scope}_{measure}_GLMM_MainEffect_{contrast}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_p_q(phase, measure, scope, contrast, unit, band_name):
    coefs_p = stats_dir(phase, measure, scope, contrast) / \
        f"{measure}_{unit}_{band_name}_coefs.csv"
    p_raw = np.nan
    if coefs_p.exists():
        coefs = pd.read_csv(coefs_p)
        row = coefs[coefs["term"] == "band_c"]
        if not row.empty:
            p_raw = float(row["p.value"].iloc[0])
    q_fdr = np.nan
    fam_p = fdr_family_path(phase, measure, scope, contrast, band_name)
    if fam_p.exists():
        fam = pd.read_csv(fam_p)
        frow = fam[fam["unit"] == unit]
        if not frow.empty:
            q_fdr = float(frow["q_FDR"].iloc[0])
    return p_raw, q_fdr


# ── Panel drawing ───────────────────────────────────────────────────────────

def draw_panel(ax, phase, measure, scope, contrast, unit, band_name, band_label):
    df = load_unit_data(phase, measure, unit, band_name, contrast)
    pred_path = (stats_dir(phase, measure, scope, contrast) /
                 f"{measure}_{unit}_{band_name}_predictions.csv")
    if not pred_path.exists() or df.empty:
        ax.text(0.5, 0.5, f"{pretty_label(unit)}\n(no fit)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="0.4")
        ax.set_axis_off()
        return
    pred = pd.read_csv(pred_path)

    p_raw, q_fdr = load_p_q(phase, measure, scope, contrast, unit, band_name)
    n_trials = len(df)
    n_subjects = int(df["Patient"].nunique())
    unit_pretty = pretty_label(unit)
    color = CONTRAST_COLOR[contrast]

    ax.fill_between(pred["band"], pred["p_lo"], pred["p_hi"],
                    color=color, alpha=0.18)
    ax.plot(pred["band"], pred["p_hat"],
            color=color, lw=2.2, label=CONTRAST_LABEL[contrast])

    if not df.empty and df[band_name].notna().any():
        try:
            bins = pd.qcut(df[band_name], 10, duplicates="drop")
            agg = df.groupby(bins, observed=True).agg(
                x=(band_name, "mean"),
                acc=("Accuracy", "mean"),
                n=("Accuracy", "size"),
            )
            agg["se"] = np.sqrt(agg["acc"] * (1 - agg["acc"]) / agg["n"])
            ax.errorbar(agg["x"], agg["acc"], yerr=agg["se"],
                        fmt="o", color=color, alpha=0.8,
                        ms=6, capsize=3, lw=1.0)
        except ValueError:
            pass

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
        f"{unit_pretty} {title_meas}\n"
        f"({p_txt}, {q_txt}, n trials = {n_trials}, "
        f"N subjects = {n_subjects})",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=11)
    ax.grid(alpha=0.25)

    if mark_star:
        ax.text(0.0, 0.97, "*", fontsize=28, fontweight="bold",
                ha="center", va="center", color="black")


# ── Grid + figure ───────────────────────────────────────────────────────────

def build_family_caption(phase, measure, scope, contrast, band_name):
    fam_path = fdr_family_path(phase, measure, scope, contrast, band_name)
    if not fam_path.exists():
        return ""
    fam = pd.read_csv(fam_path).dropna(subset=["q_FDR"])
    base = (f"{phase.capitalize()} {scope} {measure} "
            f"{band_name.replace('_', ' ')} | {CONTRAST_LABEL[contrast]} family.")
    sig = fam[fam["q_FDR"] < 0.05]
    marg = fam[(fam["q_FDR"] >= 0.05) & (fam["q_FDR"] < 0.10)]
    if not sig.empty:
        names = ", ".join(sig["unit"].astype(str).map(pretty_label))
        return f"{base} FDR-significant pairs (q < .05): {names} (* marks panel)."
    if not marg.empty:
        names = ", ".join(
            f"{pretty_label(r['unit'])} (q = {r['q_FDR']:.3f})"
            for _, r in marg.iterrows()
        )
        return f"{base} No FDR-significant survivors; marginals: {names}."
    return f"{base} No pair survives FDR within this family."


def render_grid(phase, measure, scope, contrast, band_name, band_label):
    units = SCOPE_UNITS[(measure, scope)]
    nrows, ncols, figsize = GRID[(measure, scope)]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, unit in enumerate(units):
        draw_panel(axes[i], phase, measure, scope, contrast,
                   unit, band_name, band_label)
    for j in range(len(units), len(axes)):
        axes[j].axis("off")

    family_caption = build_family_caption(phase, measure, scope, contrast,
                                          band_name)
    fig.suptitle(
        f"{phase.capitalize()} | {measure.capitalize()} | {scope} | "
        f"{CONTRAST_LABEL[contrast]} | {band_label}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.015,
        f"{CAPTION_BASE} {family_caption}",
        ha="center", va="bottom", fontsize=9, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    out_dir = fig_dir(phase, scope, measure, contrast)
    suffix = "slow_gamma" if band_name == "slow_gamma" else band_name
    if measure == "pac":
        out_name = (f"{scope}_pac_MainEffect_{contrast}_"
                    f"slow_gamma_maineffect.png")
    else:
        out_name = (f"{scope}_{measure}_MainEffect_{contrast}_"
                    f"{suffix}_maineffect.png")
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args(argv):
    phases = list(PHASES)
    measures = list(MEASURES)
    scopes = list(SCOPES)
    contrasts = list(CONTRASTS)

    args = list(argv)
    if args and args[0] in PHASES:
        phases = [args.pop(0)]
    if len(args) >= 3:
        measures = [args[0]]
        scopes = [args[1]]
        contrasts = [args[2]]
    return phases, measures, scopes, contrasts


def main():
    plt.rcParams.update({"font.family": "serif"})
    phases, measures, scopes, contrasts = parse_args(sys.argv[1:])
    for phase in phases:
        for measure in measures:
            for scope in scopes:
                for contrast in contrasts:
                    for band_name, band_label in MEASURE_BANDS[measure]:
                        render_grid(phase, measure, scope, contrast,
                                    band_name, band_label)


if __name__ == "__main__":
    main()
