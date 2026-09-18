#!/usr/bin/env python
"""
Hypothesis-targeted figures for the H1abc full retrieval/encoding pipeline.

Generates compact figures matching the style of the user's reference panels:
  - LMMcont: scatter of patient-level band-averaged feature vs mem_mod_z
             with the LMM-fit regression line + B/p annotation.
  - LMMquad: per-group mean+/-SE with patient dots overlaid.

Per-figure scope: one (measure x scope x modeltype x band) combination.
Red box around panels significant after FDR-BH per band (q < .05).

Usage:
  python build_h1abc_hypothesis_figures.py                    # all retrieval
  python build_h1abc_hypothesis_figures.py encoding           # all encoding
  python build_h1abc_hypothesis_figures.py [retrieval|encoding] BLAMTL power LMMcont theta
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


REPO_ROOT = "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_IN = os.path.join(REPO_ROOT, "OUTPUTS", "csvs")

# PHASE parsed from argv[0] in main(); defaults to retrieval. The path
# constants below are bound at module load to retrieval and rebound in
# main() if encoding is requested.
PHASE = "retrieval"
PHASE_FOLDER = "retrieval_memory_reports"
STATS_ROOT = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER, "stats", f"h1abc_full_{PHASE}_mlm")
FDR_ROOT = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER, "stats")
OUT_FIG_ROOT = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER, "figures")


def _set_phase(phase):
    global PHASE, PHASE_FOLDER, STATS_ROOT, FDR_ROOT, OUT_FIG_ROOT
    PHASE = phase
    PHASE_FOLDER = "retrieval_memory_reports" if phase == "retrieval" else "encoding_memory_reports"
    STATS_ROOT = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER, "stats", f"h1abc_full_{PHASE}_mlm")
    FDR_ROOT = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER, "stats")
    OUT_FIG_ROOT = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER, "figures")

# Band definitions match run_h1abc_full_retrieval.R.
THETA = (4.88, 7.81)
SLOW_GAMMA = (30.27, 54.69)
PAC_SLOW_GAMMA = (30, 50)
HPC_SUB = ("HPC", "CA", "DG")
ALLHPC_PAIR_SOURCES = {
    "BLA_ALLHPC": ("BLA_HPC", "BLA_CA", "BLA_DG"),
    "ALLHPC_EC":  ("EC_HPC",  "CA_EC",  "DG_EC"),
    "ALLHPC_PRC": ("HPC_PRC", "CA_PRC", "DG_PRC"),
}

SCOPE_UNITS = {
    "BLAMTL": {
        "power":     ["BLA", "ALLHPC", "EC", "PRC"],
        "coherence": ["BLA_ALLHPC", "BLA_EC", "BLA_PRC"],
        "pac":       ["BLA_ALLHPC", "BLA_EC", "BLA_PRC"],
        "needs_allhpc": True,
    },
    "HPCrhinal": {
        "power":     ["ALLHPC", "EC", "PRC"],
        "coherence": ["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"],
        "pac":       ["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"],
        "needs_allhpc": True,
    },
    "HippSubBLA": {
        "power":     ["BLA", "CA", "DG", "HPC"],
        "coherence": ["BLA_CA", "BLA_DG", "BLA_HPC"],
        "pac":       ["BLA_CA", "BLA_DG", "BLA_HPC"],
        "needs_allhpc": False,
    },
    "HippSubRhinal": {
        # EC, PRC power and EC_PRC pairs belong to HPCrhinal only.
        # ALLHPC never appears here (subregions only).
        "power":     ["CA", "DG", "HPC"],
        "coherence": ["CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"],
        "pac":       ["CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"],
        "needs_allhpc": False,
    },
}

# PAC has only slow_gamma; power/coherence have theta + slow_gamma.
MEASURE_BANDS = {
    "power":     ["theta", "slow_gamma"],
    "coherence": ["theta", "slow_gamma"],
    "pac":       ["slow_gamma"],
}

QUAD_LEVELS = ["AntiResp", "NonResp", "Moderate", "Strong"]
QUAD_DISPLAY = {
    "AntiResp": "Anti",
    "NonResp":  "Non",
    "Moderate": "Moderate",
    "Strong":   "Strong",
}
QUAD_COLORS = {
    "AntiResp": "#fcaa85",
    "NonResp":  "#f1605d",
    "Moderate": "#cd4071",
    "Strong":   "#440f76",
}
QUAD_TERM_PREFIX = "QuadResponderGroup"
QUAD_CONTRAST_TERMS = ["QuadResponderGroupAntiResp",
                       "QuadResponderGroupModerate",
                       "QuadResponderGroupStrong"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_responder() -> pd.DataFrame:
    r = pd.read_csv(os.path.join(CSV_IN, "AMMEBLAES_responder_status.csv"))
    r = r.rename(columns={"Responder status": "ResponderStatus"})
    r.loc[r["Patient"] == "BJH033", "Patient"] = "BJH032"
    r = r.drop_duplicates(subset=["Patient"])
    label_map = {
        "Non-responders":      "NonResp",
        "Anti-responders":     "AntiResp",
        "Moderate responders": "Moderate",
        "Strong responders":   "Strong",
    }
    r["QuadResponderGroup"] = r["ResponderStatus"].map(label_map)
    r["mem_mod_z"] = (
        (r["avg_stim_dprime_diff"] - r["avg_stim_dprime_diff"].mean())
        / r["avg_stim_dprime_diff"].std(ddof=0)
    )
    return r[["Patient", "QuadResponderGroup", "mem_mod_z"]].reset_index(drop=True)


def load_measure(measure: str) -> pd.DataFrame:
    fname = f"combined_{PHASE}_{measure}_all_mlmr_input.csv"
    df = pd.read_csv(os.path.join(CSV_IN, fname))
    df = df[df["yes_or_no"].isin(["yes", "no"]) & (df["trial_type"] != "new")].copy()
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"
    return df


def freq_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("diff_Freq_")]


def add_band_columns(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    fcols = freq_cols(df)
    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in fcols])
    df = df.copy()
    if measure == "pac":
        sg_cols = [c for c, f in zip(fcols, freqs)
                   if PAC_SLOW_GAMMA[0] <= f <= PAC_SLOW_GAMMA[1]]
        df["slow_gamma"] = df[sg_cols].mean(axis=1, skipna=True)
        df["theta"] = np.nan
    else:
        th_cols = [c for c, f in zip(fcols, freqs) if THETA[0] <= f <= THETA[1]]
        sg_cols = [c for c, f in zip(fcols, freqs)
                   if SLOW_GAMMA[0] <= f <= SLOW_GAMMA[1]]
        df["theta"] = df[th_cols].mean(axis=1, skipna=True)
        df["slow_gamma"] = df[sg_cols].mean(axis=1, skipna=True)
    return df


def attach_allhpc_power(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["Region"].isin(HPC_SUB)].copy()
    if sub.empty:
        return df
    sub["trial_id"] = sub.groupby(["Patient", "Region"]).cumcount()
    agg = (sub.groupby(["Patient", "trial_id", "trial_type", "yes_or_no"], dropna=False)
              [["theta", "slow_gamma"]].mean().reset_index())
    agg["Region"] = "ALLHPC"
    agg = agg.drop(columns=["trial_id"])
    return pd.concat([df[["Patient", "Region", "trial_type", "yes_or_no",
                          "theta", "slow_gamma"]],
                      agg[["Patient", "Region", "trial_type", "yes_or_no",
                           "theta", "slow_gamma"]]], ignore_index=True)


def attach_allhpc_coh(df: pd.DataFrame) -> pd.DataFrame:
    pieces = [df[["Patient", "Region", "trial_type", "yes_or_no",
                  "theta", "slow_gamma"]]]
    for tgt, sources in ALLHPC_PAIR_SOURCES.items():
        sub = df[df["Region"].isin(sources)].copy()
        if sub.empty:
            continue
        sub["trial_id"] = sub.groupby(["Patient", "Region"]).cumcount()
        agg = (sub.groupby(["Patient", "trial_id", "trial_type", "yes_or_no"], dropna=False)
                  [["theta", "slow_gamma"]].mean().reset_index())
        agg["Region"] = tgt
        pieces.append(agg[["Patient", "Region", "trial_type", "yes_or_no",
                           "theta", "slow_gamma"]])
    return pd.concat(pieces, ignore_index=True)


def patient_means(measure: str, scope: str, units: list[str], band: str,
                  responder: pd.DataFrame) -> pd.DataFrame:
    raw = load_measure(measure)
    raw = add_band_columns(raw, measure)
    if SCOPE_UNITS[scope]["needs_allhpc"]:
        raw = attach_allhpc_power(raw) if measure == "power" else attach_allhpc_coh(raw)
    else:
        raw = raw[["Patient", "Region", "trial_type", "yes_or_no",
                   "theta", "slow_gamma"]]
    raw = raw[raw["Region"].isin(units)]
    pm = (raw.groupby(["Patient", "Region"])[band]
              .mean().reset_index().rename(columns={band: "y"}))
    pm = pm.merge(responder, on="Patient", how="left")
    pm = pm.dropna(subset=["mem_mod_z", "QuadResponderGroup"])
    return pm


# ---------------------------------------------------------------------------
# Stats lookups
# ---------------------------------------------------------------------------

def coef_path(measure: str, scope: str, modeltype: str, unit: str, band: str) -> str:
    return os.path.join(STATS_ROOT, f"{measure}_{scope}_{modeltype}",
                        f"{measure}_{unit}_{band}_coefs.csv")


def fdr_path(measure: str, scope: str, modeltype: str, band: str, term: str) -> str:
    if modeltype == "LMMcont":
        return os.path.join(FDR_ROOT, f"fdr_{measure}_{scope}_{modeltype}_{PHASE}",
                            f"{measure}_{scope}_{modeltype}_{band}_mem_mod_z.csv")
    return os.path.join(FDR_ROOT, f"fdr_{measure}_{scope}_{modeltype}_{PHASE}",
                        f"{measure}_{scope}_{modeltype}_{band}_{term}.csv")


def load_coefs(measure: str, scope: str, modeltype: str, unit: str, band: str):
    p = coef_path(measure, scope, modeltype, unit, band)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def load_fdr_unit(measure: str, scope: str, modeltype: str, band: str,
                  term: str, unit: str):
    p = fdr_path(measure, scope, modeltype, band, term)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    row = df[df["unit"] == unit]
    if row.empty:
        return None
    return float(row["q_FDR"].iloc[0])


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def p_str(p):
    if pd.isna(p):
        return "n/a"
    if p < .001:
        return "p<.001"
    return f"p={p:.3f}".replace("0.", ".")


def fmt_b(b):
    if pd.isna(b):
        return ""
    return f"{b:.3f}"


def display_unit(u: str) -> str:
    # ALLHPC = macro hippocampus -> "HPC"; HPC region = subiculum -> "SUB"
    parts = []
    for p in str(u).split("_"):
        if p == "ALLHPC":
            parts.append("HPC")
        elif p == "HPC":
            parts.append("SUB")
        else:
            parts.append(p)
    return "-".join(parts)


def panel_title(unit: str, band: str) -> str:
    band_tag = "Theta" if band == "theta" else "Slow Gamma"
    return f"{display_unit(unit)} - {band_tag}"


def red_box(ax):
    ax.patch.set_edgecolor("red")
    ax.patch.set_linewidth(1.0)
    ax.patch.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("red")
        spine.set_linewidth(0.9)


# ---------------------------------------------------------------------------
# LMMcont scatter panel
# ---------------------------------------------------------------------------

def plot_lmmcont_panel(ax, measure: str, scope: str, unit: str, band: str,
                       data: pd.DataFrame):
    pdata = data[data["Region"] == unit]
    coefs = load_coefs(measure, scope, "LMMcont", unit, band)

    ax.set_title(panel_title(unit, band), fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axhline(0, color="0.85", linewidth=0.6, zorder=0)

    if coefs is None or pdata.empty:
        ax.text(0.5, 0.5, "no fit", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.4")
        return

    intercept = float(coefs.loc[coefs["term"] == "(Intercept)", "estimate"].iloc[0])
    slope_row = coefs[coefs["term"] == "mem_mod_z"]
    slope = float(slope_row["estimate"].iloc[0]) if not slope_row.empty else np.nan
    p = float(slope_row["p.value"].iloc[0]) if not slope_row.empty else np.nan

    ax.scatter(pdata["mem_mod_z"], pdata["y"], s=12, color="0.25",
               alpha=0.75, linewidths=0)

    if not np.isnan(slope):
        x_grid = np.linspace(pdata["mem_mod_z"].min(),
                             pdata["mem_mod_z"].max(), 50)
        ax.plot(x_grid, intercept + slope * x_grid, color="black", linewidth=1.4)

    q = load_fdr_unit(measure, scope, "LMMcont", band, "mem_mod_z", unit)
    sig_after_fdr = (q is not None) and (q < 0.05)
    if sig_after_fdr:
        red_box(ax)


# ---------------------------------------------------------------------------
# LMMquad mean+/-SE panel
# ---------------------------------------------------------------------------

def plot_lmmquad_panel(ax, measure: str, scope: str, unit: str, band: str,
                       data: pd.DataFrame):
    pdata = data[data["Region"] == unit]
    coefs = load_coefs(measure, scope, "LMMquad", unit, band)

    ax.set_title(panel_title(unit, band), fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axhline(0, color="0.85", linewidth=0.6, zorder=0)

    if pdata.empty or coefs is None:
        ax.text(0.5, 0.5, "no fit", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.4")
        ax.set_xticks(range(len(QUAD_LEVELS)))
        ax.set_xticklabels([QUAD_DISPLAY[g] for g in QUAD_LEVELS], fontsize=7)
        return

    sig_any = False
    q_lookup = {}
    for term in QUAD_CONTRAST_TERMS:
        grp = term.replace(QUAD_TERM_PREFIX, "")
        q = load_fdr_unit(measure, scope, "LMMquad", band, term, unit)
        q_lookup[grp] = q
        if q is not None and q < 0.05:
            sig_any = True

    xs, means, ses, ns = [], [], [], []
    for i, grp in enumerate(QUAD_LEVELS):
        sub = pdata[pdata["QuadResponderGroup"] == grp]
        if sub.empty:
            xs.append(i); means.append(np.nan); ses.append(np.nan); ns.append(0)
            continue
        m = sub["y"].mean()
        s = sub["y"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
        xs.append(i); means.append(m); ses.append(s); ns.append(len(sub))
        rng = np.random.default_rng(seed=hash((unit, band, grp)) % 2**32)
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(np.full(len(sub), i) + jitter, sub["y"],
                   s=10, color=QUAD_COLORS[grp], alpha=0.55, linewidths=0)
        ax.errorbar([i], [m], yerr=[s], fmt="o", color=QUAD_COLORS[grp],
                    markersize=6, markeredgecolor="black", markeredgewidth=0.6,
                    capsize=3, elinewidth=1.0, zorder=5)

    ax.set_xticks(range(len(QUAD_LEVELS)))
    labels = []
    sig_flags = []
    for grp, n in zip(QUAD_LEVELS, ns):
        if n == 0:
            labels.append(QUAD_DISPLAY[grp])
            sig_flags.append(False)
            continue
        q_v = q_lookup.get(grp)
        is_sig = q_v is not None and q_v < 0.05
        sig_marker = "*" if is_sig else ""
        labels.append(f"{QUAD_DISPLAY[grp]}{sig_marker}\nn={n}")
        sig_flags.append(is_sig)
    ax.set_xticklabels(labels, fontsize=7)
    for tick, is_sig in zip(ax.get_xticklabels(), sig_flags):
        if is_sig:
            tick.set_fontweight("bold")
    ax.set_xlim(-0.5, len(QUAD_LEVELS) - 0.5)

    if sig_any:
        red_box(ax)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

MEASURE_TITLES = {"power": "Power", "coherence": "Coherence", "pac": "PAC"}
MODELTYPE_TITLES = {
    "LMMcont": "Memory modulation status (continuous)",
    "LMMquad": "Responder group status (quad)",
}
BAND_TITLES = {"theta": "Theta", "slow_gamma": "Slow Gamma"}


def build_figure(scope: str, measure: str, modeltype: str, band: str):
    all_units = SCOPE_UNITS[scope][measure]
    units = [u for u in all_units
             if os.path.exists(coef_path(measure, scope, modeltype, u, band))]
    if not units:
        print(f"   skip {scope}/{measure}/{modeltype}/{band}: no fits")
        return None
    responder = load_responder()
    data = patient_means(measure, scope, units, band, responder)

    n = len(units)
    # Encoding HPCrhinal PAC only: stack the three units (ALLHPC_EC,
    # ALLHPC_PRC, EC_PRC) into one column with panel dimensions matched to
    # the encoding HPCrhinal coherence panels (~2.56" x 2.09"). Retrieval
    # PAC keeps the default 2x2-with-blank layout.
    is_hpcrhinal_pac = (scope == "HPCrhinal" and measure == "pac"
                        and PHASE == "encoding")
    if is_hpcrhinal_pac:
        ncols, nrows = 1, n
    elif n <= 2:
        ncols, nrows = n, 1
    elif n <= 4:
        ncols, nrows = 2, 2
    elif n <= 6:
        ncols, nrows = 3, 2
    else:
        ncols, nrows = 3, int(np.ceil(n / 3))

    if is_hpcrhinal_pac:
        # fig_w * (R-L)=0.73 -> axes_w ~2.70"; fig_h * (T-B)=0.80 with
        # hspace=0.36 -> per-panel height ~2.09" (matched to encoding
        # HPCrhinal coherence panel box). hspace large enough to keep
        # each panel's title clear of the x-tick numbers above.
        fig_w = 3.7
        fig_h = 9.7
    else:
        fig_w = 3.0 * ncols + 0.6
        fig_h = 2.6 * nrows + 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             squeeze=False)
    for i, unit in enumerate(units):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if modeltype == "LMMcont":
            plot_lmmcont_panel(ax, measure, scope, unit, band, data)
        else:
            plot_lmmquad_panel(ax, measure, scope, unit, band, data)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    title = (f"{scope} - {MEASURE_TITLES[measure]} - {BAND_TITLES[band]}\n"
             f"{MODELTYPE_TITLES[modeltype]} predicting "
             f"{measure} at {PHASE}")
    fig.suptitle(title,
                 fontsize=10 if is_hpcrhinal_pac else 11,
                 y=0.97,
                 fontweight="bold")

    if modeltype == "LMMcont":
        x_lab = "Z-scored memory modulation status"
    else:
        x_lab = "Responder group"
    y_unit = {"power": "Band-averaged power (dB)",
              "coherence": "Band-averaged coherence",
              "pac": "Band-averaged PAC"}[measure]
    if is_hpcrhinal_pac:
        # T=0.88 leaves a generous (~0.93") gap to the suptitle at y=0.97;
        # left=0.22 keeps the supylabel clear of the y-tick numbers;
        # hspace=0.50 leaves room between each panel's x-tick numbers and
        # the next panel's title.
        fig.subplots_adjust(left=0.22, right=0.95, bottom=0.08, top=0.88,
                            wspace=0.22, hspace=0.36)
    else:
        fig.subplots_adjust(left=0.13, right=0.99, bottom=0.13, top=0.88,
                            wspace=0.22, hspace=0.22)
    fig.supxlabel(x_lab, fontsize=12, fontweight="bold", y=0.018)
    fig.supylabel(y_unit, fontsize=12, fontweight="bold", x=0.015)

    out_dir = os.path.join(OUT_FIG_ROOT, f"{scope}_{measure}_{modeltype}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            f"{scope}_{measure}_{modeltype}_{band}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"-> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Combined-bands figure builder (theta col, slow_gamma col)
# ---------------------------------------------------------------------------

def build_combined_bands_figure(scope: str, measure: str, modeltype: str):
    """One figure per (scope, measure, modeltype): rows=units, cols=bands.
    Only applies to measures with both theta and slow_gamma (power, coherence).
    """
    bands = MEASURE_BANDS[measure]
    if len(bands) < 2:
        return None  # skip PAC (slow_gamma only)
    all_units = SCOPE_UNITS[scope][measure]
    units = [u for u in all_units
             if any(os.path.exists(coef_path(measure, scope, modeltype, u, b))
                    for b in bands)]
    if not units:
        print(f"   skip {scope}/{measure}/{modeltype} combined: no fits")
        return None

    responder = load_responder()
    band_data = {b: patient_means(measure, scope, units, b, responder)
                 for b in bands}

    nrows, ncols = len(units), len(bands)
    fig_w = 3.4 * ncols + 0.8
    fig_h = 2.7 * nrows + 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             squeeze=False)

    for r, unit in enumerate(units):
        for c, band in enumerate(bands):
            ax = axes[r][c]
            if not os.path.exists(coef_path(measure, scope, modeltype, unit, band)):
                ax.axis("off")
                continue
            if modeltype == "LMMcont":
                plot_lmmcont_panel(ax, measure, scope, unit, band, band_data[band])
            else:
                plot_lmmquad_panel(ax, measure, scope, unit, band, band_data[band])

    phase_label = "Retrieval" if PHASE == "retrieval" else "Encoding"
    title = (f"{scope} - {MEASURE_TITLES[measure]} - "
             f"{MODELTYPE_TITLES[modeltype]} - {phase_label}")
    fig.suptitle(title, fontsize=12, y=0.995)

    x_lab = ("Z-scored memory modulation status" if modeltype == "LMMcont"
             else "Responder group")
    y_unit = {"power": "Band-averaged power (dB)",
              "coherence": "Band-averaged coherence",
              "pac": "Band-averaged PAC"}[measure]
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.07, top=0.95,
                        wspace=0.22, hspace=0.32)
    fig.supxlabel(x_lab, fontsize=12, fontweight="bold", y=0.015)
    fig.supylabel(y_unit, fontsize=12, fontweight="bold", x=0.015)

    out_dir = os.path.join(OUT_FIG_ROOT, f"{scope}_{measure}_{modeltype}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            f"{scope}_{measure}_{modeltype}_combined_bands.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"-> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_RUNS = [
    (scope, measure, modeltype, band)
    for scope in ("BLAMTL", "HPCrhinal", "HippSubBLA", "HippSubRhinal")
    for measure in ("power", "coherence", "pac")
    for modeltype in ("LMMcont", "LMMquad")
    for band in MEASURE_BANDS[measure]
]

DEFAULT_COMBINED_RUNS = [
    (scope, measure, modeltype)
    for scope in ("BLAMTL", "HPCrhinal", "HippSubBLA", "HippSubRhinal")
    for measure in ("power", "coherence")
    for modeltype in ("LMMcont", "LMMquad")
]


def main(argv):
    argv = list(argv)
    if argv and argv[0] in ("retrieval", "encoding"):
        _set_phase(argv.pop(0))
    combined = "--combined-bands" in argv
    argv = [a for a in argv if a != "--combined-bands"]
    if combined:
        if len(argv) >= 3:
            runs = [(argv[0], argv[1], argv[2])]
        else:
            runs = DEFAULT_COMBINED_RUNS
        for scope, measure, modeltype in runs:
            build_combined_bands_figure(scope, measure, modeltype)
        return
    if len(argv) >= 4:
        runs = [(argv[0], argv[1], argv[2], argv[3])]
    else:
        runs = DEFAULT_RUNS
    for scope, measure, modeltype, band in runs:
        build_figure(scope, measure, modeltype, band)


if __name__ == "__main__":
    main(sys.argv[1:])
