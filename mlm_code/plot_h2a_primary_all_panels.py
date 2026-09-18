#!/usr/bin/env python
"""
H2a (encoding) primary-test figures — all panels, grayscale, separated by modality.

For each primary test (Memory Modulation Z continuous; Responder Group Status
categorical) emit three figures using the AMME-timing-filtered ENCODING data:
  - Power     (6 regions x 2 bands)
  - Coherence (12 pairs   x 2 bands)
  - PAC       (11 pairs   x slow gamma)

Usage:
  python plot_h2a_primary_all_panels.py            # imbalanced (all subjects)
  python plot_h2a_primary_all_panels.py --balanced # balanced (>= 10 rem & forg per region)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BALANCED = "--balanced" in sys.argv
FILTER_LABEL = "balanced" if BALANCED else "imbalanced"
MIN_TRIALS = 10  # mirrors the R script's filter_balanced threshold

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
DATA_DIR = ROOT / "outputs" / f"h2a_subregions_quad_lmm_{FILTER_LABEL}_encoding_mlm"
CSV_DIR = ROOT / "outputs" / "csvs"
PHASE = "encoding"

THETA = (4.88, 7.81)
SLOW_GAMMA = (30.27, 54.69)
PAC_SG = (30.0, 50.0)

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


POWER_REGIONS = ["BLA", "CA", "DG", "HPC", "EC", "PRC"]
COH_PAIRS = ["BLA_CA","BLA_DG","BLA_HPC","BLA_EC","BLA_PRC","CA_EC","CA_PRC",
             "DG_EC","DG_PRC","EC_HPC","HPC_PRC","EC_PRC"]
PAC_PAIRS = ["BLA_CA","BLA_DG","BLA_HPC","BLA_EC","BLA_PRC","CA_EC","CA_PRC",
             "DG_EC","DG_PRC","EC_HPC","HPC_PRC"]
BAND_LABELS = {"theta": "Theta", "slow_gamma": "Slow Gamma"}
BAND_RANGES = {"theta": THETA, "slow_gamma": SLOW_GAMMA}

GROUP_ORDER = ["AntiResp", "NonResp", "Moderate", "Strong"]
GROUP_LABEL = {"AntiResp": "Anti", "NonResp": "Non",
               "Moderate": "Mod", "Strong": "Strong"}

POINT_COLOR = "black"
SCATTER_COLOR = "0.55"
LINE_COLOR = "black"


def load_responder():
    r = pd.read_csv(CSV_DIR / "AMMEBLAES_responder_status.csv")
    r = r.rename(columns={"Responder.status": "ResponderStatus",
                          "Responder status": "ResponderStatus"})
    r.loc[r["Patient"] == "BJH033", "Patient"] = "BJH032"
    r = r.drop_duplicates("Patient")
    map_to = {"Non-responders": "NonResp", "Anti-responders": "AntiResp",
              "Moderate responders": "Moderate", "Strong responders": "Strong"}
    r["QuadResponderGroup"] = r["ResponderStatus"].map(map_to)
    r["mem_mod_z"] = (r["avg_stim_dprime_diff"] -
                      r["avg_stim_dprime_diff"].mean()) / \
                     r["avg_stim_dprime_diff"].std(ddof=1)
    return r[["Patient", "QuadResponderGroup", "mem_mod_z"]]


def _balanced_keep_patients(df):
    """Mirror filter_balanced from the R script: keep patients with at least
    MIN_TRIALS remembered and forgotten trials per region (averaged across
    that patient's regions)."""
    counts = (df.groupby("Patient")
                .agg(n_rem=("yes_or_no", lambda s: (s == "yes").sum()),
                     n_forg=("yes_or_no", lambda s: (s == "no").sum()),
                     n_regions=("Region", lambda s: s.nunique()))
                .reset_index())
    counts["rem_per_reg"]  = (counts["n_rem"]  / counts["n_regions"]).round()
    counts["forg_per_reg"] = (counts["n_forg"] / counts["n_regions"]).round()
    keep = counts.loc[(counts["rem_per_reg"]  >= MIN_TRIALS) &
                      (counts["forg_per_reg"] >= MIN_TRIALS), "Patient"]
    return set(keep.tolist())


def load_modality(modality):
    fname = f"combined_{PHASE}_{modality}_all_mlmr_input.csv"
    df = pd.read_csv(CSV_DIR / fname)
    df = df[df["yes_or_no"].isin(["yes", "no"])]
    df = df[df["trial_type"] != "new"]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"
    if BALANCED:
        keep = _balanced_keep_patients(df)
        before = df["Patient"].nunique()
        df = df[df["Patient"].isin(keep)]
        print(f"  [balanced] {modality}: kept "
              f"{df['Patient'].nunique()}/{before} patients")
    return df


def per_subject_feature(modality_df, unit, band_range):
    sub = modality_df[modality_df["Region"] == unit].copy()
    if sub.empty:
        return None
    freq_cols = [c for c in sub.columns if c.startswith("diff_Freq_")]
    freqs = pd.Series([float(c.replace("diff_Freq_", "")) for c in freq_cols],
                      index=freq_cols)
    use = freqs[(freqs >= band_range[0]) & (freqs <= band_range[1])].index
    sub["feature"] = sub[use].mean(axis=1)
    return sub.groupby("Patient", as_index=False)["feature"].mean()


def load_coefs(name):
    p = DATA_DIR / name
    if not p.exists():
        return None
    return pd.read_csv(p).set_index("term")


def sig_marker(p):
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    if p < .10:  return "+"
    return ""


def highlight_axes(ax, mark):
    if not mark or mark == "+":
        return
    for spine in ax.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(1.6)


def memmod_panel(ax, ps, coef, title):
    if ps is None or ps.empty or coef is None:
        ax.set_visible(False); return
    if "mem_mod_z" not in coef.index:
        ax.set_visible(False); return
    b0 = coef.loc["(Intercept)", "estimate"]
    b1 = coef.loc["mem_mod_z", "estimate"]
    p = coef.loc["mem_mod_z", "p.value"]
    ax.scatter(ps["mem_mod_z"], ps["feature"],
               s=22, alpha=0.65, color=SCATTER_COLOR,
               edgecolor="white", lw=0.4, zorder=2)
    xs = np.linspace(ps["mem_mod_z"].min() - 0.15,
                     ps["mem_mod_z"].max() + 0.15, 60)
    ax.plot(xs, b0 + b1 * xs, color=LINE_COLOR, lw=1.6, zorder=3)
    ax.axhline(0, color="0.7", lw=0.4, ls="--")
    mark = sig_marker(p)
    ax.set_title(f"{title}  (b={b1:.3g}, p={p:.3f}{mark})",
                 fontsize=8.5)
    ax.tick_params(labelsize=7)
    highlight_axes(ax, mark)


def respgroup_panel(ax, ps, coef, title):
    if ps is None or ps.empty or coef is None:
        ax.set_visible(False); return
    b0 = coef.loc["(Intercept)", "estimate"] if "(Intercept)" in coef.index else 0
    means, cis, pvals = {}, {}, {}
    means["NonResp"] = b0
    cis["NonResp"] = (coef.loc["(Intercept)", "conf.low"] if "(Intercept)" in coef.index else b0,
                      coef.loc["(Intercept)", "conf.high"] if "(Intercept)" in coef.index else b0)
    pvals["NonResp"] = np.nan
    for g in ["AntiResp", "Moderate", "Strong"]:
        term = f"QuadResponderGroup{g}"
        if term in coef.index:
            means[g] = b0 + coef.loc[term, "estimate"]
            cis[g] = (b0 + coef.loc[term, "conf.low"],
                      b0 + coef.loc[term, "conf.high"])
            pvals[g] = coef.loc[term, "p.value"]
        else:
            means[g] = np.nan
            cis[g] = (np.nan, np.nan)
            pvals[g] = np.nan

    xs = np.arange(len(GROUP_ORDER))
    for i, g in enumerate(GROUP_ORDER):
        m = means[g]
        if np.isnan(m):
            continue
        lo, hi = cis[g]
        ax.errorbar(xs[i], m, yerr=[[m - lo], [hi - m]],
                    fmt="o", color=POINT_COLOR, capsize=3,
                    markersize=5, lw=1.2, zorder=3)
        sj = ps[ps["QuadResponderGroup"] == g]
        n_sub = len(sj)
        if n_sub > 0:
            jitter = (np.random.RandomState(i)
                      .uniform(-0.10, 0.10, size=n_sub))
            ax.scatter(xs[i] + jitter, sj["feature"],
                       color=SCATTER_COLOR, alpha=0.45, s=14,
                       edgecolor="none", zorder=2)
    ax.axhline(0, color="0.7", lw=0.4, ls="--")
    ax.set_xticks(xs)
    labels = []
    panel_mark = ""
    for g in GROUP_ORDER:
        n_g = ps[ps["QuadResponderGroup"] == g]["Patient"].nunique() \
              if "Patient" in ps.columns else len(ps[ps["QuadResponderGroup"] == g])
        pv = pvals.get(g, np.nan)
        s_mark = sig_marker(pv) if not np.isnan(pv) else ""
        lab = f"{GROUP_LABEL[g]}\nn={n_g}"
        if s_mark:
            lab += f"\n{s_mark}"
        labels.append(lab)
        if s_mark and s_mark != "+":
            cur = {"*": 1, "**": 2, "***": 3}.get(s_mark, 0)
            best = {"*": 1, "**": 2, "***": 3}.get(panel_mark, 0)
            if cur > best:
                panel_mark = s_mark
    ax.set_xticklabels(labels, fontsize=7)
    title_with_mark = f"{title}{'  ' + panel_mark if panel_mark else ''}"
    ax.set_title(title_with_mark, fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    highlight_axes(ax, panel_mark)


def make_grid(ncells, cols=4):
    rows = int(np.ceil(ncells / cols))
    return rows, cols


def fig_for_modality(test, modality, panels):
    """panels: list of (label, modality_key, unit, band_range, coef_key)."""
    suffix = "primaryMemMod" if test == "memmod" else "primaryRespGroup"

    # Pre-compute valid panels (skip ones with no data / no fitted model).
    valid = []
    for label, mod_df, unit, band_range, coef_key in panels:
        ps_feat = per_subject_feature(mod_df, unit, band_range)
        if ps_feat is None:
            continue
        ps = ps_feat.merge(resp, on="Patient", how="left").dropna(
            subset=["feature", "mem_mod_z", "QuadResponderGroup"])
        if ps.empty:
            continue
        coef = load_coefs(f"{coef_key}_{suffix}_coefs.csv")
        if coef is None:
            continue
        valid.append((label, ps, coef))

    if not valid:
        print(f"  No valid panels for {test} {modality}, skipping.")
        return

    rows, cols = make_grid(len(valid), cols=4)
    fig_w = 4.0 * cols
    fig_h = 3.0 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes).flatten()

    for i, (label, ps, coef) in enumerate(valid):
        ax = axes[i]
        if test == "memmod":
            memmod_panel(ax, ps, coef, label)
        else:
            respgroup_panel(ax, ps, coef, label)

    for j in range(len(valid), len(axes)):
        axes[j].set_visible(False)

    bands_phrase = ("slow gamma" if modality == "PAC"
                    else "theta and slow gamma")
    test_phrase = ("Memory modulation status" if test == "memmod"
                   else "Responder group status")
    fig.suptitle(
        f"{test_phrase} predicting {bands_phrase} {modality.lower()} at {PHASE}",
        fontsize=18, fontweight="bold")

    if test == "memmod":
        fig.supxlabel("Memory Modulation Z (subject-level)",
                      fontsize=16, fontweight="bold")
    else:
        fig.supxlabel("Responder Group Status (vs NonResp = ref)",
                      fontsize=16, fontweight="bold")
    fig.supylabel(f"{modality} (band-avg)", fontsize=16, fontweight="bold")

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.94])
    out = DATA_DIR / (f"plot_primary_h2a_{test}_{modality.lower()}_"
                      f"{FILTER_LABEL}_all.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Build panel lists per modality ────────────────────────────────────────
resp = load_responder()

power_df = load_modality("power")
coh_df   = load_modality("coherence")
pac_df   = load_modality("pac")

power_panels = []
for reg in POWER_REGIONS:
    for bk, bl in BAND_LABELS.items():
        power_panels.append((f"{pretty_label(reg)} - {bl}", power_df, reg,
                             BAND_RANGES[bk], f"power_{reg}_{bk}"))

coh_panels = []
for pair in COH_PAIRS:
    for bk, bl in BAND_LABELS.items():
        coh_panels.append((f"{pretty_label(pair)} - {bl}", coh_df, pair,
                           BAND_RANGES[bk], f"coherence_{pair}_{bk}"))

pac_panels = []
for pair in PAC_PAIRS:
    pac_panels.append((f"{pretty_label(pair)} - Slow Gamma PAC", pac_df, pair,
                       PAC_SG, f"pac_{pair}_slow_gamma"))


for test in ["memmod", "respgroup"]:
    fig_for_modality(test, "Power",     power_panels)
    fig_for_modality(test, "Coherence", coh_panels)
    fig_for_modality(test, "PAC",       pac_panels)
