#!/usr/bin/env python
"""Encoding coherence 2x2 stim-vs-nostim bar graph for hippocampal subregion x
rhinal cortex pairs only: CA-EC, DG-EC, HPC-EC, CA-PRC, DG-PRC, HPC-PRC.

Reuses loaders / helpers from regen_combined_stim_nostim_bars.py. Stats are
re-run on this 6-pair set so BH-FDR reflects the panel. Colors use the
colorblind-safe palette (no greens). Display labels render HPC as 'SUB'.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as _stats

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/"
            "InmanLab/BLAES_data/dissertation/AMME_BLAES")
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "regen_combined_bars", ROOT / "scripts" / "regen_combined_stim_nostim_bars.py"
)
_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_base)

OUTPUTS = ROOT / "OUTPUTS"

HIPP_SUB = {"CA", "DG", "HPC"}
RHINAL = {"EC", "PRC"}

# Display rename: HPC -> SUB (data column unchanged).
DISPLAY_RENAME = {"HPC": "SUB"}

COLORBLIND_PALETTE = [
    "#E69F00",  # orange
    "#0072B2",  # blue
    "#CC79A7",  # reddish purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]

# Canonical (alphabetical) pair order for the panel.
PAIR_ORDER = ["CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"]


def _is_target_pair(region: str) -> bool:
    parts = region.split("_")
    if len(parts) != 2:
        return False
    a, b = parts
    return (a in HIPP_SUB and b in RHINAL) or (a in RHINAL and b in HIPP_SUB)


def _display_pair(region: str) -> str:
    parts = [DISPLAY_RENAME.get(p, p) for p in region.split("_")]
    return "_".join(parts)


def _palette_for(regions: list[str]) -> dict[str, str]:
    return {r: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
            for i, r in enumerate(regions)}


def _plot(df_all, df_cond_all, mod, phase: str):
    if df_all.empty:
        print("  [hippsub_rhinal] No pairs after filter, skipping.")
        return

    seen = df_all["Region"].unique().tolist()
    ordered = [p for p in PAIR_ORDER if p in seen]
    ordered += sorted(p for p in seen if p not in ordered)
    palette = _palette_for(ordered)
    display_labels = [_display_pair(p) for p in ordered]

    band_order = ["Theta", "Slow gamma"]
    mem_order = ["remembered", "forgotten"]
    mem_label = {"remembered": "Remembered Trials", "forgotten": "Forgotten Trials"}

    fig_w = max(15, 3.0 + 1.2 * len(ordered))
    fig_h = 12
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharey="row")
    fig.suptitle(
        f"Baseline-Corrected Coherence Diff (Stim - No Stim) - "
        f"{phase.capitalize()} - Hipp. Subregions x Rhinal Cortex",
        fontsize=20, fontweight="bold", y=0.995,
    )

    test_rows = []
    for r, band in enumerate(band_order):
        for c, mem in enumerate(mem_order):
            ax = axes[r, c]
            sub = df_all[(df_all["power_range"] == band) &
                         (df_all["memory_cond"] == mem)].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

            cond_sub = pd.DataFrame()
            if df_cond_all is not None and not df_cond_all.empty:
                cond_sub = df_cond_all[(df_cond_all["power_range"] == band) &
                                        (df_cond_all["memory_cond"] == mem)]
            ts, ps, ns, means, sems = [], [], [], [], []
            for roi in ordered:
                vals = sub.loc[sub["Region"] == roi, "mean_power_diff"].dropna().to_numpy()
                n_diff = len(vals)
                if not cond_sub.empty:
                    rs = cond_sub[cond_sub["Region"] == roi].dropna(subset=["stim", "nostim"])
                    stim_vals = rs["stim"].to_numpy(dtype=float)
                    nostim_vals = rs["nostim"].to_numpy(dtype=float)
                    n = len(stim_vals)
                    if n >= 2 and np.std(stim_vals - nostim_vals, ddof=1) > 0:
                        t, p = _stats.ttest_rel(stim_vals, nostim_vals)
                    else:
                        t, p = (np.nan, np.nan)
                else:
                    n = n_diff
                    t, p = (np.nan, np.nan)
                ts.append(t); ps.append(p); ns.append(n)
                means.append(float(np.nanmean(vals)) if n_diff else np.nan)
                sems.append(float(np.nanstd(vals, ddof=1)/np.sqrt(n_diff)) if n_diff >= 2 else np.nan)
            ps_arr = np.asarray(ps, dtype=float)
            valid = ~np.isnan(ps_arr)
            qs = np.full(len(ps_arr), np.nan)
            if valid.sum() > 0:
                qs[valid] = _base._bh_fdr(ps_arr[valid])
            sig_flags = [(not np.isnan(qs[i]) and qs[i] < 0.05) for i in range(len(qs))]

            for i, roi in enumerate(ordered):
                test_rows.append({
                    "panel": "hippsub_rhinal", "phase": phase, "measure": "coherence",
                    "memory": mem, "band": band, "Region": roi,
                    "n": ns[i], "df": ns[i] - 1 if ns[i] else 0,
                    "mean_diff": means[i], "sem": sems[i],
                    "t": ts[i], "p": ps[i], "q_BH": qs[i],
                })

            x = np.arange(len(ordered))
            bar_colors = [palette.get(roi, "#A0A0A0") for roi in ordered]
            ax.bar(
                x, [means[i] if not np.isnan(means[i]) else 0 for i in range(len(x))],
                yerr=[sems[i] if not np.isnan(sems[i]) else 0 for i in range(len(x))],
                color=bar_colors, edgecolor="#4D4D4D", linewidth=1.2,
                width=0.72, capsize=3, ecolor="#4D4D4D", zorder=1,
            )

            rng = np.random.default_rng(7)
            for i, roi in enumerate(ordered):
                vals = sub.loc[sub["Region"] == roi, "mean_power_diff"].dropna().to_numpy()
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.16, 0.16, len(vals))
                ax.scatter(np.full(len(vals), i, dtype=float) + jitter, vals,
                           c="#404040", s=28, alpha=0.7, edgecolors="none", zorder=3)

            ax.axhline(0, color="#4D4D4D", linewidth=1.0, zorder=0)
            ax.set_title(f"{band} - {mem_label[mem]}", fontsize=14, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Baseline-Corrected Coherence Diff"
                          if c == 0 else "",
                          fontsize=12, fontweight="bold")
            ax.tick_params(axis="y", labelsize=11)
            ax.set_xticks(x)
            tick_labels = [f"{display_labels[i]} *" if sig_flags[i]
                           else display_labels[i] for i in range(len(ordered))]
            base_fs = 13
            ax.set_xticklabels(tick_labels, rotation=35, ha="right",
                               fontsize=base_fs, fontweight="bold")
            for i, lbl in enumerate(ax.get_xticklabels()):
                lbl.set_fontweight("bold")
                if sig_flags[i]:
                    lbl.set_color("red")
                    lbl.set_fontsize(base_fs + 2)

    caption = (
        "Significance: paired-samples t-test (scipy.stats.ttest_rel) on matched "
        "per-patient stim vs no-stim baseline-corrected band means. "
        "Benjamini-Hochberg FDR-corrected separately within each band x memory cell. "
        "Pairs are hippocampal subregions (CA, DG, SUB) crossed with rhinal cortex "
        "(EC, PRC). Pairs with q<.05 are marked with a * next to the label."
    )
    fig.text(0.5, 0.012, caption, ha="center", va="bottom",
             fontsize=8, color="dimgray", wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_dir = OUTPUTS / f"{phase}_coherence" / "all"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"Bargraph_BaselineCorrected_CoherenceDiff_byROI_"
             f"{phase}_combined_hippsub_rhinal_2x2.png")
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [hippsub_rhinal] Saved: {out_dir / fname}")

    if test_rows:
        tdf = pd.DataFrame(test_rows)
        csv_name = (f"PairedTTests_StimMinusNoStim_{phase}_coherence_"
                    f"hippsub_rhinal.csv")
        tdf.to_csv(out_dir / csv_name, index=False)
        print(f"  [hippsub_rhinal] Saved t-test results: {out_dir / csv_name}")
    return test_rows


def regen(phase: str = "encoding"):
    print(f"\n=== {phase} coherence: hippsub x rhinal 2x2 ===")
    mod, data = _base.load_combined_data("coherence", phase)

    df_rem = _base._compute_mem_diff_df(mod, "rem", data, "coherence")
    df_forg = _base._compute_mem_diff_df(mod, "forg", data, "coherence")
    if df_rem.empty and df_forg.empty:
        print("  No data, skipping.")
        return []
    df_rem["memory_cond"] = "remembered"
    df_forg["memory_cond"] = "forgotten"
    df_all = pd.concat([df_rem, df_forg], ignore_index=True)
    df_all = df_all[df_all["Region"].apply(_is_target_pair)].copy()

    df_cond_rem = _base._compute_mem_cond_df(mod, "rem", data)
    df_cond_forg = _base._compute_mem_cond_df(mod, "forg", data)
    cond_parts = []
    if not df_cond_rem.empty:
        df_cond_rem["memory_cond"] = "remembered"
        cond_parts.append(df_cond_rem)
    if not df_cond_forg.empty:
        df_cond_forg["memory_cond"] = "forgotten"
        cond_parts.append(df_cond_forg)
    df_cond_all = (pd.concat(cond_parts, ignore_index=True) if cond_parts
                   else pd.DataFrame())
    if not df_cond_all.empty:
        df_cond_all = df_cond_all[df_cond_all["Region"].apply(_is_target_pair)].copy()

    return _plot(df_all, df_cond_all, mod, phase)


if __name__ == "__main__":
    rows = regen("encoding")
    if rows:
        sig = pd.DataFrame(rows)
        sig = sig[sig["q_BH"] < 0.05]
        if not sig.empty:
            print("\nSignificant cells (q<.05):")
            print(sig.to_string(index=False))
        else:
            print("\nNo cells with q<.05 in the hippsub x rhinal panel.")
    print("\nDone.")
