#!/usr/bin/env python
"""No-BLA Main Regions stim-vs-nostim 2x2 bar graphs for ENCODING ONLY.

Same layout, palette, and stats methodology as
`regen_combined_stim_nostim_bars.py`'s 'mainregions' panel, but with BLA (and
any pair containing BLA) excluded. Stats are re-run with the smaller region
set so BH-FDR is computed on the no-BLA panel.

Outputs:
  OUTPUTS/encoding_power/all/Bargraph_BaselineCorrected_PowerDiff_byROI_encoding_combined_mainregions_nobla_2x2.png
  OUTPUTS/encoding_coherence/all/Bargraph_BaselineCorrected_CoherenceDiff_byROI_encoding_combined_mainregions_nobla_2x2.png
plus a paired t-test CSV per figure.

Does not touch any retrieval figures or other encoding figures.
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

# Colorblind-safe palette (Wong / Okabe-Ito, with greens removed).
COLORBLIND_PALETTE = [
    "#E69F00",  # orange
    "#0072B2",  # blue
    "#CC79A7",  # reddish purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#8E44AD",  # purple
    "#34495E",  # dark slate
]


def _colorblind_palette(regions: list[str]) -> dict[str, str]:
    return {r: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
            for i, r in enumerate(regions)}


def _has_bla(region: str) -> bool:
    return "BLA" in [t for t in region.split("_") if t]


def _is_main_region_nobla(region: str) -> bool:
    """Main region (or main-region pair) that does NOT involve BLA."""
    if not _base._is_main_region(region):
        return False
    return not _has_bla(region)


def _plot_nobla(df_all, df_cond_all, mod, measure: str, phase: str):
    if df_all.empty:
        print("  [mainregions_nobla] No regions after filter, skipping.")
        return

    if measure == "power":
        ordered_regions = [r for r in mod.ROI_COLORS_BAR if r in df_all["Region"].unique()]
        ordered_regions += sorted(r for r in df_all["Region"].unique() if r not in ordered_regions)
        palette = {r: mod.ROI_COLORS_BAR.get(r, "#808080") for r in ordered_regions}
    else:
        ordered_regions = sorted(df_all["Region"].unique())
        palette = _colorblind_palette(ordered_regions)

    band_order = ["Theta", "Slow gamma"]
    mem_order = ["remembered", "forgotten"]
    mem_label = {"remembered": "Remembered Trials", "forgotten": "Forgotten Trials"}

    fig_h = 11 if measure == "power" else 12
    fig_w = 14 if measure == "power" else max(14, 2.5 + 0.8 * len(ordered_regions))
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharey="row")
    measure_pretty = "Power" if measure == "power" else "Coherence"
    fig.suptitle(
        f"Baseline-Corrected {measure_pretty} Diff (Stim - No Stim) - "
        f"{phase.capitalize()} - Main Regions (no BLA)",
        fontsize=20, fontweight="bold", y=0.995,
    )

    rotate_x = measure != "power" or len(ordered_regions) > 6
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
            for roi in ordered_regions:
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

            for i, roi in enumerate(ordered_regions):
                test_rows.append({
                    "panel": "mainregions_nobla", "phase": phase, "measure": measure,
                    "memory": mem, "band": band, "Region": roi,
                    "n": ns[i], "mean_diff": means[i], "sem": sems[i],
                    "t": ts[i], "p": ps[i], "q_BH": qs[i],
                })

            x = np.arange(len(ordered_regions))
            bar_colors = [palette.get(roi, "#A0A0A0") for roi in ordered_regions]
            ax.bar(
                x, [means[i] if not np.isnan(means[i]) else 0 for i in range(len(x))],
                yerr=[sems[i] if not np.isnan(sems[i]) else 0 for i in range(len(x))],
                color=bar_colors, edgecolor="#4D4D4D", linewidth=1.2,
                width=0.72, capsize=3, ecolor="#4D4D4D", zorder=1,
            )

            rng = np.random.default_rng(7)
            for i, roi in enumerate(ordered_regions):
                vals = sub.loc[sub["Region"] == roi, "mean_power_diff"].dropna().to_numpy()
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.16, 0.16, len(vals))
                ax.scatter(np.full(len(vals), i, dtype=float) + jitter, vals,
                           c="#404040", s=28, alpha=0.7, edgecolors="none", zorder=3)

            ax.axhline(0, color="#4D4D4D", linewidth=1.0, zorder=0)
            ax.set_title(f"{band} - {mem_label[mem]}", fontsize=14, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(
                f"Baseline-Corrected {measure_pretty} Diff" if c == 0 else "",
                fontsize=12, fontweight="bold",
            )
            ax.tick_params(axis="y", labelsize=11)
            ax.set_xticks(x)
            xtick_labels = [f"{roi} *" if sig_flags[i] else roi
                             for i, roi in enumerate(ordered_regions)]
            base_fs = 13 if rotate_x else 14
            ax.set_xticklabels(
                xtick_labels,
                rotation=35 if rotate_x else 0,
                ha="right" if rotate_x else "center",
                fontsize=base_fs,
                fontweight="bold",
            )
            for i, lbl in enumerate(ax.get_xticklabels()):
                lbl.set_fontweight("bold")
                if sig_flags[i]:
                    lbl.set_color("red")
                    lbl.set_fontsize(base_fs + 2)

    caption = (
        "Significance: paired-samples t-test (scipy.stats.ttest_rel) on matched "
        "per-patient stim vs no-stim baseline-corrected band means. "
        "Benjamini-Hochberg FDR-corrected separately within each panel x band x memory "
        "cell (BLA excluded; stats recomputed on the no-BLA panel). "
        "Regions with q<.05 are marked with a * next to the region label."
    )
    fig.text(0.5, 0.012, caption, ha="center", va="bottom",
             fontsize=8, color="dimgray", wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_dir = OUTPUTS / f"{phase}_{measure if measure == 'power' else 'coherence'}" / "all"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"Bargraph_BaselineCorrected_{measure_pretty}Diff_byROI_"
             f"{phase}_combined_mainregions_nobla_2x2.png")
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [mainregions_nobla] Saved: {out_dir / fname}")

    if test_rows:
        tdf = pd.DataFrame(test_rows)
        csv_name = (f"PairedTTests_StimMinusNoStim_{phase}_{measure}_"
                    f"mainregions_nobla.csv")
        tdf.to_csv(out_dir / csv_name, index=False)
        print(f"  [mainregions_nobla] Saved t-test results: {out_dir / csv_name}")


def regen(measure: str):
    phase = "encoding"
    print(f"\n=== {phase} {measure}: mainregions (no BLA) 2x2 ===")
    mod, data = _base.load_combined_data(measure, phase)

    df_rem = _base._compute_mem_diff_df(mod, "rem", data, measure)
    df_forg = _base._compute_mem_diff_df(mod, "forg", data, measure)
    if df_rem.empty and df_forg.empty:
        print("  No data, skipping.")
        return
    df_rem["memory_cond"] = "remembered"
    df_forg["memory_cond"] = "forgotten"
    df_all = pd.concat([df_rem, df_forg], ignore_index=True)
    df_all = df_all[~df_all["Region"].apply(_base._excluded_region)]
    if df_all.empty:
        print("  All rows excluded.")
        return

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
        df_cond_all = df_cond_all[~df_cond_all["Region"].apply(_base._excluded_region)]

    df_main_nobla = df_all[df_all["Region"].apply(_is_main_region_nobla)].copy()
    df_cond_main_nobla = (
        df_cond_all[df_cond_all["Region"].apply(_is_main_region_nobla)].copy()
        if not df_cond_all.empty else pd.DataFrame()
    )

    _plot_nobla(df_main_nobla, df_cond_main_nobla, mod, measure, phase)


if __name__ == "__main__":
    for measure in ("power", "coherence"):
        regen(measure)
    print("\nDone.")
