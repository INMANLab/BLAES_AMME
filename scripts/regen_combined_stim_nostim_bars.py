#!/usr/bin/env python
"""Build combined 2x2 stim-vs-nostim bar-graph figures (theta/slow-gamma x
Remembered/Forgotten) for power and coherence, retrieval and encoding.

Layout per figure:
  [1,1] Theta - Remembered      [1,2] Theta - Forgotten
  [2,1] Slow gamma - Remembered [2,2] Slow gamma - Forgotten

Each cell is the existing colored-bars + patient-dots stim-vs-nostim diff bar
graph (mean across patients of stim_band - nostim_band per patient). MTL,
PHG, and any pair containing MTL or PHG are removed. Other settings preserved.
"""
from __future__ import annotations

import re
import sys
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as _stats

# Display-label convention: ALLHPC -> "HPC"; standalone "HPC" (subiculum) -> "SUB".
# Substring-safe via lookarounds; pair names like "ALLHPC_EC" map to "HPC_EC".
_ALLHPC_RE = re.compile(r"(?<![A-Za-z])ALLHPC(?![A-Za-z])")
_HPC_RE = re.compile(r"(?<![A-Za-z])HPC(?![A-Za-z])")
_MARKER = "\x00MACROHIPP\x00"


def _display_label(text):
    if text is None:
        return text
    s = str(text)
    s = _ALLHPC_RE.sub(_MARKER, s)
    s = _HPC_RE.sub("SUB", s)
    s = s.replace(_MARKER, "HPC")
    return s

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/"
            "InmanLab/BLAES_data/dissertation/AMME_BLAES")
OUTPUTS = ROOT / "OUTPUTS"

EXCLUDE_TOKENS = ("MTL", "PHG", "PNAS")

MAIN_REGIONS = {"ALLHPC", "BLA", "EC", "PRC"}
HIPP_SUBREGIONS = {"DG", "CA", "HPC"}


def _excluded_region(r: str) -> bool:
    return any(tok in r for tok in EXCLUDE_TOKENS)


def _region_tokens(r: str):
    return [t for t in r.split("_") if t]


def _is_main_region(r: str) -> bool:
    toks = _region_tokens(r)
    return bool(toks) and all(t in MAIN_REGIONS for t in toks)


def _is_hipp_subregion(r: str) -> bool:
    toks = _region_tokens(r)
    return any(t in HIPP_SUBREGIONS for t in toks)


def _bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1, dtype=float)
    sorted_p = pvals[order]
    sorted_adj = np.minimum.accumulate((sorted_p * n / ranks)[::-1])[::-1]
    sorted_adj = np.minimum(sorted_adj, 1.0)
    adj = np.empty(n, dtype=float)
    adj[order] = sorted_adj
    return adj


def _sig_marker(q):
    if q is None or np.isnan(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def _merge(*dicts):
    merged = {}
    for d in dicts:
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                merged.setdefault(k, {}).update(v)
    return merged


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_combined_data(measure: str, phase: str):
    """Load and merge BLAES + AMME for the 'all' cohort using the existing loader."""
    if measure == "power":
        path = ROOT / f"{phase}_code" / f"combined_{phase}_power.py"
    else:
        path = ROOT / f"{phase}_code" / f"combined_{phase}_coherence.py"
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT))
    mod = _load_module(f"_{measure}_{phase}", path)

    if phase == "retrieval":
        if measure == "power":
            blaes = mod.augment_with_power_composites(mod.load_blaes_retrieval())
            amme = mod.augment_with_power_composites(mod.load_amme_retrieval())
        else:
            blaes = mod.augment_with_allhpc_pair_composites(mod.load_blaes_retrieval())
            amme = mod.augment_with_allhpc_pair_composites(mod.load_amme_retrieval())
    else:
        if measure == "power":
            blaes = mod.augment_with_power_composites(mod.load_blaes_encoding())
            amme = mod.augment_with_power_composites(mod.load_amme_encoding())
        else:
            blaes = mod.augment_with_allhpc_pair_composites(mod.load_blaes_encoding())
            amme = mod.augment_with_allhpc_pair_composites(mod.load_amme_encoding())

    all_data = {}
    keys = [
        "group_all_power", "stim_power", "nostim_power",
        "bc_stim", "bc_nostim",
        "stim_rem", "stim_forg", "nostim_rem", "nostim_forg",
        "bc_stim_rem", "bc_stim_forg", "bc_nostim_rem", "bc_nostim_forg",
    ]
    for k in keys:
        all_data[k] = _merge(blaes.get(k, {}), amme.get(k, {}))
    fp_b = blaes.get("freqs_post"); fp_a = amme.get("freqs_post")
    all_data["freqs_post"] = fp_b if fp_b is not None else fp_a
    fd_b = blaes.get("freqs_diff"); fd_a = amme.get("freqs_diff")
    all_data["freqs_diff"] = fd_b if fd_b is not None else fd_a
    if measure != "power":
        all_data = mod.augment_with_allhpc_pair_composites(mod.canonicalize_coherence_data(all_data))
    return mod, all_data


def _compute_mem_diff_df(mod, mem_key: str, data, measure: str) -> pd.DataFrame:
    """Per-patient stim-nostim band diff for a memory condition."""
    freqs = data["freqs_diff"]
    if freqs is None:
        return pd.DataFrame()
    sd = data.get(f"bc_stim_{mem_key}", {})
    nd = data.get(f"bc_nostim_{mem_key}", {})
    if not sd or not nd:
        return pd.DataFrame()
    df = mod.compute_diff_df(sd, nd, freqs, mod.POWER_RANGES)
    return df


def _compute_mem_cond_df(mod, mem_key: str, data) -> pd.DataFrame:
    """Per-patient matched stim and nostim band means for a memory condition."""
    freqs = data["freqs_diff"]
    if freqs is None:
        return pd.DataFrame()
    sd = data.get(f"bc_stim_{mem_key}", {})
    nd = data.get(f"bc_nostim_{mem_key}", {})
    if not sd or not nd:
        return pd.DataFrame()
    return mod.compute_condition_band_df(sd, nd, freqs, mod.POWER_RANGES)


def _plot_one(df_all, mod, measure: str, phase: str, region_set_label: str,
              df_cond_all: pd.DataFrame | None = None):
    if df_all.empty:
        print(f"  [{region_set_label}] No regions match filter, skipping.")
        return

    if measure == "power":
        ordered_regions = [r for r in mod.ROI_COLORS_BAR if r in df_all["Region"].unique()]
        ordered_regions += sorted(r for r in df_all["Region"].unique() if r not in ordered_regions)
        palette = {r: mod.ROI_COLORS_BAR.get(r, "#808080") for r in ordered_regions}
    else:
        ordered_regions = sorted(df_all["Region"].unique())
        palette = mod.make_roi_color_map(ordered_regions)

    # Encoding coherence: husl assigns a green to one of the mainregion pairs;
    # swap to seaborn 'colorblind' minus its green entry so all pairs stay
    # distinguishable. Retrieval coherence keeps husl.
    if measure != "power" and phase == "encoding":
        cb_no_green = ['#0173b2', '#de8f05', '#d55e00', '#cc78bc',
                       '#ca9161', '#fbafe4', '#56b4e9', '#949494', '#ece133']
        palette = {r: cb_no_green[i % len(cb_no_green)]
                   for i, r in enumerate(ordered_regions)}

    band_order = ["Theta", "Slow gamma"]
    mem_order = ["remembered", "forgotten"]
    mem_label = {"remembered": "Remembered Trials", "forgotten": "Forgotten Trials"}

    fig_h = 11 if measure == "power" else 12
    fig_w = 14 if measure == "power" else max(16, 2.5 + 0.65 * len(ordered_regions))
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h),
                             sharey="row")
    measure_pretty = "Power" if measure == "power" else "Coherence"
    region_set_pretty = {"mainregions": "Main Regions",
                         "hippsubregions": "Hippocampal Subregions"}[region_set_label]
    fig.suptitle(
        f"Baseline-Corrected {measure_pretty} Diff (Stim - No Stim) - "
        f"{phase.capitalize()} - {region_set_pretty}",
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

            # Paired-samples t-test on matched per-patient stim vs no-stim band means
            # (scipy.stats.ttest_rel). Bars/SEMs still reflect the per-patient
            # stim - no-stim difference. BH-FDR family = this single
            # (panel x band x memory) cell; theta and slow gamma are separate
            # families; remembered and forgotten are separate families;
            # mainregions and hippsubregions are separate families.
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
                qs[valid] = _bh_fdr(ps_arr[valid])
            sig_flags = [(not np.isnan(qs[i]) and qs[i] < 0.05) for i in range(len(qs))]

            for i, roi in enumerate(ordered_regions):
                test_rows.append({
                    "panel": region_set_label, "phase": phase, "measure": measure,
                    "memory": mem, "band": band, "Region": roi,
                    "n": ns[i], "mean_diff": means[i], "sem": sems[i],
                    "t": ts[i], "p": ps[i], "q_BH": qs[i],
                })

            # Bars (matplotlib so we can set per-bar edge), then patient dots
            x = np.arange(len(ordered_regions))
            bar_colors = [palette.get(roi, "#A0A0A0") for roi in ordered_regions]
            bars = ax.bar(
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
            xtick_labels = [
                f"{_display_label(roi)} *" if sig_flags[i] else _display_label(roi)
                for i, roi in enumerate(ordered_regions)
            ]
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
        "cell (theta and slow gamma are separate families; remembered and forgotten "
        "are separate families; mainregions and hippsubregions are separate families). "
        "Regions with q<.05 are marked with a * next to the region label."
    )
    fig.text(0.5, 0.012, caption, ha="center", va="bottom",
             fontsize=8, color="dimgray", wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_dir = OUTPUTS / f"{phase}_{measure if measure == 'power' else 'coherence'}" / "all"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"Bargraph_BaselineCorrected_{measure_pretty}Diff_byROI_"
             f"{phase}_combined_{region_set_label}_2x2.png")
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{region_set_label}] Saved: {out_dir / fname}")

    if test_rows:
        tdf = pd.DataFrame(test_rows)
        csv_name = (f"PairedTTests_StimMinusNoStim_{phase}_{measure}_"
                    f"{region_set_label}.csv")
        tdf.to_csv(out_dir / csv_name, index=False)
        print(f"  [{region_set_label}] Saved t-test results: {out_dir / csv_name}")
    return test_rows


def plot_combined(measure: str, phase: str):
    print(f"\n=== {phase} {measure}: combined stim-nostim bargraph 2x2 (split) ===")
    mod, data = load_combined_data(measure, phase)

    df_rem = _compute_mem_diff_df(mod, "rem", data, measure)
    df_forg = _compute_mem_diff_df(mod, "forg", data, measure)
    if df_rem.empty and df_forg.empty:
        print("  No data, skipping.")
        return
    df_rem["memory_cond"] = "remembered"
    df_forg["memory_cond"] = "forgotten"
    df_all = pd.concat([df_rem, df_forg], ignore_index=True)
    df_all = df_all[~df_all["Region"].apply(_excluded_region)]
    # Per user request: drop BLA-containing regions from all encoding 2x2
    # bargraphs (power + coherence, main + hipp). Retrieval keeps BLA.
    if phase == "encoding":
        df_all = df_all[~df_all["Region"].apply(lambda r: "BLA" in _region_tokens(r))]
    if df_all.empty:
        print("  All rows excluded.")
        return

    df_cond_rem = _compute_mem_cond_df(mod, "rem", data)
    df_cond_forg = _compute_mem_cond_df(mod, "forg", data)
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
        df_cond_all = df_cond_all[~df_cond_all["Region"].apply(_excluded_region)]
    if phase == "encoding" and not df_cond_all.empty:
        df_cond_all = df_cond_all[~df_cond_all["Region"].apply(
            lambda r: "BLA" in _region_tokens(r))]

    df_main = df_all[df_all["Region"].apply(_is_main_region)].copy()
    df_hipp = df_all[df_all["Region"].apply(_is_hipp_subregion)].copy()
    df_cond_main = (df_cond_all[df_cond_all["Region"].apply(_is_main_region)].copy()
                    if not df_cond_all.empty else pd.DataFrame())
    df_cond_hipp = (df_cond_all[df_cond_all["Region"].apply(_is_hipp_subregion)].copy()
                    if not df_cond_all.empty else pd.DataFrame())

    rows_main = _plot_one(df_main, mod, measure, phase, "mainregions",
                           df_cond_all=df_cond_main) or []
    rows_hipp = _plot_one(df_hipp, mod, measure, phase, "hippsubregions",
                           df_cond_all=df_cond_hipp) or []
    return rows_main + rows_hipp


def _print_tests(all_rows):
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    fmt = df.copy()
    fmt["mean_diff"] = fmt["mean_diff"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "NaN")
    fmt["sem"] = fmt["sem"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    fmt["t"] = fmt["t"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "NaN")
    fmt["p"] = fmt["p"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    fmt["q_BH"] = fmt["q_BH"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    for (measure, phase), block in fmt.groupby(["measure", "phase"], sort=False):
        print(f"\n############### {phase.upper()} {measure.upper()} ###############")
        for (panel, mem, band), cell in block.groupby(["panel", "memory", "band"], sort=False):
            print(f"\n--- [{panel}] {mem} | {band} ---")
            print(cell.drop(columns=["panel", "memory", "band", "phase", "measure"])
                      .to_string(index=False))


if __name__ == "__main__":
    all_rows = []
    for measure in ("power", "coherence"):
        for phase in ("retrieval", "encoding"):
            rows = plot_combined(measure, phase) or []
            all_rows.extend(rows)
    _print_tests(all_rows)
    print("\nDone.")
