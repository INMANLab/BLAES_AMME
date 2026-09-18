#!/usr/bin/env python
"""Re-render permutation-test panel figures using saved *_cluster_stats.csv
files, WITHOUT re-running any cluster permutation tests.

The figure-label convention (ALLHPC -> "HPC" in figures, "HPC" anatomical -> "SUB")
is already applied inside _permutation_helpers.plot_region_panel_dual() via
pretty_in_text(), so simply re-calling plot_region_panel_dual() with the saved
clusters + freshly built X_A/X_B paired arrays regenerates correctly-labelled
panels. The clusters' p_value / direction / family / freq range are read
verbatim from the saved cluster_stats.csv -- no statistics are recomputed.

Scope and limitations:
  - Supports both directory layouts found under OUTPUTS/PermutationOutputsAlireza/:
      <group>/<test>/<modality>/{modality}_cluster_stats.csv          (flat)
      <group>/<region_group>/<test>/<modality>/{modality}_cluster_stats.csv (nested)
    "Flat" groups are rendered with the canonical fixed layout (allregions).
    Nested groups with region_group == "allregions" are also rendered.
  - REGION_GROUP figures (mainregions / mainregions_nobla / hippsubregions /
    hippsubregions_nobla) use a region-by-region compact grid in the original
    run scripts. For simplicity this re-labelling pass only handles the
    "allregions" layout and SKIPS region-group-only figures (they're listed
    in the dry-run output and the run output for transparency).
  - null_distribution.png panels are *not* regenerated -- the raw permutation
    null arrays are not on disk. The script prints "skipping null distribution"
    once per group/test/modality.
  - Re-orientation: when matching freq_lo_hz / freq_hi_hz from the saved CSV
    to the live freq grid, start_idx / end_idx are inferred by argmin distance.

Usage:
  python relabel_permutation_figures.py --dry-run
  python relabel_permutation_figures.py --dry-run --group encoding_balancedtrials_no_bla
  python relabel_permutation_figures.py            # actually writes figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Defer matplotlib import until needed (so --dry-run is fast and doesn't pull GUI).

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
PERM_DIR = ROOT / "outputs" / "PermutationOutputsAlireza"
CSV_DIR = ROOT / "outputs" / "csvs"
CSV_ONESEC_DIR = CSV_DIR / "onesec"

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _permutation_helpers import (  # noqa: E402
    add_allhpc_region,
    get_panel_position,
    panel_grid_for,
    plot_region_panel_dual,
)

# Skip bandwise output trees -- those figures come from a different plotter.
SKIP_GROUP_PREFIXES = ("encoding_bandwise", "retreival_bandwise")

MIN_SUBJECTS = 4
MIN_TRIALS_BALANCED = 10
EXCLUDE_TOKENS_BASE = ("PHG",)
EXCLUDE_PAIRS = {
    frozenset({"CA", "DG"}),
    frozenset({"CA", "HPC"}),
    frozenset({"DG", "HPC"}),
    frozenset({"ALLHPC", "CA"}),
    frozenset({"ALLHPC", "DG"}),
    frozenset({"ALLHPC", "HPC"}),
}

MODALITY_YLABEL = {
    "power":     "Power (dB, baseline-subtracted)",
    "coherence": "Coherence (baseline-subtracted)",
    "pac":       "PAC (baseline-subtracted)",
}


def _decimals_needed(val):
    if val == 0 or not np.isfinite(val):
        return 0
    s = f"{val:.10f}".rstrip("0").rstrip(".")
    if "." in s:
        return min(4, len(s.split(".")[1]))
    return 0


def _unify_y_decimals(axes):
    """Make every axis in the figure use the same number of y-tick decimals."""
    if not axes:
        return
    from matplotlib.ticker import FormatStrFormatter
    max_dec = 0
    for ax in axes:
        for t in ax.get_yticks():
            max_dec = max(max_dec, _decimals_needed(float(t)))
    if max_dec == 0:
        return
    fmt = FormatStrFormatter(f"%.{max_dec}f")
    for ax in axes:
        ax.yaxis.set_major_formatter(fmt)


def _seed_groups(panel_results):
    """Group pair-modality panels by seed (first region in 'A_B'). Returns
    (seeds_ordered, panels_by_seed, max_partners)."""
    def _seed(p):
        return str(p["region"]).split("_", 1)[0]
    def _partner(p):
        parts = str(p["region"]).split("_", 1)
        return parts[1] if len(parts) > 1 else ""
    def _seed_prio(s):
        return {"ALLHPC": 0, "HPC": 0, "BLA": 1}.get(s, 2)

    by_seed = {}
    for p in panel_results:
        by_seed.setdefault(_seed(p), []).append(p)
    for s in by_seed:
        by_seed[s].sort(key=_partner)
    seeds_ordered = sorted(by_seed.keys(), key=lambda s: (_seed_prio(s), s))
    max_partners = max(len(by_seed[s]) for s in seeds_ordered)
    return seeds_ordered, by_seed, max_partners


# --- group-name parsing -------------------------------------------------------

def parse_group(group_name: str) -> dict | None:
    """Decode a top-level group directory name into phase / filter flags.

    Returns None for unrecognised groups (e.g. bandwise outputs)."""
    if any(group_name.startswith(p) for p in SKIP_GROUP_PREFIXES):
        return None
    name = group_name
    if name.startswith("retreival"):
        phase = "retrieval"
        suffix = name[len("retreival"):]
    elif name.startswith("encoding"):
        phase = "encoding"
        suffix = name[len("encoding"):]
    else:
        return None
    parts = [p for p in suffix.split("_") if p]
    onesec = "onesec" in parts
    balanced = "balancedtrials" in parts
    # "no_bla" appears as the two tokens "no" + "bla" because of the underscore.
    no_bla = False
    for i, p in enumerate(parts):
        if p == "no" and i + 1 < len(parts) and parts[i + 1] == "bla":
            no_bla = True
            break
    # Any other tokens after stripping known ones?
    known = {"onesec", "balancedtrials", "no", "bla"}
    unexpected = [p for p in parts if p not in known]
    if unexpected:
        # Unknown suffix tokens -- still try to proceed (treat as encoding base).
        pass
    return {"phase": phase, "onesec": onesec, "balanced": balanced,
            "no_bla": no_bla, "raw": group_name}


def input_csv_for(meta: dict, modality: str) -> Path:
    """Return expected input CSV path for the (phase, onesec, modality)."""
    phase = meta["phase"]
    if meta["onesec"]:
        if phase == "encoding":
            fname = f"combined_encoding_{modality}_onesec_mlmr_input.csv"
        else:
            fname = f"combined_retrieval_{modality}_onesec_mlmr_input.csv"
        return CSV_ONESEC_DIR / fname
    if phase == "encoding":
        # Encoding uses the AMME timing-filtered CSV by default
        # (matching run_permutation_encoding_*.py).
        fname = f"combined_encoding_{modality}_amme_timing_filtered_mlmr_input.csv"
    else:
        fname = f"combined_retrieval_{modality}_all_mlmr_input.csv"
    return CSV_DIR / fname


# --- data loading replicating run_permutation_*.py ---------------------------

def _balanced_keep(df: pd.DataFrame) -> set[str]:
    counts = (df.groupby("Patient")
                .agg(n_rem=("yes_or_no", lambda s: (s == "yes").sum()),
                     n_forg=("yes_or_no", lambda s: (s == "no").sum()),
                     n_regions=("Region", lambda s: s.nunique()))
                .reset_index())
    counts["rem_per_reg"] = (counts["n_rem"] / counts["n_regions"]).round()
    counts["forg_per_reg"] = (counts["n_forg"] / counts["n_regions"]).round()
    return set(counts.loc[
        (counts["rem_per_reg"] >= MIN_TRIALS_BALANCED)
        & (counts["forg_per_reg"] >= MIN_TRIALS_BALANCED),
        "Patient"].tolist())


def load_modality(meta: dict, modality: str, test: str):
    """Replicate the run_permutation_*.py load_modality logic."""
    csv_path = input_csv_for(meta, modality)
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"
    df = df[df["yes_or_no"].isin(["yes", "no"])]
    if test == "stim_nostim":
        df = df[df["trial_type"].notna()]
        df["StimCond"] = np.where(df["trial_type"] == "nostim", "Nostim", "Stim")
    else:  # endogenous_memory
        df = df[df["trial_type"] == "nostim"]
    df["Memory"] = np.where(df["yes_or_no"] == "yes", "Remembered", "Forgotten")

    if meta["balanced"]:
        keep = _balanced_keep(df)
        df = df[df["Patient"].isin(keep)]

    freq_cols = [c for c in df.columns if "_Freq_" in c]
    diff_cols = [c for c in freq_cols if c.startswith("diff_")]
    freqs = np.array([float(c.split("_Freq_")[1]) for c in diff_cols])
    return df, diff_cols, freqs


def per_subject_means(df: pd.DataFrame, diff_cols: list[str], test: str):
    if test == "stim_nostim":
        return df.groupby(["Patient", "Region", "StimCond", "Memory"],
                          as_index=False)[diff_cols].mean()
    return df.groupby(["Patient", "Region", "Memory"],
                      as_index=False)[diff_cols].mean()


def paired_arrays_stim_nostim(grouped, region, memory, diff_cols):
    sub = grouped[(grouped["Region"] == region)
                  & (grouped["Memory"] == memory)]
    pivoted = sub.pivot_table(index="Patient", columns="StimCond",
                              values=diff_cols, aggfunc="mean")
    if pivoted.empty:
        return None, None
    has_both = pivoted.dropna(how="any").index.tolist()
    if len(has_both) < MIN_SUBJECTS:
        return None, None
    pivoted = pivoted.loc[has_both]
    lvl1 = pivoted.columns.get_level_values(1)
    if "Stim" not in lvl1 or "Nostim" not in lvl1:
        return None, None
    X_S = np.column_stack([pivoted[(c, "Stim")].values for c in diff_cols])
    X_N = np.column_stack([pivoted[(c, "Nostim")].values for c in diff_cols])
    return X_S, X_N


def paired_arrays_endogenous(grouped, region, diff_cols):
    sub = grouped[grouped["Region"] == region]
    pivoted = sub.pivot_table(index="Patient", columns="Memory",
                              values=diff_cols, aggfunc="mean")
    if pivoted.empty:
        return None, None
    has_both = pivoted.dropna(how="any").index.tolist()
    if len(has_both) < MIN_SUBJECTS:
        return None, None
    pivoted = pivoted.loc[has_both]
    lvl1 = pivoted.columns.get_level_values(1)
    if "Remembered" not in lvl1 or "Forgotten" not in lvl1:
        return None, None
    X_R = np.column_stack([pivoted[(c, "Remembered")].values for c in diff_cols])
    X_F = np.column_stack([pivoted[(c, "Forgotten")].values for c in diff_cols])
    return X_R, X_F


def is_excluded_region(r: str, no_bla: bool) -> bool:
    exclude_tokens = ("PHG", "BLA") if no_bla else EXCLUDE_TOKENS_BASE
    if any(tok in r for tok in exclude_tokens):
        return True
    parts = r.split("_")
    if len(parts) == 2 and frozenset(parts) in EXCLUDE_PAIRS:
        return True
    return False


# --- cluster_stats -> plot dict conversion -----------------------------------

def clusters_for_panel(cluster_df: pd.DataFrame, freqs: np.ndarray,
                       region: str, memory: str | None) -> list[dict]:
    """Convert cluster_stats rows for one panel into plot_region_panel_dual()'s
    expected dict format. memory=None for endogenous-memory CSVs."""
    sub = cluster_df[cluster_df["region"] == region]
    if memory is not None and "memory" in cluster_df.columns:
        sub = sub[sub["memory"] == memory]
    clusters = []
    for _, row in sub.iterrows():
        i0 = int(np.argmin(np.abs(freqs - float(row["freq_lo_hz"]))))
        i1 = int(np.argmin(np.abs(freqs - float(row["freq_hi_hz"]))))
        if i1 < i0:
            i0, i1 = i1, i0
        clusters.append({
            "family": str(row["family"]),
            "confirmatory": bool(row["confirmatory"]),
            "start_idx": i0,
            "end_idx": i1,
            "n_freqs": int(row["n_freqs"]),
            "direction": str(row["direction"]),
            "cluster_mass": float(row["cluster_mass_t"]),
            "p_value": float(row["p_value"]),
        })
    return clusters


# --- discovery ----------------------------------------------------------------

def discover_jobs(filter_group: str | None = None) -> list[dict]:
    """Walk PERM_DIR for *_cluster_stats.csv and build a job per (group, region_group,
    test, modality). Each job points at the cluster_stats CSV and the figure
    output path."""
    jobs = []
    if not PERM_DIR.is_dir():
        return jobs
    for stats_path in PERM_DIR.rglob("*_cluster_stats.csv"):
        rel = stats_path.relative_to(PERM_DIR)
        parts = rel.parts
        # Expected: <group>/<test>/<modality>/<modality>_cluster_stats.csv (4 parts)
        # or:       <group>/<region_group>/<test>/<modality>/<modality>_cluster_stats.csv (5 parts)
        if len(parts) == 4:
            group, test, modality_dir, _file = parts
            region_group = "allregions"  # implicit
            nested = False
        elif len(parts) == 5:
            group, region_group, test, modality_dir, _file = parts
            nested = True
        else:
            continue
        if filter_group is not None and group != filter_group:
            continue
        if any(group.startswith(p) for p in SKIP_GROUP_PREFIXES):
            continue
        meta = parse_group(group)
        if meta is None:
            continue
        if test not in ("stim_nostim", "endogenous_memory"):
            continue
        modality = modality_dir.lower()
        if modality not in ("power", "coherence", "pac"):
            continue
        fig_path = stats_path.parent / f"{modality}_{test}.png"
        jobs.append({
            "stats_path": stats_path,
            "fig_path": fig_path,
            "group": group,
            "meta": meta,
            "region_group": region_group,
            "nested": nested,
            "test": test,
            "modality": modality,
        })
    jobs.sort(key=lambda j: (j["group"], j["region_group"], j["test"], j["modality"]))
    return jobs


# --- rendering ----------------------------------------------------------------

def render_job(job: dict, dry_run: bool) -> str:
    """Render (or report) one figure. Returns a status string."""
    stats_path = job["stats_path"]
    fig_path = job["fig_path"]
    meta = job["meta"]
    test = job["test"]
    modality = job["modality"]
    region_group = job["region_group"]

    csv_path = input_csv_for(meta, modality)
    if not csv_path.is_file():
        return f"skipped (no input csv: {csv_path.name}): {fig_path}"

    if dry_run:
        return f"WOULD WRITE {fig_path}  (from {stats_path.name}; input={csv_path.name})"

    # --- actually render ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded = load_modality(meta, modality, test)
    if loaded is None:
        return f"skipped (input csv unreadable): {fig_path}"
    df, diff_cols, freqs = loaded
    if df.empty:
        return f"skipped (no trials in input csv): {fig_path}"

    cluster_df = pd.read_csv(stats_path)
    if cluster_df.empty:
        return f"skipped (empty cluster_stats): {fig_path}"

    grouped = per_subject_means(df, diff_cols, test)
    cond_col = ["StimCond", "Memory"] if test == "stim_nostim" else "Memory"
    grouped = add_allhpc_region(grouped, diff_cols, cond_col, modality)

    panel_xlim = (30.0, float(freqs.max())) if modality == "pac" else None

    panel_results = []  # list of (region[, memory], X_A, X_B, clusters)
    regions = sorted(cluster_df["region"].dropna().unique().tolist())
    regions = [r for r in regions if not is_excluded_region(r, meta["no_bla"])]

    if test == "stim_nostim":
        for region in regions:
            for memory in ("Remembered", "Forgotten"):
                X_S, X_N = paired_arrays_stim_nostim(grouped, region, memory, diff_cols)
                if X_S is None:
                    continue
                clusters = clusters_for_panel(cluster_df, freqs, region, memory)
                panel_results.append({"region": region, "memory": memory,
                                      "X_A": X_S, "X_B": X_N,
                                      "clusters": clusters})
    else:
        for region in regions:
            X_R, X_F = paired_arrays_endogenous(grouped, region, diff_cols)
            if X_R is None:
                continue
            clusters = clusters_for_panel(cluster_df, freqs, region, None)
            panel_results.append({"region": region, "memory": None,
                                  "X_A": X_R, "X_B": X_F,
                                  "clusters": clusters})

    if not panel_results:
        return f"skipped (no renderable panels): {fig_path}"

    if test == "stim_nostim":
        nrows, ncols = panel_grid_for(modality, n_memory_cells=2)
        line_color_A = "#d62728"
        line_color_B = "#1f77b4"
        label_A = "Stim"
        label_B = "No stim"
        fill_color = "#b8860b"
        title_prefix = (
            f"Encoding Stim vs No-stim {modality.capitalize()} by Memory"
            if meta["phase"] == "encoding"
            else f"Stim vs No-stim {modality.capitalize()} by Memory"
        )
    else:
        nrows, ncols = panel_grid_for(modality)
        line_color_A = "#7b2d8e"
        line_color_B = "#daa520"
        label_A = "Remembered"
        label_B = "Forgotten"
        fill_color = "#c0392b"
        title_prefix = (
            f"Encoding Endogenous Memory {modality.capitalize()}"
            if meta["phase"] == "encoding"
            else f"Endogenous Memory {modality.capitalize()}"
        )

    # Encoding-style figure sizing is slightly different from retrieval -- match
    # whichever run-script produced this group.
    if meta["phase"] == "encoding":
        fig_h_factor = lambda nr: nr * 3.0 + 1.0
    else:
        fig_h_factor = lambda nr: nr * 3.2

    # --- pick layout strategy ---
    use_seed_grouping = (
        region_group != "allregions"
        and test == "endogenous_memory"
        and modality == "pac"
        and all("_" in str(p["region"]) for p in panel_results)
    )

    placed_axes = []
    if use_seed_grouping:
        # Seed-grouped layout: one row per seed region (HPC first, then BLA,
        # then alphabetical), partners centered within each row using explicit
        # axes positions so a single panel in a row sits exactly between the
        # full-row panels above it.
        seeds_ordered, panels_by_seed, max_partners = _seed_groups(panel_results)
        nrows_layout = len(seeds_ordered)
        ncols_layout = max_partners
        fig = plt.figure(
            figsize=(ncols_layout * 4.0, fig_h_factor(nrows_layout))
        )
        # Figure margins (in figure-fraction coords) and inter-panel gaps.
        # top_m leaves room for suptitle + legend; left_m for supylabel; bottom_m for supxlabel.
        # Encoding needs a wider left margin so PAC tick labels clear the supylabel.
        left_m_default = 0.18 if meta["phase"] == "encoding" else 0.10
        left_m, right_m, top_m, bottom_m = left_m_default, 0.99, 0.88, 0.10
        wspace_frac = 0.22  # gap as fraction of panel width
        hspace_frac = 0.40  # vertical gap between rows as fraction of panel height
        avail_w = right_m - left_m
        avail_h = top_m - bottom_m
        # Panel width: ncols panels + (ncols-1) gaps, each gap = wspace_frac*panel_w.
        panel_w = avail_w / (ncols_layout + (ncols_layout - 1) * wspace_frac)
        panel_h = avail_h / (nrows_layout + (nrows_layout - 1) * hspace_frac)
        gap_w = wspace_frac * panel_w
        gap_h = hspace_frac * panel_h
        for r, seed in enumerate(seeds_ordered):
            row_panels = panels_by_seed[seed]
            k = len(row_panels)
            # x-position of the leftmost panel in this row so the row is centered.
            row_width = k * panel_w + (k - 1) * gap_w
            row_left = left_m + (avail_w - row_width) / 2.0
            for i, panel in enumerate(row_panels):
                x = row_left + i * (panel_w + gap_w)
                y = top_m - (r + 1) * panel_h - r * gap_h
                ax = fig.add_axes([x, y, panel_w, panel_h])
                plot_region_panel_dual(
                    ax, freqs, panel["X_A"], panel["X_B"],
                    panel["clusters"], panel["region"],
                    line_color_A=line_color_A, line_color_B=line_color_B,
                    label_A=label_A, label_B=label_B,
                    fill_color=fill_color, xlim=panel_xlim,
                )
                placed_axes.append(ax)
    else:
        if region_group == "allregions":
            nrows_layout, ncols_layout = nrows, ncols
        elif test == "stim_nostim":
            regions_in_order = sorted({p["region"] for p in panel_results})
            nR = len(regions_in_order)
            n_panels_total = nR * 2
            best = None
            for bpr in range(1, nR + 1):
                nr = int(np.ceil(nR / bpr))
                nc = bpr * 2
                empty = nr * nc - n_panels_total
                aspect = max(nr, nc) / max(min(nr, nc), 1)
                score = (empty, aspect, -nc)
                if best is None or score < best[0]:
                    best = (score, nr, nc, bpr)
            _, nrows_layout, ncols_layout, blocks_per_row = best
        else:
            nP = len(panel_results)
            best = None
            for nc in range(1, nP + 1):
                nr = int(np.ceil(nP / nc))
                empty = nr * nc - nP
                aspect = max(nr, nc) / max(min(nr, nc), 1)
                score = (empty, aspect, -nc)
                if best is None or score < best[0]:
                    best = (score, nr, nc)
            _, nrows_layout, ncols_layout = best

        fig, axes = plt.subplots(nrows_layout, ncols_layout,
                                 figsize=(ncols_layout * 4.0, fig_h_factor(nrows_layout)),
                                 squeeze=False)
        for r in range(nrows_layout):
            for c in range(ncols_layout):
                axes[r][c].set_visible(False)

        if region_group == "allregions":
            for panel in panel_results:
                pos = get_panel_position(modality, panel["region"])
                if pos is None:
                    continue
                slot_r, slot_c = pos
                if test == "stim_nostim":
                    cell_col = slot_c * 2 + (0 if panel["memory"] == "Remembered" else 1)
                    title = f'{panel["region"]} - {panel["memory"]}'
                else:
                    cell_col = slot_c
                    title = panel["region"]
                ax = axes[slot_r][cell_col]
                ax.set_visible(True)
                plot_region_panel_dual(
                    ax, freqs, panel["X_A"], panel["X_B"],
                    panel["clusters"],
                    title,
                    line_color_A=line_color_A, line_color_B=line_color_B,
                    label_A=label_A, label_B=label_B,
                    fill_color=fill_color, xlim=panel_xlim,
                )
                placed_axes.append(ax)
        elif test == "stim_nostim":
            panel_lookup = {(p["region"], p["memory"]): p for p in panel_results}
            for i, region in enumerate(regions_in_order):
                block_row = i // blocks_per_row
                block_col_start = (i % blocks_per_row) * 2
                for offset, memory in enumerate(("Remembered", "Forgotten")):
                    cell_col = block_col_start + offset
                    panel = panel_lookup.get((region, memory))
                    if panel is None:
                        continue
                    ax = axes[block_row][cell_col]
                    ax.set_visible(True)
                    plot_region_panel_dual(
                        ax, freqs, panel["X_A"], panel["X_B"],
                        panel["clusters"],
                        f'{region} - {memory}',
                        line_color_A=line_color_A, line_color_B=line_color_B,
                        label_A=label_A, label_B=label_B,
                        fill_color=fill_color, xlim=panel_xlim,
                    )
                    placed_axes.append(ax)
        else:
            for i, panel in enumerate(panel_results):
                r, c = i // ncols_layout, i % ncols_layout
                ax = axes[r][c]
                ax.set_visible(True)
                plot_region_panel_dual(
                    ax, freqs, panel["X_A"], panel["X_B"],
                    panel["clusters"], panel["region"],
                    line_color_A=line_color_A, line_color_B=line_color_B,
                    label_A=label_A, label_B=label_B,
                    fill_color=fill_color, xlim=panel_xlim,
                )
                placed_axes.append(ax)

    if not placed_axes:
        plt.close(fig)
        return f"skipped (no panels matched layout slots): {fig_path}"

    _unify_y_decimals(placed_axes)

    flags = []
    if meta["no_bla"]:
        flags.append("no-BLA")
    if meta["balanced"]:
        flags.append("balanced")
    if meta["onesec"]:
        flags.append("onesec")
    flag_str = "".join(f" [{f}]" for f in flags)
    suptitle = f"{title_prefix}{flag_str} [{region_group}]"

    if meta["phase"] == "encoding":
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.985)
        handles, labels = placed_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", fontsize=8,
                   frameon=True, ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.955))
        if not use_seed_grouping:
            fig.tight_layout(rect=[0.05, 0.05, 1, 0.93])
        fig.supxlabel("Frequency (Hz)", fontsize=10, fontweight="bold", y=0.04)
        fig.supylabel(MODALITY_YLABEL[modality], fontsize=10,
                      fontweight="bold", x=0.04)
    else:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.985)
        handles, labels = placed_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", fontsize=11.5,
                   frameon=True, ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.955))
        # Seed-grouped uses explicit add_axes positions -- tight_layout would
        # repack them and break centering.
        if not use_seed_grouping:
            fig.tight_layout(rect=[0.05, 0.05, 1, 0.93])
        fig.supxlabel("Frequency (Hz)", fontsize=13, fontweight="bold", y=0.04)
        fig.supylabel(MODALITY_YLABEL[modality], fontsize=13,
                      fontweight="bold", x=0.04)

    fig.savefig(fig_path, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return f"wrote {fig_path}"


# --- entry point --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="List figures that would be (re-)written; don't render.")
    ap.add_argument("--group", default=None,
                    help="Limit to a single top-level group dir (e.g. encoding_balancedtrials_no_bla).")
    ap.add_argument("--region-group", default=None,
                    help="Limit to one region-group layout (allregions, mainregions, hippsubregions, ...).")
    ap.add_argument("--test", default=None,
                    help="Limit to one test (stim_nostim or endogenous_memory).")
    ap.add_argument("--modality", default=None,
                    help="Limit to one modality (power, coherence, pac).")
    args = ap.parse_args()

    jobs = discover_jobs(filter_group=args.group)
    if args.region_group:
        jobs = [j for j in jobs if j["region_group"] == args.region_group]
    if args.test:
        jobs = [j for j in jobs if j["test"] == args.test]
    if args.modality:
        jobs = [j for j in jobs if j["modality"] == args.modality]
    if not jobs:
        print("No cluster_stats.csv jobs discovered.")
        return

    # Print one "skipping null distribution" line per group/test/modality.
    seen_null = set()
    for job in jobs:
        key = (job["group"], job["region_group"], job["test"], job["modality"])
        if key not in seen_null:
            print(f"skipping null distribution for {job['modality']} in "
                  f"{job['group']}/{job['region_group']}/{job['test']}")
            seen_null.add(key)

    print()
    for job in jobs:
        status = render_job(job, dry_run=args.dry_run)
        print(status)


if __name__ == "__main__":
    main()
