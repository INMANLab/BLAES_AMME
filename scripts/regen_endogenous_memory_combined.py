#!/usr/bin/env python
"""Build 2x2 endogenous-memory figures (theta/slow-gamma x Remembered/Forgotten)
for power and coherence, retrieval and encoding.

Uses the same per-patient nostim spectra the cluster-permutation pipeline
loads (combined_{retrieval,encoding}_{power,coherence}_all_mlmr_input.csv).
Each subplot overlays all kept regions/pairs (excluding MTL and PHG, plus the
within-hippocampus pairs that get pooled into ALLHPC) as colored mean +/- SEM
lines over the band's frequency range.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/"
            "InmanLab/BLAES_data/dissertation/AMME_BLAES")
sys.path.insert(0, str(ROOT / "permutation"))
from _permutation_helpers import add_allhpc_region

CSV_DIR = ROOT / "outputs" / "csvs"
OUTPUTS = ROOT / "OUTPUTS"

THETA_LO, THETA_HI = 4.0, 8.0
SGAMMA_LO, SGAMMA_HI = 30.0, 55.0
BANDS = [("Theta", THETA_LO, THETA_HI), ("Slow gamma", SGAMMA_LO, SGAMMA_HI)]

# Pairs containing these get dropped (MTL, PHG); also drop within-hippocampus
# pairs that already roll into ALLHPC.
EXCLUDE_TOKENS = ("MTL", "PHG")
EXCLUDE_HIPP_PAIRS = {
    frozenset({"CA", "DG"}), frozenset({"CA", "HPC"}),
    frozenset({"DG", "HPC"}),
    frozenset({"ALLHPC", "CA"}), frozenset({"ALLHPC", "DG"}),
    frozenset({"ALLHPC", "HPC"}),
}

POWER_REGION_ORDER = ["ALLHPC", "HPC", "CA", "DG", "BLA", "EC", "PRC"]
COH_PAIR_ORDER = [
    "ALLHPC_BLA", "ALLHPC_EC", "ALLHPC_PRC",
    "BLA_EC", "BLA_PRC", "EC_PRC",
    "HPC_BLA", "HPC_EC", "HPC_PRC",
    "CA_BLA", "CA_EC", "CA_PRC",
    "DG_BLA", "DG_EC", "DG_PRC",
]
REGION_COLORS = {
    "ALLHPC": "#5B4B8A", "HPC": "#FFA500", "CA": "#000000", "DG": "#87CEEB",
    "BLA": "#008080", "EC": "#800080", "PRC": "#E61C59",
}


def _is_excluded_region(r: str) -> bool:
    if any(tok in r for tok in EXCLUDE_TOKENS):
        return True
    parts = r.split("_")
    if len(parts) == 2 and frozenset(parts) in EXCLUDE_HIPP_PAIRS:
        return True
    return False


def _canonical_pair(r: str) -> str:
    parts = r.split("_")
    if len(parts) == 2:
        return "_".join(sorted(parts))
    return r


def _pair_color(pair: str) -> tuple[str, str]:
    """Return color for a coherence pair label (sort-canonical)."""
    parts = pair.split("_")
    if len(parts) != 2:
        return ("#444444", pair)
    a, b = sorted(parts)
    canonical = f"{a}_{b}"
    # Color = blend of the two endpoints; for simplicity use the first.
    return (REGION_COLORS.get(a, REGION_COLORS.get(b, "#444444")), canonical)


def load_data(measure: str, phase: str):
    """Returns df, freqs (np.array), diff_cols (list of column names)."""
    fname = f"combined_{phase}_{measure}_all_mlmr_input.csv"
    df = pd.read_csv(CSV_DIR / fname)
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"
    df = df[df["yes_or_no"].isin(["yes", "no"])]
    df = df[df["trial_type"] == "nostim"].copy()
    df["Memory"] = np.where(df["yes_or_no"] == "yes", "Remembered", "Forgotten")
    diff_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    freqs = np.array([float(c.split("_Freq_")[1]) for c in diff_cols])
    return df, freqs, diff_cols


def per_patient_means(df: pd.DataFrame, diff_cols: list[str], measure: str):
    """Patient-level mean spectra per (Patient, Region, Memory) and pool ALLHPC."""
    grouped = df.groupby(["Patient", "Region", "Memory"], as_index=False)[diff_cols].mean()
    modality_for_helper = "power" if measure == "power" else "coherence"
    grouped = add_allhpc_region(grouped, diff_cols, "Memory", modality_for_helper)
    if measure != "power":
        grouped["Region"] = grouped["Region"].map(_canonical_pair)
    grouped = grouped[~grouped["Region"].apply(_is_excluded_region)]
    return grouped


def regions_to_plot(grouped: pd.DataFrame, measure: str) -> list[str]:
    seen = grouped["Region"].unique().tolist()
    if measure == "power":
        order = [r for r in POWER_REGION_ORDER if r in seen]
        order += [r for r in seen if r not in order]
    else:
        order = [r for r in COH_PAIR_ORDER if r in seen]
        order += sorted(r for r in seen if r not in order)
    return order


def color_for(region: str, measure: str):
    if measure == "power":
        return REGION_COLORS.get(region, "#444444")
    return _pair_color(region)[0]


def plot_combined(measure: str, phase: str):
    print(f"\n=== {phase} {measure}: endogenous memory combined figure ===")
    df, freqs, diff_cols = load_data(measure, phase)
    grouped = per_patient_means(df, diff_cols, measure)
    if grouped.empty:
        print("  No data after filtering, skipping.")
        return
    region_order = regions_to_plot(grouped, measure)
    print(f"  Plotting {len(region_order)} {'regions' if measure=='power' else 'pairs'}: {region_order}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    measure_label = "Power (dB)" if measure == "power" else "Coherence (Fisher Z)"
    fig.suptitle(
        f"Baseline-Corrected {'Power' if measure == 'power' else 'Coherence'} - "
        f"Endogenous Memory ({phase.capitalize()})",
        fontsize=18, fontweight="bold", y=0.995,
    )

    band_row = {"Theta": 0, "Slow gamma": 1}
    mem_col = {"Remembered": 0, "Forgotten": 1}

    legend_handles = []
    for band_name, lo, hi in BANDS:
        band_mask = (freqs >= lo) & (freqs <= hi)
        band_freqs = freqs[band_mask]
        band_cols = [c for c, m in zip(diff_cols, band_mask) if m]
        for mem in ("Remembered", "Forgotten"):
            ax = axes[band_row[band_name], mem_col[mem]]
            sub = grouped[grouped["Memory"] == mem]
            for region in region_order:
                rsub = sub[sub["Region"] == region]
                if rsub.empty or len(rsub) < 4:
                    continue
                arr = rsub[band_cols].to_numpy(dtype=float)
                arr = arr[~np.any(np.isnan(arr), axis=1)]
                if arr.shape[0] < 4:
                    continue
                mean = arr.mean(axis=0)
                sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                col = color_for(region, measure)
                line, = ax.plot(band_freqs, mean, color=col, lw=2,
                                label=f"{region} (n={arr.shape[0]})")
                ax.fill_between(band_freqs, mean - sem, mean + sem,
                                color=col, alpha=0.18, lw=0)
                if band_name == "Theta" and mem == "Remembered":
                    legend_handles.append(line)
            ax.axhline(0, color="0.6", lw=0.5, ls="--")
            ax.set_title(f"{band_name} - {mem}", fontsize=14, fontweight="bold")
            ax.set_xlim(lo, hi)
            ax.tick_params(axis="both", labelsize=11)
            if mem_col[mem] == 0:
                ax.set_ylabel(measure_label, fontsize=12, fontweight="bold")
            if band_row[band_name] == 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight="bold")

    # Equalize y-limits within each row so theta and slow-gamma rows are
    # internally comparable across mem conds.
    for row in (0, 1):
        ymins, ymaxs = zip(*(axes[row, c].get_ylim() for c in (0, 1)))
        ylo, yhi = min(ymins), max(ymaxs)
        for c in (0, 1):
            axes[row, c].set_ylim(ylo, yhi)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=11, frameon=True, framealpha=0.9,
        )

    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    out_dir = OUTPUTS / f"{phase}_{measure if measure == 'power' else 'coherence'}" / "all"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"EndogenousMemory_BaselineCorrected_"
             f"{'Power' if measure=='power' else 'Coherence'}_"
             f"{phase}_combined.png")
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


if __name__ == "__main__":
    for measure in ("power", "coherence"):
        for phase in ("retrieval", "encoding"):
            plot_combined(measure, phase)
    print("\nDone.")
