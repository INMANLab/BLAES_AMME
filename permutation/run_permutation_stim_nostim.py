#!/usr/bin/env python
"""Cluster-based permutation test (Maris & Oostenveld, 2007) of stim vs
no-stim retrieval-trial differences, run separately per Region.

Now uses the dual-family FWE scheme:
  - Confirmatory (theta 4-8 Hz, slow gamma 35-50 Hz): each tested with its
    own null distribution.
  - Exploratory (all other freqs): tested with its own null distribution.
ALLHPC (pooled CA/DG/HPC) is added as an extra region.

Outputs to outputs/PermutationOutputsAlireza/{retreival|retreival_balancedtrials}/stim_nostim/.

Usage:
  python run_permutation_stim_nostim.py
  python run_permutation_stim_nostim.py power
  python run_permutation_stim_nostim.py --balanced
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _permutation_helpers import (
    add_allhpc_region,
    get_panel_position,
    panel_grid_for,
    plot_region_panel_dual,
    run_family_permutations,
)

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
INPUT_DIR = ROOT / "outputs" / "csvs"

_raw_args = sys.argv[1:]
BALANCED = "--balanced" in _raw_args
ONESEC = "--onesec" in _raw_args
REGION_GROUP = None
for _a in _raw_args:
    if _a.startswith("--region-group="):
        REGION_GROUP = _a.split("=", 1)[1]
assert REGION_GROUP in (None, "mainregions", "hippsubregions")
if REGION_GROUP and (BALANCED or ONESEC):
    raise SystemExit(
        "--region-group is only supported for the vanilla branch "
        "(no --balanced, no --onesec)")
_pos_args = [a for a in _raw_args
             if not a.startswith("--") and "=" not in a]

_branch = ("retreival_balancedtrials_onesec" if (BALANCED and ONESEC)
           else "retreival_onesec" if ONESEC
           else "retreival_balancedtrials" if BALANCED
           else "retreival")
_segments = [_branch]
_segments.append(REGION_GROUP if REGION_GROUP else "allregions")
_segments.append("stim_nostim")
OUT_BASE = (ROOT / "outputs" / "PermutationOutputsAlireza"
            / Path(*_segments))
OUT_BASE.mkdir(parents=True, exist_ok=True)

MAIN_REGIONS = {"BLA", "EC", "PRC", "ALLHPC"}
HIPPSUB_REGIONS = {"CA", "DG", "HPC"}


def in_region_group(region, group):
    """True if `region` (single or pair) belongs to the group."""
    parts = region.split("_")
    if group == "mainregions":
        return all(p in MAIN_REGIONS for p in parts)
    if group == "hippsubregions":
        return any(p in HIPPSUB_REGIONS for p in parts)
    return True

N_PERMUTATIONS = 5000
SEED = 100
ALPHA = 0.05
MIN_SUBJECTS = 4
MIN_TRIALS_BALANCED = 10
EXCLUDE_TOKENS = ("PHG",)
EXCLUDE_PAIRS = {
    frozenset({"CA", "DG"}),
    frozenset({"CA", "HPC"}),
    frozenset({"DG", "HPC"}),
    frozenset({"ALLHPC", "CA"}),
    frozenset({"ALLHPC", "DG"}),
    frozenset({"ALLHPC", "HPC"}),
}


def is_excluded_region(r):
    if any(tok in r for tok in EXCLUDE_TOKENS):
        return True
    parts = r.split("_")
    if len(parts) == 2 and frozenset(parts) in EXCLUDE_PAIRS:
        return True
    return False


if ONESEC:
    INPUT_DIR = ROOT / "outputs" / "csvs" / "onesec"
    MODALITY_TO_FILE = {
        "power":     "combined_retrieval_power_onesec_mlmr_input.csv",
        "coherence": "combined_retrieval_coherence_onesec_mlmr_input.csv",
        "pac":       "combined_retrieval_pac_onesec_mlmr_input.csv",
    }
else:
    MODALITY_TO_FILE = {
        "power":     "combined_retrieval_power_all_mlmr_input.csv",
        "coherence": "combined_retrieval_coherence_all_mlmr_input.csv",
        "pac":       "combined_retrieval_pac_all_mlmr_input.csv",
    }
MODALITY_YLABEL = {
    "power":     "Power (dB, baseline-subtracted)",
    "coherence": "Coherence (baseline-subtracted)",
    "pac":       "PAC (baseline-subtracted)",
}

modalities = _pos_args if _pos_args else ["power", "coherence", "pac"]


def _balanced_keep(df):
    counts = (df.groupby("Patient")
                .agg(n_rem=("yes_or_no", lambda s: (s == "yes").sum()),
                     n_forg=("yes_or_no", lambda s: (s == "no").sum()),
                     n_regions=("Region", lambda s: s.nunique()))
                .reset_index())
    counts["rem_per_reg"] = (counts["n_rem"] / counts["n_regions"]).round()
    counts["forg_per_reg"] = (counts["n_forg"] / counts["n_regions"]).round()
    return set(counts.loc[(counts["rem_per_reg"] >= MIN_TRIALS_BALANCED) &
                          (counts["forg_per_reg"] >= MIN_TRIALS_BALANCED),
                          "Patient"].tolist())


def load_modality(modality):
    df = pd.read_csv(INPUT_DIR / MODALITY_TO_FILE[modality])
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"
    df = df[df["yes_or_no"].isin(["yes", "no"])]
    df = df[df["trial_type"].notna()]
    if BALANCED:
        keep = _balanced_keep(df)
        before = df["Patient"].nunique()
        df = df[df["Patient"].isin(keep)]
        print(f"  [balanced] {modality}: kept "
              f"{df['Patient'].nunique()}/{before} patients")
    df["StimCond"] = np.where(df["trial_type"] == "nostim", "Nostim", "Stim")

    freq_cols = [c for c in df.columns if "_Freq_" in c]
    diff_cols = [c for c in freq_cols if c.startswith("diff_")]
    freqs = [float(c.split("_Freq_")[1]) for c in diff_cols]
    return df, diff_cols, np.array(freqs)


def per_subject_means(df, diff_cols):
    df = df.copy()
    df["Memory"] = np.where(df["yes_or_no"] == "yes", "Remembered", "Forgotten")
    grouped = df.groupby(["Patient", "Region", "StimCond", "Memory"],
                         as_index=False)[diff_cols].mean()
    return grouped


def paired_arrays_by_memory(grouped, region, memory, freq_cols):
    sub = grouped[(grouped["Region"] == region)
                  & (grouped["Memory"] == memory)]
    pivoted = sub.pivot_table(index="Patient", columns="StimCond",
                              values=freq_cols, aggfunc="mean")
    if pivoted.empty:
        return None, None, None
    has_both = pivoted.dropna(how="any").index.tolist()
    if len(has_both) < MIN_SUBJECTS:
        return None, None, None
    pivoted = pivoted.loc[has_both]
    if "Stim" not in pivoted.columns.get_level_values(1) or \
       "Nostim" not in pivoted.columns.get_level_values(1):
        return None, None, None
    X_S = np.column_stack([pivoted[(c, "Stim")].values for c in freq_cols])
    X_N = np.column_stack([pivoted[(c, "Nostim")].values for c in freq_cols])
    return X_S, X_N, has_both


def run_modality(modality):
    print(f"\n========== {modality} ==========")
    df, diff_cols, freqs = load_modality(modality)
    if df.empty:
        print("  No retrieval trials found.")
        return
    print(f"  {len(diff_cols)} freq bins, "
          f"{df['Patient'].nunique()} patients in raw data")

    grouped = per_subject_means(df, diff_cols)
    grouped = add_allhpc_region(grouped, diff_cols,
                                ["StimCond", "Memory"], modality)

    out_dir = OUT_BASE / modality
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = sorted(grouped["Region"].unique())
    excluded = [r for r in regions if is_excluded_region(r)]
    if excluded:
        print(f"  Excluding: {', '.join(excluded)}")
    regions = [r for r in regions if r not in excluded]
    if REGION_GROUP:
        before = list(regions)
        regions = [r for r in regions if in_region_group(r, REGION_GROUP)]
        dropped = sorted(set(before) - set(regions))
        if dropped:
            print(f"  [{REGION_GROUP}] dropping out-of-group: "
                  f"{', '.join(dropped)}")

    panel_results = []
    cluster_table = []
    for region in regions:
        for memory in ("Remembered", "Forgotten"):
            X_S, X_N, _ = paired_arrays_by_memory(grouped, region, memory, diff_cols)
            if X_S is None:
                print(f"  - {region} | {memory}: skipped (n < {MIN_SUBJECTS})")
                continue
            print(f"  - {region} | {memory}: n={X_S.shape[0]} subjects ...",
                  flush=True)
            res = run_family_permutations(
                X_S, X_N, freqs,
                n_perm=N_PERMUTATIONS, alpha=ALPHA, seed=SEED,
                direction_labels=("Stim>Nostim", "Stim<Nostim"),
            )
            for c in res["clusters"]:
                cluster_table.append({
                    "modality": modality, "region": region,
                    "memory": memory,
                    "n_subjects": res["n_subj"],
                    "family": c["family"],
                    "confirmatory": c["confirmatory"],
                    "freq_lo_hz": float(freqs[c["start_idx"]]),
                    "freq_hi_hz": float(freqs[c["end_idx"]]),
                    "n_freqs": c["n_freqs"], "direction": c["direction"],
                    "cluster_mass_t": c["cluster_mass"],
                    "t_threshold": res["t_thresh"],
                    "p_value": c["p_value"],
                    "significant": bool(c["p_value"] < .05),
                })
            panel_results.append({
                "region": region, "memory": memory,
                "X_S": X_S, "X_N": X_N,
                "clusters": res["clusters"],
                "null_by_family": res["null_by_family"],
            })

    pd.DataFrame(cluster_table).to_csv(
        out_dir / f"{modality}_cluster_stats.csv", index=False)
    print(f"  Saved cluster stats: {out_dir / f'{modality}_cluster_stats.csv'}")

    if not panel_results:
        print("  No panels to plot.")
        return
    panel_xlim = (30.0, float(freqs.max())) if modality == "pac" else None

    if REGION_GROUP:
        # Compact layout: pack region-blocks sequentially. Choose (nrows,
        # blocks_per_row) that minimizes empty cells then most-square aspect.
        regions_in_order = sorted({p["region"] for p in panel_results})
        nR = len(regions_in_order)
        n_panels = nR * 2
        best = None
        for bpr in range(1, nR + 1):
            nr = int(np.ceil(nR / bpr))
            nc = bpr * 2
            empty = nr * nc - n_panels
            aspect = max(nr, nc) / max(min(nr, nc), 1)
            # tiebreaker: prefer wider (more cols, fewer rows) for paper layout.
            score = (empty, aspect, -nc)
            if best is None or score < best[0]:
                best = (score, nr, nc, bpr)
        _, nrows, ncols, blocks_per_row = best
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.0, nrows * 3.2),
                                 squeeze=False)
        for r in range(nrows):
            for c in range(ncols):
                axes[r][c].set_visible(False)
        panel_lookup = {(p["region"], p["memory"]): p for p in panel_results}
        placed_axes = []
        for i, region in enumerate(regions_in_order):
            block_row = i // blocks_per_row
            block_col_start = (i % blocks_per_row) * 2
            for offset, memory in enumerate(("Remembered", "Forgotten")):
                cell_col = block_col_start + offset
                ax = axes[block_row][cell_col]
                panel = panel_lookup.get((region, memory))
                if panel is None:
                    continue
                ax.set_visible(True)
                plot_region_panel_dual(
                    ax, freqs, panel["X_S"], panel["X_N"],
                    panel["clusters"],
                    f'{region} - {memory}',
                    line_color_A="#d62728", line_color_B="#1f77b4",
                    label_A="Stim", label_B="No stim",
                    fill_color="#b8860b",
                    xlim=panel_xlim,
                )
                placed_axes.append(ax)
        axes_flat = placed_axes if placed_axes else [axes[0][0]]
    else:
        nrows, ncols = panel_grid_for(modality, n_memory_cells=2)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.0, nrows * 3.2),
                                 squeeze=False)
        for r in range(nrows):
            for c in range(ncols):
                axes[r][c].set_visible(False)
        placed_axes = []
        for panel in panel_results:
            pos = get_panel_position(modality, panel["region"])
            if pos is None:
                print(f"  - {panel['region']}: no layout slot, skipping plot")
                continue
            slot_r, slot_c = pos
            cell_col = slot_c * 2 + (0 if panel["memory"] == "Remembered" else 1)
            ax = axes[slot_r][cell_col]
            ax.set_visible(True)
            plot_region_panel_dual(
                ax, freqs, panel["X_S"], panel["X_N"],
                panel["clusters"],
                f'{panel["region"]} - {panel["memory"]}',
                line_color_A="#d62728", line_color_B="#1f77b4",
                label_A="Stim", label_B="No stim",
                fill_color="#b8860b",
                xlim=panel_xlim,
            )
            placed_axes.append(ax)
        axes_flat = placed_axes if placed_axes else [axes[0][0]]

    _region_tag = REGION_GROUP if REGION_GROUP else "allregions"
    fig.suptitle(
        f"Stim vs No-stim {modality.capitalize()} by Memory"
        f"{' [balanced]' if BALANCED else ''}"
        f"{' [onesec]' if ONESEC else ''}"
        f" [{_region_tag}]",
        fontsize=13, fontweight="bold", y=0.985)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fontsize=18,
               frameon=True, ncol=len(labels),
               bbox_to_anchor=(0.5, 0.955))
    fig.supxlabel("Frequency (Hz)", fontsize=18, fontweight="bold", y=0.04)
    fig.supylabel(MODALITY_YLABEL[modality], fontsize=18, fontweight="bold",
                  x=0.035)
    fig.tight_layout(rect=[0.045, 0.05, 1, 0.93])

    fig_path = out_dir / f"{modality}_stim_nostim.png"
    fig.savefig(fig_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, family in zip(axes, ("theta", "slow_gamma", "exploratory")):
        all_max = np.concatenate(
            [p["null_by_family"][family] for p in panel_results])
        ax.hist(all_max, bins=60, color="0.7", edgecolor="0.4")
        ax.set_title(f"Null max |cluster mass| - {family}", fontsize=10)
        ax.set_xlabel("Max |cluster mass|")
        ax.set_ylabel("Count")
    fig.suptitle(
        f"Null distributions - {modality.upper()}"
        f" ({N_PERMUTATIONS} permutations x {len(panel_results)} panels, pooled)",
        fontsize=11)
    fig.tight_layout()
    null_path = out_dir / f"{modality}_null_distribution.png"
    fig.savefig(null_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved null distribution: {null_path}")


if __name__ == "__main__":
    for m in modalities:
        run_modality(m)
    print("\n===== ALL DONE =====")
