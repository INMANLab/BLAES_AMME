#!/usr/bin/env python
"""Band-averaged paired permutation test of endogenous memory effects
(Remembered vs Forgotten) on no-stim retrieval trials, per Region.

Vanilla only (no --balanced, no --onesec).
Outputs: outputs/PermutationOutputsAlireza/retreival_bandwise/endogenous_memory/<modality>/.
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
    plot_region_panel_bandwise,
    run_band_permutations,
)

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
INPUT_DIR = ROOT / "outputs" / "csvs"
OUT_BASE = (ROOT / "outputs" / "PermutationOutputsAlireza"
            / "retreival_bandwise" / "endogenous_memory")
OUT_BASE.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 5000
SEED = 100
ALPHA = 0.05
MIN_SUBJECTS = 4
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

modalities = sys.argv[1:] if sys.argv[1:] else ["power", "coherence", "pac"]


def load_modality(modality):
    df = pd.read_csv(INPUT_DIR / MODALITY_TO_FILE[modality])
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"
    df = df[df["yes_or_no"].isin(["yes", "no"])]
    df = df[df["trial_type"] == "nostim"]
    df["Memory"] = np.where(df["yes_or_no"] == "yes", "Remembered", "Forgotten")
    freq_cols = [c for c in df.columns if "_Freq_" in c]
    diff_cols = [c for c in freq_cols if c.startswith("diff_")]
    freqs = [float(c.split("_Freq_")[1]) for c in diff_cols]
    return df, diff_cols, np.array(freqs)


def per_subject_means(df, diff_cols):
    return df.groupby(["Patient", "Region", "Memory"],
                      as_index=False)[diff_cols].mean()


def paired_arrays(grouped, region, freq_cols):
    sub = grouped[grouped["Region"] == region]
    pivoted = sub.pivot_table(index="Patient", columns="Memory",
                              values=freq_cols, aggfunc="mean")
    if pivoted.empty:
        return None, None, None
    has_both = pivoted.dropna(how="any").index.tolist()
    if len(has_both) < MIN_SUBJECTS:
        return None, None, None
    pivoted = pivoted.loc[has_both]
    if "Remembered" not in pivoted.columns.get_level_values(1) or \
       "Forgotten" not in pivoted.columns.get_level_values(1):
        return None, None, None
    X_R = np.column_stack([pivoted[(c, "Remembered")].values for c in freq_cols])
    X_F = np.column_stack([pivoted[(c, "Forgotten")].values for c in freq_cols])
    return X_R, X_F, has_both


def run_modality(modality):
    print(f"\n========== {modality} ==========")
    df, diff_cols, freqs = load_modality(modality)
    if df.empty:
        print("  No nostim trials found.")
        return
    print(f"  {len(diff_cols)} freq bins, "
          f"{df['Patient'].nunique()} patients in raw data")
    grouped = per_subject_means(df, diff_cols)
    grouped = add_allhpc_region(grouped, diff_cols, "Memory", modality)

    out_dir = OUT_BASE / modality
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = sorted(grouped["Region"].unique())
    excluded = [r for r in regions if is_excluded_region(r)]
    if excluded:
        print(f"  Excluding: {', '.join(excluded)}")
    regions = [r for r in regions if r not in excluded]

    panel_results = []
    band_table = []
    for region in regions:
        X_R, X_F, _ = paired_arrays(grouped, region, diff_cols)
        if X_R is None:
            print(f"  - {region}: skipped (n < {MIN_SUBJECTS})")
            continue
        print(f"  - {region}: n={X_R.shape[0]} subjects ...", flush=True)
        res = run_band_permutations(
            X_R, X_F, freqs,
            n_perm=N_PERMUTATIONS, alpha=ALPHA, seed=SEED,
            direction_labels=("Remembered>Forgotten", "Remembered<Forgotten"),
        )
        for b in res["bands"]:
            band_table.append({
                "modality": modality, "region": region,
                "n_subjects": res["n_subj"],
                "band": b["band"],
                "family": b["family"],
                "confirmatory": b["confirmatory"],
                "freq_lo_hz": b["freq_lo_hz"],
                "freq_hi_hz": b["freq_hi_hz"],
                "t_stat": b["t_stat"],
                "direction": b["direction"],
                "p_value": b["p_value"],
                "significant": bool(b["p_value"] < .05),
            })
        panel_results.append({
            "region": region, "X_R": X_R, "X_F": X_F,
            "bands": res["bands"],
            "null_by_family": res["null_by_family"],
        })

    pd.DataFrame(band_table).to_csv(
        out_dir / f"{modality}_band_stats.csv", index=False)
    print(f"  Saved band stats: {out_dir / f'{modality}_band_stats.csv'}")

    if not panel_results:
        return
    n = len(panel_results)
    ncols = min(4, max(2, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.0, nrows * 3.2),
                             squeeze=False)
    axes_flat = axes.flatten()
    panel_xlim = (30.0, float(freqs.max())) if modality == "pac" else None
    for ax, panel in zip(axes_flat, panel_results):
        plot_region_panel_bandwise(
            ax, freqs, panel["X_R"], panel["X_F"],
            panel["bands"], panel["region"],
            line_color_A="#7b2d8e", line_color_B="#daa520",
            label_A="Remembered", label_B="Forgotten",
            fill_color="#c0392b",
            xlim=panel_xlim,
        )
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(
        f"Endogenous memory (Remembered vs Forgotten) Band-wise - {modality.upper()}\n"
        f"No-stim retrieval trials, paired band permutation "
        f"({N_PERMUTATIONS} samples) | dark=confirmatory sig, "
        f"light=exploratory sig, none=ns",
        fontsize=14, fontweight="bold", y=0.985)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fontsize=20,
               frameon=True, ncol=len(labels),
               bbox_to_anchor=(0.5, 0.955))
    fig.supxlabel("Frequency (Hz)", fontsize=20, fontweight="bold", y=0.04)
    fig.supylabel(MODALITY_YLABEL[modality], fontsize=20, fontweight="bold",
                  x=0.035)
    fig.tight_layout(rect=[0.045, 0.05, 1, 0.93])
    fig.savefig(out_dir / f"{modality}_endogenous_memory.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_dir / f'{modality}_endogenous_memory.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, family in zip(axes, ("confirmatory", "exploratory")):
        all_max = np.concatenate(
            [p["null_by_family"][family] for p in panel_results])
        ax.hist(all_max, bins=60, color="0.7", edgecolor="0.4")
        ax.set_title(f"Null max |t| - {family}", fontsize=10)
        ax.set_xlabel("Max |t|")
        ax.set_ylabel("Count")
    fig.suptitle(
        f"Null distributions - {modality.upper()}"
        f" ({N_PERMUTATIONS} permutations x {n} panels, pooled)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{modality}_null_distribution.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for m in modalities:
        run_modality(m)
    print("\n===== ALL DONE =====")
