#!/usr/bin/env python
"""Cluster-based permutation test (Maris & Oostenveld, 2007) of endogenous
memory effects (Remembered vs Forgotten) on no-stim retrieval trials.

Now uses the dual-family FWE scheme:
  - Confirmatory (theta 4-8 Hz, slow gamma 35-50 Hz): each tested with its
    own null distribution.
  - Exploratory (all other freqs): tested with its own null distribution.
ALLHPC (pooled CA/DG/HPC) is added as an extra region.

Outputs to outputs/PermutationOutputsAlireza/{retreival|retreival_balancedtrials}/endogenous_memory/.

Usage:
  python run_permutation_endogenous_memory.py
  python run_permutation_endogenous_memory.py power
  python run_permutation_endogenous_memory.py --balanced
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
    plot_region_panel_dual,
    run_family_permutations,
)

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
INPUT_DIR = ROOT / "outputs" / "csvs"

_raw_args = sys.argv[1:]
BALANCED = "--balanced" in _raw_args
ONESEC = "--onesec" in _raw_args
_pos_args = [a for a in _raw_args if not a.startswith("--")]

_branch = ("retreival_balancedtrials_onesec" if (BALANCED and ONESEC)
           else "retreival_onesec" if ONESEC
           else "retreival_balancedtrials" if BALANCED
           else "retreival")
OUT_BASE = ROOT / "outputs" / "PermutationOutputsAlireza" / _branch / "endogenous_memory"
OUT_BASE.mkdir(parents=True, exist_ok=True)

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
    df = df[df["trial_type"] == "nostim"]
    if BALANCED:
        keep = _balanced_keep(df)
        before = df["Patient"].nunique()
        df = df[df["Patient"].isin(keep)]
        print(f"  [balanced] {modality}: kept "
              f"{df['Patient'].nunique()}/{before} patients")
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
    cluster_table = []
    for region in regions:
        X_R, X_F, _ = paired_arrays(grouped, region, diff_cols)
        if X_R is None:
            print(f"  - {region}: skipped (n < {MIN_SUBJECTS})")
            continue
        print(f"  - {region}: n={X_R.shape[0]} subjects, "
              f"{len(diff_cols)} freqs ...", flush=True)
        res = run_family_permutations(
            X_R, X_F, freqs,
            n_perm=N_PERMUTATIONS, alpha=ALPHA, seed=SEED,
            direction_labels=("Remembered>Forgotten", "Remembered<Forgotten"),
        )

        for c in res["clusters"]:
            cluster_table.append({
                "modality": modality, "region": region,
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
            "region": region, "X_R": X_R, "X_F": X_F,
            "clusters": res["clusters"],
            "null_by_family": res["null_by_family"],
        })

    pd.DataFrame(cluster_table).to_csv(
        out_dir / f"{modality}_cluster_stats.csv", index=False)
    print(f"  Saved cluster stats: {out_dir / f'{modality}_cluster_stats.csv'}")

    n_panels = len(panel_results)
    if n_panels == 0:
        return
    ncols = min(4, max(2, int(np.ceil(np.sqrt(n_panels)))))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.0, nrows * 3.2),
                             squeeze=False)
    axes_flat = axes.flatten()
    panel_xlim = (30.0, float(freqs.max())) if modality == "pac" else None
    for ax, panel in zip(axes_flat, panel_results):
        plot_region_panel_dual(
            ax, freqs, panel["X_R"], panel["X_F"],
            panel["clusters"], panel["region"],
            line_color_A="#7b2d8e", line_color_B="#daa520",
            label_A="Remembered", label_B="Forgotten",
            fill_color="#c0392b",
            xlim=panel_xlim,
        )
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Endogenous memory (Remembered vs Forgotten) - {modality.upper()}"
        f"{' [balanced]' if BALANCED else ''}"
        f"{' [onesec]' if ONESEC else ''}\n"
        f"No-stim retrieval trials, paired cluster permutation "
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

    fig_path = out_dir / f"{modality}_endogenous_memory.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
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
        f" ({N_PERMUTATIONS} permutations x {n_panels} regions, pooled)",
        fontsize=11)
    fig.tight_layout()
    null_path = out_dir / f"{modality}_null_distribution.png"
    fig.savefig(null_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for m in modalities:
        run_modality(m)
    print("\n===== ALL DONE =====")
