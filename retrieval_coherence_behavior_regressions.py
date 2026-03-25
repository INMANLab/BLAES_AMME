#!/usr/bin/env python
"""
Patient-level regressions between retrieval coherence baseline connectivity and
memory modulation (avg_stim_dprime_diff).

Baseline functional connectivity measure:
1. Match the retrieval notebook logic by averaging each patient's available
   stim x memory retrieval coherence spectra within each region pair to get one
   mean spectrum.
2. Collapse that mean spectrum within a target band to obtain one scalar
   baseline FC value.
3. For composite ROIs, first average the patient's mean spectra across source
   BLA pairs, then take the band-average scalar.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib


def running_in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


IN_NOTEBOOK = running_in_notebook()
if not IN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

from combined_encoding_coherence import BLA_ALLHPC, BLA_MTL, build_bla_composite_data
from combined_retrieval_coherence import (
    get_overall_power_for_plot,
    load_amme_retrieval,
    load_blaes_retrieval,
    merge_dicts,
    merge_sets,
    visible_rois,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "retrieval_coherence_behavior_regressions"
BEHAVIOR_CSV = SCRIPT_DIR / "AMMEBLAES_includedpts_firstsession_behavioral.csv"
TARGET_COLUMN = "avg_stim_dprime_diff"
BAND_SPECS = {
    "theta": (4.0, 8.0),
    "slow_gamma": (35.0, 50.0),
}
BAND_LABELS = {
    "theta": "Theta (4-8 Hz)",
    "slow_gamma": "Slow Gamma (35-50 Hz)",
}
LOGIC_NOTE = (
    "X-axis baseline FC: within each patient and retrieval region pair, average the available "
    "stim x memory-condition spectra to one mean coherence spectrum (matching retrieval notebook logic), "
    "then average within the selected band. Composite ROIs average patient mean spectra across source "
    "BLA pairs before the band-average scalar is taken. Y-axis memory modulation: avg_stim_dprime_diff."
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_all_retrieval_data():
    blaes = load_blaes_retrieval()
    amme = load_amme_retrieval()
    all_data = {
        "group_all_power": merge_dicts(blaes["group_all_power"], amme["group_all_power"]),
        "stim_power": merge_dicts(blaes["stim_power"], amme["stim_power"]),
        "nostim_power": merge_dicts(blaes["nostim_power"], amme["nostim_power"]),
        "bc_stim": merge_dicts(blaes["bc_stim"], amme["bc_stim"]),
        "bc_nostim": merge_dicts(blaes["bc_nostim"], amme["bc_nostim"]),
        "freqs_post": blaes["freqs_post"] if blaes["freqs_post"] is not None else amme["freqs_post"],
        "freqs_diff": blaes["freqs_diff"] if blaes["freqs_diff"] is not None else amme["freqs_diff"],
        "has_memory": True,
        "use_memory_collapsed_overall": True,
        "overall_plot_exclude_substrings": merge_sets(
            blaes.get("overall_plot_exclude_substrings"),
            amme.get("overall_plot_exclude_substrings"),
        ),
        "stim_plot_exclude_substrings": merge_sets(
            blaes.get("stim_plot_exclude_substrings"),
            amme.get("stim_plot_exclude_substrings"),
        ),
        "memory_plot_exclude_substrings": merge_sets(
            blaes.get("memory_plot_exclude_substrings"),
            amme.get("memory_plot_exclude_substrings"),
        ),
        "bc_plot_exclude_substrings": merge_sets(
            blaes.get("bc_plot_exclude_substrings"),
            amme.get("bc_plot_exclude_substrings"),
        ),
        "stim_rem": merge_dicts(blaes["stim_rem"], amme["stim_rem"]),
        "stim_forg": merge_dicts(blaes["stim_forg"], amme["stim_forg"]),
        "nostim_rem": merge_dicts(blaes["nostim_rem"], amme["nostim_rem"]),
        "nostim_forg": merge_dicts(blaes["nostim_forg"], amme["nostim_forg"]),
        "bc_stim_rem": merge_dicts(blaes["bc_stim_rem"], amme["bc_stim_rem"]),
        "bc_stim_forg": merge_dicts(blaes["bc_stim_forg"], amme["bc_stim_forg"]),
        "bc_nostim_rem": merge_dicts(blaes["bc_nostim_rem"], amme["bc_nostim_rem"]),
        "bc_nostim_forg": merge_dicts(blaes["bc_nostim_forg"], amme["bc_nostim_forg"]),
    }
    composite_data = build_bla_composite_data(all_data)
    return all_data, composite_data


def regression_rois(all_data, composite_data):
    base_gap = get_overall_power_for_plot(all_data)
    composite_gap = get_overall_power_for_plot(composite_data)
    original_rois = visible_rois(base_gap.keys(), all_data.get("overall_plot_exclude_substrings", set()))
    return sorted(set(original_rois) | set(composite_gap.keys()) | {BLA_ALLHPC, BLA_MTL})


def build_baseline_fc_table(group_all_power, freqs):
    rows = []
    freqs = np.asarray(freqs, dtype=np.float64)
    for roi, subj_dict in group_all_power.items():
        for patient, spectrum in subj_dict.items():
            arr = np.asarray(spectrum, dtype=np.float64)
            if arr.size == 0:
                continue
            for band_name, (lo, hi) in BAND_SPECS.items():
                mask = (freqs >= lo) & (freqs <= hi)
                if not np.any(mask):
                    continue
                rows.append(
                    {
                        "Patient": str(patient),
                        "Region": roi,
                        "Band": band_name,
                        "baseline_fc": float(np.nanmean(arr[mask])),
                    }
                )
    return pd.DataFrame(rows)


def load_behavior():
    behavior = pd.read_csv(BEHAVIOR_CSV)
    if "Patient" not in behavior.columns or TARGET_COLUMN not in behavior.columns:
        raise ValueError(
            f"Behavior CSV must contain 'Patient' and '{TARGET_COLUMN}' columns: {BEHAVIOR_CSV}"
        )
    behavior = behavior[["Patient", TARGET_COLUMN]].copy()
    behavior["Patient"] = behavior["Patient"].astype(str)
    behavior[TARGET_COLUMN] = pd.to_numeric(behavior[TARGET_COLUMN], errors="coerce")
    behavior = behavior.dropna(subset=[TARGET_COLUMN]).copy()
    behavior = behavior[np.isfinite(behavior[TARGET_COLUMN])].copy()
    return behavior.drop_duplicates(subset=["Patient"])


def fit_regression(df):
    if df["baseline_fc"].nunique() < 2:
        return {
            "status": "undefined_baseline_fc",
            "n": int(len(df)),
            "slope": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "stderr": np.nan,
            "intercept_stderr": np.nan,
        }
    if df[TARGET_COLUMN].nunique() < 2:
        return {
            "status": "undefined_memory_modulation",
            "n": int(len(df)),
            "slope": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "stderr": np.nan,
            "intercept_stderr": np.nan,
        }

    result = stats.linregress(df["baseline_fc"], df[TARGET_COLUMN])
    out = {
        "status": "ok",
        "n": int(len(df)),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r": float(result.rvalue),
        "r_squared": float(result.rvalue ** 2),
        "p_value": float(result.pvalue),
        "stderr": float(result.stderr),
        "intercept_stderr": float(result.intercept_stderr),
    }
    if not np.isfinite(out["slope"]) or not np.isfinite(out["r"]) or not np.isfinite(out["p_value"]):
        out["status"] = "undefined_fit"
    return out


def make_annotation(stats_row):
    if stats_row["status"] != "ok":
        reason = {
            "undefined_baseline_fc": "baseline FC has no variability",
            "undefined_memory_modulation": "memory modulation has no variability",
            "undefined_fit": "fit returned non-finite values",
        }.get(stats_row["status"], stats_row["status"])
        return "\n".join([f"n = {stats_row['n']}", "Regression undefined", reason])
    return "\n".join(
        [
            f"n = {stats_row['n']}",
            f"slope = {stats_row['slope']:.4f}",
            f"intercept = {stats_row['intercept']:.4f}",
            f"r = {stats_row['r']:.3f}",
            f"R^2 = {stats_row['r_squared']:.3f}",
            f"p = {stats_row['p_value']:.4g}",
        ]
    )


def plot_regression(df, stats_row, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    x = df["baseline_fc"].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    ax.scatter(x, y, s=55, alpha=0.8, color="#1f4e79", edgecolors="white", linewidths=0.6)
    if stats_row["status"] == "ok":
        order = np.argsort(x)
        x_sorted = x[order]
        y_fit = stats_row["intercept"] + stats_row["slope"] * x_sorted
        ax.plot(x_sorted, y_fit, color="#b22222", linewidth=2)

    band_label = BAND_LABELS[df["Band"].iloc[0]]
    ax.set_xlabel(f"Baseline Functional Connectivity ({band_label})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Memory Modulation (avg_stim_dprime_diff)", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Retrieval Coherence vs Memory Modulation: {df['Region'].iloc[0]} - {band_label}",
        fontsize=15,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=11)
    ax.text(
        0.98,
        0.98,
        make_annotation(stats_row),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#808080"),
    )

    fig.text(0.015, 0.015, LOGIC_NOTE, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def plot_behavior_histogram(behavior_df, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    values = behavior_df[TARGET_COLUMN].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    bins = min(12, max(5, int(np.sqrt(len(values)))))
    ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Memory Modulation (avg_stim_dprime_diff)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=13, fontweight="bold")
    ax.set_title("Distribution of avg_stim_dprime_diff", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def plot_baseline_fc_histogram(df, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    values = df["baseline_fc"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    bins = min(12, max(5, int(np.sqrt(len(values)))))
    ax.hist(values, bins=bins, color="#72b7b2", edgecolor="white", alpha=0.9)
    band_label = BAND_LABELS[df["Band"].iloc[0]]
    ax.set_xlabel(f"Baseline Functional Connectivity ({band_label})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Baseline FC Distribution: {df['Region'].iloc[0]} - {band_label}",
        fontsize=15,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def main():
    reset_dir(OUTPUT_DIR)
    behavior = load_behavior()
    all_data, composite_data = load_all_retrieval_data()
    source_rois = regression_rois(all_data, composite_data)

    base_gap = get_overall_power_for_plot(all_data)
    composite_gap = get_overall_power_for_plot(composite_data)
    base_df = build_baseline_fc_table(base_gap, all_data["freqs_post"])
    composite_df = build_baseline_fc_table(composite_gap, composite_data["freqs_post"])
    fc_df = pd.concat([base_df, composite_df], ignore_index=True)
    fc_df = fc_df[fc_df["Region"].isin(source_rois)].copy()
    fc_df = fc_df[np.isfinite(fc_df["baseline_fc"])].copy()

    joined = fc_df.merge(behavior, on="Patient", how="inner")
    joined = joined.dropna(subset=["baseline_fc", TARGET_COLUMN]).copy()
    joined = joined[np.isfinite(joined["baseline_fc"]) & np.isfinite(joined[TARGET_COLUMN])].copy()
    if joined.empty:
        raise RuntimeError("No overlapping patients between retrieval coherence and behavioral memory-modulation data.")

    detail_dir = ensure_dir(OUTPUT_DIR / "plots")
    hist_dir = ensure_dir(OUTPUT_DIR / "histograms")
    summary_rows = []

    matched_behavior = joined[["Patient", TARGET_COLUMN]].drop_duplicates(subset=["Patient"]).sort_values("Patient")
    plot_behavior_histogram(matched_behavior, hist_dir / "avg_stim_dprime_diff_histogram.png")

    for roi in source_rois:
        for band_name in BAND_SPECS:
            roi_df = joined[(joined["Region"] == roi) & (joined["Band"] == band_name)].copy()
            roi_df = roi_df.drop_duplicates(subset=["Patient"])
            if len(roi_df) < 3:
                continue

            plot_baseline_fc_histogram(
                roi_df,
                hist_dir / f"{roi}_{band_name}_baseline_fc_histogram.png",
            )
            stats_row = fit_regression(roi_df)
            stats_row["Region"] = roi
            stats_row["Band"] = band_name
            summary_rows.append(stats_row)

            roi_csv = detail_dir / f"{roi}_{band_name}_baseline_fc_vs_memory_modulation.csv"
            roi_df.sort_values("Patient").to_csv(roi_csv, index=False)
            plot_regression(roi_df, stats_row, detail_dir / f"{roi}_{band_name}_baseline_fc_vs_memory_modulation.png")

    if not summary_rows:
        raise RuntimeError("No retrieval coherence regressions could be fit; check patient overlap and baseline connectivity variability.")

    summary_df = pd.DataFrame(summary_rows)[
        ["Region", "Band", "status", "n", "slope", "intercept", "r", "r_squared", "p_value", "stderr", "intercept_stderr"]
    ].sort_values(["Region", "Band"])
    summary_df.to_csv(OUTPUT_DIR / "retrieval_coherence_behavior_regression_summary.csv", index=False)
    joined.sort_values(["Region", "Band", "Patient"]).to_csv(
        OUTPUT_DIR / "retrieval_coherence_behavior_regression_joined_data.csv",
        index=False,
    )
    (OUTPUT_DIR / "logic.txt").write_text(LOGIC_NOTE + "\n", encoding="utf-8")

    print(f"Wrote retrieval coherence regression outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
