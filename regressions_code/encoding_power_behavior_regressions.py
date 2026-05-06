#!/usr/bin/env python
"""
Patient-level regressions between encoding baseline-corrected power and memory
modulation (avg_stim_dprime_diff).

Baseline-corrected power measure:
1. Within each patient and ROI, average the baseline-corrected stim and
   nostim spectra separately for remembered and forgotten trials.
2. Collapse each remembered/forgotten spectrum within a target band to obtain
   the x-axis values for the two regression subfigures.
3. Composite ROIs average patient mean spectra across source single-region
   MTL and hippocampal ROIs before the band-average scalar is taken.
"""

# --- AMME_BLAES sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
for _p in (_root, _root / 'encoding', _root / 'retrieval', _root / 'endogenous_memory',
          _root / 'regressions', _root / 'behavioral', _root / 'balanced_memory'):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---

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

from behavior_regression_memory_panels import (
    MEMORY_ORDER,
    average_condition_region_dicts,
    build_memory_band_table,
    fit_regression,
    plot_memory_panel_histogram,
    plot_memory_panel_regression,
)
from combined_encoding_power import (
    load_amme_encoding,
    load_blaes_encoding,
    merge_dicts,
    merge_sets,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "encoding_power_behavior_regressions"
TARGET_COLUMN = "avg_stim_dprime_diff"
TARGET_LABEL = "dprime difference"
BAND_SPECS = {
    "theta": (4.0, 8.0),
    "slow_gamma": (35.0, 50.0),
}
BAND_LABELS = {
    "theta": "Theta (4-8 Hz)",
    "slow_gamma": "Slow Gamma (35-50 Hz)",
}
LOGIC_NOTE = (
    "X-axis baseline-corrected power: within each patient and encoding ROI, average the "
    "baseline-corrected stim and nostim spectra separately for remembered and forgotten trials, "
    "then average within the selected band. Composite ROIs average patient mean spectra across "
    "source single-region ROIs before the band-average scalar is taken. Y-axis memory modulation: "
    "dprime difference."
)


def resolve_behavior_csv():
    candidates = [
        SCRIPT_DIR / "AMMEBLAES_includedpts_firstsession_behavioral.csv",
        SCRIPT_DIR.parent / "AMMEBLAES_includedpts_firstsession_behavioral.csv",
        SCRIPT_DIR / "behavioral figures" / "AMMEBLAES_includedpts_firstsession_behavioral.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate AMMEBLAES_includedpts_firstsession_behavioral.csv")


BEHAVIOR_CSV = resolve_behavior_csv()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_all_encoding_data():
    blaes = load_blaes_encoding()
    amme = load_amme_encoding()
    return {
        "group_all_power": merge_dicts(blaes["group_all_power"], amme["group_all_power"]),
        "stim_power": merge_dicts(blaes["stim_power"], amme["stim_power"]),
        "nostim_power": merge_dicts(blaes["nostim_power"], amme["nostim_power"]),
        "bc_stim": merge_dicts(blaes["bc_stim"], amme["bc_stim"]),
        "bc_nostim": merge_dicts(blaes["bc_nostim"], amme["bc_nostim"]),
        "freqs_post": blaes["freqs_post"] if blaes["freqs_post"] is not None else amme["freqs_post"],
        "freqs_diff": blaes["freqs_diff"] if blaes["freqs_diff"] is not None else amme["freqs_diff"],
        "has_memory": True,
        "use_memory_collapsed_overall": False,
        "bc_bar_exclude_rois": merge_sets(
            blaes.get("bc_bar_exclude_rois"),
            amme.get("bc_bar_exclude_rois"),
        ),
        "bc_memory_collapsed_exclude_rois": merge_sets(
            blaes.get("bc_memory_collapsed_exclude_rois"),
            amme.get("bc_memory_collapsed_exclude_rois"),
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


def regression_rois(group_all_power):
    return sorted(roi for roi in group_all_power.keys() if not str(roi).startswith("PNAS"))


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


def plot_behavior_histogram(behavior_df, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    values = behavior_df[TARGET_COLUMN].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    bins = min(12, max(5, int(np.sqrt(len(values)))))
    ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white", alpha=0.9)
    ax.set_xlabel(f"Memory Modulation ({TARGET_LABEL})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=13, fontweight="bold")
    ax.set_title(f"Distribution of {TARGET_LABEL}", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def main():
    reset_dir(OUTPUT_DIR)
    behavior = load_behavior()
    all_data = load_all_encoding_data()
    memory_power = {
        "remembered": average_condition_region_dicts(all_data["bc_stim_rem"], all_data["bc_nostim_rem"]),
        "forgotten": average_condition_region_dicts(all_data["bc_stim_forg"], all_data["bc_nostim_forg"]),
    }
    source_rois = regression_rois(average_condition_region_dicts(*memory_power.values()))
    power_df = build_memory_band_table(memory_power, all_data["freqs_diff"], BAND_SPECS, "baseline_power")
    power_df = power_df[power_df["Region"].isin(source_rois)].copy()
    power_df = power_df[np.isfinite(power_df["baseline_power"])].copy()

    joined = power_df.merge(behavior, on="Patient", how="inner")
    joined = joined.dropna(subset=["baseline_power", TARGET_COLUMN]).copy()
    joined = joined[np.isfinite(joined["baseline_power"]) & np.isfinite(joined[TARGET_COLUMN])].copy()
    if joined.empty:
        raise RuntimeError("No overlapping patients between encoding power and behavioral memory-modulation data.")

    detail_dir = ensure_dir(OUTPUT_DIR / "plots")
    hist_dir = ensure_dir(OUTPUT_DIR / "histograms")
    summary_rows = []

    matched_behavior = joined[["Patient", TARGET_COLUMN]].drop_duplicates(subset=["Patient"]).sort_values("Patient")
    plot_behavior_histogram(matched_behavior, hist_dir / "dprime_difference_histogram.png")

    for roi in source_rois:
        for band_name in BAND_SPECS:
            roi_df = joined[(joined["Region"] == roi) & (joined["Band"] == band_name)].copy()
            roi_df = roi_df.drop_duplicates(subset=["Patient", "Memory"])
            if roi_df.empty:
                continue

            plot_memory_panel_histogram(
                roi_df,
                "baseline_power",
                BAND_LABELS[band_name],
                "Baseline-Corrected Power",
                "Encoding Power Distribution",
                hist_dir / f"{roi}_{band_name}_baseline_power_histogram.png",
            )
            stats_by_memory = {}
            for memory_name in MEMORY_ORDER:
                memory_df = roi_df[roi_df["Memory"] == memory_name].copy()
                if memory_df.empty:
                    continue
                stats_row = fit_regression(memory_df, "baseline_power", TARGET_COLUMN, "baseline_power")
                stats_row["Region"] = roi
                stats_row["Band"] = band_name
                stats_row["Memory"] = memory_name
                summary_rows.append(stats_row)
                stats_by_memory[memory_name] = stats_row

            roi_csv = detail_dir / f"{roi}_{band_name}_baseline_power_vs_memory_modulation.csv"
            roi_df.sort_values(["Memory", "Patient"]).to_csv(roi_csv, index=False)
            plot_memory_panel_regression(
                roi_df,
                stats_by_memory,
                "baseline_power",
                TARGET_COLUMN,
                TARGET_LABEL,
                BAND_LABELS[band_name],
                "Baseline-Corrected Power",
                "Encoding Power vs Memory Modulation",
                detail_dir / f"{roi}_{band_name}_baseline_power_vs_memory_modulation.png",
                LOGIC_NOTE,
                point_color="#355c7d",
                line_color="#c06c84",
                undefined_label="baseline_power",
            )

    if not summary_rows:
        raise RuntimeError("No encoding power regressions could be fit; check patient overlap and baseline power variability.")

    summary_df = pd.DataFrame(summary_rows)[
        ["Region", "Band", "Memory", "status", "n", "slope", "intercept", "r", "r_squared", "p_value", "stderr", "intercept_stderr"]
    ].sort_values(["Region", "Band", "Memory"])
    summary_df.to_csv(OUTPUT_DIR / "encoding_power_behavior_regression_summary.csv", index=False)
    joined.sort_values(["Region", "Band", "Memory", "Patient"]).to_csv(
        OUTPUT_DIR / "encoding_power_behavior_regression_joined_data.csv",
        index=False,
    )
    (OUTPUT_DIR / "logic.txt").write_text(LOGIC_NOTE + "\n", encoding="utf-8")

    print(f"Wrote encoding power regression outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
