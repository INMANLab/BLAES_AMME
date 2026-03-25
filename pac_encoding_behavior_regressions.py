#!/usr/bin/env python
"""
Patient-level regressions between combined-cohort encoding PAC baseline values
and memory modulation (avg_stim_dprime_diff).

Baseline PAC measure:
1. Average encoding PAC within each patient and original region pair across
   all trials to obtain one mean PAC spectrum.
2. For BLA composites, average those already-averaged per-pair spectra within
   each patient after the original region-pair means are formed.
3. Collapse the resulting spectrum within each configured PAC band to obtain
   one scalar baseline PAC value for regression.
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


SCRIPT_DIR = Path(__file__).resolve().parent

from combined_encoding_pac import load_grouped_encoding_pac_data
from combined_pac_common import COMPOSITE_LOGIC_TEXT, augment_with_bla_composites
from to_combine.BLAES_Group_PAC_analyses_from_matlab_encoding import PAC_BANDS


OUTPUT_DIR = SCRIPT_DIR / "outputs" / "PAC_encoding_behavior_regressions"
TARGET_COLUMN = "avg_stim_dprime_diff"
TARGET_LABEL = "dprime difference"
LOGIC_NOTE = (
    "X-axis baseline PAC: mean post_Freq encoding PAC across trials within patient and region pair, "
    "with BLA_ALLHPC and BLA_MTL formed only after those per-pair patient averages are computed, "
    "then mean within the selected PAC band. Y-axis memory modulation: dprime difference."
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


def build_baseline_pac_table(region_dict, freqs):
    rows = []
    freqs = np.asarray(freqs, dtype=np.float64)
    for roi, subj_dict in (region_dict or {}).items():
        for patient, values in subj_dict.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            for band_name, (lo, hi) in PAC_BANDS.items():
                mask = (freqs >= lo) & (freqs <= hi)
                if not np.any(mask):
                    continue
                rows.append(
                    {
                        "Patient": str(patient),
                        "Region": roi,
                        "Band": band_name,
                        "baseline_pac": float(np.nanmean(arr[mask])),
                    }
                )
    return pd.DataFrame(rows)


def load_behavior():
    behavior = pd.read_csv(BEHAVIOR_CSV)
    behavior = behavior[["Patient", TARGET_COLUMN]].copy()
    behavior["Patient"] = behavior["Patient"].astype(str)
    behavior[TARGET_COLUMN] = pd.to_numeric(behavior[TARGET_COLUMN], errors="coerce")
    behavior = behavior.dropna(subset=[TARGET_COLUMN]).copy()
    behavior = behavior[np.isfinite(behavior[TARGET_COLUMN])].copy()
    return behavior.drop_duplicates(subset=["Patient"])


def fit_regression(df):
    if df["baseline_pac"].nunique() < 2:
        return {
            "status": "undefined_baseline_pac",
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

    result = stats.linregress(df["baseline_pac"], df[TARGET_COLUMN])
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
            "undefined_baseline_pac": "baseline PAC has no variability",
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
    x = df["baseline_pac"].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    ax.scatter(x, y, s=55, alpha=0.8, color="#2f6c8f", edgecolors="white", linewidths=0.6)
    if stats_row["status"] == "ok":
        order = np.argsort(x)
        x_sorted = x[order]
        y_fit = stats_row["intercept"] + stats_row["slope"] * x_sorted
        ax.plot(x_sorted, y_fit, color="#b24c63", linewidth=2)

    band = df["Band"].iloc[0]
    lo, hi = PAC_BANDS[band]
    band_label = f"{band} ({int(lo)}-{int(hi)} Hz)"
    ax.set_xlabel(f"Baseline PAC ({band_label})", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Memory Modulation ({TARGET_LABEL})", fontsize=13, fontweight="bold")
    ax.set_title(f"Encoding PAC vs Memory Modulation: {df['Region'].iloc[0]} - {band_label}", fontsize=15, fontweight="bold")
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
    ax.set_xlabel(f"Memory Modulation ({TARGET_LABEL})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=13, fontweight="bold")
    ax.set_title(f"Distribution of {TARGET_LABEL}", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def plot_baseline_pac_histogram(df, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    values = df["baseline_pac"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    bins = min(12, max(5, int(np.sqrt(len(values)))))
    band = df["Band"].iloc[0]
    lo, hi = PAC_BANDS[band]
    band_label = f"{band} ({int(lo)}-{int(hi)} Hz)"
    ax.hist(values, bins=bins, color="#72b7b2", edgecolor="white", alpha=0.9)
    ax.set_xlabel(f"Baseline PAC ({band_label})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=13, fontweight="bold")
    ax.set_title(f"Baseline PAC Distribution: {df['Region'].iloc[0]} - {band_label}", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def main():
    reset_dir(OUTPUT_DIR)
    data = augment_with_bla_composites(load_grouped_encoding_pac_data()["all"])
    behavior = load_behavior()

    pac_df = build_baseline_pac_table(data["post_all"], data["freqs_post"])
    pac_df = pac_df[np.isfinite(pac_df["baseline_pac"])].copy()
    joined = pac_df.merge(behavior, on="Patient", how="inner")
    joined = joined.dropna(subset=["baseline_pac", TARGET_COLUMN]).copy()
    joined = joined[np.isfinite(joined["baseline_pac"]) & np.isfinite(joined[TARGET_COLUMN])].copy()
    if joined.empty:
        raise RuntimeError("No overlapping patients between encoding PAC and behavioral memory-modulation data.")

    detail_dir = ensure_dir(OUTPUT_DIR / "plots")
    hist_dir = ensure_dir(OUTPUT_DIR / "histograms")
    summary_rows = []

    matched_behavior = joined[["Patient", TARGET_COLUMN]].drop_duplicates(subset=["Patient"]).sort_values("Patient")
    plot_behavior_histogram(matched_behavior, hist_dir / "dprime_difference_histogram.png")

    for roi in sorted(joined["Region"].unique()):
        for band in PAC_BANDS:
            roi_df = joined[(joined["Region"] == roi) & (joined["Band"] == band)].copy()
            roi_df = roi_df.drop_duplicates(subset=["Patient"])
            if len(roi_df) < 3:
                continue
            plot_baseline_pac_histogram(roi_df, hist_dir / f"{roi}_{band}_baseline_pac_histogram.png")
            stats_row = fit_regression(roi_df)
            stats_row["Region"] = roi
            stats_row["Band"] = band
            summary_rows.append(stats_row)
            roi_df.sort_values("Patient").to_csv(detail_dir / f"{roi}_{band}_baseline_pac_vs_memory_modulation.csv", index=False)
            plot_regression(roi_df, stats_row, detail_dir / f"{roi}_{band}_baseline_pac_vs_memory_modulation.png")

    if not summary_rows:
        raise RuntimeError("No encoding PAC regressions could be fit; check patient overlap and baseline PAC variability.")

    summary_df = pd.DataFrame(summary_rows)[
        ["Region", "Band", "status", "n", "slope", "intercept", "r", "r_squared", "p_value", "stderr", "intercept_stderr"]
    ].sort_values(["Region", "Band"])
    summary_df.to_csv(OUTPUT_DIR / "pac_encoding_behavior_regression_summary.csv", index=False)
    joined.sort_values(["Region", "Band", "Patient"]).to_csv(
        OUTPUT_DIR / "pac_encoding_behavior_regression_joined_data.csv",
        index=False,
    )
    (OUTPUT_DIR / "logic.txt").write_text(LOGIC_NOTE + "\n" + COMPOSITE_LOGIC_TEXT + "\n", encoding="utf-8")
    print(f"Wrote PAC encoding regression outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
