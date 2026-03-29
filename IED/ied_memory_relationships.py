# %% [markdown]
# # IED Measures vs Subsequent Memory
#
# This notebook relates patient-level IED summary measures to the behavioral memory outcome
# in `AMMEBLAES_includedpts_firstsession_behavioral.csv`.
#
# Note:
# - The behavioral file does not contain a column literally named `avg_dprime_diff`.
# - The available memory outcome in that file is `avg_stim_dprime_diff`, which is used here.
# - Each point in the plots is one patient.
# - `Spearman rho` is the rank-based correlation coefficient.
#
# Outputs from this analysis are written under `outputs/ied_memory_relationships/`.

# %%
from pathlib import Path
import os

import matplotlib
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


def running_in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except ImportError:
        return False
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


if not running_in_notebook():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None
    spearmanr = None

BASE_DIR = Path.cwd()
BEHAVIOR_PATH = BASE_DIR / "AMMEBLAES_includedpts_firstsession_behavioral.csv"
if not BEHAVIOR_PATH.exists():
    BEHAVIOR_PATH = BASE_DIR / "behavioral figures" / "AMMEBLAES_includedpts_firstsession_behavioral.csv"
IED_DIR = Path(os.environ.get("IED_SUMMARY_DIR", str(BASE_DIR / "outputs" / "ied_trial_level_summary")))
OUTPUT_DIR = Path(os.environ.get("IED_MEMORY_OUTPUT_DIR", str(BASE_DIR / "outputs" / "ied_memory_relationships")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("default")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300


def compute_correlation_frame(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    subset = df[[x_col, y_col]].dropna()
    result = {
        "x_measure": x_col,
        "y_measure": y_col,
        "n": len(subset),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(subset) < 2:
        return result
    if pearsonr is not None:
        pearson_r, pearson_p = pearsonr(subset[x_col], subset[y_col])
        result["pearson_r"] = pearson_r
        result["pearson_p"] = pearson_p
    else:
        result["pearson_r"] = subset[x_col].corr(subset[y_col], method="pearson")
    if spearmanr is not None:
        spearman_rho, spearman_p = spearmanr(subset[x_col], subset[y_col])
        result["spearman_rho"] = spearman_rho
        result["spearman_p"] = spearman_p
    else:
        result["spearman_rho"] = subset[x_col].corr(subset[y_col], method="spearman")
    return result


def scatter_with_fit(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    color: str,
) -> dict:
    subset = df[[x_col, y_col, "Patient"]].dropna().copy()

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(
        subset[x_col],
        subset[y_col],
        color=color,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9,
        s=55,
    )

    if len(subset) >= 2:
        fit = np.polyfit(subset[x_col], subset[y_col], 1)
        x_line = np.linspace(subset[x_col].min(), subset[x_col].max(), 100)
        y_line = fit[0] * x_line + fit[1]
        ax.plot(x_line, y_line, color="black", linewidth=1.2)

    stats = compute_correlation_frame(subset, x_col, y_col)

    annotation_lines = [f"n = {stats['n']}"]
    if not np.isnan(stats["pearson_r"]):
        if not np.isnan(stats["pearson_p"]):
            annotation_lines.append(f"Pearson r = {stats['pearson_r']:.2f}, p = {stats['pearson_p']:.3f}")
        else:
            annotation_lines.append(f"Pearson r = {stats['pearson_r']:.2f}")
    if not np.isnan(stats["spearman_rho"]):
        if not np.isnan(stats["spearman_p"]):
            annotation_lines.append(
                f"Spearman rho = {stats['spearman_rho']:.2f}, p = {stats['spearman_p']:.3f}"
            )
        else:
            annotation_lines.append(f"Spearman rho = {stats['spearman_rho']:.2f}")
    ax.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)
    return stats


# %% [markdown]
# ## Load Behavioral And IED Summary Data

# %%
behavior_df = pd.read_csv(BEHAVIOR_PATH)
ied_rate = pd.read_csv(IED_DIR / "patient_iedrateavg_summary.csv")
ied_channel = pd.read_csv(IED_DIR / "patient_channelspread_summary.csv")
ied_region = pd.read_csv(IED_DIR / "patient_regionspread_summary.csv")
ied_trial = pd.read_csv(IED_DIR / "patient_trial_summary.csv")

behavior_df["Patient"] = behavior_df["Patient"].astype("string").str.strip()
behavior_df["avg_stim_dprime_diff"] = pd.to_numeric(behavior_df["avg_stim_dprime_diff"], errors="coerce")

for frame in [ied_rate, ied_channel, ied_region, ied_trial]:
    frame["Patient"] = frame["Patient"].astype("string").str.strip()

print(f"Loaded behavioral rows: {len(behavior_df):,}")
print(f"Loaded IED-rate rows: {len(ied_rate):,}")


# %% [markdown]
# ## Merge At The Patient Level

# %%
merged = (
    behavior_df[
        [
            "Patient",
            "Study",
            "avg_stim_dprime_diff",
            "IED_freq",
            "IED_laterality",
            "avg_stim",
            "nostim",
        ]
    ]
    .merge(ied_rate[["Patient", "IEDRateAvg", "trials_with_weighted_iedrate"]], on="Patient", how="inner")
    .merge(ied_channel[["Patient", "PatientChannelSpread"]], on="Patient", how="left")
    .merge(ied_region[["Patient", "PatientRegionSpread"]], on="Patient", how="left")
    .merge(ied_trial[["Patient", "unique_ied_trials"]], on="Patient", how="left")
)

merged.to_csv(OUTPUT_DIR / "memory_vs_ied_merged.csv", index=False)

print(f"Merged patient count: {len(merged):,}")
display(merged.head(12))


# %% [markdown]
# ## Relationship Measures

# %%
memory_measure = "avg_stim_dprime_diff"
ied_measures = {
    "IEDRateAvg": {
        "title": "Memory vs IEDRateAvg",
        "xlabel": "IEDRateAvg (average of trial-level WeightedIEDRate)",
        "color": "#C97C10",
        "filename": "memory_vs_IEDRateAvg.png",
    },
    "PatientChannelSpread": {
        "title": "Memory vs PatientChannelSpread",
        "xlabel": "PatientChannelSpread",
        "color": "#2A6F97",
        "filename": "memory_vs_PatientChannelSpread.png",
    },
    "PatientRegionSpread": {
        "title": "Memory vs PatientRegionSpread",
        "xlabel": "PatientRegionSpread",
        "color": "#7F5539",
        "filename": "memory_vs_PatientRegionSpread.png",
    },
    "unique_ied_trials": {
        "title": "Memory vs Unique IED-Positive Trials",
        "xlabel": "Unique IED-positive trials",
        "color": "#5B8E7D",
        "filename": "memory_vs_unique_ied_trials.png",
    },
}


# %% [markdown]
# ## Scatter Plots And Correlations

# %%
correlation_rows = []
for measure, meta in ied_measures.items():
    stats = scatter_with_fit(
        merged,
        x_col=measure,
        y_col=memory_measure,
        title=meta["title"],
        xlabel=meta["xlabel"],
        ylabel="avg_stim_dprime_diff",
        filename=meta["filename"],
        color=meta["color"],
    )
    correlation_rows.append(stats)

correlation_summary = pd.DataFrame(correlation_rows)
correlation_summary.to_csv(OUTPUT_DIR / "memory_ied_correlation_summary.csv", index=False)
display(correlation_summary)


# %% [markdown]
# ## Combined Overview Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, (measure, meta) in zip(axes, ied_measures.items()):
    subset = merged[[measure, memory_measure]].dropna()
    ax.scatter(
        subset[measure],
        subset[memory_measure],
        color=meta["color"],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9,
        s=50,
    )
    if len(subset) >= 2:
        fit = np.polyfit(subset[measure], subset[memory_measure], 1)
        x_line = np.linspace(subset[measure].min(), subset[measure].max(), 100)
        y_line = fit[0] * x_line + fit[1]
        ax.plot(x_line, y_line, color="black", linewidth=1.1)

    stats_row = correlation_summary.loc[correlation_summary["x_measure"] == measure].iloc[0]
    ax.set_title(meta["title"])
    ax.set_xlabel(meta["xlabel"])
    ax.set_ylabel("avg_stim_dprime_diff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.03,
        0.97,
        (
            f"n = {int(stats_row['n'])}\n"
            f"Pearson r = {stats_row['pearson_r']:.2f}, p = {stats_row['pearson_p']:.3f}\n"
            f"Spearman rho = {stats_row['spearman_rho']:.2f}, p = {stats_row['spearman_p']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

fig.suptitle("Patient-Level IED Measures vs avg_stim_dprime_diff", y=1.02, fontsize=16)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "memory_vs_ied_combined.png", bbox_inches="tight")
if running_in_notebook():
    plt.show()
plt.close(fig)


# %% [markdown]
# ## Summary

# %%
print("Behavioral memory column used: avg_stim_dprime_diff")
print(f"Merged patients: {len(merged)}")

for _, row in correlation_summary.iterrows():
    print(
        f"{row['x_measure']}: "
        f"Pearson r = {row['pearson_r']:.3f} (p = {row['pearson_p']:.3f}), "
        f"Spearman rho = {row['spearman_rho']:.3f} (p = {row['spearman_p']:.3f}), "
        f"n = {int(row['n'])}"
    )

print(f"\nOutputs saved in: {OUTPUT_DIR}")
