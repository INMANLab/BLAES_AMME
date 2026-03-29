# %% [markdown]
# # AMMEBLAES IED Trial-Level Summary
#
# This notebook reads the cleaned CSV in `IED/AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned.csv`
# and summarizes IED-positive trials within and across patients.
#
# Analysis rules used here:
# - Duplicate rows with the same `Patient + Trial` are collapsed to a single trial for trial-based summaries.
# - `StimCond` is taken as the unique trial label within each `Patient + Trial`.
# - The timing flags (`DuringImg`, `DuringStim`, `BeforeImgITI`, `AfterImgITI`) are collapsed with `any(Y)`.
#   If any row for that patient/trial is `Y`, the collapsed trial is marked `Y` for that condition.
# - The `GrayMatter` summary is computed from row-level data because tissue is a channel-level attribute.
#   The `G` vs `W` distribution excludes `B` and missing values; those are saved separately for QA.
#
# Outputs from this analysis are written under `outputs/ied_trial_level_summary/`.

# %%
from pathlib import Path
import os

import matplotlib
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

BASE_DIR = Path.cwd()
DEFAULT_INPUT_NAME = "AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned.csv"
INPUT_NAME = os.environ.get("IED_TRIAL_INPUT_NAME", DEFAULT_INPUT_NAME)
CSV_PATH = BASE_DIR / "IED" / INPUT_NAME

if INPUT_NAME == DEFAULT_INPUT_NAME:
    OUTPUT_DIR = BASE_DIR / "outputs" / "ied_trial_level_summary"
else:
    OUTPUT_DIR = BASE_DIR / "outputs" / "ied_trial_level_summary" / Path(INPUT_NAME).stem

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUGMENTED_CSV_PATH = OUTPUT_DIR / f"{Path(INPUT_NAME).stem}_with_IEDRateAvg.csv"

plt.style.use("default")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300


def first_nonmissing(series: pd.Series):
    nonmissing = series.dropna()
    if nonmissing.empty:
        return pd.NA
    return nonmissing.iloc[0]


def any_yes(series: pd.Series) -> str:
    return "Y" if (series.fillna("N") == "Y").any() else "N"


def split_channel_tokens(value) -> list[str]:
    if pd.isna(value):
        return []
    tokens = [token.strip() for token in str(value).split("_")]
    return [token for token in tokens if token]


def save_barplot(series: pd.Series, title: str, ylabel: str, filename: str, color: str) -> None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(series.index, series.values, color=color, edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for idx, value in enumerate(series.values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def save_grouped_barplot(
    frame: pd.DataFrame,
    title: str,
    ylabel: str,
    filename: str,
    colors: list[str],
) -> None:
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x_positions = range(len(frame.index))
    width = 0.8 / max(len(frame.columns), 1)

    for idx, column in enumerate(frame.columns):
        offset = (idx - (len(frame.columns) - 1) / 2) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            frame[column].values,
            width=width,
            label=column,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(frame.index, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def save_horizontal_grouped_barplot(
    frame: pd.DataFrame,
    title: str,
    xlabel: str,
    filename: str,
    colors: list[str],
) -> None:
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(frame.index))))
    y_positions = range(len(frame.index))
    height = 0.8 / max(len(frame.columns), 1)

    for idx, column in enumerate(frame.columns):
        offset = (idx - (len(frame.columns) - 1) / 2) * height
        ax.barh(
            [y + offset for y in y_positions],
            frame[column].values,
            height=height,
            label=column,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(frame.index)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def save_horizontal_barplot(
    series: pd.Series,
    title: str,
    xlabel: str,
    filename: str,
    color: str,
) -> None:
    series = pd.to_numeric(series, errors="coerce")
    series = series[series.index.to_series().notna()].dropna()
    if series.empty:
        return
    labels = series.index.astype(str)
    values = series.astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(series.index))))
    ax.barh(labels, values, color=color, edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:.2f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def save_horizontal_boxplot(
    grouped_series: pd.Series,
    title: str,
    xlabel: str,
    filename: str,
    color: str,
) -> None:
    grouped = grouped_series.groupby(level=0)
    labels = list(grouped.groups.keys())
    values = [grouped.get_group(label).dropna().tolist() for label in labels]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(labels))))
    box = ax.boxplot(values, vert=False, patch_artist=True, tick_labels=labels)
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## Load And Standardize The CSV

# %%
df = pd.read_csv(CSV_PATH)

# Harmonize alternate column labels so study and test CSVs can flow through one pipeline.
if "DuringImg" not in df.columns and "DuringImgITI" in df.columns:
    df["DuringImg"] = df["DuringImgITI"]
if "StimCond" not in df.columns and "Test" in df.columns:
    df["StimCond"] = df["Test"]

for required_col, default_value in {
    "StimCond": pd.NA,
    "DuringImg": "N",
    "DuringStim": "N",
    "BeforeImgITI": "N",
    "AfterImgITI": "N",
}.items():
    if required_col not in df.columns:
        df[required_col] = default_value

string_columns = [
    "Patient",
    "Region",
    "GrayMatter",
    "Hemisphere",
    "ChannelName",
    "StimCond",
    "DuringImg",
    "DuringStim",
    "BeforeImgITI",
    "AfterImgITI",
]

for column in string_columns:
    if column not in df.columns:
        continue
    df[column] = df[column].astype("string").str.strip()
    df[column] = df[column].replace({"": pd.NA})

df["StimCond"] = df["StimCond"].str.upper()
df["StimCond"] = df["StimCond"].replace({"Y": "S", "N": "NS"})
for column in ["DuringImg", "DuringStim", "BeforeImgITI", "AfterImgITI", "GrayMatter"]:
    df[column] = df[column].str.upper()

df["Trial"] = pd.to_numeric(df["Trial"], errors="coerce").astype("Int64")
df["WeightedIEDRate"] = pd.to_numeric(df["WeightedIEDRate"], errors="coerce")
df["IEDRateAvg"] = pd.to_numeric(df["IEDRateAvg"], errors="coerce")
condition_columns = ["DuringImg", "DuringStim", "BeforeImgITI", "AfterImgITI"]

print(f"Loaded {len(df):,} rows from {CSV_PATH}")
display(df.head())


# %% [markdown]
# ## Compute TrialChannelSpread And PatientChannelSpread
#
# `TrialChannelSpread` counts the number of unique channels observed within each `Patient + Trial`
# after splitting `ChannelName` on `_`.
# `PatientChannelSpread` is the average `TrialChannelSpread` across all trials within a patient.

# %%
trial_channelspread = (
    df.groupby(["Patient", "Trial"], dropna=False)["ChannelName"]
    .agg(
        TrialChannelSpread=lambda s: len(
            {
                channel
                for value in s.dropna()
                for channel in split_channel_tokens(value)
            }
        )
    )
    .reset_index()
)

patient_channelspread = (
    trial_channelspread.groupby("Patient", dropna=False, as_index=False)
    .agg(PatientChannelSpread=("TrialChannelSpread", "mean"))
)

df = df.merge(trial_channelspread, on=["Patient", "Trial"], how="left")
df = df.merge(patient_channelspread, on="Patient", how="left")

trial_channelspread.to_csv(OUTPUT_DIR / "trial_channelspread_summary.csv", index=False)
patient_channelspread.to_csv(OUTPUT_DIR / "patient_channelspread_summary.csv", index=False)

print("TrialChannelSpread and PatientChannelSpread added.")
display(trial_channelspread.head())
display(patient_channelspread.head())


# %% [markdown]
# ## Compute TrialRegionSpread And PatientRegionSpread
#
# `TrialRegionSpread` counts the number of unique non-missing `Region` labels within each
# `Patient + Trial`. `PatientRegionSpread` is the average `TrialRegionSpread` across all trials
# within a patient.

# %%
trial_regionspread = (
    df.groupby(["Patient", "Trial"], dropna=False)["Region"]
    .agg(TrialRegionSpread=lambda s: s.dropna().nunique())
    .reset_index()
)

patient_regionspread = (
    trial_regionspread.groupby("Patient", dropna=False, as_index=False)
    .agg(PatientRegionSpread=("TrialRegionSpread", "mean"))
)

df = df.merge(trial_regionspread, on=["Patient", "Trial"], how="left")
df = df.merge(patient_regionspread, on="Patient", how="left")

trial_regionspread.to_csv(OUTPUT_DIR / "trial_regionspread_summary.csv", index=False)
patient_regionspread.to_csv(OUTPUT_DIR / "patient_regionspread_summary.csv", index=False)

print("TrialRegionSpread and PatientRegionSpread added.")
display(trial_regionspread.head())
display(patient_regionspread.head())


# %% [markdown]
# ## Compute Subject-Level IEDRateAvg From Trial-Level WeightedIEDRate
#
# `IEDRateAvg` is computed in two steps:
# 1. Collapse duplicate `Patient + Trial` rows by averaging `WeightedIEDRate` within each trial.
# 2. Average those trial-level values across all available trials within each patient.
#
# The subject-level result is then written back onto every row for that patient.

# %%
trial_weighted_iedrate = (
    df.groupby(["Patient", "Trial"], dropna=False, as_index=False)
    .agg(trial_weighted_iedrate_mean=("WeightedIEDRate", "mean"))
)

patient_iedrateavg = (
    trial_weighted_iedrate.groupby("Patient", dropna=False, as_index=False)
    .agg(
        IEDRateAvg=("trial_weighted_iedrate_mean", "mean"),
        trials_with_weighted_iedrate=("trial_weighted_iedrate_mean", lambda s: s.notna().sum()),
    )
)

df = df.drop(columns=["IEDRateAvg"]).merge(patient_iedrateavg[["Patient", "IEDRateAvg"]], on="Patient", how="left")

trial_weighted_iedrate.to_csv(OUTPUT_DIR / "trial_weighted_iedrate_summary.csv", index=False)
patient_iedrateavg.to_csv(OUTPUT_DIR / "patient_iedrateavg_summary.csv", index=False)
df.to_csv(AUGMENTED_CSV_PATH, index=False)

print(f"Updated CSV with populated IEDRateAvg written to: {AUGMENTED_CSV_PATH}")
display(patient_iedrateavg.head())


# %% [markdown]
# ## Patient-Level PatientChannelSpread Figure

# %%
patient_channelspread_plot = (
    patient_channelspread.sort_values("PatientChannelSpread", ascending=True, kind="stable")
    .set_index("Patient")["PatientChannelSpread"]
)

save_horizontal_barplot(
    patient_channelspread_plot,
    title="PatientChannelSpread By Patient",
    xlabel="Average unique channels per trial",
    filename="patient_channelspread_by_patient.png",
    color="#2A6F97",
)


# %% [markdown]
# ## Patient-Level IEDRateAvg Figure
#
# `IEDRateAvg` shown here is the patient-level average of trial-level `WeightedIEDRate`.

# %%
patient_iedrateavg_plot = (
    patient_iedrateavg.sort_values("IEDRateAvg", ascending=True, kind="stable")
    .set_index("Patient")["IEDRateAvg"]
)

save_horizontal_barplot(
    patient_iedrateavg_plot,
    title="IEDRateAvg By Patient (Average of Trial-Level WeightedIEDRate)",
    xlabel="IEDRateAvg",
    filename="patient_iedrateavg_by_patient.png",
    color="#C97C10",
)


# %% [markdown]
# ## Patient-Level PatientRegionSpread Figure

# %%
patient_regionspread_plot = (
    patient_regionspread.sort_values("PatientRegionSpread", ascending=True, kind="stable")
    .set_index("Patient")["PatientRegionSpread"]
)

save_horizontal_barplot(
    patient_regionspread_plot,
    title="PatientRegionSpread By Patient",
    xlabel="Average unique regions per trial",
    filename="patient_regionspread_by_patient.png",
    color="#7F5539",
)


# %% [markdown]
# ## Trial-Level RegionSpread Distribution By Patient

# %%
region_distribution_order = (
    patient_regionspread.sort_values("PatientRegionSpread", ascending=True, kind="stable")["Patient"]
)
trial_regionspread_distribution = (
    trial_regionspread.set_index("Patient").loc[region_distribution_order, "TrialRegionSpread"]
)

save_horizontal_boxplot(
    trial_regionspread_distribution,
    title="Trial-Level RegionSpread Distribution By Patient",
    xlabel="Unique regions within a trial",
    filename="trial_regionspread_distribution_by_patient.png",
    color="#B08968",
)


# %% [markdown]
# ## Collapse To Unique Patient-Trial Rows

# %%
trial_level = (
    df.groupby(["Patient", "Trial"], dropna=False, as_index=False)
    .agg(
        StimCond=("StimCond", first_nonmissing),
        DuringImg=("DuringImg", any_yes),
        DuringStim=("DuringStim", any_yes),
        BeforeImgITI=("BeforeImgITI", any_yes),
        AfterImgITI=("AfterImgITI", any_yes),
    )
    .sort_values(["Patient", "Trial"], kind="stable")
)

trial_level.to_csv(OUTPUT_DIR / "collapsed_trial_level.csv", index=False)

print(f"Collapsed to {len(trial_level):,} unique patient-trial rows")
display(trial_level.head())


# %% [markdown]
# ## Per-Patient Trial Summary

# %%
patient_trial_summary = (
    trial_level.groupby("Patient", dropna=False)
    .agg(
        unique_ied_trials=("Trial", "count"),
        stimulated_trials=("StimCond", lambda s: (s == "S").sum()),
        nonstim_trials=("StimCond", lambda s: (s == "NS").sum()),
        during_img_trials=("DuringImg", lambda s: (s == "Y").sum()),
        during_stim_trials=("DuringStim", lambda s: (s == "Y").sum()),
        before_img_iti_trials=("BeforeImgITI", lambda s: (s == "Y").sum()),
        after_img_iti_trials=("AfterImgITI", lambda s: (s == "Y").sum()),
    )
    .reset_index()
)

count_columns = [
    "stimulated_trials",
    "nonstim_trials",
    "during_img_trials",
    "during_stim_trials",
    "before_img_iti_trials",
    "after_img_iti_trials",
]
for column in count_columns:
    patient_trial_summary[f"{column}_pct"] = (
        patient_trial_summary[column] / patient_trial_summary["unique_ied_trials"]
    )

patient_trial_summary.to_csv(OUTPUT_DIR / "patient_trial_summary.csv", index=False)
display(patient_trial_summary.head())


# %% [markdown]
# ## Across-Patient Averages

# %%
overall_trial_summary = pd.DataFrame(
    {
        "metric": [
            "Average unique IED-positive trials per patient",
            "Average stimulated trials per patient",
            "Average non-stimulated trials per patient",
            "Average proportion DuringImg",
            "Average proportion DuringStim",
            "Average proportion BeforeImgITI",
            "Average proportion AfterImgITI",
        ],
        "value": [
            patient_trial_summary["unique_ied_trials"].mean(),
            patient_trial_summary["stimulated_trials"].mean(),
            patient_trial_summary["nonstim_trials"].mean(),
            patient_trial_summary["during_img_trials_pct"].mean(),
            patient_trial_summary["during_stim_trials_pct"].mean(),
            patient_trial_summary["before_img_iti_trials_pct"].mean(),
            patient_trial_summary["after_img_iti_trials_pct"].mean(),
        ],
    }
)

overall_trial_summary.to_csv(OUTPUT_DIR / "overall_trial_summary.csv", index=False)
display(overall_trial_summary)


# %% [markdown]
# ## Plot Average Trial-Level Condition Distribution

# %%
condition_plot_values = pd.Series(
    {
        "DuringImg": patient_trial_summary["during_img_trials_pct"].mean() * 100,
        "DuringStim": patient_trial_summary["during_stim_trials_pct"].mean() * 100,
        "BeforeImgITI": patient_trial_summary["before_img_iti_trials_pct"].mean() * 100,
        "AfterImgITI": patient_trial_summary["after_img_iti_trials_pct"].mean() * 100,
    }
)

save_barplot(
    condition_plot_values,
    title="Average Timing Distribution Across Patients",
    ylabel="Average % of unique IED-positive trials",
    filename="average_condition_distribution.png",
    color="#5B8E7D",
)

condition_plot_values.to_csv(OUTPUT_DIR / "average_condition_distribution.csv", header=["average_percent"])
condition_plot_values


# %% [markdown]
# ## Plot Average Stimulated vs Non-Stimulated Trial Counts

# %%
stim_plot_values = pd.Series(
    {
        "S": patient_trial_summary["stimulated_trials"].mean(),
        "NS": patient_trial_summary["nonstim_trials"].mean(),
    }
)

save_barplot(
    stim_plot_values,
    title="Average Unique IED-Positive Trials Per Patient by StimCond",
    ylabel="Average count of unique trials",
    filename="average_stim_distribution.png",
    color="#D98E04",
)

stim_plot_values.to_csv(OUTPUT_DIR / "average_stim_distribution.csv", header=["average_count"])
stim_plot_values


# %% [markdown]
# ## Gray Matter vs White Matter Distribution

# %%
graymatter_qa = (
    df.groupby("Patient", dropna=False)["GrayMatter"]
    .value_counts(dropna=False)
    .unstack(fill_value=0)
    .reset_index()
)
graymatter_qa.to_csv(OUTPUT_DIR / "patient_graymatter_qa_counts.csv", index=False)

graymatter_gw = df[df["GrayMatter"].isin(["G", "W"])].copy()

patient_graymatter_summary = (
    graymatter_gw.groupby("Patient", dropna=False)["GrayMatter"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
    .pivot(index="Patient", columns="GrayMatter", values="proportion")
    .fillna(0)
    .reset_index()
)

patient_graymatter_summary.to_csv(OUTPUT_DIR / "patient_graymatter_gw_summary.csv", index=False)
display(patient_graymatter_summary.head())


# %% [markdown]
# ## Plot Average Gray Matter vs White Matter Distribution

# %%
graymatter_plot_values = pd.Series(
    {
        "G": patient_graymatter_summary.get("G", pd.Series(dtype=float)).mean() * 100,
        "W": patient_graymatter_summary.get("W", pd.Series(dtype=float)).mean() * 100,
    }
)

save_barplot(
    graymatter_plot_values,
    title="Average G vs W Distribution Across Patients",
    ylabel="Average % of row-level IED observations",
    filename="average_graymatter_distribution.png",
    color="#4C78A8",
)

graymatter_plot_values.to_csv(OUTPUT_DIR / "average_graymatter_distribution.csv", header=["average_percent"])
graymatter_plot_values


# %% [markdown]
# ## Condition Distribution In The Context Of Stimulation

# %%
trial_condition_binary = trial_level.copy()
for column in condition_columns:
    trial_condition_binary[column] = trial_condition_binary[column].eq("Y").astype(int)

condition_by_stim_counts = (
    trial_condition_binary.groupby("StimCond")[condition_columns].sum().reindex(["S", "NS"])
)
stim_trial_totals = trial_condition_binary["StimCond"].value_counts().reindex(["S", "NS"])
condition_by_stim_percent = condition_by_stim_counts.div(stim_trial_totals, axis=0) * 100

condition_by_stim_counts.to_csv(OUTPUT_DIR / "condition_by_stimulation_counts.csv")
condition_by_stim_percent.to_csv(OUTPUT_DIR / "condition_by_stimulation_percent.csv")

display(condition_by_stim_counts)
display(condition_by_stim_percent.round(2))

save_grouped_barplot(
    condition_by_stim_percent.T,
    title="Condition Distribution Within Stimulated vs Non-Stimulated Trials",
    ylabel="% of unique trials within StimCond",
    filename="condition_by_stimulation_percent.png",
    colors=["#D98E04", "#7A8FA6"],
)


# %% [markdown]
# ## Condition Distribution By Gray Matter vs White Matter
#
# This section uses row-level observations restricted to `GrayMatter` values `G` and `W`.

# %%
condition_by_gray_counts = (
    graymatter_gw.groupby("GrayMatter")[condition_columns]
    .agg(lambda s: (s == "Y").sum())
    .reindex(["G", "W"])
)
graymatter_totals = graymatter_gw["GrayMatter"].value_counts().reindex(["G", "W"])
condition_by_gray_percent = condition_by_gray_counts.div(graymatter_totals, axis=0) * 100

condition_by_gray_counts.to_csv(OUTPUT_DIR / "condition_by_graymatter_counts.csv")
condition_by_gray_percent.to_csv(OUTPUT_DIR / "condition_by_graymatter_percent.csv")

display(condition_by_gray_counts)
display(condition_by_gray_percent.round(2))

save_grouped_barplot(
    condition_by_gray_percent.T,
    title="Condition Distribution Within Gray vs White Matter",
    ylabel="% of row-level observations within tissue class",
    filename="condition_by_graymatter_percent.png",
    colors=["#4C78A8", "#E45756"],
)


# %% [markdown]
# ## Region Distribution And Dominant Gray vs White Matter
#
# Dominance is called from `G` vs `W` counts only. Regions with no `G` or `W` rows are labeled
# `No G/W observations`.

# %%
region_tissue_counts = pd.crosstab(
    df["Region"].fillna("Missing"),
    df["GrayMatter"].fillna("Missing"),
)
for column in ["G", "W", "B", "Missing"]:
    if column not in region_tissue_counts.columns:
        region_tissue_counts[column] = 0

region_tissue_summary = (
    region_tissue_counts[["G", "W", "B", "Missing"]]
    .reset_index()
    .rename(columns={"Region": "Region"})
)
region_tissue_summary["gw_total"] = region_tissue_summary["G"] + region_tissue_summary["W"]
region_tissue_summary["gray_pct_within_gw"] = (
    region_tissue_summary["G"] / region_tissue_summary["gw_total"]
).where(region_tissue_summary["gw_total"] > 0)
region_tissue_summary["white_pct_within_gw"] = (
    region_tissue_summary["W"] / region_tissue_summary["gw_total"]
).where(region_tissue_summary["gw_total"] > 0)


def dominant_gw_tissue(row: pd.Series) -> str:
    if row["gw_total"] == 0:
        return "No G/W observations"
    if row["G"] > row["W"]:
        return "G"
    if row["W"] > row["G"]:
        return "W"
    return "Tie"


region_tissue_summary["dominant_gw_tissue"] = region_tissue_summary.apply(dominant_gw_tissue, axis=1)
region_tissue_summary = region_tissue_summary.sort_values(
    ["gw_total", "G", "W"], ascending=[False, False, False], kind="stable"
).reset_index(drop=True)

region_tissue_summary.to_csv(OUTPUT_DIR / "region_graywhite_summary.csv", index=False)
display(region_tissue_summary.head(20))

region_plot_df = (
    region_tissue_summary.loc[region_tissue_summary["gw_total"] > 0, ["Region", "G", "W"]]
    .head(15)
    .set_index("Region")
)

save_horizontal_grouped_barplot(
    region_plot_df,
    title="Top Regions By Gray vs White Matter Counts",
    xlabel="Row-level observation count",
    filename="region_graywhite_top15.png",
    colors=["#4C78A8", "#E45756"],
)


# %% [markdown]
# ## Key Summary Numbers

# %%
print("Average unique IED-positive trials per patient:")
print(f"  {patient_trial_summary['unique_ied_trials'].mean():.2f}")

print("\nSubject-level IEDRateAvg from trial-level WeightedIEDRate:")
print(f"  Mean across patients: {patient_iedrateavg['IEDRateAvg'].mean():.4f}")
print(f"  Patients with usable WeightedIEDRate trials: {patient_iedrateavg['IEDRateAvg'].notna().sum()}")

print("\nChannel spread summaries:")
print(f"  Mean TrialChannelSpread across patient-trials: {trial_channelspread['TrialChannelSpread'].mean():.2f}")
print(f"  Mean PatientChannelSpread across patients: {patient_channelspread['PatientChannelSpread'].mean():.2f}")

print("\nAverage timing-flag percentages across patients:")
for label, value in condition_plot_values.items():
    print(f"  {label}: {value:.2f}%")

print("\nAverage stimulated vs non-stimulated unique trial counts per patient:")
for label, value in stim_plot_values.items():
    print(f"  {label}: {value:.2f}")

print("\nAverage GrayMatter distribution across patients (G/W only):")
for label, value in graymatter_plot_values.items():
    print(f"  {label}: {value:.2f}%")

print("\nCondition distribution within stimulation groups (% of unique trials):")
print(condition_by_stim_percent.round(2))

print("\nCondition distribution within gray vs white matter (% of row-level observations):")
print(condition_by_gray_percent.round(2))

print("\nRegion gray/white dominance summary:")
dominance_counts = region_tissue_summary["dominant_gw_tissue"].value_counts()
for label, value in dominance_counts.items():
    print(f"  {label}: {value}")

print(f"\nAugmented CSV saved to: {AUGMENTED_CSV_PATH}")
print(f"\nOutputs saved in: {OUTPUT_DIR}")
