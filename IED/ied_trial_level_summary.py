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
# Outputs from this analysis are written under `outputs/IED_trial_level_summary/`.

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
    OUTPUT_DIR = BASE_DIR / "outputs" / "IED_trial_level_summary"
else:
    OUTPUT_DIR = BASE_DIR / "outputs" / "IED_trial_level_summary" / Path(INPUT_NAME).stem

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUGMENTED_CSV_PATH = OUTPUT_DIR / f"{Path(INPUT_NAME).stem}_with_IEDRateAvg.csv"

plt.style.use("default")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300

TIMING_LABELS = {
    "DuringImg": "During Image",
    "DuringStim": "During Stimulation",
    "BeforeImgITI": "Before Image ITI",
    "AfterImgITI": "After Image ITI",
}
STIM_LABELS = {"S": "Stim", "NS": "No Stim"}
TISSUE_LABELS = {"G": "Gray Matter", "W": "White Matter"}
TISSUE_COLORS = {"Gray Matter": "#4D4D4D", "White Matter": "#CFCFCF"}


def normalize_region_value(value):
    if pd.isna(value):
        return pd.NA
    raw = str(value).strip()
    if not raw:
        return pd.NA
    canonical = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    if canonical in {"hippocampalgrey", "hippocampus", "hippocampal"}:
        return "Hippocampus"
    return raw


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


def save_barplot(
    series: pd.Series,
    title: str,
    ylabel: str,
    filename: str,
    color: str,
    output_dir: Path,
    bar_color_map: dict[str, str] | None = None,
) -> None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if bar_color_map:
        bar_colors = [bar_color_map.get(str(label), color) for label in series.index]
    else:
        bar_colors = color
    ax.bar(series.index, series.values, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for idx, value in enumerate(series.values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def save_grouped_barplot(
    frame: pd.DataFrame,
    title: str,
    ylabel: str,
    filename: str,
    colors: list[str],
    output_dir: Path,
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
    fig.savefig(output_dir / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def save_horizontal_grouped_barplot(
    frame: pd.DataFrame,
    title: str,
    xlabel: str,
    filename: str,
    colors: list[str],
    output_dir: Path,
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
    fig.savefig(output_dir / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def save_horizontal_barplot(
    series: pd.Series,
    title: str,
    xlabel: str,
    filename: str,
    color: str,
    output_dir: Path,
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
    fig.savefig(output_dir / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def save_horizontal_boxplot(
    grouped_series: pd.Series,
    title: str,
    xlabel: str,
    filename: str,
    color: str,
    output_dir: Path,
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
    fig.savefig(output_dir / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def dominant_gw_tissue(row: pd.Series) -> str:
    if row["gw_total"] == 0:
        return "No Gray Matter/White Matter observations"
    if row["G"] > row["W"]:
        return "Gray Matter"
    if row["W"] > row["G"]:
        return "White Matter"
    return "Tie"


def standardize_ied_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    if "DuringImg" not in frame.columns and "DuringImgITI" in frame.columns:
        frame["DuringImg"] = frame["DuringImgITI"]
    if "StimCond" not in frame.columns and "Test" in frame.columns:
        frame["StimCond"] = frame["Test"]

    for required_col, default_value in {
        "StimCond": pd.NA,
        "DuringImg": "N",
        "DuringStim": "N",
        "BeforeImgITI": "N",
        "AfterImgITI": "N",
    }.items():
        if required_col not in frame.columns:
            frame[required_col] = default_value

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
        "Encoding",
        "Test",
    ]
    for column in string_columns:
        if column not in frame.columns:
            continue
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace({"": pd.NA})

    frame["StimCond"] = frame["StimCond"].str.upper()
    frame["StimCond"] = frame["StimCond"].replace({"Y": "S", "N": "NS"})
    for column in ["DuringImg", "DuringStim", "BeforeImgITI", "AfterImgITI", "GrayMatter", "Encoding", "Test"]:
        if column in frame.columns:
            frame[column] = frame[column].str.upper()

    if "Region" in frame.columns:
        frame["Region"] = frame["Region"].apply(normalize_region_value)

    frame["Trial"] = pd.to_numeric(frame["Trial"], errors="coerce").astype("Int64")
    frame["WeightedIEDRate"] = pd.to_numeric(frame.get("WeightedIEDRate"), errors="coerce")
    if "IEDRateAvg" in frame.columns:
        frame["IEDRateAvg"] = pd.to_numeric(frame["IEDRateAvg"], errors="coerce")
    else:
        frame["IEDRateAvg"] = pd.NA
    return frame


def load_input_frames() -> tuple[pd.DataFrame, list[str]]:
    if INPUT_NAME != DEFAULT_INPUT_NAME:
        frame = pd.read_csv(CSV_PATH)
        return standardize_ied_frame(frame), [str(CSV_PATH)]

    default_candidates = [
        BASE_DIR / "IED" / "AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned.csv",
        BASE_DIR / "IED" / "AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned.csv",
    ]
    existing = [p for p in default_candidates if p.exists()]
    if not existing:
        frame = pd.read_csv(CSV_PATH)
        return standardize_ied_frame(frame), [str(CSV_PATH)]

    frames = [standardize_ied_frame(pd.read_csv(path)) for path in existing]
    return pd.concat(frames, ignore_index=True), [str(path) for path in existing]


def build_phase_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    encoding_mask = pd.Series(False, index=df.index)
    retrieval_mask = pd.Series(False, index=df.index)
    if "Encoding" in df.columns:
        encoding_mask = encoding_mask | (df["Encoding"] == "Y")
    if "Test" in df.columns:
        retrieval_mask = retrieval_mask | (df["Test"] == "Y")

    return {
        "combined": df.copy(),
        "encoding": df.loc[encoding_mask].copy(),
        "retrieval": df.loc[retrieval_mask].copy(),
    }


def run_summary_for_subset(df_in: pd.DataFrame, output_dir: Path, subset_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_title = subset_label.title()
    if subset_label == "retrieval":
        condition_columns = ["DuringImg", "BeforeImgITI"]
    else:
        condition_columns = ["DuringImg", "DuringStim", "BeforeImgITI", "AfterImgITI"]
    condition_display = {col: TIMING_LABELS[col] for col in condition_columns}

    if subset_label == "retrieval":
        for stale_name in [
            "condition_by_stimulation_counts.csv",
            "condition_by_stimulation_percent.csv",
            "condition_by_stimulation_percent.png",
        ]:
            stale_path = output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

    if df_in.empty:
        print(f"[{subset_label}] No rows available. Skipping.")
        return

    print(f"[{subset_label}] Running summary on {len(df_in):,} rows -> {output_dir}")
    df = df_in.copy()

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
    trial_channelspread.to_csv(output_dir / "trial_channelspread_summary.csv", index=False)
    patient_channelspread.to_csv(output_dir / "patient_channelspread_summary.csv", index=False)

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
    trial_regionspread.to_csv(output_dir / "trial_regionspread_summary.csv", index=False)
    patient_regionspread.to_csv(output_dir / "patient_regionspread_summary.csv", index=False)

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
    df = df.drop(columns=["IEDRateAvg"], errors="ignore").merge(
        patient_iedrateavg[["Patient", "IEDRateAvg"]],
        on="Patient",
        how="left",
    )
    trial_weighted_iedrate.to_csv(output_dir / "trial_weighted_iedrate_summary.csv", index=False)
    patient_iedrateavg.to_csv(output_dir / "patient_iedrateavg_summary.csv", index=False)
    df.to_csv(output_dir / f"{subset_label}_with_IEDRateAvg.csv", index=False)

    patient_channelspread_plot = (
        patient_channelspread.sort_values("PatientChannelSpread", ascending=True, kind="stable")
        .set_index("Patient")["PatientChannelSpread"]
    )
    save_horizontal_barplot(
        patient_channelspread_plot,
        title=f"{phase_title}: PatientChannelSpread By Patient",
        xlabel="Average unique channels per trial",
        filename="patient_channelspread_by_patient.png",
        color="#2A6F97",
        output_dir=output_dir,
    )

    patient_iedrateavg_plot = (
        patient_iedrateavg.sort_values("IEDRateAvg", ascending=True, kind="stable")
        .set_index("Patient")["IEDRateAvg"]
    )
    save_horizontal_barplot(
        patient_iedrateavg_plot,
        title=f"{phase_title}: IEDRateAvg By Patient (Average of Trial-Level WeightedIEDRate)",
        xlabel="IEDRateAvg",
        filename="patient_iedrateavg_by_patient.png",
        color="#C97C10",
        output_dir=output_dir,
    )

    patient_regionspread_plot = (
        patient_regionspread.sort_values("PatientRegionSpread", ascending=True, kind="stable")
        .set_index("Patient")["PatientRegionSpread"]
    )
    save_horizontal_barplot(
        patient_regionspread_plot,
        title=f"{phase_title}: PatientRegionSpread By Patient",
        xlabel="Average unique regions per trial",
        filename="patient_regionspread_by_patient.png",
        color="#7F5539",
        output_dir=output_dir,
    )

    region_distribution_order = (
        patient_regionspread.sort_values("PatientRegionSpread", ascending=True, kind="stable")["Patient"]
    )
    trial_regionspread_distribution = (
        trial_regionspread.set_index("Patient").loc[region_distribution_order, "TrialRegionSpread"]
    )
    save_horizontal_boxplot(
        trial_regionspread_distribution,
        title=f"{phase_title}: Trial-Level RegionSpread Distribution By Patient",
        xlabel="Unique regions within a trial",
        filename="trial_regionspread_distribution_by_patient.png",
        color="#B08968",
        output_dir=output_dir,
    )

    trial_agg = {"StimCond": ("StimCond", first_nonmissing)}
    for col in condition_columns:
        trial_agg[col] = (col, any_yes)
    trial_level = (
        df.groupby(["Patient", "Trial"], dropna=False, as_index=False)
        .agg(**trial_agg)
        .sort_values(["Patient", "Trial"], kind="stable")
    )
    trial_level.to_csv(output_dir / "collapsed_trial_level.csv", index=False)

    patient_trial_agg = {
        "unique_ied_trials": ("Trial", "count"),
        "stimulated_trials": ("StimCond", lambda s: (s == "S").sum()),
        "nonstim_trials": ("StimCond", lambda s: (s == "NS").sum()),
    }
    condition_metric_map = {
        "DuringImg": "during_img_trials",
        "DuringStim": "during_stim_trials",
        "BeforeImgITI": "before_img_iti_trials",
        "AfterImgITI": "after_img_iti_trials",
    }
    for col in condition_columns:
        metric_name = condition_metric_map[col]
        patient_trial_agg[metric_name] = (col, lambda s: (s == "Y").sum())

    patient_trial_summary = trial_level.groupby("Patient", dropna=False).agg(**patient_trial_agg).reset_index()
    pct_count_columns = ["stimulated_trials", "nonstim_trials"] + [condition_metric_map[col] for col in condition_columns]
    for column in pct_count_columns:
        patient_trial_summary[f"{column}_pct"] = patient_trial_summary[column] / patient_trial_summary["unique_ied_trials"]

    patient_trial_summary.to_csv(output_dir / "patient_trial_summary.csv", index=False)

    overall_metrics = [
        ("Average unique IED-positive trials per patient", patient_trial_summary["unique_ied_trials"].mean()),
        ("Average stimulated trials per patient", patient_trial_summary["stimulated_trials"].mean()),
        ("Average non-stimulated trials per patient", patient_trial_summary["nonstim_trials"].mean()),
    ]
    for col in condition_columns:
        metric_name = condition_metric_map[col]
        overall_metrics.append((f"Average proportion {condition_display[col]}", patient_trial_summary[f"{metric_name}_pct"].mean()))
    overall_trial_summary = pd.DataFrame(overall_metrics, columns=["metric", "value"])
    overall_trial_summary.to_csv(output_dir / "overall_trial_summary.csv", index=False)

    condition_plot_values = pd.Series(
        {
            condition_display[col]: patient_trial_summary[f"{condition_metric_map[col]}_pct"].mean() * 100
            for col in condition_columns
        }
    )
    save_barplot(
        condition_plot_values,
        title=f"{phase_title}: Average Timing Distribution Across Patients",
        ylabel="Average % of unique IED-positive trials",
        filename="average_condition_distribution.png",
        color="#5B8E7D",
        output_dir=output_dir,
    )
    condition_plot_values.to_csv(output_dir / "average_condition_distribution.csv", header=["average_percent"])

    stim_plot_values = pd.Series(
        {
            "Stim": patient_trial_summary["stimulated_trials"].mean(),
            "No Stim": patient_trial_summary["nonstim_trials"].mean(),
        }
    )
    save_barplot(
        stim_plot_values,
        title=f"{phase_title}: Average Unique IED-Positive Trials Per Patient by Stim Condition",
        ylabel="Average count of unique trials",
        filename="average_stim_distribution.png",
        color="#D98E04",
        output_dir=output_dir,
    )
    stim_plot_values.to_csv(output_dir / "average_stim_distribution.csv", header=["average_count"])

    graymatter_qa = (
        df.groupby("Patient", dropna=False)["GrayMatter"]
        .value_counts(dropna=False)
        .unstack(fill_value=0)
        .reset_index()
    )
    graymatter_qa.to_csv(output_dir / "patient_graymatter_qa_counts.csv", index=False)

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
    patient_graymatter_summary = patient_graymatter_summary.rename(columns=TISSUE_LABELS)
    patient_graymatter_summary.to_csv(output_dir / "patient_graymatter_gw_summary.csv", index=False)

    graymatter_plot_values = pd.Series(
        {
            "Gray Matter": patient_graymatter_summary.get("Gray Matter", pd.Series(dtype=float)).mean() * 100,
            "White Matter": patient_graymatter_summary.get("White Matter", pd.Series(dtype=float)).mean() * 100,
        }
    )
    save_barplot(
        graymatter_plot_values,
        title=f"{phase_title}: Average Gray Matter vs White Matter Distribution Across Patients",
        ylabel="Average % of row-level IED observations",
        filename="average_graymatter_distribution.png",
        color="#4D4D4D",
        output_dir=output_dir,
        bar_color_map=TISSUE_COLORS,
    )
    graymatter_plot_values.to_csv(output_dir / "average_graymatter_distribution.csv", header=["average_percent"])

    if subset_label != "retrieval":
        trial_condition_binary = trial_level.copy()
        for column in condition_columns:
            trial_condition_binary[column] = trial_condition_binary[column].eq("Y").astype(int)

        condition_by_stim_counts = (
            trial_condition_binary.groupby("StimCond")[condition_columns].sum().reindex(["S", "NS"])
        )
        stim_trial_totals = trial_condition_binary["StimCond"].value_counts().reindex(["S", "NS"])
        condition_by_stim_percent = condition_by_stim_counts.div(stim_trial_totals, axis=0) * 100
        condition_by_stim_counts = condition_by_stim_counts.rename(index=STIM_LABELS, columns=TIMING_LABELS)
        condition_by_stim_percent = condition_by_stim_percent.rename(index=STIM_LABELS, columns=TIMING_LABELS)
        condition_by_stim_counts.to_csv(output_dir / "condition_by_stimulation_counts.csv")
        condition_by_stim_percent.to_csv(output_dir / "condition_by_stimulation_percent.csv")
        save_grouped_barplot(
            condition_by_stim_percent.T,
            title=f"{phase_title}: Condition Distribution Within Stimulated vs Non-Stimulated Trials",
            ylabel="% of unique trials within stimulation condition",
            filename="condition_by_stimulation_percent.png",
            colors=["#D98E04", "#7A8FA6"],
            output_dir=output_dir,
        )

    condition_by_gray_counts = (
        graymatter_gw.groupby("GrayMatter")[condition_columns]
        .agg(lambda s: (s == "Y").sum())
        .reindex(["G", "W"])
    )
    graymatter_totals = graymatter_gw["GrayMatter"].value_counts().reindex(["G", "W"])
    condition_by_gray_percent = condition_by_gray_counts.div(graymatter_totals, axis=0) * 100
    condition_by_gray_counts = condition_by_gray_counts.rename(index=TISSUE_LABELS, columns=TIMING_LABELS)
    condition_by_gray_percent = condition_by_gray_percent.rename(index=TISSUE_LABELS, columns=TIMING_LABELS)
    condition_by_gray_counts.to_csv(output_dir / "condition_by_graymatter_counts.csv")
    condition_by_gray_percent.to_csv(output_dir / "condition_by_graymatter_percent.csv")
    save_grouped_barplot(
        condition_by_gray_percent.T,
        title=f"{phase_title}: Condition Distribution Within Gray Matter vs White Matter",
        ylabel="% of row-level observations within tissue class",
        filename="condition_by_graymatter_percent.png",
        colors=[TISSUE_COLORS["Gray Matter"], TISSUE_COLORS["White Matter"]],
        output_dir=output_dir,
    )

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
    region_tissue_summary["dominant_gw_tissue"] = region_tissue_summary.apply(dominant_gw_tissue, axis=1)
    region_tissue_summary = region_tissue_summary.sort_values(
        ["gw_total", "G", "W"], ascending=[False, False, False], kind="stable"
    ).reset_index(drop=True)
    region_tissue_summary_export = region_tissue_summary.rename(
        columns={
            "G": "Gray Matter",
            "W": "White Matter",
            "gray_pct_within_gw": "gray_matter_pct_within_gw",
            "white_pct_within_gw": "white_matter_pct_within_gw",
        }
    )
    region_tissue_summary_export.to_csv(output_dir / "region_graywhite_summary.csv", index=False)

    region_plot_df = (
        region_tissue_summary.loc[region_tissue_summary["gw_total"] > 0, ["Region", "G", "W"]]
        .head(15)
        .set_index("Region")
    )
    region_plot_df = region_plot_df.rename(columns=TISSUE_LABELS)
    save_horizontal_grouped_barplot(
        region_plot_df,
        title=f"{phase_title}: Top Regions By Gray Matter vs White Matter Counts",
        xlabel="Row-level observation count",
        filename="region_graywhite_top15.png",
        colors=[TISSUE_COLORS["Gray Matter"], TISSUE_COLORS["White Matter"]],
        output_dir=output_dir,
    )

    print(f"[{subset_label}] Completed outputs in {output_dir}")


if __name__ == "__main__":
    full_df, loaded_sources = load_input_frames()
    print("Loaded sources:")
    for src in loaded_sources:
        print(f"  - {src}")
    print(f"Total rows loaded: {len(full_df):,}")

    phase_frames = build_phase_splits(full_df)
    for phase_name in ["encoding", "retrieval", "combined"]:
        run_summary_for_subset(phase_frames[phase_name], OUTPUT_DIR / phase_name, phase_name)

    print(f"\nOutputs saved in: {OUTPUT_DIR}")
