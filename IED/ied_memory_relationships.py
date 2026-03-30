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

try:
    from scipy.stats import f as f_dist
    from scipy.stats import t as t_dist
except ImportError:
    f_dist = None
    t_dist = None

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

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

RAW_MEMORY_COLUMN = "avg_stim_dprime_diff"
NORMALIZED_MEMORY_COLUMN = "normalized_subsequent_memory_score"
RAW_MEMORY_LABEL = "Subsequent memory"
NORMALIZED_MEMORY_LABEL = "Normalized subsequent memory score"


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
    output_dir: Path,
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
    fig.savefig(output_dir / filename, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)
    return stats


def resolve_phase_dirs(summary_root: Path) -> dict[str, Path]:
    phase_dirs = {
        "encoding": summary_root / "encoding",
        "retrieval": summary_root / "retrieval",
        "combined": summary_root / "combined",
    }
    required = [
        "patient_iedrateavg_summary.csv",
        "patient_channelspread_summary.csv",
        "patient_regionspread_summary.csv",
        "patient_trial_summary.csv",
    ]
    has_phase_layout = all((phase_dirs["combined"] / req).exists() for req in required)
    if has_phase_layout:
        return phase_dirs
    return {"combined": summary_root}


def available_timing_pct_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "during_img_trials_pct",
        "during_stim_trials_pct",
        "before_img_iti_trials_pct",
        "after_img_iti_trials_pct",
    ]
    return [col for col in preferred if col in df.columns]


def timing_label(col: str) -> str:
    mapping = {
        "during_img_trials_pct": "During Image",
        "during_stim_trials_pct": "During Stimulation",
        "before_img_iti_trials_pct": "Before Image ITI",
        "after_img_iti_trials_pct": "After Image ITI",
    }
    return mapping.get(col, col)


def zscore_series(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if pd.isna(std) or np.isclose(std, 0):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def memory_label(y_col: str) -> str:
    return NORMALIZED_MEMORY_LABEL if y_col == NORMALIZED_MEMORY_COLUMN else RAW_MEMORY_LABEL


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def run_figure2a_like_encoding(
    merged_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    predictor_meta = {
        "IEDRateAvg": "IED rate (per trial)",
        "left_lateralized_ied": "Left-lateralized IED",
        "PatientRegionSpread": "IED regional spread",
        "White Matter": "White matter IED proportion",
        "stimulated_trials_pct": "Stimulated trial proportion",
        "nonstim_trials_pct": "Non-stimulated trial proportion",
    }
    predictors = [col for col in predictor_meta if col in merged_df.columns]
    if len(predictors) < 2:
        return

    work = merged_df[[NORMALIZED_MEMORY_COLUMN] + predictors].copy()
    for col in predictors + [NORMALIZED_MEMORY_COLUMN]:
        work[col] = safe_numeric(work[col])

    usable_predictors = [c for c in predictors if work[c].nunique(dropna=True) > 1]
    if len(usable_predictors) < 2:
        return
    work = work[[NORMALIZED_MEMORY_COLUMN] + usable_predictors].dropna().copy()
    if len(work) < len(usable_predictors) + 5:
        return

    # Standardize predictors to mirror Figure 2A's standardized continuous effects.
    for col in usable_predictors:
        work[col] = zscore_series(work[col])
    work = work.dropna().copy()
    if len(work) < len(usable_predictors) + 5:
        return

    rows = []
    if sm is not None:
        x = sm.add_constant(work[usable_predictors], has_constant="add")
        y = work[NORMALIZED_MEMORY_COLUMN]
        model = sm.OLS(y, x, missing="drop").fit(cov_type="HC3")
        for col in usable_predictors:
            rows.append(
                {
                    "predictor": col,
                    "label": predictor_meta.get(col, col),
                    "beta": float(model.params.get(col, np.nan)),
                    "std_err": float(model.bse.get(col, np.nan)),
                    "t": float(model.tvalues.get(col, np.nan)),
                    "p_value": float(model.pvalues.get(col, np.nan)),
                    "n": int(model.nobs),
                    "model_r2": float(model.rsquared),
                    "model_adj_r2": float(model.rsquared_adj),
                }
            )
    else:
        # Fallback: univariate approximations when statsmodels is unavailable.
        for col in usable_predictors:
            tmp = work[[NORMALIZED_MEMORY_COLUMN, col]].dropna()
            if len(tmp) < 3:
                continue
            slope, intercept = np.polyfit(tmp[col], tmp[NORMALIZED_MEMORY_COLUMN], 1)
            r = np.corrcoef(tmp[col], tmp[NORMALIZED_MEMORY_COLUMN])[0, 1]
            p = np.nan
            if t_dist is not None and np.isfinite(r) and abs(r) < 1:
                t_val = r * np.sqrt((len(tmp) - 2) / (1 - r**2))
                p = float(2 * t_dist.sf(np.abs(t_val), len(tmp) - 2))
            rows.append(
                {
                    "predictor": col,
                    "label": predictor_meta.get(col, col),
                    "beta": float(slope),
                    "std_err": np.nan,
                    "t": np.nan,
                    "p_value": p,
                    "n": int(len(tmp)),
                    "model_r2": float(r**2) if np.isfinite(r) else np.nan,
                    "model_adj_r2": np.nan,
                }
            )

    if not rows:
        return

    coef_df = pd.DataFrame(rows)
    ranked = coef_df["p_value"].rank(method="min")
    m = max(int(ranked.max()), 1)
    coef_df["p_adjusted_bh"] = (coef_df["p_value"] * m / ranked).clip(upper=1.0)
    coef_df = coef_df.sort_values(["p_adjusted_bh", "p_value"], na_position="last")
    coef_df.to_csv(output_dir / "figure2a_like_encoding_glm_summary.csv", index=False)

    plot_df = coef_df.copy()
    plot_df["ci95"] = 1.96 * plot_df["std_err"].fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(
        plot_df["label"],
        plot_df["beta"],
        xerr=plot_df["ci95"],
        color="#4C78A8",
        edgecolor="black",
        linewidth=1.0,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Standardized effect size (beta)")
    ax.set_title("Encoding Figure 2A-like IED feature model")
    fig.text(
        0.01,
        0.01,
        "Model approximation of Quon Figure 2A: multivariable standardized effects with BH-adjusted p-values.",
        fontsize=8.5,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_dir / "figure2a_like_encoding_glm_coefficients.png", bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def run_encoding_stim_split_regressions(
    merged_df: pd.DataFrame,
    phase_output_dir: Path,
) -> None:
    spec = {
        "stimulated_trials_pct": {
            "title": "Encoding: Subsequent memory vs Stimulated IED-trial proportion",
            "xlabel": "Stimulated IED-positive trial proportion",
            "filename": "memory_vs_stimulated_trials_pct_encoding.png",
            "color": "#D62728",
        },
        "nonstim_trials_pct": {
            "title": "Encoding: Subsequent memory vs Non-stimulated IED-trial proportion",
            "xlabel": "Non-stimulated IED-positive trial proportion",
            "filename": "memory_vs_nonstim_trials_pct_encoding.png",
            "color": "#1F77B4",
        },
    }

    rows = []
    for x_col, meta in spec.items():
        if x_col not in merged_df.columns:
            continue
        stats = scatter_with_fit(
            merged_df,
            x_col=x_col,
            y_col=RAW_MEMORY_COLUMN,
            title=meta["title"],
            xlabel=meta["xlabel"],
            ylabel=RAW_MEMORY_LABEL,
            filename=meta["filename"],
            color=meta["color"],
            output_dir=phase_output_dir,
        )
        stats["display_label"] = meta["xlabel"]
        rows.append(stats)

    if rows:
        pd.DataFrame(rows).to_csv(
            phase_output_dir / "encoding_stim_split_correlation_summary.csv",
            index=False,
        )


def run_timing_relationships(
    merged_df: pd.DataFrame,
    timing_cols: list[str],
    phase_name: str,
    output_dir: Path,
    memory_measure: str,
) -> None:
    if not timing_cols:
        return

    timing_palette = {
        "during_img_trials_pct": "#2E8B57",
        "during_stim_trials_pct": "#D1495B",
        "before_img_iti_trials_pct": "#4C78A8",
        "after_img_iti_trials_pct": "#8E6C8A",
    }

    timing_rows = []
    for t_col in timing_cols:
        lbl = timing_label(t_col)
        stats = scatter_with_fit(
            merged_df,
            x_col=t_col,
            y_col=memory_measure,
            title=f"{phase_name.title()}: Memory vs {lbl} IED Timing",
            xlabel=f"{lbl} (% of unique IED-positive trials)",
            ylabel=memory_label(memory_measure),
            filename=f"memory_vs_{t_col}.png",
            color=timing_palette.get(t_col, "#555555"),
            output_dir=output_dir,
        )
        stats["timing_label"] = lbl
        timing_rows.append(stats)

    timing_corr = pd.DataFrame(timing_rows)
    timing_corr.to_csv(output_dir / "memory_timing_correlation_summary.csv", index=False)

    n_cols = 2
    n_rows = int(np.ceil(len(timing_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, t_col in enumerate(timing_cols):
        ax = axes[idx]
        subset = merged_df[[t_col, memory_measure]].dropna()
        ax.scatter(
            subset[t_col],
            subset[memory_measure],
            color=timing_palette.get(t_col, "#555555"),
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            s=50,
        )
        if len(subset) >= 2:
            fit = np.polyfit(subset[t_col], subset[memory_measure], 1)
            x_line = np.linspace(subset[t_col].min(), subset[t_col].max(), 100)
            y_line = fit[0] * x_line + fit[1]
            ax.plot(x_line, y_line, color="black", linewidth=1.1)

        row = timing_corr.loc[timing_corr["x_measure"] == t_col].iloc[0]
        ax.set_title(timing_label(t_col))
        ax.set_xlabel("Proportion of unique IED-positive trials")
        ax.set_ylabel(memory_label(memory_measure))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        p_text = "nan" if pd.isna(row["pearson_p"]) else f"{row['pearson_p']:.3f}"
        s_text = "nan" if pd.isna(row["spearman_p"]) else f"{row['spearman_p']:.3f}"
        ax.text(
            0.03,
            0.97,
            (
                f"n = {int(row['n'])}\n"
                f"Pearson r = {row['pearson_r']:.2f}, p = {p_text}\n"
                f"Spearman rho = {row['spearman_rho']:.2f}, p = {s_text}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    for j in range(len(timing_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{phase_name.title()}: IED Timing Correlations with {memory_label(memory_measure)}",
        y=1.01,
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "memory_vs_ied_timing_combined.png", bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)

    # Compare timing impacts with a multiple linear model using standardized variables.
    model_cols = [memory_measure] + timing_cols
    model_df = merged_df[model_cols].dropna().copy()
    if model_df.empty:
        return
    usable_cols = [col for col in timing_cols if model_df[col].nunique(dropna=True) > 1]
    if len(usable_cols) < 2 or len(model_df) <= len(usable_cols) + 1:
        return

    xz = np.column_stack([zscore_series(model_df[col]).to_numpy(dtype=np.float64) for col in usable_cols])
    yz = zscore_series(model_df[memory_measure]).to_numpy(dtype=np.float64)

    finite_mask = np.isfinite(yz)
    finite_mask &= np.all(np.isfinite(xz), axis=1)
    xz = xz[finite_mask]
    yz = yz[finite_mask]

    n = xz.shape[0]
    k = xz.shape[1]
    if n <= k + 1:
        return

    x_design = np.column_stack([np.ones(n), xz])
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    beta = xtx_inv @ x_design.T @ yz
    y_hat = x_design @ beta
    resid = yz - y_hat

    sse = float(np.sum(resid**2))
    sst = float(np.sum((yz - yz.mean()) ** 2))
    ssr = sst - sse
    r2 = np.nan if np.isclose(sst, 0) else 1 - (sse / sst)
    adj_r2 = np.nan if n <= k + 1 else 1 - (1 - r2) * ((n - 1) / (n - k - 1))

    mse = sse / (n - k - 1)
    se_beta = np.sqrt(np.diag(xtx_inv) * mse)
    t_vals = beta / se_beta

    model_f = (ssr / k) / (sse / (n - k - 1)) if k > 0 and not np.isclose(sse, 0) else np.nan
    model_p = np.nan
    if f_dist is not None and not np.isnan(model_f):
        model_p = float(f_dist.sf(model_f, k, n - k - 1))

    coef_rows = []
    predictors = ["Intercept"] + usable_cols
    for i, name in enumerate(predictors):
        p_val = np.nan
        if t_dist is not None and n - k - 1 > 0 and np.isfinite(t_vals[i]):
            p_val = float(2 * t_dist.sf(np.abs(t_vals[i]), n - k - 1))
        coef_rows.append(
            {
                "term": name,
                "timing_label": timing_label(name) if name != "Intercept" else "Intercept",
                "beta": beta[i],
                "std_error": se_beta[i],
                "t": t_vals[i],
                "p": p_val,
            }
        )

    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(output_dir / "timing_impact_regression_coefficients.csv", index=False)

    model_summary = pd.DataFrame(
        [
            {
                "n": n,
                "predictors": k,
                "r_squared": r2,
                "adj_r_squared": adj_r2,
                "model_f": model_f,
                "model_p": model_p,
            }
        ]
    )
    model_summary.to_csv(output_dir / "timing_impact_regression_model_summary.csv", index=False)

    coef_plot_df = coef_df[coef_df["term"] != "Intercept"].copy()
    if not coef_plot_df.empty:
        coef_plot_df["ci95"] = 1.96 * coef_plot_df["std_error"]
        coef_plot_df = coef_plot_df.sort_values("beta", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.bar(
            coef_plot_df["timing_label"],
            coef_plot_df["beta"],
            yerr=coef_plot_df["ci95"],
            color=[timing_palette.get(term, "#777777") for term in coef_plot_df["term"]],
            edgecolor="black",
            linewidth=1.0,
            capsize=4,
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Standardized beta (95% CI)")
        ax.set_title(f"{phase_name.title()}: Relative Timing Impact on {memory_label(memory_measure)}")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / "timing_impact_regression_betas.png", bbox_inches="tight")
        if running_in_notebook():
            plt.show()
        plt.close(fig)


def run_phase_relationships(phase_name: str, summary_dir: Path, output_root: Path, behavior_df: pd.DataFrame) -> bool:
    required_files = {
        "ied_rate": summary_dir / "patient_iedrateavg_summary.csv",
        "ied_channel": summary_dir / "patient_channelspread_summary.csv",
        "ied_region": summary_dir / "patient_regionspread_summary.csv",
        "ied_trial": summary_dir / "patient_trial_summary.csv",
        "ied_graywhite": summary_dir / "patient_graymatter_gw_summary.csv",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        print(f"[{phase_name}] Skipping: missing summary inputs.")
        for path in missing:
            print(f"  - {path}")
        return False

    phase_output_dir = output_root / phase_name
    phase_output_dir.mkdir(parents=True, exist_ok=True)

    ied_rate = pd.read_csv(required_files["ied_rate"])
    ied_channel = pd.read_csv(required_files["ied_channel"])
    ied_region = pd.read_csv(required_files["ied_region"])
    ied_trial = pd.read_csv(required_files["ied_trial"])
    ied_graywhite = pd.read_csv(required_files["ied_graywhite"])

    for frame in [ied_rate, ied_channel, ied_region, ied_trial, ied_graywhite]:
        frame["Patient"] = frame["Patient"].astype("string").str.strip()

    timing_cols = available_timing_pct_columns(ied_trial)
    stim_split_cols = [
        col
        for col in ["stimulated_trials_pct", "nonstim_trials_pct"]
        if col in ied_trial.columns
    ]

    merged = (
        behavior_df[
            [
                "Patient",
                "Study",
                RAW_MEMORY_COLUMN,
                NORMALIZED_MEMORY_COLUMN,
                "IED_freq",
                "IED_laterality",
                "avg_stim",
                "nostim",
            ]
        ]
        .merge(ied_rate[["Patient", "IEDRateAvg", "trials_with_weighted_iedrate"]], on="Patient", how="inner")
        .merge(ied_channel[["Patient", "PatientChannelSpread"]], on="Patient", how="left")
        .merge(ied_region[["Patient", "PatientRegionSpread"]], on="Patient", how="left")
        .merge(ied_graywhite[["Patient", "Gray Matter", "White Matter"]], on="Patient", how="left")
        .merge(
            ied_trial[["Patient", "unique_ied_trials"] + stim_split_cols + timing_cols],
            on="Patient",
            how="left",
        )
    )

    if "IED_laterality" in merged.columns:
        merged["left_lateralized_ied"] = (
            merged["IED_laterality"].astype("string").str.strip().str.upper() == "Y"
        ).astype(float)

    for col in [
        RAW_MEMORY_COLUMN,
        NORMALIZED_MEMORY_COLUMN,
        "IEDRateAvg",
        "PatientChannelSpread",
        "PatientRegionSpread",
        "White Matter",
        "Gray Matter",
        "stimulated_trials_pct",
        "nonstim_trials_pct",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged.to_csv(phase_output_dir / "memory_vs_ied_merged.csv", index=False)

    memory_measure = RAW_MEMORY_COLUMN
    ied_measures = {
        "IEDRateAvg": {
            "title": f"{phase_name.title()}: Memory vs IEDRateAvg",
            "xlabel": "IEDRateAvg (average of trial-level WeightedIEDRate)",
            "color": "#C97C10",
            "filename": "memory_vs_IEDRateAvg.png",
        },
        "PatientChannelSpread": {
            "title": f"{phase_name.title()}: Memory vs PatientChannelSpread",
            "xlabel": "PatientChannelSpread",
            "color": "#2A6F97",
            "filename": "memory_vs_PatientChannelSpread.png",
        },
        "PatientRegionSpread": {
            "title": f"{phase_name.title()}: Memory vs PatientRegionSpread",
            "xlabel": "PatientRegionSpread",
            "color": "#7F5539",
            "filename": "memory_vs_PatientRegionSpread.png",
        },
        "unique_ied_trials": {
            "title": f"{phase_name.title()}: Memory vs Unique IED-Positive Trials",
            "xlabel": "Unique IED-positive trials",
            "color": "#5B8E7D",
            "filename": "memory_vs_unique_ied_trials.png",
        },
    }

    correlation_rows = []
    for measure, meta in ied_measures.items():
        stats = scatter_with_fit(
            merged,
            x_col=measure,
            y_col=memory_measure,
            title=meta["title"],
            xlabel=meta["xlabel"],
            ylabel=RAW_MEMORY_LABEL,
            filename=meta["filename"],
            color=meta["color"],
            output_dir=phase_output_dir,
        )
        correlation_rows.append(stats)

    correlation_summary = pd.DataFrame(correlation_rows)
    correlation_summary.to_csv(phase_output_dir / "memory_ied_correlation_summary.csv", index=False)

    run_timing_relationships(
        merged_df=merged,
        timing_cols=timing_cols,
        phase_name=phase_name,
        output_dir=phase_output_dir,
        memory_measure=memory_measure,
    )

    if phase_name == "encoding":
        run_encoding_stim_split_regressions(merged, phase_output_dir)
        run_figure2a_like_encoding(merged, phase_output_dir)

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
        ax.set_ylabel(RAW_MEMORY_LABEL)
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

    fig.suptitle(f"{phase_name.title()}: Patient-Level IED Measures vs {RAW_MEMORY_LABEL}", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(phase_output_dir / "memory_vs_ied_combined.png", bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)

    print(f"[{phase_name}] Merged patients: {len(merged)}")
    return True


if __name__ == "__main__":
    behavior_df = pd.read_csv(BEHAVIOR_PATH)
    behavior_df["Patient"] = behavior_df["Patient"].astype("string").str.strip()
    behavior_df[RAW_MEMORY_COLUMN] = pd.to_numeric(behavior_df[RAW_MEMORY_COLUMN], errors="coerce")
    behavior_df[NORMALIZED_MEMORY_COLUMN] = zscore_series(behavior_df[RAW_MEMORY_COLUMN])

    phase_dirs = resolve_phase_dirs(IED_DIR)
    ran_any = False
    for phase_name in ["encoding", "retrieval", "combined"]:
        if phase_name not in phase_dirs:
            continue
        ran_any = run_phase_relationships(
            phase_name=phase_name,
            summary_dir=phase_dirs[phase_name],
            output_root=OUTPUT_DIR,
            behavior_df=behavior_df,
        ) or ran_any

    if not ran_any:
        print("No phase outputs were generated; required summary CSVs were not found.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
