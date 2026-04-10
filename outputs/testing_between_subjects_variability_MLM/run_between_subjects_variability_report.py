#!/usr/bin/env python
"""
Patient-level modeling of between-subject variability in memory modulation.

This analysis uses the behavioral summary file with one row per patient.
Because the requested outcome is already collapsed to a patient-level value,
there is no within-subject replication left for a subject random intercept.
The appropriate analysis here is patient-level linear modeling rather than an
MLM with (1 | subject).

All outputs are written under outputs/testing_between_subjects_variability_MLM/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = OUTPUT_DIR / "between_subjects_variability_report.pdf"
CSV_DIR = OUTPUT_DIR / "csv"
SUMMARY_DIR = CSV_DIR / "summary"
COEFFICIENTS_DIR = CSV_DIR / "coefficients"
CONTRIBUTIONS_DIR = CSV_DIR / "contributions"
SEQUENTIAL_STEPS_DIR = CSV_DIR / "sequential_steps"

BEHAVIOR_CANDIDATES = [
    BASE_DIR.parent / "AMMEBLAES_includedpts_firstsession_behavioral.csv",
    BASE_DIR / "AMMEBLAES_includedpts_firstsession_behavioral.csv",
    BASE_DIR / "behavioral figures" / "AMMEBLAES_includedpts_firstsession_behavioral.csv",
]

OUTCOME_COLUMN = "avg_stim_dprime_diff"
OUTCOME_LABEL = "avg_stim_dprime_diff"
MEMORY_LABEL = "Memory modulation (avg_stim_dprime_diff)"

PREDICTOR_INFO = {
    "sex": {
        "label": "Sex (male vs female)",
        "short_label": "Sex (male vs female)",
        "type": "binary",
        "reference": "female",
    },
    "stim_trajectory": {
        "label": "Stim trajectory (STG vs MTG; frontal excluded)",
        "short_label": "Trajectory (STG vs MTG)",
        "type": "binary",
        "reference": "MTG",
    },
    "age": {
        "label": "Age (years)",
        "short_label": "Age",
        "type": "continuous",
    },
    "stim_hemisphere": {
        "label": "Stim hemisphere (R vs L)",
        "short_label": "Hemisphere (R vs L)",
        "type": "binary",
        "reference": "L",
    },
    "stim_DB": {
        "label": "Stim amplitude (DB)",
        "short_label": "Amplitude (DB)",
        "type": "continuous",
    },
    "IED_freq": {
        "label": "IED frequency (Likert)",
        "short_label": "IED frequency",
        "type": "continuous",
    },
    "Memory_Z": {
        "label": "Pre-surgical baseline memory (Memory_Z)",
        "short_label": "Memory_Z",
        "type": "continuous",
    },
}

MODEL_SPECS = [
    {
        "slug": "primary_full_sample",
        "title": "Primary Full-Sample Model",
        "requested_predictors": ["sex", "stim_trajectory", "age", "stim_hemisphere", "stim_DB"],
        "description": (
            "Primary patient-level model using the full cohort. "
            "This is the cleanest adjusted test because sex, age, stimulation hemisphere, "
            "and stimulation amplitude are available broadly, and stimulation trajectory is "
            "available for the non-frontal MTG/STG subset."
        ),
    },
    {
        "slug": "ied_sensitivity",
        "title": "IED Frequency Sensitivity Model",
        "requested_predictors": ["sex", "stim_trajectory", "age", "stim_hemisphere", "stim_DB", "IED_freq"],
        "description": (
            "Sensitivity model adding IED frequency after the primary covariates. "
            "This costs additional patients because IED_freq is missing for part of the sample."
        ),
    },
    {
        "slug": "memoryz_sensitivity",
        "title": "Memory_Z Sensitivity Model",
        "requested_predictors": ["sex", "stim_trajectory", "age", "stim_hemisphere", "stim_DB", "Memory_Z"],
        "description": (
            "Sensitivity model adding pre-surgical baseline memory after the primary covariates. "
            "The available sample is restricted to the studies where Memory_Z exists."
        ),
    },
    {
        "slug": "complete_case_exploratory",
        "title": "Complete-Case Exploratory Model",
        "requested_predictors": ["sex", "stim_trajectory", "age", "stim_hemisphere", "stim_DB", "IED_freq", "Memory_Z"],
        "description": (
            "Exploratory complete-case model including all requested between-subject predictors. "
            "This is the most data-limited model and is shown mainly as a sensitivity check."
        ),
    },
]


@dataclass
class LinearModelResult:
    slug: str
    title: str
    description: str
    requested_predictors: list[str]
    included_predictors: list[str]
    omitted_predictors: list[tuple[str, str]]
    n: int
    df_model: int
    df_resid: int
    r2: float
    adj_r2: float
    f_stat: float
    f_p: float
    sigma: float
    coefficients: pd.DataFrame
    contributions: pd.DataFrame
    sequential_steps: pd.DataFrame
    data: pd.DataFrame


def p_str(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}".lstrip("0")


def fmt(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def ci_str(low: float, high: float, decimals: int = 3) -> str:
    return f"[{fmt(low, decimals)}, {fmt(high, decimals)}]"


def wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width)


def short_label(predictor: str) -> str:
    return PREDICTOR_INFO[predictor].get("short_label", PREDICTOR_INFO[predictor]["label"])


def load_behavior() -> pd.DataFrame:
    for path in BEHAVIOR_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            break
    else:
        raise FileNotFoundError("Could not locate AMMEBLAES_includedpts_firstsession_behavioral.csv")

    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df["stim_hemisphere"] = df["stim_hemisphere"].astype(str).str.strip().str.upper()
    df["stim_trajectory"] = (
        df["stim_trajectory"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "NaN": np.nan, "nan": np.nan})
    )
    df.loc[df["stim_trajectory"].str.contains("frontal", case=False, na=False), "stim_trajectory"] = np.nan
    df["stim_trajectory"] = df["stim_trajectory"].str.upper()
    df[OUTCOME_COLUMN] = pd.to_numeric(df[OUTCOME_COLUMN], errors="coerce")
    for col in ["age", "stim_DB", "IED_freq", "Memory_Z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def select_active_predictors(model_df: pd.DataFrame, requested_predictors: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    included: list[str] = []
    omitted: list[tuple[str, str]] = []
    for predictor in requested_predictors:
        values = model_df[predictor]
        if values.isna().all():
            omitted.append((predictor, "all values missing"))
            continue

        if PREDICTOR_INFO[predictor]["type"] == "binary":
            valid_values = values.dropna().astype(str).str.strip()
            if valid_values.nunique() < 2:
                omitted.append((predictor, "no variation in analysis subset"))
                continue
        else:
            numeric_values = pd.to_numeric(values, errors="coerce").dropna()
            if numeric_values.empty or np.isclose(float(numeric_values.std(ddof=0)), 0.0):
                omitted.append((predictor, "no variation in analysis subset"))
                continue

        included.append(predictor)
    return included, omitted


def build_design_matrix(model_df: pd.DataFrame, predictors: list[str]) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(model_df))]
    names = ["Intercept"]

    for predictor in predictors:
        info = PREDICTOR_INFO[predictor]
        if info["type"] == "binary":
            if predictor == "sex":
                encoded = (model_df[predictor] == "male").astype(float).to_numpy()
            elif predictor == "stim_trajectory":
                encoded = (model_df[predictor] == "STG").astype(float).to_numpy()
            elif predictor == "stim_hemisphere":
                encoded = (model_df[predictor] == "R").astype(float).to_numpy()
            else:
                raise ValueError(f"Unhandled binary predictor: {predictor}")
        else:
            encoded = pd.to_numeric(model_df[predictor], errors="coerce").to_numpy(dtype=float)
        columns.append(encoded)
        names.append(predictor)

    return np.column_stack(columns), names


def fit_ols_from_matrix(y: np.ndarray, X: np.ndarray, term_names: list[str]) -> tuple[dict, pd.DataFrame]:
    n_obs, n_params = X.shape
    dof_resid = n_obs - n_params
    if dof_resid <= 0:
        raise ValueError("Not enough observations to fit the requested model.")

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    sse = float(np.dot(resid, resid))
    sst = float(np.dot(y - y.mean(), y - y.mean()))
    mse = sse / dof_resid
    sigma = float(np.sqrt(mse))
    cov_beta = mse * xtx_inv
    se = np.sqrt(np.diag(cov_beta))
    t_values = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), dof_resid))
    t_crit = stats.t.ppf(0.975, dof_resid)
    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se

    if sst > 0:
        r2 = 1 - sse / sst
    else:
        r2 = np.nan
    if np.isnan(r2):
        adj_r2 = np.nan
        f_stat = np.nan
        f_p = np.nan
    elif n_params == 1:
        adj_r2 = 0.0
        f_stat = np.nan
        f_p = np.nan
    else:
        adj_r2 = 1 - (1 - r2) * (n_obs - 1) / dof_resid
        f_stat = (r2 / (n_params - 1)) / ((1 - r2) / dof_resid)
        f_p = 1 - stats.f.cdf(f_stat, n_params - 1, dof_resid)

    coefficients = pd.DataFrame(
        {
            "term": term_names,
            "B": beta,
            "SE": se,
            "t": t_values,
            "p": p_values,
            "CI_low": ci_low,
            "CI_high": ci_high,
        }
    )

    fit_stats = {
        "n": n_obs,
        "df_model": n_params - 1,
        "df_resid": dof_resid,
        "sse": sse,
        "sst": sst,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": f_stat,
        "f_p": f_p,
        "sigma": sigma,
    }
    return fit_stats, coefficients


def fit_linear_model(df: pd.DataFrame, spec: dict) -> LinearModelResult:
    required_columns = [OUTCOME_COLUMN] + spec["requested_predictors"]
    model_df = df[required_columns + ["Patient", "Study"]].dropna(subset=required_columns).copy()

    included_predictors, omitted_predictors = select_active_predictors(model_df, spec["requested_predictors"])
    design_df = model_df[[OUTCOME_COLUMN] + included_predictors].copy()
    X, term_names = build_design_matrix(design_df, included_predictors)
    y = design_df[OUTCOME_COLUMN].to_numpy(dtype=float)
    fit_stats, coefficients = fit_ols_from_matrix(y, X, term_names)

    sequential_rows = []
    prev_X, prev_names = build_design_matrix(design_df, [])
    prev_fit_stats, _ = fit_ols_from_matrix(y, prev_X, prev_names)
    for step_index, predictor in enumerate(included_predictors, start=1):
        current_predictors = included_predictors[:step_index]
        step_X, step_names = build_design_matrix(design_df, current_predictors)
        step_fit_stats, _ = fit_ols_from_matrix(y, step_X, step_names)
        delta_r2 = step_fit_stats["r2"] - prev_fit_stats["r2"]
        f_change = ((prev_fit_stats["sse"] - step_fit_stats["sse"]) / 1.0) / (
            step_fit_stats["sse"] / step_fit_stats["df_resid"]
        )
        p_change = 1 - stats.f.cdf(f_change, 1, step_fit_stats["df_resid"])
        sequential_rows.append(
            {
                "step": step_index,
                "predictor": predictor,
                "label": PREDICTOR_INFO[predictor]["label"],
                "model_predictors": "; ".join(PREDICTOR_INFO[item]["label"] for item in current_predictors),
                "R2": step_fit_stats["r2"],
                "Adj_R2": step_fit_stats["adj_r2"],
                "Delta_R2": delta_r2,
                "F_change": f_change,
                "p_change": p_change,
            }
        )
        prev_fit_stats = step_fit_stats

    contributions = []
    for predictor in included_predictors:
        reduced_predictors = [item for item in included_predictors if item != predictor]
        reduced_X, reduced_names = build_design_matrix(design_df, reduced_predictors)
        reduced_fit_stats, _ = fit_ols_from_matrix(y, reduced_X, reduced_names)
        delta_r2 = fit_stats["r2"] - reduced_fit_stats["r2"]
        f_change = ((reduced_fit_stats["sse"] - fit_stats["sse"]) / 1.0) / (
            fit_stats["sse"] / fit_stats["df_resid"]
        )
        p_change = 1 - stats.f.cdf(f_change, 1, fit_stats["df_resid"])
        contributions.append(
            {
                "predictor": predictor,
                "label": PREDICTOR_INFO[predictor]["label"],
                "Delta_R2": delta_r2,
                "F_change": f_change,
                "p_change": p_change,
            }
        )

    return LinearModelResult(
        slug=spec["slug"],
        title=spec["title"],
        description=spec["description"],
        requested_predictors=spec["requested_predictors"],
        included_predictors=included_predictors,
        omitted_predictors=omitted_predictors,
        n=int(fit_stats["n"]),
        df_model=int(fit_stats["df_model"]),
        df_resid=int(fit_stats["df_resid"]),
        r2=float(fit_stats["r2"]),
        adj_r2=float(fit_stats["adj_r2"]),
        f_stat=float(fit_stats["f_stat"]) if pd.notna(fit_stats["f_stat"]) else np.nan,
        f_p=float(fit_stats["f_p"]) if pd.notna(fit_stats["f_p"]) else np.nan,
        sigma=float(fit_stats["sigma"]),
        coefficients=coefficients,
        contributions=pd.DataFrame(contributions),
        sequential_steps=pd.DataFrame(sequential_rows),
        data=model_df.reset_index(drop=True),
    )


def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [OUTCOME_COLUMN, "sex", "stim_trajectory", "age", "stim_hemisphere", "stim_DB", "IED_freq", "Memory_Z"]:
        rows.append(
            {
                "Variable": column,
                "Label": MEMORY_LABEL if column == OUTCOME_COLUMN else PREDICTOR_INFO[column]["label"],
                "Non-missing": int(df[column].notna().sum()),
                "Missing": int(df[column].isna().sum()),
                "Percent available": df[column].notna().mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def compute_sex_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("sex")[OUTCOME_COLUMN]
    summary = pd.DataFrame(
        {
            "N": grouped.size(),
            "Mean": grouped.mean(),
            "SD": grouped.std(ddof=1),
            "Median": grouped.median(),
            "Min": grouped.min(),
            "Max": grouped.max(),
        }
    )
    return summary.reindex(["female", "male"]).reset_index().rename(columns={"sex": "Group"})


def compute_unadjusted_sex_test(df: pd.DataFrame) -> dict:
    female = df.loc[df["sex"] == "female", OUTCOME_COLUMN].dropna().to_numpy(dtype=float)
    male = df.loc[df["sex"] == "male", OUTCOME_COLUMN].dropna().to_numpy(dtype=float)
    test = stats.ttest_ind(male, female, equal_var=False)
    mean_diff = male.mean() - female.mean()
    var_f = female.var(ddof=1)
    var_m = male.var(ddof=1)
    se_diff = np.sqrt(var_f / len(female) + var_m / len(male))
    df_num = (var_f / len(female) + var_m / len(male)) ** 2
    df_den = (var_f**2) / ((len(female) ** 2) * (len(female) - 1)) + (var_m**2) / (
        (len(male) ** 2) * (len(male) - 1)
    )
    df_welch = df_num / df_den
    t_crit = stats.t.ppf(0.975, df_welch)
    return {
        "female_mean": float(female.mean()),
        "male_mean": float(male.mean()),
        "female_sd": float(female.std(ddof=1)),
        "male_sd": float(male.std(ddof=1)),
        "mean_diff": float(mean_diff),
        "t": float(test.statistic),
        "df": float(df_welch),
        "p": float(test.pvalue),
        "ci_low": float(mean_diff - t_crit * se_diff),
        "ci_high": float(mean_diff + t_crit * se_diff),
    }


def build_model_summary_table(results: list[LinearModelResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        included_labels = [short_label(p) for p in result.included_predictors]
        omitted_labels = [
            f"{short_label(p)} ({reason})" for p, reason in result.omitted_predictors
        ]
        rows.append(
            {
                "Model": result.title,
                "N": result.n,
                "Predictors used": "; ".join(included_labels),
                "Omitted in subset": "; ".join(omitted_labels) if omitted_labels else "",
                "R2": result.r2,
                "Adj R2": result.adj_r2,
                "Model p": result.f_p,
            }
        )
    return pd.DataFrame(rows)


def pretty_coefficients(result: LinearModelResult) -> pd.DataFrame:
    rows = []
    for _, row in result.coefficients.iterrows():
        term = row["term"]
        if term == "Intercept":
            label = "Intercept"
        else:
            label = short_label(term)
        rows.append(
            {
                "Term": label,
                "B": fmt(row["B"]),
                "SE": fmt(row["SE"]),
                "t": fmt(row["t"]),
                "p": p_str(row["p"]),
                "95% CI": ci_str(row["CI_low"], row["CI_high"]),
            }
        )
    return pd.DataFrame(rows)


def pretty_contributions(result: LinearModelResult) -> pd.DataFrame:
    if result.contributions.empty:
        return pd.DataFrame(columns=["Predictor", "Delta R2", "F change", "p"])
    out = result.contributions.copy()
    out["Predictor"] = out["predictor"].map(short_label)
    out["Delta R2"] = out["Delta_R2"].map(fmt)
    out["F change"] = out["F_change"].map(fmt)
    out["p"] = out["p_change"].map(p_str)
    return out[["Predictor", "Delta R2", "F change", "p"]]


def pretty_sequential_steps(result: LinearModelResult) -> pd.DataFrame:
    if result.sequential_steps.empty:
        return pd.DataFrame(columns=["Step", "Added predictor", "Model R2", "Delta R2", "F change", "p"])
    out = result.sequential_steps.copy()
    out["Step"] = out["step"].astype(int)
    out["Added predictor"] = out["predictor"].map(short_label)
    out["Model R2"] = out["R2"].map(fmt)
    out["Delta R2"] = out["Delta_R2"].map(fmt)
    out["F change"] = out["F_change"].map(fmt)
    out["p"] = out["p_change"].map(p_str)
    return out[["Step", "Added predictor", "Model R2", "Delta R2", "F change", "p"]]


def export_tables(
    missingness: pd.DataFrame,
    sex_descriptives: pd.DataFrame,
    model_summary: pd.DataFrame,
    results: list[LinearModelResult],
) -> None:
    for folder in [SUMMARY_DIR, COEFFICIENTS_DIR, CONTRIBUTIONS_DIR, SEQUENTIAL_STEPS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    missingness.to_csv(SUMMARY_DIR / "missingness_table.csv", index=False)
    sex_descriptives.to_csv(SUMMARY_DIR / "sex_descriptives_table.csv", index=False)
    model_summary.to_csv(SUMMARY_DIR / "model_summary_table.csv", index=False)

    for result in results:
        pretty_coefficients(result).to_csv(
            COEFFICIENTS_DIR / f"{result.slug}_coefficients.csv",
            index=False,
        )
        pretty_sequential_steps(result).to_csv(
            SEQUENTIAL_STEPS_DIR / f"{result.slug}_sequential_steps.csv",
            index=False,
        )
        pretty_contributions(result).to_csv(
            CONTRIBUTIONS_DIR / f"{result.slug}_contributions.csv",
            index=False,
        )


def start_page() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def draw_wrapped_text(ax: plt.Axes, x: float, y: float, text: str, fontsize: int = 11, weight: str = "normal") -> None:
    ax.text(
        x,
        y,
        wrap(text, 100),
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight=weight,
        family="DejaVu Serif",
    )


def draw_apa_table(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    note: str | None = None,
    bbox: list[float] | None = None,
    font_size: int = 9,
    col_widths: list[float] | None = None,
) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.04,
        title,
        ha="left",
        va="bottom",
        fontsize=11,
        style="italic",
        family="DejaVu Serif",
        transform=ax.transAxes,
    )

    display_df = df.copy()
    if bbox is None:
        bbox = [0, 0.05, 1, 0.9]
    table = ax.table(
        cellText=display_df.values,
        colLabels=list(display_df.columns),
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=bbox,
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("white")
        cell.set_linewidth(0.0)
        cell.PAD = 0.08
        if row == 0:
            cell.set_text_props(weight="bold", family="DejaVu Serif")
            cell.visible_edges = "TB"
            cell.set_edgecolor("black")
            cell.set_linewidth(1.0)
        else:
            cell.set_text_props(family="DejaVu Serif")

    last_row = len(display_df)
    for col in range(len(display_df.columns)):
        cell = table[(last_row, col)]
        cell.visible_edges = "B"
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

    if note:
        ax.text(
            0.0,
            0.01,
            wrap(f"Note. {note}", 115),
            ha="left",
            va="bottom",
            fontsize=8.5,
            style="italic",
            family="DejaVu Serif",
            transform=ax.transAxes,
        )


def add_title_page(pdf: PdfPages, sex_test: dict, primary_result: LinearModelResult) -> None:
    fig, ax = start_page()
    primary_steps = primary_result.sequential_steps.set_index("predictor", drop=False)
    ax.text(
        0.5,
        0.95,
        "Between-Subject Variability in Memory Modulation",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Serif",
    )
    ax.text(
        0.5,
        0.915,
        "Patient-level linear models using the behavioral first-session summary file",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
    )

    draw_wrapped_text(
        ax,
        0.08,
        0.84,
        "Requested outcome: avg_dprime_diff. The behavioral file does not contain a column with that exact name; "
        "the available patient-level outcome is avg_stim_dprime_diff, which was used here.",
    )
    draw_wrapped_text(
        ax,
        0.08,
        0.76,
        "Critical modeling point: this file has one row per patient (54 rows, 54 unique patients). "
        "Because the outcome is already collapsed to a single number per person, a subject random intercept is not identifiable. "
        "The correct analysis at this level is a patient-level linear model, not an MLM with (1 | subject).",
    )

    bullets = [
        f"Unadjusted sex difference: males M = {fmt(sex_test['male_mean'])}, females M = {fmt(sex_test['female_mean'])}, "
        f"mean difference = {fmt(sex_test['mean_diff'])}, Welch t({fmt(sex_test['df'], 2)}) = {fmt(sex_test['t'])}, p = {p_str(sex_test['p'])}.",
        f"Primary adjusted model: R2 = {fmt(primary_result.r2)}, adjusted R2 = {fmt(primary_result.adj_r2)}, "
        f"F({primary_result.df_model}, {primary_result.df_resid}) = {fmt(primary_result.f_stat)}, p = {p_str(primary_result.f_p)}.",
        f"Sequential build: sex entered first (model R2 = {fmt(primary_steps.loc['sex', 'R2'])}, "
        f"step p = {p_str(primary_steps.loc['sex', 'p_change'])}); stim trajectory entered second "
        f"(delta R2 = {fmt(primary_steps.loc['stim_trajectory', 'Delta_R2'])}, "
        f"step p = {p_str(primary_steps.loc['stim_trajectory', 'p_change'])}).",
        "IED_freq and Memory_Z did not improve model fit in the smaller subsets where they were available.",
    ]
    ax.text(
        0.08,
        0.62,
        "Executive Summary",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        family="DejaVu Serif",
    )
    y = 0.58
    for bullet in bullets:
        draw_wrapped_text(ax, 0.10, y, f"• {bullet}", fontsize=11)
        y -= 0.085

    ax.text(
        0.08,
        0.18,
        "Interpretive frame",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        family="DejaVu Serif",
    )
    draw_wrapped_text(
        ax,
        0.08,
        0.145,
        "These models do not explain the original trial-level ICC directly. Instead, they test whether the requested "
        "patient-level covariates are associated with between-subject differences in the already-collapsed memory modulation score.",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_data_page(pdf: PdfPages, missingness: pd.DataFrame, sex_descriptives: pd.DataFrame, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax_top = fig.add_axes([0.08, 0.56, 0.84, 0.32])
    ax_bottom = fig.add_axes([0.08, 0.18, 0.84, 0.24])
    ax_text = fig.add_axes([0.08, 0.90, 0.84, 0.07])
    ax_text.axis("off")
    ax_text.text(
        0.0,
        0.9,
        "Data Availability and Descriptives",
        fontsize=15,
        fontweight="bold",
        family="DejaVu Serif",
        va="top",
    )

    missing_display = missingness.copy()
    missing_display["Percent available"] = missing_display["Percent available"].map(lambda v: f"{v:.1f}%")
    draw_apa_table(
        ax_top,
        missing_display[["Variable", "Label", "Non-missing", "Missing", "Percent available"]],
        "Table 1\nVariable availability in the behavioral summary file",
        note=(
            "Stim trajectory is available for 48/54 patients after excluding the 3 frontal-trajectory cases from modeling. "
            "Memory_Z is available for 31/54 patients and is concentrated in the Original, Timing, and Duration cohorts. "
            "IED_freq is available for 44/54 patients."
        ),
        font_size=8.8,
        bbox=[0, 0.08, 1, 0.84],
        col_widths=[0.15, 0.39, 0.14, 0.12, 0.20],
    )

    sex_display = sex_descriptives.copy()
    for col in ["Mean", "SD", "Median", "Min", "Max"]:
        sex_display[col] = sex_display[col].map(fmt)
    draw_apa_table(
        ax_bottom,
        sex_display[["Group", "N", "Mean", "SD", "Median", "Min", "Max"]],
        "Table 2\nOutcome descriptives by sex",
        note="Means are computed on avg_stim_dprime_diff. Positive values indicate greater memory performance on stimulated relative to non-stimulated trials.",
        font_size=9,
        bbox=[0, 0.10, 1, 0.78],
        col_widths=[0.17, 0.09, 0.14, 0.14, 0.14, 0.14, 0.14],
    )

    fig.text(
        0.08,
        0.49,
        wrap(
            "In the raw patient-level outcome, males showed higher avg_stim_dprime_diff than females. "
            "That unadjusted difference is useful descriptively, but the primary question is whether it survives "
            "after accounting for stimulation trajectory, age, stimulation hemisphere, and stimulation amplitude.",
            110,
        ),
        fontsize=10.5,
        family="DejaVu Serif",
        va="top",
    )

    study_note = (
        df.loc[df["Memory_Z"].notna(), "Study"]
        .value_counts()
        .rename_axis("Study")
        .reset_index(name="Count")
    )
    study_line = ", ".join(f"{row.Study} (n={row.Count})" for row in study_note.itertuples(index=False))
    fig.text(
        0.08,
        0.11,
        wrap(
            f"Memory_Z subset composition: {study_line}. Within that subset, stim_DB has no variation: all available cases are 0.5 DB. "
            "Stim trajectory is modeled as STG versus MTG, with frontal trajectories excluded to match the separate trajectory test.",
            110,
        ),
        fontsize=9.5,
        family="DejaVu Serif",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_primary_pages(pdf: PdfPages, result: LinearModelResult, sex_test: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax_title = fig.add_axes([0.08, 0.90, 0.84, 0.07])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.9,
        result.title,
        fontsize=15,
        fontweight="bold",
        family="DejaVu Serif",
        va="top",
    )
    ax_title.text(
        0.0,
        0.35,
        wrap(result.description, 110),
        fontsize=10.5,
        family="DejaVu Serif",
        va="top",
    )

    summary_df = pd.DataFrame(
        [
            {
                "N": result.n,
                "Predictors": len(result.included_predictors),
                "R2": fmt(result.r2),
                "Adj R2": fmt(result.adj_r2),
                "F(df)": f"{fmt(result.f_stat)} ({result.df_model}, {result.df_resid})",
                "Model p": p_str(result.f_p),
            }
        ]
    )
    ax_summary = fig.add_axes([0.08, 0.74, 0.84, 0.10])
    draw_apa_table(
        ax_summary,
        summary_df,
        "Table 3\nPrimary model fit summary",
        note="The primary model includes sex first, then stim trajectory, age, stimulation hemisphere, and stimulation amplitude.",
        font_size=9.3,
        bbox=[0, 0.18, 1, 0.58],
        col_widths=[0.10, 0.14, 0.12, 0.12, 0.28, 0.12],
    )

    ax_coef = fig.add_axes([0.08, 0.36, 0.84, 0.34])
    draw_apa_table(
        ax_coef,
        pretty_coefficients(result),
        "Table 4\nPrimary model coefficients",
        note="Female, MTG trajectory, and left hemisphere are the reference groups for binary predictors.",
        font_size=9.0,
        bbox=[0, 0.08, 1, 0.84],
        col_widths=[0.36, 0.11, 0.11, 0.11, 0.11, 0.20],
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8.5, 11))
    ax_seq = fig.add_axes([0.08, 0.58, 0.84, 0.24])
    draw_apa_table(
        ax_seq,
        pretty_sequential_steps(result),
        "Table 5\nSequential model building in the requested order",
        note="Each row shows the change produced by adding that predictor after all predictors above it were already in the model.",
        font_size=8.6,
        bbox=[0, 0.10, 1, 0.82],
        col_widths=[0.08, 0.36, 0.14, 0.14, 0.14, 0.10],
    )

    coef_sex = result.coefficients.loc[result.coefficients["term"] == "sex"].iloc[0]
    coef_traj = result.coefficients.loc[result.coefficients["term"] == "stim_trajectory"].iloc[0]
    sex_step = result.sequential_steps.loc[result.sequential_steps["predictor"] == "sex"].iloc[0]
    traj_step = result.sequential_steps.loc[result.sequential_steps["predictor"] == "stim_trajectory"].iloc[0]
    narrative = (
        f"The primary adjusted model was significant overall, F({result.df_model}, {result.df_resid}) = "
        f"{fmt(result.f_stat)}, p = {p_str(result.f_p)}, and accounted for {fmt(result.r2)} of the variance in "
        f"{OUTCOME_LABEL}. In the requested sequence, sex entered first and produced model R2 = {fmt(sex_step['R2'])} "
        f"(step p = {p_str(sex_step['p_change'])}); adding stim trajectory next increased R2 by {fmt(traj_step['Delta_R2'])} "
        f"(step p = {p_str(traj_step['p_change'])}). The adjusted coefficient for sex was B = {fmt(coef_sex['B'])} "
        f"(95% CI {ci_str(coef_sex['CI_low'], coef_sex['CI_high'])}), t = {fmt(coef_sex['t'])}, p = {p_str(coef_sex['p'])}, "
        f"and the adjusted coefficient for STG versus MTG trajectory was B = {fmt(coef_traj['B'])} "
        f"(95% CI {ci_str(coef_traj['CI_low'], coef_traj['CI_high'])}), t = {fmt(coef_traj['t'])}, p = {p_str(coef_traj['p'])}."
    )
    fig.text(0.08, 0.90, "Primary Model Interpretation", fontsize=15, fontweight="bold", family="DejaVu Serif")
    fig.text(0.08, 0.54, wrap(narrative, 112), fontsize=10.2, family="DejaVu Serif")

    ax_plot = fig.add_axes([0.11, 0.14, 0.78, 0.28])
    plot_df = result.data.copy()
    colors = {"female": "#c96f4a", "male": "#2f5d7e"}
    x_positions = {"female": 0, "male": 1}
    rng = np.random.default_rng(42)
    for sex_value in ["female", "male"]:
        sub = plot_df.loc[plot_df["sex"] == sex_value, OUTCOME_COLUMN].dropna().to_numpy(dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=len(sub))
        ax_plot.scatter(
            np.full(len(sub), x_positions[sex_value]) + jitter,
            sub,
            s=55,
            alpha=0.85,
            color=colors[sex_value],
            edgecolor="black",
            linewidth=0.5,
        )
        mean_value = sub.mean()
        se_value = sub.std(ddof=1) / np.sqrt(len(sub))
        ci_half = stats.t.ppf(0.975, len(sub) - 1) * se_value
        ax_plot.errorbar(
            x_positions[sex_value],
            mean_value,
            yerr=ci_half,
            fmt="s",
            ms=9,
            mfc="white",
            mec="black",
            ecolor="black",
            capsize=5,
            zorder=4,
        )
    ax_plot.axhline(0, color="0.65", linestyle="--", linewidth=1)
    ax_plot.set_xticks([0, 1], ["Female", "Male"])
    ax_plot.set_ylabel(OUTCOME_LABEL)
    ax_plot.set_title("Raw patient-level outcome by sex", fontsize=13, family="DejaVu Serif")
    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)

    fig.text(
        0.08,
        0.49,
        "Unadjusted Sex Difference",
        fontsize=15,
        fontweight="bold",
        family="DejaVu Serif",
    )
    fig.text(
        0.08,
        0.08,
        wrap(
            f"Welch's test on the raw patient-level outcome gave t({fmt(sex_test['df'], 2)}) = {fmt(sex_test['t'])}, "
            f"p = {p_str(sex_test['p'])}, with males higher than females by {fmt(sex_test['mean_diff'])} "
            f"(95% CI {ci_str(sex_test['ci_low'], sex_test['ci_high'])}). "
            "The adjusted primary model then evaluated sex first, followed by stimulation trajectory and the remaining covariates in that order.",
            110,
        ),
        fontsize=10.5,
        family="DejaVu Serif",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_sensitivity_pages(pdf: PdfPages, results: list[LinearModelResult]) -> None:
    summary_df = build_model_summary_table(results).copy()
    summary_df["R2"] = summary_df["R2"].map(fmt)
    summary_df["Adj R2"] = summary_df["Adj R2"].map(fmt)
    summary_df["Model p"] = summary_df["Model p"].map(p_str)

    fig = plt.figure(figsize=(8.5, 11))
    ax_title = fig.add_axes([0.08, 0.90, 0.84, 0.07])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.9,
        "Sensitivity Models",
        fontsize=15,
        fontweight="bold",
        family="DejaVu Serif",
        va="top",
    )
    ax_title.text(
        0.0,
        0.35,
        wrap(
            "The sensitivity models show what happens when IED_freq and Memory_Z are brought into the analysis. "
            "The cost is lower N and, for Memory_Z, a narrower set of study cohorts. "
            "Within those reduced subsets, no additional requested predictor produced a clear signal.",
            112,
        ),
        fontsize=10.5,
        family="DejaVu Serif",
        va="top",
    )
    ax_table = fig.add_axes([0.08, 0.17, 0.84, 0.66])
    draw_apa_table(
        ax_table,
        summary_df,
        "Table 6\nComparison of fitted patient-level models",
        note=(
            "In the Memory_Z-only and complete-case subsets, stim_DB was automatically omitted because all available cases "
            "were at 0.5 DB, leaving no within-subset variation to estimate its effect."
        ),
        font_size=8.2,
        bbox=[0, 0.06, 1, 0.90],
        col_widths=[0.16, 0.07, 0.29, 0.24, 0.07, 0.08, 0.09],
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    for result in results[1:]:
        fig = plt.figure(figsize=(8.5, 11))
        ax_title = fig.add_axes([0.08, 0.90, 0.84, 0.07])
        ax_title.axis("off")
        ax_title.text(
            0.0,
            0.9,
            result.title,
            fontsize=15,
            fontweight="bold",
            family="DejaVu Serif",
            va="top",
        )
        ax_title.text(
            0.0,
            0.35,
            wrap(result.description, 112),
            fontsize=10.5,
            family="DejaVu Serif",
            va="top",
        )

        ax_coef = fig.add_axes([0.08, 0.28, 0.84, 0.50])
        draw_apa_table(
            ax_coef,
            pretty_coefficients(result),
            f"Table\n{result.title} coefficients",
            note="Female, MTG trajectory, and left hemisphere are the reference groups when present. Constant predictors in a subset were omitted automatically.",
            font_size=8.9,
            bbox=[0, 0.10, 1, 0.82],
            col_widths=[0.36, 0.11, 0.11, 0.11, 0.11, 0.20],
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        ax_title = fig.add_axes([0.08, 0.90, 0.84, 0.07])
        ax_title.axis("off")
        ax_title.text(
            0.0,
            0.9,
            f"{result.title}: Stepwise Build",
            fontsize=15,
            fontweight="bold",
            family="DejaVu Serif",
            va="top",
        )

        ax_contrib = fig.add_axes([0.08, 0.48, 0.84, 0.30])
        draw_apa_table(
            ax_contrib,
            pretty_sequential_steps(result),
            f"Table\n{result.title} sequential model-building steps",
            note="Predictors are entered in the requested order for that model and delta R2 values are subset-specific.",
            font_size=8.6,
            bbox=[0, 0.12, 1, 0.78],
            col_widths=[0.08, 0.36, 0.14, 0.14, 0.14, 0.10],
        )

        omitted_line = "None"
        if result.omitted_predictors:
            omitted_line = "; ".join(
                f"{short_label(p)} ({reason})" for p, reason in result.omitted_predictors
            )
        fig.text(
            0.08,
            0.38,
            wrap(
                f"N = {result.n}, R2 = {fmt(result.r2)}, adjusted R2 = {fmt(result.adj_r2)}, "
                f"F({result.df_model}, {result.df_resid}) = {fmt(result.f_stat)}, p = {p_str(result.f_p)}. "
                f"Omitted predictors: {omitted_line}.",
                112,
            ),
            fontsize=10.2,
            family="DejaVu Serif",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_conclusion_page(pdf: PdfPages, primary_result: LinearModelResult, results: list[LinearModelResult]) -> None:
    fig, ax = start_page()
    ax.text(
        0.08,
        0.93,
        "Bottom Line",
        fontsize=15,
        fontweight="bold",
        family="DejaVu Serif",
        va="top",
    )

    primary_sex = primary_result.coefficients.loc[primary_result.coefficients["term"] == "sex"].iloc[0]
    primary_traj = primary_result.coefficients.loc[primary_result.coefficients["term"] == "stim_trajectory"].iloc[0]
    ied_result = next(item for item in results if item.slug == "ied_sensitivity")
    mem_result = next(item for item in results if item.slug == "memoryz_sensitivity")
    primary_steps = primary_result.sequential_steps.set_index("predictor", drop=False)
    ied_steps = ied_result.sequential_steps.set_index("predictor", drop=False)
    mem_steps = mem_result.sequential_steps.set_index("predictor", drop=False)

    paragraphs = [
        "This file supports a between-subject regression question, not a within-subject MLM question. "
        "The outcome is already one number per patient, so there is no residual subject-level nesting to model.",
        f"In the full-sample adjusted model, sex was entered first and stimulation trajectory second. "
        f"B = {fmt(primary_sex['B'])}, 95% CI {ci_str(primary_sex['CI_low'], primary_sex['CI_high'])}, "
        f"p = {p_str(primary_sex['p'])} for sex, and B = {fmt(primary_traj['B'])}, "
        f"95% CI {ci_str(primary_traj['CI_low'], primary_traj['CI_high'])}, p = {p_str(primary_traj['p'])} "
        "for STG versus MTG trajectory after adjustment.",
        f"Sequentially, sex produced model R2 = {fmt(primary_steps.loc['sex', 'R2'])} and adding trajectory increased R2 by "
        f"{fmt(primary_steps.loc['stim_trajectory', 'Delta_R2'])}. Later additions were smaller and did not reverse the earlier pattern.",
        f"IED_freq did not contribute meaningfully in the sensitivity model when it was added after the primary covariates "
        f"(step p = {p_str(ied_steps.loc['IED_freq', 'p_change'])}).",
        f"Memory_Z was only available for 31 patients from a restricted subset of studies, and its coefficient was not significant "
        f"in that subset (step p = {p_str(mem_steps.loc['Memory_Z', 'p_change'])}). Because all cases in that subset were at 0.5 DB, stim_DB could not be estimated there.",
        "If the substantive question is still what explains the high ICC from the original trial-level null model, the next step is to return to the trial-level dataset "
        "and add these patient-level covariates there as fixed effects in a mixed model. The present report instead shows which patient-level covariates relate to the collapsed patient summary outcome.",
    ]

    y = 0.86
    for paragraph in paragraphs:
        draw_wrapped_text(ax, 0.08, y, paragraph, fontsize=11)
        y -= 0.12

    ax.text(
        0.08,
        0.25,
        "Recommended interpretation",
        fontsize=13,
        fontweight="bold",
        family="DejaVu Serif",
        va="top",
    )
    draw_wrapped_text(
        ax,
        0.08,
        0.21,
        "Among the requested subject-level variables, sex and stimulation trajectory now enter the patient-level model first and should be interpreted together. "
        "The later covariates mainly test whether that earlier between-subject pattern survives adjustment rather than replacing it. Age, hemisphere, amplitude, IED_freq, and Memory_Z added comparatively modest information in these patient-level models.",
        fontsize=11,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_behavior()
    missingness = compute_missingness(df)
    sex_descriptives = compute_sex_descriptives(df)
    sex_test = compute_unadjusted_sex_test(df)
    results = [fit_linear_model(df, spec) for spec in MODEL_SPECS]
    export_tables(missingness, sex_descriptives, build_model_summary_table(results), results)

    with PdfPages(OUTPUT_PDF) as pdf:
        add_title_page(pdf, sex_test, results[0])
        add_data_page(pdf, missingness, sex_descriptives, df)
        add_primary_pages(pdf, results[0], sex_test)
        add_sensitivity_pages(pdf, results)
        add_conclusion_page(pdf, results[0], results)

    print(f"Wrote report to {OUTPUT_PDF}")
    print(f"Wrote tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
