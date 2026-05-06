#!/usr/bin/env python
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STATS_ROOT = SCRIPT_DIR / "outputs" / "retrieval_memory_reports" / "stats"
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "retrieval_memory_reports"

MEASURE_LABELS = {
    "power": "Power",
    "coherence": "Coherence",
    "pac": "PAC",
}

OUTCOME_LABELS = {
    "mlm": {
        "title": "MLM",
        "outcome": "avg_stim_dprime_diff",
        "estimate_label": "Estimate",
    },
    "glmm": {
        "title": "GLMM",
        "outcome": "trial-level memory accuracy",
        "estimate_label": "Odds Ratio",
    },
}

ANALYSIS_LABELS = {
    "frequency_across_regions": "Within Frequency Across Regions",
    "region_across_bands": "Within Region Across Bands",
}

REGION_LABELS = {
    "BLA": "BLA",
    "CA": "CA",
    "DG": "DG",
    "HPC": "HPC",
    "EC": "EC",
    "PRC": "PRC",
}

TERM_LABELS = {
    "(Intercept)": "Intercept",
    "StimCondstim": "StimCond [stim]",
    "band_c": "Band",
    "band_c:StimCondstim": "Band x StimCond [stim]",
    "StimCondstim:band_c": "Band x StimCond [stim]",
    "theta_c": "Theta",
    "slow_gamma_c": "Slow Gamma",
    "slow_gamma_pac_c": "Slow Gamma PAC",
    "hfa_pac_c": "HFA PAC",
    "theta_c:StimCondstim": "Theta x StimCond [stim]",
    "StimCondstim:theta_c": "Theta x StimCond [stim]",
    "slow_gamma_c:StimCondstim": "Slow Gamma x StimCond [stim]",
    "StimCondstim:slow_gamma_c": "Slow Gamma x StimCond [stim]",
    "slow_gamma_pac_c:StimCondstim": "Slow Gamma PAC x StimCond [stim]",
    "StimCondstim:slow_gamma_pac_c": "Slow Gamma PAC x StimCond [stim]",
    "hfa_pac_c:StimCondstim": "HFA PAC x StimCond [stim]",
    "StimCondstim:hfa_pac_c": "HFA PAC x StimCond [stim]",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
    }
)


def p_str(value):
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "< .001"
    return f"{value:.3f}".lstrip("0")


def fmt_num(value, decimals=3):
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


def display_region(region: str) -> str:
    parts = str(region).split("_")
    return " - ".join(REGION_LABELS.get(part, part) for part in parts)


def display_target(measure_slug: str, analysis_type: str, target: str) -> str:
    if analysis_type == "frequency_across_regions":
        if target == "theta":
            return "Theta"
        if target == "slow_gamma":
            return "Slow Gamma"
        if target == "slow_gamma_pac":
            return "Slow Gamma PAC"
        if target == "hfa_pac":
            return "HFA PAC"
    return display_region(target)


def label_term(term: str) -> str:
    if term in TERM_LABELS:
        return TERM_LABELS[term]
    if term.startswith("Region") and ":" not in term:
        return f"Region [{display_region(term.replace('Region', ''))}]"
    if term.startswith("Region") and ":band_c" in term:
        return f"Band x Region [{display_region(term.replace('Region', '').replace(':band_c', ''))}]"
    if term.startswith("band_c:Region"):
        return f"Band x Region [{display_region(term.replace('band_c:Region', ''))}]"
    return term


def interaction_direction(row: pd.Series, outcome_type: str) -> str:
    estimate = float(row["estimate"])
    if outcome_type == "glmm":
        return "positive" if estimate > 1 else "negative"
    return "positive" if estimate > 0 else "negative"


def interaction_strength_text(row: pd.Series, outcome_type: str) -> str:
    p_value = row["p_value"]
    if pd.isna(p_value):
        return "could not be tested cleanly"
    if p_value < 0.05:
        level = "significant"
    elif p_value < 0.10:
        level = "marginal"
    else:
        level = "not significant"
    est_label = "OR" if outcome_type == "glmm" else "b"
    return f"{level} ({est_label} = {fmt_num(row['estimate'])}, p = {p_str(p_value)})"


def build_model_finding(interactions: pd.DataFrame, outcome_type: str) -> str:
    if interactions.empty:
        return "No StimCond interaction term was returned for this model."
    parts = []
    for _, row in interactions.sort_values("p_value", na_position="last").iterrows():
        label = label_term(row["term"]).replace(" x StimCond [stim]", "")
        direction = interaction_direction(row, outcome_type)
        parts.append(
            f"{label}: {interaction_strength_text(row, outcome_type)} and the interaction direction was {direction}."
        )
    return " ".join(parts)


def make_text_page(title: str, paragraphs: list[str], subtitle: str | None = None):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.9])
    ax.axis("off")
    y = 0.98
    ax.text(0, y, title, fontsize=16, fontweight="bold", va="top")
    y -= 0.05
    if subtitle:
        ax.text(0, y, subtitle, fontsize=11, va="top")
        y -= 0.05
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=100)
        ax.text(0, y, wrapped, fontsize=10.5, va="top")
        y -= 0.04 * (wrapped.count("\n") + 1) + 0.02
    return fig


def make_table_pages(title: str, df: pd.DataFrame, note: str | None = None, rows_per_page: int = 22):
    if df.empty:
        return [make_text_page(title, ["No rows were available for this table."])]

    pages = []
    for start in range(0, len(df), rows_per_page):
        chunk = df.iloc[start:start + rows_per_page].copy()
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
        ax.axis("off")
        page_title = title if start == 0 else f"{title} (continued)"
        ax.text(0, 1.02, page_title, fontsize=14, fontweight="bold", va="bottom")
        table = ax.table(
            cellText=chunk.values.tolist(),
            colLabels=list(chunk.columns),
            loc="upper left",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.6)
        table.scale(1, 1.35)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#404040" if row == 0 else "#b0b0b0")
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#e8e8e8")
            else:
                cell.set_facecolor("white")
        if note and start == 0:
            ax.text(0, -0.03, textwrap.fill(f"Note. {note}", width=110), fontsize=8.5, style="italic", va="top")
        pages.append(fig)
    return pages


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_summary_df(report_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in report_manifest.iterrows():
        interaction_df = read_csv(row["interaction_summary_path"])
        if interaction_df.empty:
            interaction_label = "no interaction term"
            p_value = ""
        else:
            best = interaction_df.sort_values("p_value", na_position="last").iloc[0]
            interaction_label = label_term(best["term"])
            p_value = p_str(best["p_value"])
        rows.append(
            {
                "Analysis": ANALYSIS_LABELS[row["analysis_type"]],
                "Family": row["family_label"],
                "Target": display_target(row["measure_slug"], row["analysis_type"], row["target"]),
                "N": int(row["n_patients"]),
                "Rows": int(row["n_rows"]),
                "Interaction": interaction_label,
                "p": p_value,
            }
        )
    return pd.DataFrame(rows)


def build_overall_findings(report_manifest: pd.DataFrame, outcome_type: str) -> list[str]:
    findings = []
    for _, row in report_manifest.iterrows():
        interaction_df = read_csv(row["interaction_summary_path"])
        if interaction_df.empty:
            continue
        sig = interaction_df[interaction_df["p_value"] < 0.05].copy()
        if sig.empty:
            continue
        for _, sig_row in sig.sort_values("p_value").iterrows():
            est_label = "OR" if outcome_type == "glmm" else "b"
            findings.append(
                f"{row['family_label']} | {ANALYSIS_LABELS[row['analysis_type']]} | "
                f"{display_target(row['measure_slug'], row['analysis_type'], row['target'])}: "
                f"{label_term(sig_row['term'])} ({est_label} = {fmt_num(sig_row['estimate'])}, p = {p_str(sig_row['p_value'])})."
            )
    if not findings:
        return ["No StimCond interaction reached p < .05 in this report."]
    return findings[:10]


def write_report(measure_slug: str, outcome_type: str):
    manifest_path = STATS_ROOT / measure_slug / "manifest.csv"
    if not manifest_path.exists():
        return

    manifest = read_csv(manifest_path)
    if "outcome_type" not in manifest.columns:
        manifest["outcome_type"] = manifest["model_id"].astype(str).str.split("__").str[1]
    report_manifest = manifest[manifest["outcome_type"] == outcome_type].copy()
    if report_manifest.empty:
        return

    report_manifest = report_manifest.sort_values(["analysis_type", "family_label", "target"])
    measure_label = MEASURE_LABELS[measure_slug]
    outcome_cfg = OUTCOME_LABELS[outcome_type]
    out_path = OUTPUT_ROOT / f"Retrieval_{measure_label}_{outcome_cfg['title']}_Report.pdf"

    with PdfPages(out_path) as pdf:
        pdf.savefig(
            make_text_page(
                f"Retrieval {measure_label} {outcome_cfg['title']} Report",
                [
                    "These retrieval reports separate power, coherence, and PAC into independent modeling tracks so the analyses do not pool all neural measures into the same model.",
                    f"The {outcome_cfg['title']} outcome is {outcome_cfg['outcome']}. Stimulation was entered before the neural predictor terms and the interaction terms were then tested explicitly.",
                    "Balanced-trial filtering was applied within each retrieval family so that patients with fewer than 10 remembered or 10 forgotten trials in the analyzed subset were excluded before fitting models.",
                    "Model families: (1) Within Frequency Across Regions, which fits one frequency-specific model at a time across the allowed regions or region pairs; and (2) Within Region Across Bands, which fits one region or region-pair model at a time across the relevant band predictors for that measure.",
                ],
            )
        )
        plt.close("all")

        summary_df = build_summary_df(report_manifest)
        for fig in make_table_pages(
            f"Retrieval {measure_label} {outcome_cfg['title']} Model Index",
            summary_df,
            note="The p column shows the smallest interaction p-value returned for each fitted model.",
            rows_per_page=24,
        ):
            pdf.savefig(fig)
            plt.close(fig)

        for _, model_row in report_manifest.iterrows():
            metadata = read_csv(Path(model_row["coefficient_path"]).with_name("metadata.csv")).iloc[0]
            coef_df = read_csv(model_row["coefficient_path"])
            compare_df = read_csv(model_row["model_comparison_path"])
            interaction_df = read_csv(model_row["interaction_summary_path"])
            target_label = display_target(measure_slug, metadata["analysis_type"], metadata["target"])

            final_compare = compare_df[compare_df["Model"] == "m_full"]
            paragraphs = [
                f"Analysis: {ANALYSIS_LABELS[metadata['analysis_type']]}",
                f"Family: {metadata['family_label']}",
                f"Target: {target_label}",
                f"N = {int(metadata['n_patients'])} patients; rows = {int(metadata['n_rows'])}.",
                build_model_finding(interaction_df, outcome_type),
            ]
            if not final_compare.empty:
                paragraphs.insert(
                    4,
                    f"Final nested-model comparison for m_full: AIC = {fmt_num(final_compare.iloc[0]['AIC'])}, "
                    f"BIC = {fmt_num(final_compare.iloc[0]['BIC'])}, p = {p_str(final_compare.iloc[0]['p_value'])}.",
                )

            pdf.savefig(
                make_text_page(
                    f"{ANALYSIS_LABELS[metadata['analysis_type']]} | {target_label}",
                    paragraphs,
                    subtitle=f"Retrieval {measure_label} {outcome_cfg['title']}",
                )
            )
            plt.close("all")

            coef_table = pd.DataFrame(
                {
                    "Predictor": [label_term(term) for term in coef_df["term"]],
                    coef_df["estimate_label"].iloc[0] if not coef_df.empty else outcome_cfg["estimate_label"]: [fmt_num(v) for v in coef_df["estimate"]],
                    "SE": [fmt_num(v) for v in coef_df["std_error"]],
                    "95% CI": [f"[{fmt_num(lo)}, {fmt_num(hi)}]" for lo, hi in zip(coef_df["conf_low"], coef_df["conf_high"])],
                    "Stat": [fmt_num(v) for v in coef_df["statistic"]],
                    "p": [p_str(v) for v in coef_df["p_value"]],
                }
            )
            for fig in make_table_pages(
                f"Fixed Effects | {target_label}",
                coef_table,
                rows_per_page=22,
            ):
                pdf.savefig(fig)
                plt.close(fig)

        pdf.savefig(
            make_text_page(
                f"Overall Findings | Retrieval {measure_label} {outcome_cfg['title']}",
                build_overall_findings(report_manifest, outcome_type),
            )
        )
        plt.close("all")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for measure_slug in ("power", "coherence", "pac"):
        for outcome_type in ("mlm", "glmm"):
            write_report(measure_slug, outcome_type)


if __name__ == "__main__":
    main()
