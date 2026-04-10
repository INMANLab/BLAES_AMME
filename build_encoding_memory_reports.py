#!/usr/bin/env python3
"""
Build APA-formatted PDF reports for encoding memory analyses.
Matches the layout of Does_Baseline_Neural_Activity_x_Stim_Impact_Memory_Encoding.

Structure per measure (Power / Coherence / PAC):
  1. Overview
  2. Per-Region Models  (within each region, across frequency bands)
     - Summary interaction table (one row per region, showing band x stim OR/p)
     - Full coefficient table for each region
  3. Across-Region Models  (within each frequency, across regions)
     - Summary interaction table per frequency band
     - Full coefficient table per frequency
  4. Model building detail  (step-by-step AIC/BIC/ICC for each model)
  5. Summary of all interactions
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF


SCRIPT_DIR = Path(__file__).resolve().parent
STATS_ROOT = SCRIPT_DIR / "outputs" / "encoding_memory_reports" / "stats"
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "encoding_memory_reports"

MEASURE_LABELS = {"power": "Power", "coherence": "Coherence", "pac": "PAC"}

OUTCOME_LABELS = {
    "mlm": {
        "title": "MLM",
        "outcome": "avg_stim_dprime_diff (patient-level d-prime difference)",
        "estimate_label": "Estimate",
        "stat_label": "t",
    },
    "glmm": {
        "title": "GLMM",
        "outcome": "trial-level memory accuracy (binomial)",
        "estimate_label": "OR",
        "stat_label": "z",
    },
}

ANALYSIS_LABELS = {
    "frequency_across_regions": "Across-Region",
    "region_across_bands": "Per-Region",
}

REGION_LABELS = {
    "BLA": "Amygdala", "CA": "CA", "DG": "DG", "HPC": "HPC",
    "EC": "EC", "PRC": "PRC",
}

TERM_LABELS = {
    "(Intercept)": "Intercept",
    "StimCondstim": "StimCond [stim]",
    "band_c": "Band (centered)",
    "band_c:StimCondstim": "Band x StimCond",
    "StimCondstim:band_c": "Band x StimCond",
    "theta_c": "Theta (centered)",
    "slow_gamma_c": "Slow Gamma (centered)",
    "slow_gamma_pac_c": "SG PAC (centered)",
    "hfa_pac_c": "HFA PAC (centered)",
    "theta_c:StimCondstim": "Theta x StimCond",
    "StimCondstim:theta_c": "Theta x StimCond",
    "slow_gamma_c:StimCondstim": "Slow Gamma x StimCond",
    "StimCondstim:slow_gamma_c": "Slow Gamma x StimCond",
    "slow_gamma_pac_c:StimCondstim": "SG PAC x StimCond",
    "StimCondstim:slow_gamma_pac_c": "SG PAC x StimCond",
    "hfa_pac_c:StimCondstim": "HFA PAC x StimCond",
    "StimCondstim:hfa_pac_c": "HFA PAC x StimCond",
}

FAMILY_LABELS = {
    "bla_hpc_subregions": "Amygdala + Hippocampal Subfields (Amygdala, CA, DG, HPC)",
    "subregions_ec_prc": "Hippocampal Subfields + Rhinal Cortices (CA, DG, HPC, EC, PRC)",
}

BAND_LABELS = {
    "theta": "Theta (4-8 Hz)",
    "slow_gamma": "Slow Gamma (30-55 Hz)",
    "slow_gamma_pac": "SG PAC (30-50 Hz)",
    "hfa_pac": "HFA PAC (70-100 Hz)",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def p_str(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}".lstrip("0")


def sig_str(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "+"
    return ""


def fmt_num(value, decimals=4):
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1e5 or (abs(value) < 1e-4 and value != 0):
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


def fmt_short(value, decimals=1):
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def ci_str(lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{fmt_num(lo)}, {fmt_num(hi)}]"


def display_region(region: str) -> str:
    parts = str(region).split("_")
    return "-".join(REGION_LABELS.get(p, p) for p in parts)


def display_target(analysis_type, target):
    if analysis_type == "frequency_across_regions":
        return BAND_LABELS.get(target, target)
    return display_region(target)


def label_term(term):
    if term in TERM_LABELS:
        return TERM_LABELS[term]
    if term.startswith("Region") and ":" not in term:
        return f"Region [{display_region(term.replace('Region', ''))}]"
    if term.startswith("Region") and ":band_c" in term:
        return f"Band x Region [{display_region(term.replace('Region', '').replace(':band_c', ''))}]"
    if term.startswith("band_c:Region"):
        return f"Band x Region [{display_region(term.replace('band_c:Region', ''))}]"
    return term


# ---------------------------------------------------------------------------
# APA PDF class
# ---------------------------------------------------------------------------
class APAReport(FPDF):
    def __init__(self, report_title):
        super().__init__()
        self._report_title = report_title
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, self._report_title, align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def big_title(self, text):
        self.set_font("Helvetica", "B", 16)
        self.multi_cell(0, 9, text)
        self.ln(2)

    def subtitle(self, text):
        self.set_font("Helvetica", "", 11)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(6)

    def section_title(self, text):
        self.set_font("Helvetica", "B", 13)
        self.ln(4)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.ln(2)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Times", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_font("Times", "B", 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def italic_text(self, text):
        self.set_font("Times", "I", 9)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def apa_table(self, title, headers, rows, col_widths=None, note=None):
        self.ln(3)
        self.set_font("Times", "I", 10)
        self.multi_cell(0, 5, title)
        self.ln(1)
        if col_widths is None:
            avail = self.w - self.l_margin - self.r_margin
            col_widths = [avail / len(headers)] * len(headers)
        x_start = self.get_x()
        total_w = sum(col_widths)
        self.set_line_width(0.5)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)
        self.set_font("Times", "B", 9)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, str(h), align="C")
        self.ln()
        self.set_line_width(0.3)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)
        self.set_font("Times", "", 9)
        for row in rows:
            if self.get_y() > self.h - 35:
                self.set_line_width(0.5)
                self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
                self.add_page()
                self.set_font("Times", "I", 10)
                self.multi_cell(0, 5, title + " (continued)")
                self.ln(1)
                self.set_line_width(0.5)
                x_start = self.get_x()
                self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
                self.ln(1)
                self.set_font("Times", "B", 9)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 5, str(h), align="C")
                self.ln()
                self.set_line_width(0.3)
                self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
                self.ln(1)
                self.set_font("Times", "", 9)
            for i, val in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 5, str(val), align=align)
            self.ln()
        self.set_line_width(0.5)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)
        if note:
            self.set_font("Times", "I", 8)
            self.multi_cell(0, 4, f"Note. {note}")
            self.ln(2)


# ---------------------------------------------------------------------------
# Helpers to pull data from manifest
# ---------------------------------------------------------------------------
def get_model_data(model_row):
    """Load all CSVs for a model row from the manifest."""
    coef_path = Path(model_row["coefficient_path"])
    return {
        "metadata": pd.read_csv(coef_path.with_name("metadata.csv")).iloc[0].to_dict(),
        "coefs": pd.read_csv(model_row["coefficient_path"]),
        "comparison": pd.read_csv(model_row["model_comparison_path"]),
        "interactions": pd.read_csv(model_row["interaction_summary_path"]),
    }


def get_interaction_row(coef_df, pattern):
    """Get the first interaction term matching pattern."""
    mask = coef_df["term"].str.contains(pattern, na=False)
    if mask.any():
        return coef_df[mask].iloc[0]
    return None


def coef_table_rows(coef_df):
    """Build rows for a full fixed-effects table."""
    rows = []
    for _, r in coef_df.iterrows():
        p = r.get("p_value", np.nan)
        rows.append([
            label_term(r["term"]),
            fmt_num(r["estimate"]),
            ci_str(r.get("conf_low", np.nan), r.get("conf_high", np.nan)),
            fmt_num(r.get("statistic", np.nan)),
            p_str(p),
            sig_str(p),
        ])
    return rows


def model_building_rows(compare_df):
    """Build rows for the model-building step table."""
    if compare_df.empty:
        return []
    base_aic = compare_df["AIC"].iloc[0]
    base_bic = compare_df["BIC"].iloc[0]
    has_icc = "ICC" in compare_df.columns
    rows = []
    for i, (_, r) in enumerate(compare_df.iterrows()):
        p = r.get("p_value", np.nan)
        d_aic = fmt_short(r["AIC"] - base_aic) if i > 0 else "--"
        d_bic = fmt_short(r["BIC"] - base_bic) if i > 0 else "--"
        row = [
            r["Model"],
            str(int(r["npar"])),
            fmt_short(r["AIC"]),
            d_aic,
            fmt_short(r["BIC"]),
            d_bic,
            p_str(p),
            sig_str(p),
        ]
        if has_icc:
            icc = r.get("ICC", np.nan)
            row.append(f"{icc:.3f}" if not pd.isna(icc) else "")
        rows.append(row)
    return rows


def model_building_headers(compare_df):
    headers = ["Model", "npar", "AIC", "dAIC", "BIC", "dBIC", "p", ""]
    if "ICC" in compare_df.columns:
        headers.append("ICC")
    return headers


def model_building_widths(compare_df):
    widths = [16, 10, 22, 16, 22, 16, 16, 7]
    if "ICC" in compare_df.columns:
        widths.append(16)
    return widths


def saturation_text(compare_df):
    """One-paragraph saturation summary."""
    if compare_df.empty or len(compare_df) < 2:
        return "Insufficient models for assessment."
    parts = []

    # Null ICC
    if "ICC" in compare_df.columns and not pd.isna(compare_df["ICC"].iloc[0]):
        null_icc = compare_df["ICC"].iloc[0]
        parts.append(f"Null model ICC = {null_icc:.3f}.")

    # Best AIC/BIC
    best_aic = compare_df.loc[compare_df["AIC"].idxmin(), "Model"]
    best_bic = compare_df.loc[compare_df["BIC"].idxmin(), "Model"]
    parts.append(f"Best AIC: {best_aic}. Best BIC: {best_bic}.")

    # Full model
    full = compare_df[compare_df["Model"] == "m_full"]
    if not full.empty:
        fp = full.iloc[0].get("p_value", np.nan)
        if not pd.isna(fp):
            if fp < 0.05:
                parts.append(f"Full model significantly improved fit (p = {p_str(fp)}).")
            else:
                parts.append(f"Full model did not improve fit (p = {p_str(fp)}).")

    # R2
    if "R2_marginal" in compare_df.columns and not full.empty:
        r2m = full.iloc[0].get("R2_marginal", np.nan)
        r2c = full.iloc[0].get("R2_conditional", np.nan)
        if not pd.isna(r2m):
            parts.append(f"R2 marginal = {r2m:.3f}, conditional = {r2c:.3f}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main report writer
# ---------------------------------------------------------------------------
def write_report(measure_slug, outcome_type):
    manifest_path = STATS_ROOT / measure_slug / "manifest.csv"
    if not manifest_path.exists():
        return

    manifest = pd.read_csv(manifest_path)
    if "outcome_type" not in manifest.columns:
        manifest["outcome_type"] = (
            manifest["model_id"].astype(str).str.split("__").str[1])
    report_manifest = manifest[manifest["outcome_type"] == outcome_type].copy()
    if report_manifest.empty:
        return

    measure_label = MEASURE_LABELS[measure_slug]
    outcome_cfg = OUTCOME_LABELS[outcome_type]
    est_label = outcome_cfg["estimate_label"]
    stat_label = outcome_cfg["stat_label"]
    report_title = f"Encoding {measure_label} - {outcome_cfg['title']}"
    out_path = OUTPUT_ROOT / f"Encoding_{measure_label}_{outcome_cfg['title']}_Report.pdf"

    # Split manifest by analysis type
    per_region = report_manifest[
        report_manifest["analysis_type"] == "region_across_bands"
    ].sort_values(["family_label", "target"])
    per_freq = report_manifest[
        report_manifest["analysis_type"] == "frequency_across_regions"
    ].sort_values(["family_label", "target"])

    pdf = APAReport(report_title)
    pdf.add_page()

    # ====================================================================
    # 1. Title + Overview
    # ====================================================================
    pdf.big_title(
        f"Does Baseline {measure_label} x Stimulation\n"
        "Predict Encoding Memory?")
    pdf.subtitle(f"Encoding Phase - {outcome_cfg['title']} - Balanced Trials")

    pdf.section_title("1. Overview")
    pdf.body_text(
        f"This report tests whether baseline neural {measure_label.lower()} "
        "during encoding differentially predicts memory accuracy as a "
        "function of stimulation condition (BLA theta-modulated gamma "
        "stimulation vs. no stimulation).")
    pdf.body_text(
        "Balanced-trials filter: patients with fewer than 10 remembered or "
        "10 forgotten trials were excluded.")

    pdf.subsection_title("Statistical Approach")
    if outcome_type == "glmm":
        pdf.body_text(
            "All models are trial-level GLMMs (binomial, logit link, bobyqa "
            "optimizer, maxfun = 200,000, random intercept for patient). "
            "Results reported as odds ratios (OR) with 95% Wald CIs.")
    else:
        pdf.body_text(
            "All models are patient-level LMMs (ML estimation, bobyqa "
            "optimizer, random intercept for patient). Outcome: "
            "avg_stim_dprime_diff. Results reported as unstandardized "
            "estimates with 95% Wald CIs.")

    pdf.body_text(
        "Per-region models: fit within each region separately, including "
        "both frequency bands (e.g. theta + slow gamma) and their "
        "interactions with StimCond. Tests whether stim modulates the "
        "band-memory relationship within a specific brain region.")
    pdf.body_text(
        "Across-region models: fit within a single frequency band, pooling "
        "across regions with Region as a covariate. Tests whether stim "
        "modulates the band-memory relationship across regions.")

    sec = 2  # running section counter

    # ====================================================================
    # 2. Per-Region Models
    # ====================================================================
    if not per_region.empty:
        pdf.section_title(f"{sec}. Per-Region {measure_label} Models")
        pdf.body_text(
            "Each model includes one region at a time with both frequency "
            "band predictors and their interactions with StimCond. "
            "The critical tests are the Band x StimCond interaction terms.")

        # Group by family
        for family_slug, family_group in per_region.groupby("family_slug"):
            family_label = FAMILY_LABELS.get(
                family_slug, family_group.iloc[0]["family_label"])
            pdf.subsection_title(f"{sec}.{list(per_region['family_slug'].unique()).index(family_slug) + 1}. {family_label}")

            # --- Interaction summary table ---
            summary_rows = []
            model_data_cache = {}
            for _, mrow in family_group.iterrows():
                md = get_model_data(mrow)
                region_label = display_region(md["metadata"]["target"])
                model_data_cache[mrow["target"]] = md

                # Get all interaction terms from full model coefficients
                interactions = md["interactions"]
                if interactions.empty:
                    # Try from coefficients
                    stim_terms = md["coefs"][
                        md["coefs"]["term"].str.contains("StimCond", na=False) &
                        ~md["coefs"]["term"].isin(["StimCondstim", "(Intercept)"])
                    ]
                else:
                    stim_terms = interactions

                for _, irow in stim_terms.iterrows():
                    summary_rows.append([
                        region_label,
                        label_term(irow["term"]).replace(" x StimCond", ""),
                        fmt_num(irow["estimate"]),
                        ci_str(irow.get("conf_low", np.nan),
                               irow.get("conf_high", np.nan)),
                        fmt_num(irow.get("statistic", np.nan)),
                        p_str(irow["p_value"]),
                        sig_str(irow["p_value"]),
                    ])

            if summary_rows:
                pdf.apa_table(
                    f"Table. Per-Region {measure_label} x StimCond Interactions - "
                    f"{family_label}",
                    ["Region", "Band", est_label, "95% CI", stat_label, "p", ""],
                    summary_rows,
                    col_widths=[24, 30, 18, 38, 14, 14, 7],
                    note=f"{est_label} for the Band x StimCond [stim] interaction. "
                         f"* p < .05, ** p < .01, + p < .10.",
                )

            # --- Full coefficient table per region ---
            for _, mrow in family_group.iterrows():
                md = model_data_cache.get(mrow["target"])
                if md is None:
                    md = get_model_data(mrow)
                region_label = display_region(md["metadata"]["target"])
                n_pat = int(md["metadata"]["n_patients"])
                n_obs = int(md["metadata"]["n_rows"])

                pdf.apa_table(
                    f"Table. Full Model: {measure_label} in {region_label} "
                    f"(N = {n_pat}, obs = {n_obs})",
                    ["Predictor", est_label, "95% CI", stat_label, "p", ""],
                    coef_table_rows(md["coefs"]),
                    col_widths=[44, 18, 38, 14, 18, 8],
                )

                # Model building + saturation as compact note
                sat = saturation_text(md["comparison"])
                if sat:
                    pdf.italic_text(f"Model building: {sat}")

        sec += 1

    # ====================================================================
    # 3. Across-Region Models
    # ====================================================================
    if not per_freq.empty:
        pdf.section_title(f"{sec}. Across-Region {measure_label} Models")
        pdf.body_text(
            "Each model fits one frequency band at a time, pooling across "
            "regions within the region set. Region is included as a covariate "
            "with Band x Region and Band x StimCond interaction terms.")

        for family_slug, family_group in per_freq.groupby("family_slug"):
            family_label = FAMILY_LABELS.get(
                family_slug, family_group.iloc[0]["family_label"])
            pdf.subsection_title(f"{sec}.{list(per_freq['family_slug'].unique()).index(family_slug) + 1}. {family_label}")

            # --- Summary interaction table ---
            summary_rows = []
            model_data_cache = {}
            for _, mrow in family_group.iterrows():
                md = get_model_data(mrow)
                band_label = BAND_LABELS.get(
                    md["metadata"]["target"], md["metadata"]["target"])
                model_data_cache[mrow["target"]] = md

                irow = get_interaction_row(
                    md["coefs"], "StimCond.*band_c|band_c.*StimCond")
                if irow is not None:
                    summary_rows.append([
                        band_label,
                        fmt_num(irow["estimate"]),
                        ci_str(irow.get("conf_low", np.nan),
                               irow.get("conf_high", np.nan)),
                        fmt_num(irow.get("statistic", np.nan)),
                        p_str(irow["p_value"]),
                        sig_str(irow["p_value"]),
                    ])

            if summary_rows:
                pdf.apa_table(
                    f"Table. Across-Region {measure_label} x StimCond - "
                    f"{family_label}",
                    ["Band", est_label, "95% CI", stat_label, "p", ""],
                    summary_rows,
                    col_widths=[40, 18, 42, 14, 18, 8],
                    note="Pooled across regions within each frequency band. "
                         "* p < .05, ** p < .01, + p < .10.",
                )

            # --- Full coefficient table per frequency ---
            for _, mrow in family_group.iterrows():
                md = model_data_cache.get(mrow["target"])
                if md is None:
                    md = get_model_data(mrow)
                band_label = BAND_LABELS.get(
                    md["metadata"]["target"], md["metadata"]["target"])
                n_pat = int(md["metadata"]["n_patients"])
                n_obs = int(md["metadata"]["n_rows"])

                pdf.apa_table(
                    f"Table. Full Model: {measure_label} {band_label} - "
                    f"{family_label} (N = {n_pat}, obs = {n_obs})",
                    ["Predictor", est_label, "95% CI", stat_label, "p", ""],
                    coef_table_rows(md["coefs"]),
                    col_widths=[44, 18, 38, 14, 18, 8],
                )

                sat = saturation_text(md["comparison"])
                if sat:
                    pdf.italic_text(f"Model building: {sat}")

        sec += 1

    # ====================================================================
    # 4. Model Building Detail
    # ====================================================================
    pdf.section_title(f"{sec}. Model Building Detail")
    pdf.body_text(
        "Step-by-step model building for each analysis. Models are nested: "
        "m0 (intercept only) -> m1 (+ StimCond) -> ... -> m_full "
        "(+ StimCond interactions). Chi-squared tests compare each step. "
        "ICC = intraclass correlation (patient-level variance / total). "
        "dAIC/dBIC = change from baseline m0 (lower = better).")

    for _, mrow in report_manifest.sort_values(
            ["analysis_type", "family_label", "target"]).iterrows():
        md = get_model_data(mrow)
        analysis_label = ANALYSIS_LABELS[md["metadata"]["analysis_type"]]
        target_label = display_target(
            md["metadata"]["analysis_type"], md["metadata"]["target"])
        title = f"{analysis_label}: {target_label} ({md['metadata']['family_label']})"

        rows = model_building_rows(md["comparison"])
        if rows:
            pdf.apa_table(
                f"Table. {title}",
                model_building_headers(md["comparison"]),
                rows,
                col_widths=model_building_widths(md["comparison"]),
                note="* p < .05, + p < .10. "
                     "Formulas shown for reference. "
                     + " | ".join(
                    f"{r['Model']}: {r['formula_text']}"
                    for _, r in md["comparison"].iterrows()
                    if pd.notna(r.get("formula_text", np.nan))
                ),
            )

    sec += 1

    # ====================================================================
    # 5. Summary of All Interactions
    # ====================================================================
    pdf.section_title(f"{sec}. Summary of All Band x StimCond Interactions")

    all_rows = []
    for _, mrow in report_manifest.sort_values(
            ["analysis_type", "family_label", "target"]).iterrows():
        md = get_model_data(mrow)
        analysis_label = ANALYSIS_LABELS[md["metadata"]["analysis_type"]]
        target_label = display_target(
            md["metadata"]["analysis_type"], md["metadata"]["target"])

        interactions = md["interactions"]
        if interactions.empty:
            continue
        for _, irow in interactions.sort_values("p_value", na_position="last").iterrows():
            all_rows.append([
                analysis_label,
                target_label,
                label_term(irow["term"]).replace(" x StimCond", ""),
                fmt_num(irow["estimate"]),
                p_str(irow["p_value"]),
                sig_str(irow["p_value"]),
            ])

    if all_rows:
        pdf.apa_table(
            f"Table. All {measure_label} x StimCond Interactions",
            ["Type", "Target", "Band", est_label, "p", ""],
            all_rows,
            col_widths=[24, 30, 30, 20, 18, 8],
            note="Per-region and across-region models combined. "
                 "* p < .05, ** p < .01, *** p < .001, + p < .10.",
        )
    else:
        pdf.body_text("No StimCond interaction terms were extracted.")

    # ====================================================================
    # 6. Interpretation
    # ====================================================================
    pdf.section_title(f"{sec}. Interpretation")

    # Collect significant and marginal findings for narrative
    sig_findings = []
    marg_findings = []
    all_ns = []
    for _, mrow in report_manifest.iterrows():
        md = get_model_data(mrow)
        all_ns.append(int(md["metadata"]["n_patients"]))
        interactions = md["interactions"]
        if interactions.empty:
            continue
        analysis_label = ANALYSIS_LABELS[md["metadata"]["analysis_type"]]
        target_label = display_target(
            md["metadata"]["analysis_type"], md["metadata"]["target"])
        family_label = FAMILY_LABELS.get(
            mrow.get("family_slug", ""), md["metadata"]["family_label"])
        for _, irow in interactions.iterrows():
            p = irow["p_value"]
            est = irow["estimate"]
            term_label = label_term(irow["term"]).replace(" x StimCond", "")
            if outcome_type == "glmm":
                direction = "higher" if est > 1 else "lower"
                effect_desc = (
                    f"OR = {fmt_num(est)}, meaning that the positive "
                    f"relationship between {term_label.lower()} and memory "
                    f"was {'stronger' if est > 1 else 'weaker'} on "
                    "stimulation trials compared to no-stimulation trials")
            else:
                direction = "positive" if est > 0 else "negative"
                effect_desc = f"b = {fmt_num(est)}, direction = {direction}"
            entry = {
                "region": target_label,
                "family": family_label,
                "analysis": analysis_label,
                "band": term_label,
                "est": est,
                "p": p,
                "effect_desc": effect_desc,
            }
            if not pd.isna(p) and p < 0.05:
                sig_findings.append(entry)
            elif not pd.isna(p) and p < 0.10:
                marg_findings.append(entry)

    sig_findings.sort(key=lambda x: x["p"])
    marg_findings.sort(key=lambda x: x["p"])
    n_models = len(report_manifest)
    n_range = f"{min(all_ns)}-{max(all_ns)}" if all_ns else "?"

    # Opening summary
    pdf.body_text(
        f"This report examined whether baseline {measure_label.lower()} "
        "during the encoding phase interacted with BLA stimulation "
        "condition to predict trial-level memory accuracy. A total of "
        f"{n_models} GLMMs were fit across per-region and across-region "
        f"analyses (N = {n_range} patients per model, after balanced-trials "
        "exclusion).")

    # Significant findings
    if sig_findings:
        pdf.bold_text("Significant Interactions (p < .05)")
        for f in sig_findings:
            pdf.body_text(
                f"{f['analysis']} model in {f['region']} "
                f"({f['family']}): The {f['band']} x StimCond interaction "
                f"was significant (p = {p_str(f['p'])}; {f['effect_desc']}). "
                f"In plain language, {f['band'].lower()} "
                f"{'had a different relationship with memory accuracy on '}"
                "stimulation trials than on no-stimulation trials in this "
                "region/set.")
    else:
        pdf.bold_text("Significant Interactions (p < .05)")
        pdf.body_text(
            f"No {measure_label.lower()} x stimulation interactions reached "
            "statistical significance. This suggests that baseline "
            f"{measure_label.lower()} during encoding did not differentially "
            "predict memory accuracy as a function of whether BLA stimulation "
            "was delivered. The relationship between neural activity and "
            "memory was similar regardless of stimulation condition.")

    # Marginal findings
    if marg_findings:
        pdf.bold_text("Marginal Trends (.05 <= p < .10)")
        for f in marg_findings:
            pdf.body_text(
                f"{f['analysis']} model in {f['region']}: "
                f"The {f['band']} x StimCond interaction showed a marginal "
                f"trend (p = {p_str(f['p'])}; {f['effect_desc']}). "
                "This did not reach conventional significance but may "
                "warrant further investigation.")

    # Overall interpretation
    pdf.bold_text("Overall Interpretation")
    if sig_findings:
        n_sig = len(sig_findings)
        regions_with_sig = sorted(set(f["region"] for f in sig_findings))
        bands_with_sig = sorted(set(f["band"] for f in sig_findings))
        pdf.body_text(
            f"Out of {n_models} models tested, {n_sig} "
            f"{'interaction' if n_sig == 1 else 'interactions'} reached "
            f"significance (p < .05), involving "
            f"{', '.join(regions_with_sig)} and "
            f"{', '.join(bands_with_sig)} "
            f"{measure_label.lower()}. This indicates that stimulation "
            "modulated the relationship between baseline neural activity "
            "and memory in specific region-frequency combinations. "
            "However, given the number of models tested, these results "
            "should be interpreted cautiously and in the context of "
            "effect sizes and theoretical expectations.")
        if outcome_type == "glmm":
            pdf.body_text(
                "For significant interactions with OR > 1: higher baseline "
                f"{measure_label.lower()} was more beneficial for memory on "
                "stimulation trials than no-stimulation trials. For OR < 1: "
                f"higher baseline {measure_label.lower()} was less beneficial "
                "for memory on stimulation trials, consistent with a "
                "potential ceiling or saturation effect.")
    else:
        pdf.body_text(
            f"Across all {n_models} models, no {measure_label.lower()} x "
            "stimulation interactions reached significance. Baseline "
            f"{measure_label.lower()} during encoding does not appear to "
            "modulate the effect of BLA stimulation on memory accuracy. "
            "This null finding suggests that the impact of stimulation on "
            "encoding-phase memory is not contingent on the pre-existing "
            f"level of {measure_label.lower()} in the tested regions and "
            "frequency bands.")

    pdf.output(str(out_path))
    print(f"  Wrote {out_path.name}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for measure_slug in ("power", "coherence", "pac"):
        write_report(measure_slug, "glmm")
    print("Done.")


if __name__ == "__main__":
    main()
