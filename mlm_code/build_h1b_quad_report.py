#!/usr/bin/env python
"""
H1b QUAD RESPONDER CATEGORY report (Anti / Non / Moderate / Strong).

Reads coefs/anovas from outputs/h1b_quad_<lmm|glmm>_<filter>_retrieval_mlm/
and writes outputs/Hypothesis_1b_QuadResponderCategory_<KIND>_Retrieval_*.pdf.

Usage:
  python build_h1b_quad_report.py <lmm|glmm> <balanced|imbalanced>
"""

import os
import sys

import numpy as np
import pandas as pd
from fpdf import FPDF


if len(sys.argv) < 3:
    print("Usage: python build_h1b_quad_report.py <lmm|glmm> <balanced|imbalanced>")
    sys.exit(1)

KIND = sys.argv[1]
FILTER = sys.argv[2]
assert KIND in ("lmm", "glmm")
assert FILTER in ("balanced", "imbalanced")

KIND_LABEL = "LMM" if KIND == "lmm" else "GLMM (3-way)"
EFFECT_LABEL = "Estimate" if KIND == "lmm" else "OR"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "outputs",
                        f"h1b_quad_{KIND}_{FILTER}_retrieval_mlm")
FILTER_LABEL = "Balanced Trials" if FILTER == "balanced" else "ImBalanced Trials"

REPORT_TITLE = (
    f"Hypothesis 1b - Quad Responder Category ({KIND_LABEL}) - "
    f"Retrieval - {FILTER_LABEL}"
)
OUTPUT_PDF = os.path.join(
    SCRIPT_DIR, "outputs",
    f"Hypothesis_1b_QuadResponderCategory_{KIND.upper()}_Retrieval_"
    f"{FILTER_LABEL.replace(' ', '_')}.pdf",
)

POWER_REGIONS = ["BLA", "HPC", "EC", "PRC"]
COH_PAIRS = ["BLA_HPC", "BLA_EC", "BLA_PRC", "EC_HPC", "EC_PRC", "HPC_PRC"]
PAC_PAIRS = ["BLA_HPC", "BLA_EC", "BLA_PRC"]
BAND_LABELS = {"theta": "Theta (4-8 Hz)", "slow_gamma": "Slow Gamma (30-55 Hz)"}
REGION_LABELS = {"BLA": "BLA", "HPC": "HPC", "CA": "CA", "DG": "DG",
                 "EC": "EC", "PRC": "PRC", "PHG": "PHG"}


def display_region(r):
    return "-".join(REGION_LABELS.get(p, p) for p in r.split("_"))


class APAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, REPORT_TITLE, align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.ln(2)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def subsubsection_title(self, title):
        self.set_font("Helvetica", "B", 10)
        self.ln(1)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")

    def body_text(self, text):
        self.set_font("Times", "", 11)
        self.multi_cell(0, 5.5, text)
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


def p_str(p):
    if isinstance(p, str):
        return p
    if pd.isna(p):
        return ""
    if p < .001:
        return "< .001"
    return f"{p:.3f}".lstrip("0")


def sig_str(p):
    if isinstance(p, str) or pd.isna(p):
        return ""
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    if p < .10:  return "+"
    return ""


def num_str(v, decimals=4):
    if pd.isna(v):
        return ""
    if abs(v) > 1e4 or (abs(v) < 1e-4 and v != 0):
        return f"{v:.2e}"
    return f"{v:.{decimals}f}"


def ci_str(lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{num_str(lo)}, {num_str(hi)}]"


def load_csv(filename):
    p = os.path.join(DATA_DIR, filename)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def coef_table_rows(df):
    rows = []
    for _, r in df.iterrows():
        stat_label = r.get("statistic", float("nan"))
        rows.append([
            str(r["term"]),
            num_str(r["estimate"]),
            ci_str(r["conf.low"], r["conf.high"]),
            f"{stat_label:.3f}" if pd.notna(stat_label) else "",
            p_str(r["p.value"]) if "p.value" in df.columns else "",
            sig_str(r["p.value"]) if "p.value" in df.columns else "",
        ])
    return rows


def fmt_fit(value, decimals=2):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}"


def anova_table_rows(df):
    rows = []
    for _, r in df.iterrows():
        rows.append([
            str(r["Model"]),
            str(r["Formula"]),
            f"{int(r['npar'])}" if not pd.isna(r["npar"]) else "-",
            fmt_fit(r["AIC"], 1),
            fmt_fit(r["BIC"], 1),
            fmt_fit(r["logLik"], 1),
            fmt_fit(r["Chisq"], 2),
            fmt_fit(r["Df"], 0),
            p_str(r["p.value"]) if not pd.isna(r["p.value"]) else "-",
            sig_str(r["p.value"]) if not pd.isna(r["p.value"]) else "",
        ])
    return rows


def render_panel(pdf, title_prefix, modality, unit, band_label):
    has_any = False
    flavors = [
        ("Continuous",       "memory modulation (z-scored stim-no-stim d')"),
        ("QuadCategorical",  "quad responder (Anti / Non=ref / Moderate / Strong)"),
    ]
    for mod_kind, mod_label in flavors:
        coef_fn = f"{modality}_{unit}_{band_label}_mod{mod_kind}_coefs.csv"
        anova_fn = f"{modality}_{unit}_{band_label}_mod{mod_kind}_anova.csv"
        df_coef = load_csv(coef_fn)
        df_anova = load_csv(anova_fn)
        if df_coef is None and df_anova is None:
            continue
        has_any = True
        pdf.subsubsection_title(f"{title_prefix} - moderator: {mod_label}")
        if df_coef is not None and not df_coef.empty:
            stat_col = "t" if KIND == "lmm" else "z"
            pdf.apa_table(
                f"Table. Full Model - {title_prefix} ({mod_kind} moderator)",
                ["Predictor", EFFECT_LABEL, "95% CI", stat_col, "p", ""],
                coef_table_rows(df_coef),
                col_widths=[78, 18, 38, 14, 14, 8],
            )
        if df_anova is not None and not df_anova.empty:
            pdf.apa_table(
                f"Table. Sequential Model Build - {title_prefix} ({mod_kind} moderator)",
                ["Model", "Formula", "npar", "AIC", "BIC", "logLik",
                 "Chisq", "Df", "p", ""],
                anova_table_rows(df_anova),
                col_widths=[14, 92, 10, 14, 14, 14, 14, 8, 14, 6],
                note=("Sequential model build (LR test vs previous row). "
                      "* p < .05, ** p < .01, *** p < .001, + p < .10."),
            )
    return has_any


def build_report():
    pdf = APAReport()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9,
                   "Hypothesis 1b - Quad Responder Category\n"
                   f"({KIND_LABEL})")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Retrieval Phase - {FILTER_LABEL}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    sec = 1
    pdf.section_title(f"{sec}. Overview")
    pdf.body_text(
        "H1b: A decrease or no change in neural activity and connectivity "
        "between the BLA, hippocampus, perirhinal, and entorhinal cortices "
        "will occur for participants whose memory was impaired by prior "
        "stimulation, and these neural features will be distinct from those "
        "of participants whose memory was enhanced."
    )
    pdf.body_text(
        "This report uses the four-level (quad) responder grouping. Two "
        "moderators are evaluated in parallel for every panel: "
        "(1) continuous memory modulation = z-scored avg_stim_dprime_diff; "
        "(2) categorical 4-level responder status with Non-responders as "
        "reference. Subject IDs BJH032 and BJH033 are folded to a single "
        "subject (54 unique participants)."
    )
    pdf.body_text(
        "Group definitions and ranges of avg_stim_dprime_diff (stim - "
        "no-stim d'):"
    )
    pdf.apa_table(
        "Table. Quad responder definitions",
        ["Group (factor level)", "Source label", "n", "min", "median", "max"],
        [
            ["AntiResp",  "Anti-responders",     "14", "-0.779", "-0.306", "-0.139"],
            ["NonResp (ref)", "Non-responders",  "13", "-0.130", "0.000",  "+0.076"],
            ["Moderate", "Moderate responders",  "13", "+0.077", "+0.160", "+0.210"],
            ["Strong",   "Strong responders",    "14", "+0.250", "+0.363", "+0.840"],
        ],
        col_widths=[42, 56, 14, 22, 22, 20],
        note=("Reference category: NonResp. Coefficients labelled "
              "QuadResponderGroupAntiResp/Moderate/Strong are contrasts "
              "vs NonResp. The H1b prediction maps onto: AntiResp x Stim "
              "<= 0 and distinct from Strong (and Moderate) x Stim > 0."),
    )

    pdf.subsection_title("Statistical Approach")
    if KIND == "lmm":
        pdf.body_text(
            "Per region (power) or pair (coherence/PAC), per band, the "
            "neural feature is regressed on stim condition x moderator "
            "with a patient random intercept. Continuous moderator: "
            "feature ~ StimCond * mem_mod_z + (1|Patient). Categorical "
            "(quad) moderator: feature ~ StimCond * QuadResponderGroup + "
            "(1|Patient). Three StimCond:Group contrasts (AntiResp / "
            "Moderate / Strong vs NonResp) test whether stim moves the "
            "feature differently in each group relative to non-responders."
        )
        pdf.body_text(
            "Fixed effects fit with REML; sequential model comparisons "
            "refit with ML for likelihood-ratio tests."
        )
    else:
        pdf.body_text(
            "Per region (power) or pair (coherence/PAC), per band, "
            "trial-level binomial GLMMs test 3-way interactions: "
            "Accuracy ~ StimCond * feature_c * mem_mod_z + (1|Patient) "
            "and Accuracy ~ StimCond * feature_c * QuadResponderGroup + "
            "(1|Patient). Sequential model build adds main effects, all "
            "2-way interactions, then the 3-way term."
        )
    pdf.body_text(
        "Bands tested: theta (4-8 Hz) and slow gamma (30-55 Hz). "
        "PAC: SG PAC (30-50 Hz) only. Per-panel patients >= 5 required."
    )
    pdf.body_text(
        "Power regions: BLA, HPC, EC, PRC. Coherence pairs: BLA-HPC, "
        "BLA-EC, BLA-PRC, EC-HPC, EC-PRC, HPC-PRC. PAC pairs: BLA-HPC, "
        "BLA-EC, BLA-PRC."
    )

    sec += 1
    pdf.section_title(f"{sec}. Power")
    sub_idx = 1
    for reg in POWER_REGIONS:
        for bk, bl in BAND_LABELS.items():
            anchor = load_csv(f"power_{reg}_{bk}_modContinuous_coefs.csv")
            if anchor is None:
                continue
            title_prefix = f"Power - {display_region(reg)} - {bl}"
            pdf.subsection_title(f"{sec}.{sub_idx}. {title_prefix}")
            sub_idx += 1
            render_panel(pdf, title_prefix, "power", reg, bk)

    sec += 1
    pdf.section_title(f"{sec}. Coherence")
    sub_idx = 1
    for pair in COH_PAIRS:
        for bk, bl in BAND_LABELS.items():
            anchor = load_csv(f"coherence_{pair}_{bk}_modContinuous_coefs.csv")
            if anchor is None:
                continue
            title_prefix = f"Coherence - {display_region(pair)} - {bl}"
            pdf.subsection_title(f"{sec}.{sub_idx}. {title_prefix}")
            sub_idx += 1
            render_panel(pdf, title_prefix, "coherence", pair, bk)

    sec += 1
    pdf.section_title(f"{sec}. PAC (SG PAC, 30-50 Hz)")
    sub_idx = 1
    for pair in PAC_PAIRS:
        anchor = load_csv(f"pac_{pair}_slow_gamma_modContinuous_coefs.csv")
        if anchor is None:
            continue
        title_prefix = f"PAC - {display_region(pair)} - SG PAC"
        pdf.subsection_title(f"{sec}.{sub_idx}. {title_prefix}")
        sub_idx += 1
        render_panel(pdf, title_prefix, "pac", pair, "slow_gamma")

    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    pdf.output(OUTPUT_PDF)
    print(f"Report saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_report()
