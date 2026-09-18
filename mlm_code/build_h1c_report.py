#!/usr/bin/env python
"""
Build the Hypothesis 1c PDF report:
  Coherence and PAC ONLY, restricted to non-BLA region pairs
  (CA, DG, HPC, EC, PRC; 10 pairs).

Inputs are coefficient and anova CSVs produced by run_h1c_retrieval.R, located
in outputs/h1c_<filter>_retrieval_mlm/.

Usage:
  python build_h1c_report.py imbalanced
  python build_h1c_report.py balanced
"""

import os
import sys

import numpy as np
import pandas as pd
from fpdf import FPDF


if len(sys.argv) < 2:
    print("Usage: python build_h1c_report.py <balanced|imbalanced>")
    sys.exit(1)

FILTER = sys.argv[1]
assert FILTER in ("balanced", "imbalanced"), f"Invalid filter: {FILTER}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "outputs", f"h1c_{FILTER}_retrieval_mlm")

FILTER_LABEL = "Balanced Trials" if FILTER == "balanced" else "ImBalanced Trials"

REPORT_TITLE = (
    f"Hypothesis 1c - Coherence + PAC (non-BLA pairs) - "
    f"Retrieval - {FILTER_LABEL}"
)
OUTPUT_PDF = os.path.join(
    SCRIPT_DIR, "outputs",
    f"Hypothesis_1c_Report_Retrieval_{FILTER_LABEL.replace(' ', '_')}.pdf",
)


# ── Pair list ───────────────────────────────────────────────────────────────
NONBLA_PAIRS = [
    "CA_DG", "CA_EC", "CA_HPC", "CA_PRC",
    "DG_EC", "DG_HPC", "DG_PRC",
    "EC_HPC", "EC_PRC",
    "HPC_PRC",
]

BAND_LABELS = {
    "theta": "Theta (4-8 Hz)",
    "slow_gamma": "Slow Gamma (30-55 Hz)",
}
PAC_TYPES = {
    "slow_gamma": "SG PAC (30-50 Hz)",
    "hfa": "HFA PAC (70-100 Hz)",
}

# ALLHPC = macro hippocampus -> "HPC"; HPC region = subiculum -> "SUB"
REGION_LABELS = {
    "BLA": "BLA", "ALLHPC": "HPC", "HPC": "SUB", "CA": "CA", "DG": "DG",
    "EC": "EC", "PRC": "PRC", "PHG": "PHG",
}


def display_region(r):
    parts = r.split("_")
    return "-".join(REGION_LABELS.get(p, p) for p in parts)


NONBLA_PAIRS_LIST = ", ".join(display_region(p) for p in NONBLA_PAIRS)


# ── PDF class ───────────────────────────────────────────────────────────────

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


# ── Helpers ─────────────────────────────────────────────────────────────────

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
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    if p < .10:
        return "+"
    return ""


def or_str(v):
    if pd.isna(v):
        return ""
    if abs(v) > 1e4 or abs(v) < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"


def ci_str(lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return ""
    if abs(lo) > 1e4 or abs(hi) > 1e4 or abs(lo) < 1e-4:
        return f"[{lo:.2e}, {hi:.2e}]"
    return f"[{lo:.4f}, {hi:.4f}]"


def load_coefs(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def format_term(term):
    labels = {
        "(Intercept)": "Intercept",
        "StimCondstim": "StimCond [stim]",
        "band_c": "Band",
        "band_c:StimCondstim": "Band x StimCond",
        "StimCondstim:band_c": "Band x StimCond",
        "pac_z": "PAC (z)",
        "StimCondstim:pac_z": "PAC (z) x StimCond",
        "pac_z:StimCondstim": "PAC (z) x StimCond",
        "theta_c": "theta_c",
        "slow_gamma_c": "slow_gamma_c",
        "theta_c:StimCondstim": "StimCondstim:theta_c",
        "StimCondstim:theta_c": "StimCondstim:theta_c",
        "slow_gamma_c:StimCondstim": "StimCondstim:slow_gamma_c",
        "StimCondstim:slow_gamma_c": "StimCondstim:slow_gamma_c",
        "sg_pac_z": "sg_pac_z",
        "hfa_pac_z": "hfa_pac_z",
        "sg_pac_z:StimCondstim": "StimCondstim:sg_pac_z",
        "StimCondstim:sg_pac_z": "StimCondstim:sg_pac_z",
        "hfa_pac_z:StimCondstim": "StimCondstim:hfa_pac_z",
        "StimCondstim:hfa_pac_z": "StimCondstim:hfa_pac_z",
    }
    if term in labels:
        return labels[term]
    if term.startswith("Region") and ":" not in term:
        return f"Region [{display_region(term.replace('Region', ''))}]"
    if term.startswith("StimCondstim:Region"):
        reg = term.replace("StimCondstim:Region", "")
        return f"StimCondstim:Region{reg}"
    return term


def coef_table_rows(df):
    rows = []
    for _, r in df.iterrows():
        rows.append([
            format_term(r["term"]),
            or_str(r["estimate"]),
            ci_str(r["conf.low"], r["conf.high"]),
            f"{r['statistic']:.3f}",
            p_str(r["p.value"]),
            sig_str(r["p.value"]),
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


def render_anova(pdf, title, anova_filename):
    df = load_coefs(anova_filename)
    if df is None or df.empty:
        return
    pdf.apa_table(
        title,
        ["Model", "Formula", "npar", "AIC", "BIC", "logLik",
         "Chisq", "Df", "p", ""],
        anova_table_rows(df),
        col_widths=[14, 92, 10, 14, 14, 14, 14, 8, 14, 6],
        note=("Sequential glmer model build. Chisq, Df, p compare each model "
              "against the previous row (likelihood-ratio test). "
              "* p < .05, ** p < .01, *** p < .001, + p < .10."),
    )


# ── Build report ────────────────────────────────────────────────────────────

def build_report():
    pdf = APAReport()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9,
                   "Hypothesis 1c\n"
                   "Coherence and PAC between non-BLA MTL/HPC regions")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Retrieval Phase - {FILTER_LABEL}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    sec = 1
    pdf.section_title(f"{sec}. Overview")
    pdf.body_text(
        "This report tests whether retrieval-phase coherence and PAC between "
        "non-BLA medial temporal lobe / hippocampal regions (CA, DG, HPC, EC, "
        "PRC) differentially predict subsequent memory as a function of "
        "stimulation condition."
    )
    if FILTER == "balanced":
        pdf.body_text(
            "Balanced-trials filter applied: subjects with fewer than 10 "
            "trials in either remembered or forgotten conditions were "
            "excluded."
        )
    else:
        pdf.body_text(
            "No trial-balance filter applied: ALL subjects are included "
            "regardless of the distribution of remembered vs. forgotten "
            "trials. This maximizes power but may include subjects whose "
            "extreme memory rates could bias estimates."
        )
    pdf.body_text(
        f"Pairs analyzed (10 total): {NONBLA_PAIRS_LIST}. "
        f"Per-pair tests require >= 5 patients."
    )

    pdf.subsection_title("Statistical Approach")
    pdf.body_text(
        "All models are trial-level GLMMs (binomial, logit link, bobyqa "
        "optimizer, maxfun = 200,000, random intercept for patient). "
        "Results reported as odds ratios (OR) with 95% Wald CIs."
    )
    pdf.body_text(
        "Pair x StimCond models (one per band / per PAC type): "
        "Accuracy ~ band_c (or pac_z) + StimCond + Region + Region:StimCond + "
        "(1|Patient). Tests overall Region, StimCond, and Region x StimCond "
        "effects across all eligible non-BLA pairs."
    )
    pdf.body_text(
        "Per-pair two-band coherence model: Accuracy ~ theta_c + "
        "slow_gamma_c + StimCond + theta_c:StimCond + slow_gamma_c:StimCond "
        "+ (1|Patient). Per-pair two-PAC-type model: Accuracy ~ sg_pac_z + "
        "hfa_pac_z + StimCond + sg_pac_z:StimCond + hfa_pac_z:StimCond + "
        "(1|Patient). Fit separately for each pair."
    )
    pdf.body_text(
        "Frequency bands (coherence): theta (4-8 Hz), slow gamma (30-55 Hz). "
        "PAC types: SG PAC (30-50 Hz), HFA PAC (70-100 Hz)."
    )

    # ───────────────────────────────────────────────────────────────────────
    # COHERENCE
    # ───────────────────────────────────────────────────────────────────────
    sec += 1
    pdf.section_title(f"{sec}. Coherence x Stimulation Condition")

    # ── 2.1 Across-pair Pair x StimCond per band ──
    pdf.subsection_title(f"{sec}.1. Coherence: Pair x StimCond - All Pairs (per band)")
    pdf.body_text(
        f"Model: Accuracy ~ band_c + StimCond + Region + Region:StimCond + "
        f"(1|Patient). Restricted to non-BLA coherence pairs "
        f"({NONBLA_PAIRS_LIST}). Region [CA-DG] (the first alphabetical "
        f"non-BLA pair) is the reference category."
    )
    for bk, bl in BAND_LABELS.items():
        df = load_coefs(f"coherence_allpairs_pairbystim_{bk}_coefs.csv")
        if df is None:
            continue
        pdf.apa_table(
            f"Table. Coherence - Non-BLA Pairs x StimCond - {bl}",
            ["Predictor", "OR", "95% CI", "z", "p", ""],
            coef_table_rows(df),
            col_widths=[60, 18, 38, 14, 14, 8],
            note=("Region [CA-DG] is the reference pair (absorbed into the "
                  "Intercept). Other Region rows are baseline differences vs "
                  "CA-DG; Region:StimCond[stim] rows are stim-effect "
                  "differences vs CA-DG."),
        )
        render_anova(
            pdf,
            f"Table. Sequential Model Build - Coherence Non-BLA Pairs x StimCond - {bl}",
            f"coherence_allpairs_pairbystim_{bk}_anova.csv",
        )

    # ── 2.2 Per-pair two-band coherence ──
    pdf.subsection_title(f"{sec}.2. Coherence: Per-Pair Two-Band Models")
    pdf.body_text(
        "Model (per pair): Accuracy ~ theta_c + slow_gamma_c + StimCond + "
        "theta_c:StimCond + slow_gamma_c:StimCond + (1|Patient). "
        "One full coefficient table per pair."
    )
    for pair in NONBLA_PAIRS:
        df = load_coefs(f"coherence_twoband_{pair}_coefs.csv")
        if df is None:
            continue
        pdf.apa_table(
            f"Table. Coherence Two-Band Model - {display_region(pair)}",
            ["Predictor", "OR", "95% CI", "z", "p", ""],
            coef_table_rows(df),
            col_widths=[58, 18, 38, 14, 14, 8],
        )
        render_anova(
            pdf,
            f"Table. Sequential Model Build - Coherence Two-Band - {display_region(pair)}",
            f"coherence_twoband_{pair}_anova.csv",
        )

    # ───────────────────────────────────────────────────────────────────────
    # PAC
    # ───────────────────────────────────────────────────────────────────────
    sec += 1
    pdf.section_title(f"{sec}. PAC x Stimulation Condition")
    pdf.body_text(
        "PAC measures theta-phase x amplitude coupling. PAC values are "
        "z-scored within the modeled pair set."
    )

    # ── 3.1 Across-pair Pair x StimCond per PAC type ──
    pdf.subsection_title(f"{sec}.1. PAC: Pair x StimCond - All Pairs (per PAC type)")
    pdf.body_text(
        f"Model: Accuracy ~ pac_z + StimCond + Region + Region:StimCond + "
        f"(1|Patient). Restricted to non-BLA PAC pairs ({NONBLA_PAIRS_LIST})."
    )
    for pt, pl in PAC_TYPES.items():
        df = load_coefs(f"pac_allpairs_pairbystim_{pt}_coefs.csv")
        if df is None:
            continue
        pdf.apa_table(
            f"Table. PAC ({pl}) - Non-BLA Pairs x StimCond",
            ["Predictor", "OR", "95% CI", "z", "p", ""],
            coef_table_rows(df),
            col_widths=[60, 18, 38, 14, 14, 8],
            note=("Region [CA-DG] is the reference pair (absorbed into the "
                  "Intercept). Other Region rows are baseline differences vs "
                  "CA-DG; Region:StimCond[stim] rows are stim-effect "
                  "differences vs CA-DG."),
        )
        render_anova(
            pdf,
            f"Table. Sequential Model Build - PAC ({pl}) - Non-BLA Pairs x StimCond",
            f"pac_allpairs_pairbystim_{pt}_anova.csv",
        )

    # ── 3.2 Per-pair two-PAC-type ──
    pdf.subsection_title(f"{sec}.2. PAC: Per-Pair Two-PAC-Type Models")
    pdf.body_text(
        "Model (per pair): Accuracy ~ sg_pac_z + hfa_pac_z + StimCond + "
        "sg_pac_z:StimCond + hfa_pac_z:StimCond + (1|Patient). "
        "PAC values z-scored within each pair. One full coefficient table "
        "per pair."
    )
    for pair in NONBLA_PAIRS:
        df = load_coefs(f"pac_twotype_{pair}_coefs.csv")
        if df is None:
            continue
        pdf.apa_table(
            f"Table. PAC Two-PAC-Type Model - {display_region(pair)}",
            ["Predictor", "OR", "95% CI", "z", "p", ""],
            coef_table_rows(df),
            col_widths=[58, 18, 38, 14, 14, 8],
        )
        render_anova(
            pdf,
            f"Table. Sequential Model Build - PAC Two-PAC-Type - {display_region(pair)}",
            f"pac_twotype_{pair}_anova.csv",
        )

    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    pdf.output(OUTPUT_PDF)
    print(f"Report saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_report()
