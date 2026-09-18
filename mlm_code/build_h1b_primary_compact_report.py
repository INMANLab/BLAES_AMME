#!/usr/bin/env python
"""
H1b PRIMARY TEST compact reports — paste-into-paper version.

Same primary tests as build_h1b_primary_report.py but trimmed:
  - drops the sequential model-build (anova) tables
  - drops the per-table Note. line on every table EXCEPT the very last one
    in the document (which keeps the reference-category note for posterity)

Two reports:
  - Memory Modulation Z (continuous)
        Neural Feature ~ Memory Modulation Z + (1 | Patient)
  - Responder Group Status (categorical, NonResp = reference)
        Neural Feature ~ Responder Group Status + (1 | Patient)

Usage:
  python build_h1b_primary_compact_report.py <memmod|respgroup> <balanced|imbalanced>
"""

import os
import sys

import pandas as pd
from fpdf import FPDF


if len(sys.argv) < 3:
    print("Usage: python build_h1b_primary_compact_report.py "
          "<memmod|respgroup> <balanced|imbalanced>")
    sys.exit(1)

TEST = sys.argv[1]
FILTER = sys.argv[2]
assert TEST in ("memmod", "respgroup")
assert FILTER in ("balanced", "imbalanced")

if TEST == "memmod":
    SUFFIX = "primaryMemMod"
    TEST_LABEL = "Memory Modulation Status Z"
    OUT_TAG = "MemMod"
    REF_NOTE = ("Memory Modulation Z is the per-subject z-scored "
                "stim-minus-no-stim d' (one value per patient). "
                "Reference: Memory Modulation Z = 0 (sample mean). "
                "* p < .05, ** p < .01, *** p < .001, + p < .10.")
else:
    SUFFIX = "primaryRespGroup"
    TEST_LABEL = "Responder Group Status"
    OUT_TAG = "RespGroup"
    REF_NOTE = ("Reference category for Responder Group Status: "
                "NonResp (Non-responders). Coefficients labelled "
                "AntiResp / Moderate / Strong are contrasts vs NonResp. "
                "* p < .05, ** p < .01, *** p < .001, + p < .10.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "outputs",
                        f"h1b_subregions_quad_lmm_{FILTER}_retrieval_mlm")
FILTER_LABEL = "Balanced Trials" if FILTER == "balanced" else "ImBalanced Trials"

REPORT_TITLE = (
    f"Hypothesis 1b - Primary Test (compact): {TEST_LABEL} - "
    f"Retrieval - {FILTER_LABEL}"
)
OUTPUT_PDF = os.path.join(
    SCRIPT_DIR, "outputs",
    f"Hypothesis_1b_PrimaryTest_{OUT_TAG}_LMM_Compact_Retrieval_"
    f"{FILTER_LABEL.replace(' ', '_')}.pdf",
)

POWER_REGIONS = ["BLA", "CA", "DG", "HPC", "EC", "PRC"]
COH_PAIRS = ["BLA_CA","BLA_DG","BLA_HPC","BLA_EC","BLA_PRC","CA_EC","CA_PRC","DG_EC","DG_PRC","EC_HPC","HPC_PRC","EC_PRC"]
PAC_PAIRS = ["BLA_CA","BLA_DG","BLA_HPC","BLA_EC","BLA_PRC","CA_EC","CA_PRC","DG_EC","DG_PRC","EC_HPC","HPC_PRC"]
BAND_LABELS = {"theta": "Theta (4-8 Hz)", "slow_gamma": "Slow Gamma (30-55 Hz)"}


def display_region(r):
    # ALLHPC = macro hippocampus -> "HPC"; HPC region = subiculum -> "SUB"
    parts = []
    for p in str(r).split("_"):
        if p == "ALLHPC":
            parts.append("HPC")
        elif p == "HPC":
            parts.append("SUB")
        else:
            parts.append(p)
    return "-".join(parts)


def feature_name(modality):
    if modality == "power":      return "Power"
    if modality == "coherence":  return "Coherence"
    if modality == "pac":        return "PAC"
    return modality


class APAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, REPORT_TITLE, align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def apa_table(self, title, headers, rows, col_widths=None, note=None):
        self.ln(2)
        self.set_font("Times", "I", 10)
        self.multi_cell(0, 5, title)
        self.ln(0.5)
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
            if self.get_y() > self.h - 30:
                self.set_line_width(0.5)
                self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
                self.add_page()
                self.set_font("Times", "I", 10)
                self.multi_cell(0, 5, title + " (continued)")
                self.ln(0.5)
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
            self.ln(1)


def p_str(p):
    if isinstance(p, str): return p
    if pd.isna(p):         return ""
    if p < .001:           return "< .001"
    return f"{p:.3f}".lstrip("0")


def sig_str(p):
    if isinstance(p, str) or pd.isna(p): return ""
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    if p < .10:  return "+"
    return ""


def num_str(v, decimals=4):
    if pd.isna(v): return ""
    if abs(v) > 1e4 or (abs(v) < 1e-4 and v != 0):
        return f"{v:.2e}"
    return f"{v:.{decimals}f}"


def ci_str(lo, hi):
    if pd.isna(lo) or pd.isna(hi): return ""
    return f"[{num_str(lo)}, {num_str(hi)}]"


def pretty_term(term):
    if term == "(Intercept)": return "(Intercept)"
    if term == "mem_mod_z":   return "Memory Modulation Z"
    if term.startswith("QuadResponderGroup"):
        return f"Responder Group Status: {term.replace('QuadResponderGroup','')}"
    return term


def load_csv(filename):
    p = os.path.join(DATA_DIR, filename)
    if not os.path.exists(p): return None
    return pd.read_csv(p)


def coef_table_rows(df):
    rows = []
    for _, r in df.iterrows():
        stat = r.get("statistic", float("nan"))
        rows.append([
            pretty_term(str(r["term"])),
            num_str(r["estimate"]),
            ci_str(r["conf.low"], r["conf.high"]),
            f"{stat:.3f}" if pd.notna(stat) else "",
            p_str(r["p.value"]) if "p.value" in df.columns else "",
            sig_str(r["p.value"]) if "p.value" in df.columns else "",
        ])
    return rows


def collect_by_modality():
    """Return dict modality -> list of (panel_label, df_coef)."""
    out = {"Power": [], "Coherence": [], "PAC": []}
    for reg in POWER_REGIONS:
        for bk, bl in BAND_LABELS.items():
            df = load_csv(f"power_{reg}_{bk}_{SUFFIX}_coefs.csv")
            if df is not None and not df.empty:
                out["Power"].append(
                    (f"{display_region(reg)} - {bl}", df))
    for pair in COH_PAIRS:
        for bk, bl in BAND_LABELS.items():
            df = load_csv(f"coherence_{pair}_{bk}_{SUFFIX}_coefs.csv")
            if df is not None and not df.empty:
                out["Coherence"].append(
                    (f"{display_region(pair)} - {bl}", df))
    for pair in PAC_PAIRS:
        df = load_csv(f"pac_{pair}_slow_gamma_{SUFFIX}_coefs.csv")
        if df is not None and not df.empty:
            out["PAC"].append(
                (f"{display_region(pair)} - Slow Gamma PAC", df))
    return out


def coef_table_rows(df):
    rows = []
    for _, r in df.iterrows():
        stat = r.get("statistic", float("nan"))
        rows.append([
            pretty_term(str(r["term"])),
            num_str(r["estimate"]),
            ci_str(r["conf.low"], r["conf.high"]),
            f"{stat:.3f}" if pd.notna(stat) else "",
            p_str(r["p.value"]) if "p.value" in df.columns else "",
            sig_str(r["p.value"]) if "p.value" in df.columns else "",
        ])
    return rows


# Memmod report uses tables 5-7; respgroup uses tables 8-10.
TABLE_NUM_BASE = 5 if TEST == "memmod" else 8
MODALITY_ORDER = ["Power", "Coherence", "PAC"]


def build_report():
    pdf = APAReport()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 7,
                   f"Hypothesis 1b - {TEST_LABEL}\n"
                   f"Retrieval - {FILTER_LABEL}")
    pdf.ln(2)

    by_mod = collect_by_modality()

    last_modality_idx = len(MODALITY_ORDER) - 1
    for i, modality in enumerate(MODALITY_ORDER):
        panels = by_mod[modality]
        if not panels:
            continue
        table_num = TABLE_NUM_BASE + i

        # New page per modality so Power / Coherence / PAC are visually
        # separated and easy to find when pasting.
        if i > 0:
            pdf.add_page()

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, f"Section {table_num}. {TEST_LABEL} - {modality}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        last_panel_idx = len(panels) - 1
        for j, (panel_label, df_coef) in enumerate(panels):
            is_last_table = (i == last_modality_idx and j == last_panel_idx)
            title = (f"Table {table_num}.{j + 1}. {TEST_LABEL} - "
                     f"{modality} - {panel_label}")
            pdf.apa_table(
                title,
                ["Predictor", "Estimate", "95% CI", "t", "p", ""],
                coef_table_rows(df_coef),
                col_widths=[78, 18, 38, 14, 14, 8],
                note=REF_NOTE if is_last_table else None,
            )

    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    pdf.output(OUTPUT_PDF)
    print(f"Report saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_report()
