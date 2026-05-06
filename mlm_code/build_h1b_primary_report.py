#!/usr/bin/env python
"""
H1b PRIMARY TEST reports (between-subject only, no StimCond).

Two reports:
  - Memory Modulation Status Z (continuous moderator)
        Neural Feature ~ Memory Modulation Z + (1 | Patient)
  - Responder Group Status (categorical, NonResp = reference)
        Neural Feature ~ Responder Group Status + (1 | Patient)

Reads from outputs/h1b_subregions_quad_lmm_<filter>_retrieval_mlm/
and writes outputs/Hypothesis_1b_PrimaryTest_<MemMod|RespGroup>_LMM_Retrieval_*.pdf.

Usage:
  python build_h1b_primary_report.py <memmod|respgroup> <balanced|imbalanced>
"""

import os
import sys

import pandas as pd
from fpdf import FPDF


if len(sys.argv) < 3:
    print("Usage: python build_h1b_primary_report.py <memmod|respgroup> <balanced|imbalanced>")
    sys.exit(1)

TEST = sys.argv[1]
FILTER = sys.argv[2]
assert TEST in ("memmod", "respgroup")
assert FILTER in ("balanced", "imbalanced")

if TEST == "memmod":
    SUFFIX = "primaryMemMod"
    TEST_LABEL = "Memory Modulation Status Z"
    OUT_TAG = "MemMod"
else:
    SUFFIX = "primaryRespGroup"
    TEST_LABEL = "Responder Group Status"
    OUT_TAG = "RespGroup"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "outputs",
                        f"h1b_subregions_quad_lmm_{FILTER}_retrieval_mlm")
FILTER_LABEL = "Balanced Trials" if FILTER == "balanced" else "ImBalanced Trials"

REPORT_TITLE = (
    f"Hypothesis 1b - Primary Test: {TEST_LABEL} - "
    f"Retrieval - {FILTER_LABEL}"
)
OUTPUT_PDF = os.path.join(
    SCRIPT_DIR, "outputs",
    f"Hypothesis_1b_PrimaryTest_{OUT_TAG}_LMM_Retrieval_"
    f"{FILTER_LABEL.replace(' ', '_')}.pdf",
)

POWER_REGIONS = ["BLA", "CA", "DG", "HPC", "EC", "PRC"]
COH_PAIRS = ["BLA_CA","BLA_DG","BLA_HPC","BLA_EC","BLA_PRC","CA_EC","CA_PRC","DG_EC","DG_PRC","EC_HPC","HPC_PRC","EC_PRC"]
PAC_PAIRS = ["BLA_CA","BLA_DG","BLA_HPC","BLA_EC","BLA_PRC","CA_EC","CA_PRC","DG_EC","DG_PRC","EC_HPC","HPC_PRC"]
BAND_LABELS = {"theta": "Theta (4-8 Hz)", "slow_gamma": "Slow Gamma (30-55 Hz)"}


def display_region(r):
    return r.replace("_", "-")


def feature_name(modality):
    if modality == "power":
        return "Power"
    if modality == "coherence":
        return "Coherence"
    if modality == "pac":
        return "PAC"
    return modality


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


def fmt_fit(value, decimals=2):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}"


def pretty_term(term):
    if term == "(Intercept)":
        return "(Intercept)"
    if term == "mem_mod_z":
        return "Memory Modulation Z"
    if term.startswith("QuadResponderGroup"):
        level = term.replace("QuadResponderGroup", "")
        return f"Responder Group Status: {level}"
    return term


def pretty_formula(formula):
    return (formula
            .replace("feature", "NeuralFeature")
            .replace("QuadResponderGroup", "ResponderGroupStatus")
            .replace("mem_mod_z", "MemoryModulationZ"))


def load_csv(filename):
    p = os.path.join(DATA_DIR, filename)
    if not os.path.exists(p):
        return None
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


def anova_table_rows(df):
    rows = []
    for _, r in df.iterrows():
        rows.append([
            str(r["Model"]),
            pretty_formula(str(r["Formula"])),
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


def render_panel(pdf, panel_label, modality, unit, band_label):
    coef_fn = f"{modality}_{unit}_{band_label}_{SUFFIX}_coefs.csv"
    anova_fn = f"{modality}_{unit}_{band_label}_{SUFFIX}_anova.csv"
    df_coef = load_csv(coef_fn)
    df_anova = load_csv(anova_fn)
    if df_coef is None and df_anova is None:
        return False

    pdf.subsubsection_title(panel_label)

    if TEST == "memmod":
        ref_note = ("Memory Modulation Z is the per-subject z-scored "
                    "stim-minus-no-stim d' (one value per patient). "
                    "Reference: Memory Modulation Z = 0 (sample mean).")
    else:
        ref_note = ("Reference category for Responder Group Status: "
                    "NonResp (Non-responders). Coefficients labelled "
                    "AntiResp / Moderate / Strong are contrasts vs NonResp.")

    if df_coef is not None and not df_coef.empty:
        pdf.apa_table(
            f"Table. Primary H1b Test - {panel_label}",
            ["Predictor", "Estimate", "95% CI", "t", "p", ""],
            coef_table_rows(df_coef),
            col_widths=[78, 18, 38, 14, 14, 8],
            note=ref_note,
        )
    if df_anova is not None and not df_anova.empty:
        pdf.apa_table(
            f"Table. Sequential Model Build - {panel_label}",
            ["Model", "Formula", "npar", "AIC", "BIC", "logLik",
             "Chisq", "Df", "p", ""],
            anova_table_rows(df_anova),
            col_widths=[14, 92, 10, 14, 14, 14, 14, 8, 14, 6],
            note=("LR test vs previous row. * p < .05, ** p < .01, "
                  "*** p < .001, + p < .10."),
        )
    return True


def build_report():
    pdf = APAReport()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9,
                   f"Hypothesis 1b - Primary Test\n"
                   f"({TEST_LABEL})")
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

    if TEST == "memmod":
        pdf.body_text(
            "This report tests H1b directly with a between-subject model "
            "that asks whether the Neural Feature (Power / Coherence / PAC) "
            "varies across patients as a function of Memory Modulation "
            "Status Z (continuous moderator)."
        )
        pdf.body_text(
            "Memory Modulation Status Z is each patient's z-scored "
            "stim-minus-no-stim d' (a single subject-level value, computed "
            "from behavioral accuracy on the AMME recognition test). The "
            "Neural Feature is the band-averaged spectral or connectivity "
            "value extracted at retrieval. Trial-level stim condition is "
            "NOT a predictor in this primary test."
        )
    else:
        pdf.body_text(
            "This report tests H1b directly with a between-subject model "
            "that asks whether the Neural Feature (Power / Coherence / PAC) "
            "differs across the four Responder Group Status levels "
            "(Anti / Non = ref / Moderate / Strong)."
        )
        pdf.body_text(
            "Responder Group Status is each patient's discretized "
            "behavioral memory benefit from prior BLA stimulation. The "
            "Neural Feature is the band-averaged spectral or connectivity "
            "value extracted at retrieval. Trial-level stim condition is "
            "NOT a predictor in this primary test."
        )

    pdf.body_text(
        "Subject IDs BJH032 and BJH033 are folded to a single subject "
        "(54 unique participants)."
    )

    pdf.subsection_title("Statistical Approach")
    if TEST == "memmod":
        pdf.body_text(
            "Per region (Power) or pair (Coherence / PAC), per band, "
            "the Neural Feature is regressed on Memory Modulation Z with a "
            "patient random intercept: NeuralFeature ~ MemoryModulationZ + "
            "(1 | Patient). Sequential model build compares the intercept-only "
            "model (m0) to the model with Memory Modulation Z added (m1)."
        )
    else:
        pdf.body_text(
            "Per region (Power) or pair (Coherence / PAC), per band, "
            "the Neural Feature is regressed on Responder Group Status with "
            "a patient random intercept: NeuralFeature ~ ResponderGroupStatus + "
            "(1 | Patient). The three contrasts (AntiResp / Moderate / Strong "
            "vs NonResp) test whether each group's mean Neural Feature differs "
            "from Non-responders'."
        )
    pdf.body_text(
        "Fixed effects fit with REML; sequential model comparisons refit "
        "with ML for likelihood-ratio tests. Bands tested: theta (4-8 Hz) "
        "and slow gamma (30-55 Hz). PAC: Slow Gamma PAC (30-50 Hz) only."
    )

    sec += 1
    pdf.section_title(f"{sec}. Power")
    sub_idx = 1
    for reg in POWER_REGIONS:
        for bk, bl in BAND_LABELS.items():
            anchor = load_csv(f"power_{reg}_{bk}_{SUFFIX}_coefs.csv")
            if anchor is None:
                continue
            label = f"{feature_name('power')} - {display_region(reg)} - {bl}"
            pdf.subsection_title(f"{sec}.{sub_idx}. {label}")
            sub_idx += 1
            render_panel(pdf, label, "power", reg, bk)

    sec += 1
    pdf.section_title(f"{sec}. Coherence")
    sub_idx = 1
    for pair in COH_PAIRS:
        for bk, bl in BAND_LABELS.items():
            anchor = load_csv(f"coherence_{pair}_{bk}_{SUFFIX}_coefs.csv")
            if anchor is None:
                continue
            label = f"{feature_name('coherence')} - {display_region(pair)} - {bl}"
            pdf.subsection_title(f"{sec}.{sub_idx}. {label}")
            sub_idx += 1
            render_panel(pdf, label, "coherence", pair, bk)

    sec += 1
    pdf.section_title(f"{sec}. PAC (Slow Gamma PAC, 30-50 Hz)")
    sub_idx = 1
    for pair in PAC_PAIRS:
        anchor = load_csv(f"pac_{pair}_slow_gamma_{SUFFIX}_coefs.csv")
        if anchor is None:
            continue
        label = f"{feature_name('pac')} - {display_region(pair)} - Slow Gamma PAC"
        pdf.subsection_title(f"{sec}.{sub_idx}. {label}")
        sub_idx += 1
        render_panel(pdf, label, "pac", pair, "slow_gamma")

    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    pdf.output(OUTPUT_PDF)
    print(f"Report saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_report()
