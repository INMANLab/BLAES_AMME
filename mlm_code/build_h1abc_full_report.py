#!/usr/bin/env python
"""
Build a single FDR-corrected retrieval PDF for one
(measure, scope, modeltype) combination produced by
run_h1abc_full_retrieval.R.

Family = (this PDF) x (band). Theta and slow gamma are corrected separately.

LMM (cont/quad)
  - One panel per region/pair per band.
  - FDR-BH applied to the omnibus LR test (m1 vs m0) p-value, one per panel.
  - Coefficients table shows raw p only.
  - Sequential model build table m1 row shows raw p AND q (FDR).

GLMM
  - One model per band (Region as fixed factor).
  - FDR-BH applied to the Region:StimCond interaction p-values within band.
  - Coefficient table shows raw p AND q (FDR) inline; q only on
    Region:StimCond rows (other rows blank).

Usage:
  python build_h1abc_full_report.py <measure> <scope> <modeltype>
    measure   power | coherence | pac
    scope     BLAMTL | HPCrhinal | HippSubBLA | HippSubRhinal
    modeltype LMMcont | LMMquad | GLMM
"""

import os
import sys

import numpy as np
import pandas as pd
from fpdf import FPDF


# First arg may be phase ("retrieval" / "encoding"); defaults to retrieval.
_argv = list(sys.argv[1:])
if _argv and _argv[0] in ("retrieval", "encoding"):
    PHASE = _argv.pop(0)
else:
    PHASE = "retrieval"

if len(_argv) < 3:
    print("Usage: python build_h1abc_full_report.py "
          "[retrieval|encoding] <measure> <scope> <modeltype>")
    sys.exit(1)

MEASURE = _argv[0]
SCOPE = _argv[1]
MODELTYPE = _argv[2]
assert MEASURE in ("power", "coherence", "pac")
assert SCOPE in ("BLAMTL", "HPCrhinal", "HippSubBLA", "HippSubRhinal")
assert MODELTYPE in ("LMMcont", "LMMquad", "GLMM")

REPO_ROOT = "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
PHASE_FOLDER = "retrieval_memory_reports" if PHASE == "retrieval" else "encoding_memory_reports"
CSV_DIR = os.path.join(
    REPO_ROOT, "OUTPUTS", PHASE_FOLDER,
    "stats", f"h1abc_full_{PHASE}_mlm",
    f"{MEASURE}_{SCOPE}_{MODELTYPE}",
)
OUT_DIR = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER)
FAMILY_DIR = os.path.join(OUT_DIR, "stats",
                          f"fdr_{MEASURE}_{SCOPE}_{MODELTYPE}_{PHASE}")
os.makedirs(FAMILY_DIR, exist_ok=True)

SCOPE_LABELS = {
    "BLAMTL":        "BLA-MTL (uses ALLHPC)",
    "HPCrhinal":     "HPC-rhinal (uses ALLHPC)",
    "HippSubBLA":    "Hippocampal subregions vs BLA",
    "HippSubRhinal": "Hippocampal subregions vs rhinal cortices",
}
MODEL_LABELS = {
    "LMMcont": "LMM, memory modulation continuous (mem_mod_z)",
    "LMMquad": "LMM, quad responder group",
    "GLMM":    "GLMM, Accuracy ~ band_c + StimCond + Region + Region:StimCond",
}
PHASE_LABEL = "Retrieval" if PHASE == "retrieval" else "Encoding"

# Default scopes mirror the R driver.
SCOPES = {
    "BLAMTL": dict(
        power_regions=["BLA", "ALLHPC", "EC", "PRC"],
        coh_pairs=["BLA_ALLHPC", "BLA_EC", "BLA_PRC"],
        pac_pairs=["BLA_ALLHPC", "BLA_EC", "BLA_PRC"],
    ),
    "HPCrhinal": dict(
        power_regions=["ALLHPC", "EC", "PRC"],
        coh_pairs=["ALLHPC_EC", "ALLHPC_PRC"],
        pac_pairs=["ALLHPC_EC", "ALLHPC_PRC"],
    ),
    "HippSubBLA": dict(
        power_regions=["BLA", "CA", "DG", "HPC"],
        coh_pairs=["BLA_CA", "BLA_DG", "BLA_HPC"],
        pac_pairs=["BLA_CA", "BLA_DG", "BLA_HPC"],
    ),
    "HippSubRhinal": dict(
        power_regions=["CA", "DG", "HPC", "EC", "PRC"],
        coh_pairs=["CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"],
        pac_pairs=["CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"],
    ),
}

if MEASURE == "power":
    UNITS = SCOPES[SCOPE]["power_regions"]
elif MEASURE == "coherence":
    UNITS = SCOPES[SCOPE]["coh_pairs"]
else:
    UNITS = SCOPES[SCOPE]["pac_pairs"]

BANDS = ["slow_gamma"] if MEASURE == "pac" else ["theta", "slow_gamma"]
BAND_LABELS = {
    "theta": "Theta (4-8 Hz)",
    "slow_gamma": ("SG PAC (30-50 Hz)" if MEASURE == "pac"
                   else "Slow Gamma (30-55 Hz)"),
}

REPORT_TITLE = (
    f"H1abc - {MEASURE.capitalize()} - {SCOPE_LABELS[SCOPE]} - "
    f"{MODEL_LABELS[MODELTYPE]} - {PHASE_LABEL} - ImBalanced Trials"
)
PDF_NAME = (f"{PHASE_LABEL}_{MEASURE.capitalize()}_{SCOPE}_{MODELTYPE}"
            f"_FDRcorrected_ImBalanced.pdf")
OUTPUT_PDF = os.path.join(OUT_DIR, PDF_NAME)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def display_unit(u):
    return u.replace("_", "-")


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


def fmt_fit(v, decimals=2):
    if pd.isna(v):
        return "-"
    return f"{float(v):.{decimals}f}"


def pretty_term(term):
    if term == "(Intercept)":
        return "(Intercept)"
    if term == "mem_mod_z":
        return "Memory Modulation Z"
    if term == "band_c":
        return "band (grand-mean centered)"
    if term == "StimCondstim":
        return "StimCond (stim vs nostim)"
    if term.startswith("QuadResponderGroup"):
        return "Responder Status: " + term.replace("QuadResponderGroup", "")
    if term.startswith("Region") and ":" not in term:
        return f"Region: {term.replace('Region', '')}"
    if term.startswith("StimCondstim:Region"):
        return f"StimCond x Region: {term.replace('StimCondstim:Region', '')}"
    if term.startswith("Region") and ":StimCondstim" in term:
        # broom.mixed sometimes flips order
        return ("StimCond x Region: "
                + term.replace(":StimCondstim", "").replace("Region", ""))
    return term


# ---------------------------------------------------------------------------
# Data loading & FDR
# ---------------------------------------------------------------------------

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    pv = p[valid]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q_raw = ranked * n / (np.arange(n) + 1)
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_mono = np.minimum(q_mono, 1.0)
    q_full = np.empty(n, dtype=float)
    q_full[order] = q_mono
    out[valid] = q_full
    return out


def load_panel(unit, band):
    """Load one panel's coef + anova CSVs (works for LMM and per-region GLMM)."""
    coefs_p = os.path.join(CSV_DIR, f"{MEASURE}_{unit}_{band}_coefs.csv")
    anova_p = os.path.join(CSV_DIR, f"{MEASURE}_{unit}_{band}_anova.csv")
    if not (os.path.exists(coefs_p) and os.path.exists(anova_p)):
        return None, None
    return pd.read_csv(coefs_p), pd.read_csv(anova_p)


# Focal-coefficient targets for FDR within each model type.
QUAD_TERMS = ["QuadResponderGroupAntiResp",
              "QuadResponderGroupModerate",
              "QuadResponderGroupStrong"]
GLMM_INTERACTION_TERM = "band_c:StimCondstim"
LMM_CONT_TERM = "mem_mod_z"


def get_term_p(coefs_df, term):
    """Return raw p-value for a term, or NaN if not present."""
    if coefs_df is None or coefs_df.empty:
        return np.nan
    row = coefs_df[coefs_df["term"] == term]
    if row.empty:
        return np.nan
    return float(row["p.value"].iloc[0])


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FDR family computation
#
# All three model types render per-region/pair panels; what differs is the
# focal coefficient that gets FDR-corrected within each band.
#
#   LMMcont : focal term = mem_mod_z                       (1 family per band)
#   LMMquad : focal terms = QuadResponderGroupAntiResp /
#             Moderate / Strong                            (3 families per band)
#   GLMM    : focal term = band_c:StimCondstim             (1 family per band)
# ---------------------------------------------------------------------------

def focal_terms_for_modeltype():
    if MODELTYPE == "LMMcont":
        return [LMM_CONT_TERM]
    if MODELTYPE == "LMMquad":
        return list(QUAD_TERMS)
    return [GLMM_INTERACTION_TERM]


def compute_families():
    """Return {band: {term: {unit -> (p_raw, q_FDR)}}}."""
    out = {}
    terms = focal_terms_for_modeltype()
    for band in BANDS:
        out[band] = {}
        for term in terms:
            rows = []
            for u in UNITS:
                coefs, _ = load_panel(u, band)
                rows.append({"unit": u, "p_raw": get_term_p(coefs, term)})
            df = pd.DataFrame(rows)
            df["q_FDR"] = bh_fdr(df["p_raw"].to_numpy())
            df["sig"] = df["q_FDR"].apply(sig_str)
            df.to_csv(
                os.path.join(
                    FAMILY_DIR,
                    f"{MEASURE}_{SCOPE}_{MODELTYPE}_{band}_{term}.csv".replace(
                        ":", "_x_"
                    ),
                ),
                index=False,
            )
            mapping = {r["unit"]: (r["p_raw"], r["q_FDR"])
                       for _, r in df.iterrows()}
            out[band][term] = mapping
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def model_overview_text():
    if MODELTYPE == "LMMcont":
        return ("Per region/pair, fit:  feature ~ mem_mod_z + (1 | Patient). "
                "mem_mod_z = z-scored avg_stim_dprime_diff (between-patient). "
                "FDR-BH is applied to the mem_mod_z slope p-value across all "
                "panels within each band (theta and slow gamma corrected "
                "separately).")
    if MODELTYPE == "LMMquad":
        return ("Per region/pair, fit:  feature ~ QuadResponderGroup + "
                "(1 | Patient). Reference: Non-responders. Three contrasts "
                "(AntiResp, Moderate, Strong vs NonResp) each FDR-BH corrected "
                "as their own family within each band - 3 sub-families per "
                "band, one per contrast.")
    return ("Per region/pair, fit:  Accuracy ~ band_c + StimCond + "
            "band_c:StimCond + (1 | Patient). band_c is grand-mean-centered "
            "(matches H1c convention). Estimates are odds ratios. FDR-BH is "
            "applied to the band_c:StimCond interaction p-value across "
            "panels within each band.")


def is_focal_term(term):
    return term in focal_terms_for_modeltype()


def render_panel(pdf, u, band, families_band):
    coefs, anova = load_panel(u, band)
    if coefs is None:
        pdf.subsection_title(
            f"{display_unit(u)} -- skipped (no fit; <4 patients or missing "
            f"data)"
        )
        return

    pdf.subsection_title(display_unit(u))

    is_glmm = MODELTYPE == "GLMM"
    headers = ["Predictor",
               "OR" if is_glmm else "Estimate",
               "95% CI",
               "z" if is_glmm else "t",
               "p", "q (FDR)", ""]
    col_widths = [78, 18, 38, 14, 14, 18, 8]

    coef_rows = []
    for _, r in coefs.iterrows():
        term = str(r["term"])
        q = np.nan
        if is_focal_term(term):
            mapping = families_band.get(term, {})
            tup = mapping.get(u)
            if tup is not None:
                q = tup[1]
        coef_rows.append([
            pretty_term(term),
            num_str(r["estimate"]),
            ci_str(r["conf.low"], r["conf.high"]),
            f"{r['statistic']:.3f}" if pd.notna(r.get("statistic")) else "",
            p_str(r["p.value"]),
            p_str(q) if pd.notna(q) else "-",
            sig_str(q) if pd.notna(q) else "",
        ])
    pdf.apa_table(
        ("Table. Fixed-effect Coefficients (odds ratios)" if is_glmm
         else "Table. Fixed-effect Coefficients"),
        headers, coef_rows, col_widths=col_widths,
        note=("q (FDR-BH) within this band is applied only on the focal "
              "term(s) for this model type; other rows show raw p only."),
    )

    anova_rows = []
    for _, r in anova.iterrows():
        anova_rows.append([
            str(r["Model"]),
            (str(r["Formula"])
                .replace("QuadResponderGroup", "Responder Status")
                .replace("mem_mod_z", "Memory Modulation Z")),
            f"{int(r['npar'])}" if pd.notna(r["npar"]) else "-",
            fmt_fit(r["AIC"], 1),
            fmt_fit(r["BIC"], 1),
            fmt_fit(r["logLik"], 1),
            fmt_fit(r["Chisq"], 2),
            fmt_fit(r["Df"], 0),
            p_str(r["p.value"]) if pd.notna(r["p.value"]) else "-",
            sig_str(r["p.value"]) if pd.notna(r["p.value"]) else "",
        ])
    pdf.apa_table(
        "Table. Sequential Model Build",
        ["Model", "Formula", "npar", "AIC", "BIC", "logLik",
         "Chisq", "Df", "p", ""],
        anova_rows,
        col_widths=[12, 100, 10, 14, 14, 14, 14, 8, 14, 6],
        note="Sequential LR test vs previous row.",
    )


def render_report():
    pdf = APAReport()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(
        0, 9,
        f"{PHASE_LABEL} | {MEASURE.capitalize()} | {SCOPE_LABELS[SCOPE]}\n"
        f"{MODEL_LABELS[MODELTYPE]}"
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, "ImBalanced Trials | FDR-BH per band (coefficient-based)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.section_title("1. Overview")
    pdf.body_text(model_overview_text())
    pdf.body_text(
        "Subjects BJH032 and BJH033 are folded to a single subject. Panels "
        "with < 4 unique patients are dropped (do not enter FDR)."
    )

    families = compute_families()

    sec = 2
    for band in BANDS:
        pdf.section_title(f"{sec}. {BAND_LABELS[band]}")
        sec += 1
        for u in UNITS:
            render_panel(pdf, u, band, families[band])

    pdf.output(OUTPUT_PDF)


render_report()

print(f"PDF -> {OUTPUT_PDF}")
print(f"Family CSVs -> {FAMILY_DIR}/")
