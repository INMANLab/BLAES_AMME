#!/usr/bin/env python
"""Build narrow-band retrieval PDF report for one (measure, scope, modeltype).

Mirrors build_h1abc_full_report.py but reads from the narrow-band stats folder
and adds per-region freq ranges in the panel headers.

Usage:
  python build_narrowband_h1abc_report.py <measure> <scope> <modeltype>
    measure   power | coherence | pac
    scope     BLAMTL | HPCrhinal | HippSubBLA | HippSubRhinal
    modeltype LMMcont | LMMquad | GLMM
"""

import os
import sys

import numpy as np
import pandas as pd
from fpdf import FPDF


if len(sys.argv) < 4:
    print("Usage: python build_narrowband_h1abc_report.py "
          "<measure> <scope> <modeltype>")
    sys.exit(1)

MEASURE = sys.argv[1]
SCOPE = sys.argv[2]
MODELTYPE = sys.argv[3]
assert MEASURE in ("power", "coherence", "pac")
assert SCOPE in ("BLAMTL", "HPCrhinal", "HippSubBLA", "HippSubRhinal")
assert MODELTYPE in ("LMMcont", "LMMquad", "GLMM")

REPO_ROOT = "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
PHASE = "retrieval"
PHASE_LABEL = "Retrieval"
PHASE_FOLDER = "retrieval memory reports permutation test based"

CSV_DIR = os.path.join(
    REPO_ROOT, "OUTPUTS", PHASE_FOLDER,
    "stats", "narrowband_h1abc_retrieval_mlm",
    f"{MEASURE}_{SCOPE}_{MODELTYPE}",
)
OUT_DIR = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER)
FAMILY_DIR = os.path.join(OUT_DIR, "stats",
                          f"fdr_{MEASURE}_{SCOPE}_{MODELTYPE}_narrowband")
os.makedirs(FAMILY_DIR, exist_ok=True)

SCOPE_LABELS = {
    "BLAMTL":        "BLA-MTL (uses HPC)",
    "HPCrhinal":     "HPC-rhinal (uses HPC)",
    "HippSubBLA":    "Hippocampal subregions vs BLA",
    "HippSubRhinal": "Hippocampal subregions vs rhinal cortices",
}
MODEL_LABELS = {
    "LMMcont": "LMM, memory modulation continuous (mem_mod_z)",
    "LMMquad": "LMM, quad responder group",
    "GLMM":    "GLMM, Accuracy ~ band_c + StimCond + band_c:StimCond",
}

# Load the narrow-band freq ranges that were used. If file is missing,
# the combo had no panels.
NB_USED_PATH = os.path.join(CSV_DIR, "_narrow_bands_used.csv")
if not os.path.exists(NB_USED_PATH):
    print(f"No narrow bands used for {MEASURE}/{SCOPE}/{MODELTYPE}; "
          f"nothing to render.")
    sys.exit(0)
NB_USED = pd.read_csv(NB_USED_PATH)
# UNITS_BANDS = (region, family) pairs in the order they appear in the CSV.
UNITS_BANDS = [(r["region"], f"narrow_{r['family']}")
               for _, r in NB_USED.iterrows()]
# The 'basis' column was removed when narrow bands moved to per-modality;
# the source-cluster summary is captured in 'source_summary' instead.
def _get(row, col, default=""):
    return row[col] if col in row.index else default
NB_LOOKUP = {
    (r["region"], f"narrow_{r['family']}"): (
        r["narrow_lo_hz"], r["narrow_hi_hz"],
        _get(r, "source_summary",
             _get(r, "basis", ""))
    )
    for _, r in NB_USED.iterrows()
}

# Bands present in this combo (theta and/or slow_gamma).
BANDS = []
for _, family in UNITS_BANDS:
    if family not in BANDS:
        BANDS.append(family)

REPORT_TITLE = (
    f"NarrowBand - {MEASURE.capitalize()} - {SCOPE_LABELS[SCOPE]} - "
    f"{MODEL_LABELS[MODELTYPE]} - {PHASE_LABEL} - cluster-derived bands"
)
PDF_NAME = (f"{PHASE_LABEL}_{MEASURE.capitalize()}_{SCOPE}_{MODELTYPE}"
            f"_FDRcorrected_NarrowBand.pdf")
OUTPUT_PDF = os.path.join(OUT_DIR, PDF_NAME)


# ---------------------------------------------------------------------------
# Formatting helpers (mirrored from build_h1abc_full_report.py)
# ---------------------------------------------------------------------------
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


def display_unit(u):
    # ALLHPC = macro hippocampus -> "HPC"; HPC region = subiculum -> "SUB"
    parts = []
    for p in str(u).split("_"):
        if p == "ALLHPC":
            parts.append("HPC")
        elif p == "HPC":
            parts.append("SUB")
        else:
            parts.append(p)
    return "-".join(parts)


def pretty_term(term):
    if term == "(Intercept)":
        return "(Intercept)"
    if term == "mem_mod_z":
        return "Memory Modulation Z"
    if term == "band_c":
        return "Narrow Band"
    if term == "StimCondstim":
        return "StimCond (stim vs nostim)"
    if term == "band_c:StimCondstim":
        return "Narrow Band x StimCond"
    if term == "StimCondstim:band_c":
        return "Narrow Band x StimCond"
    if term.startswith("QuadResponderGroup"):
        return "Responder Status: " + term.replace("QuadResponderGroup", "")
    return term


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


def load_panel(unit, band_label):
    coefs_p = os.path.join(CSV_DIR, f"{MEASURE}_{unit}_{band_label}_coefs.csv")
    anova_p = os.path.join(CSV_DIR, f"{MEASURE}_{unit}_{band_label}_anova.csv")
    if not (os.path.exists(coefs_p) and os.path.exists(anova_p)):
        return None, None
    return pd.read_csv(coefs_p), pd.read_csv(anova_p)


QUAD_TERMS = ["QuadResponderGroupAntiResp",
              "QuadResponderGroupModerate",
              "QuadResponderGroupStrong"]
GLMM_INTERACTION_TERM = "band_c:StimCondstim"
LMM_CONT_TERM = "mem_mod_z"


def get_term_p(coefs_df, term):
    if coefs_df is None or coefs_df.empty:
        return np.nan
    row = coefs_df[coefs_df["term"] == term]
    if row.empty:
        return np.nan
    return float(row["p.value"].iloc[0])


def focal_terms_for_modeltype():
    if MODELTYPE == "LMMcont":
        return [LMM_CONT_TERM]
    if MODELTYPE == "LMMquad":
        return list(QUAD_TERMS)
    return [GLMM_INTERACTION_TERM]


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


def model_overview_text():
    if MODELTYPE == "LMMcont":
        return ("Per region/pair, fit:  feature ~ mem_mod_z + (1 | Patient). "
                "feature = mean of diff_Freq columns within the region's "
                "narrow band (cluster-derived freq range from the per-freq "
                "permutation test). FDR-BH applied to mem_mod_z slope p-value "
                "across panels within each band family (narrow_theta and "
                "narrow_slow_gamma corrected separately).")
    if MODELTYPE == "LMMquad":
        return ("Per region/pair, fit:  feature ~ QuadResponderGroup + "
                "(1 | Patient). Reference: Non-responders. Three contrasts "
                "(AntiResp, Moderate, Strong vs NonResp) each FDR-BH corrected "
                "as their own family within each narrow band.")
    return ("Per region/pair, fit:  Accuracy ~ band_c + StimCond + "
            "band_c:StimCond + (1 | Patient). band_c = mean within the "
            "region's narrow band (cluster-derived). FDR-BH applied to the "
            "band_c:StimCond interaction p-value across panels within each "
            "narrow band family.")


def is_focal_term(term):
    return term in focal_terms_for_modeltype()


def compute_families():
    """{band_label: {term: {unit -> (p_raw, q_FDR)}}}"""
    out = {}
    terms = focal_terms_for_modeltype()
    for band in BANDS:
        out[band] = {}
        units_in_band = [u for (u, b) in UNITS_BANDS if b == band]
        for term in terms:
            rows = []
            for u in units_in_band:
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


def render_panel(pdf, u, band, families_band):
    coefs, anova = load_panel(u, band)
    lo, hi, _src = NB_LOOKUP.get((u, band), (None, None, ""))
    range_str = (f" [{lo:.2f}-{hi:.2f} Hz]" if lo is not None else "")
    if coefs is None:
        pdf.subsection_title(
            f"{display_unit(u)}{range_str} -- skipped (no fit; <4 patients "
            f"or missing data)")
        return

    pdf.subsection_title(f"{display_unit(u)}{range_str}")

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
        note=("q (FDR-BH) within this narrow band is applied only on the "
              "focal term(s) for this model type; other rows show raw p only."),
    )

    anova_rows = []
    for _, r in anova.iterrows():
        anova_rows.append([
            str(r["Model"]),
            (str(r["Formula"])
                .replace("QuadResponderGroup", "Responder Status")
                .replace("mem_mod_z", "Memory Modulation Z")
                .replace("band_c", "NarrowBand")),
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


BAND_DISPLAY = {
    "narrow_theta": "Narrow Theta (cluster-derived freq range, per region)",
    "narrow_slow_gamma": "Narrow Slow Gamma (cluster-derived freq range, per region)",
}


def render_report():
    pdf = APAReport()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(
        0, 9,
        f"{PHASE_LABEL} | {MEASURE.capitalize()} | {SCOPE_LABELS[SCOPE]}\n"
        f"{MODEL_LABELS[MODELTYPE]} - NarrowBand"
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6,
             "Cluster-derived per-region narrow bands | "
             "FDR-BH per band family",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.section_title("1. Overview")
    pdf.body_text(model_overview_text())
    pdf.body_text(
        "Narrow band freq ranges come from the per-freq cluster permutation "
        "test (retrieval, stim_nostim contrast). Per region: if a "
        "significant cluster appeared in both Remembered and Forgotten "
        "subsets, the band is set to their freq-range intersection. "
        "Otherwise the single available cluster's range is used. Subjects "
        "BJH032/BJH033 folded; panels with <4 unique patients dropped."
    )

    families = compute_families()

    sec = 2
    for band in BANDS:
        pdf.section_title(f"{sec}. {BAND_DISPLAY.get(band, band)}")
        sec += 1
        for u, b in UNITS_BANDS:
            if b != band:
                continue
            render_panel(pdf, u, band, families[band])

    pdf.output(OUTPUT_PDF)


render_report()
print(f"PDF -> {OUTPUT_PDF}")
print(f"Family CSVs -> {FAMILY_DIR}/")
