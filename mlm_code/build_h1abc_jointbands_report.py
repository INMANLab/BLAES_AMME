#!/usr/bin/env python
"""
Build a joint-bands GLMM PDF (test report) for one (measure, scope) combo
produced by run_h1abc_jointbands_retrieval.R.

Each region's GLMM has BOTH theta_c and slow_gamma_c entered jointly:
    Accuracy ~ theta_c + slow_gamma_c + StimCond
             + theta_c:StimCond + slow_gamma_c:StimCond + (1 | Patient)

FDR-BH is applied separately within this PDF to:
  - theta_c:StimCondstim p-values across regions (FDR family A)
  - slow_gamma_c:StimCondstim p-values across regions (FDR family B)
Main effects are NOT FDR-corrected (per user spec).

Output:
  OUTPUTS/<phase>_memory_reports/testing/
      Jointbands_<phase>_<measure>_<scope>.pdf

Usage:
  python build_h1abc_jointbands_report.py [retrieval|encoding] <measure> <scope>
"""

import os
import sys

import numpy as np
import pandas as pd
from fpdf import FPDF


_argv = list(sys.argv[1:])
if _argv and _argv[0] in ("retrieval", "encoding"):
    PHASE = _argv.pop(0)
else:
    PHASE = "retrieval"

if len(_argv) < 2:
    print("Usage: python build_h1abc_jointbands_report.py "
          "[retrieval|encoding] <measure> <scope>")
    sys.exit(1)

MEASURE = _argv[0]
SCOPE = _argv[1]
assert MEASURE in ("power", "coherence")
assert SCOPE in ("BLAMTL", "HPCrhinal", "HippSubBLA", "HippSubRhinal")

REPO_ROOT = "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
PHASE_FOLDER = "retrieval_memory_reports" if PHASE == "retrieval" else "encoding_memory_reports"
CSV_DIR = os.path.join(
    REPO_ROOT, "OUTPUTS", PHASE_FOLDER,
    "testing", "stats", "jointbands_glmm",
    f"{MEASURE}_{SCOPE}",
)
OUT_DIR = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER, "testing")
os.makedirs(OUT_DIR, exist_ok=True)

SCOPE_LABELS = {
    "BLAMTL":        "BLA-MTL (uses ALLHPC)",
    "HPCrhinal":     "HPC-rhinal (uses ALLHPC)",
    "HippSubBLA":    "Hippocampal subregions vs BLA",
    "HippSubRhinal": "Hippocampal subregions vs rhinal cortices",
}
PHASE_LABEL = "Retrieval" if PHASE == "retrieval" else "Encoding"

SCOPES = {
    "BLAMTL": dict(
        power_regions=["BLA", "ALLHPC", "EC", "PRC"],
        coh_pairs=["BLA_ALLHPC", "BLA_EC", "BLA_PRC"],
    ),
    "HPCrhinal": dict(
        power_regions=["ALLHPC", "EC", "PRC"],
        coh_pairs=["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"],
    ),
    "HippSubBLA": dict(
        power_regions=["BLA", "CA", "DG", "HPC"],
        coh_pairs=["BLA_CA", "BLA_DG", "BLA_HPC"],
    ),
    "HippSubRhinal": dict(
        power_regions=["CA", "DG", "HPC", "EC", "PRC"],
        coh_pairs=["CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC", "EC_PRC"],
    ),
}

UNITS = SCOPES[SCOPE]["power_regions"] if MEASURE == "power" else SCOPES[SCOPE]["coh_pairs"]

REPORT_TITLE = (
    f"H1abc Joint-Bands GLMM (TEST) - {MEASURE.capitalize()} - "
    f"{SCOPE_LABELS[SCOPE]} - {PHASE_LABEL}"
)
PDF_NAME = f"Jointbands_{PHASE_LABEL}_{MEASURE.capitalize()}_{SCOPE}.pdf"
OUTPUT_PDF = os.path.join(OUT_DIR, PDF_NAME)

THETA_INT_TERM = "theta_c:StimCondstim"
SG_INT_TERM = "slow_gamma_c:StimCondstim"

# ---------------------------------------------------------------------------
# Formatting helpers (mirrored from build_h1abc_full_report.py style)
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
    mapping = {
        "(Intercept)":            "(Intercept)",
        "theta_c":                "Theta",
        "slow_gamma_c":           "Slow Gamma",
        "StimCondstim":           "StimCond (stim vs nostim)",
        "theta_c:StimCondstim":   "Theta x StimCond",
        "slow_gamma_c:StimCondstim": "Slow Gamma x StimCond",
        "StimCondstim:theta_c":   "Theta x StimCond",
        "StimCondstim:slow_gamma_c": "Slow Gamma x StimCond",
    }
    return mapping.get(term, term)


# ---------------------------------------------------------------------------
# FDR
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


def load_panel(unit):
    coefs_p = os.path.join(CSV_DIR, f"{MEASURE}_{unit}_jointbands_coefs.csv")
    anova_p = os.path.join(CSV_DIR, f"{MEASURE}_{unit}_jointbands_anova.csv")
    if not (os.path.exists(coefs_p) and os.path.exists(anova_p)):
        return None, None
    return pd.read_csv(coefs_p), pd.read_csv(anova_p)


def get_term_p(coefs_df, term_options):
    if coefs_df is None or coefs_df.empty:
        return np.nan
    for term in term_options:
        row = coefs_df[coefs_df["term"] == term]
        if not row.empty:
            return float(row["p.value"].iloc[0])
    return np.nan


# ---------------------------------------------------------------------------
# Build per-unit FDR families across regions in this scope
# ---------------------------------------------------------------------------

theta_int_terms = ["theta_c:StimCondstim", "StimCondstim:theta_c"]
sg_int_terms    = ["slow_gamma_c:StimCondstim", "StimCondstim:slow_gamma_c"]

panels = {}        # unit -> (coefs_df, anova_df)
theta_p = []       # one entry per unit
sg_p    = []
for u in UNITS:
    coefs, anova = load_panel(u)
    panels[u] = (coefs, anova)
    theta_p.append(get_term_p(coefs, theta_int_terms))
    sg_p.append(get_term_p(coefs, sg_int_terms))

theta_q = bh_fdr(theta_p)
sg_q    = bh_fdr(sg_p)

theta_q_map = {u: q for u, q in zip(UNITS, theta_q)}
sg_q_map    = {u: q for u, q in zip(UNITS, sg_q)}

# Save the per-family FDR table so it's auditable.
fdr_rows = []
for u in UNITS:
    fdr_rows.append({
        "unit": u,
        "theta_int_p":    theta_p[UNITS.index(u)],
        "theta_int_q_BH": theta_q_map[u],
        "sg_int_p":       sg_p[UNITS.index(u)],
        "sg_int_q_BH":    sg_q_map[u],
    })
fdr_df = pd.DataFrame(fdr_rows)
fdr_csv = os.path.join(CSV_DIR, "_fdr_summary.csv")
fdr_df.to_csv(fdr_csv, index=False)
print(f"FDR summary -> {fdr_csv}")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class APAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, REPORT_TITLE, ln=True, align="C")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, text, ln=True)
        self.ln(1)

    def subsection_title(self, text):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, text, ln=True)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, text)
        self.ln(1)


def coef_table(pdf, coefs_df, unit):
    if coefs_df is None or coefs_df.empty:
        pdf.body_text("(no coefficients available)")
        return

    # Build a rendered df with q for the two interaction terms.
    rows = []
    for _, r in coefs_df.iterrows():
        term = r["term"]
        est = float(r["estimate"]) if "estimate" in r else np.nan
        lo = float(r["conf.low"]) if "conf.low" in r else np.nan
        hi = float(r["conf.high"]) if "conf.high" in r else np.nan
        p = float(r["p.value"]) if "p.value" in r else np.nan
        q = ""
        if term in theta_int_terms:
            q_val = theta_q_map.get(unit)
            q = "" if q_val is None or pd.isna(q_val) else p_str(q_val)
        elif term in sg_int_terms:
            q_val = sg_q_map.get(unit)
            q = "" if q_val is None or pd.isna(q_val) else p_str(q_val)
        rows.append([
            pretty_term(term),
            num_str(est),
            ci_str(lo, hi),
            p_str(p),
            sig_str(p),
            q,
        ])

    headers = ["Term", "OR", "95% CI", "p", "", "q (BH)"]
    widths = [55, 22, 38, 18, 8, 22]

    pdf.set_font("Helvetica", "B", 8.5)
    for h, w in zip(headers, widths):
        pdf.cell(w, 5, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8.5)
    for row in rows:
        for val, w in zip(row, widths):
            pdf.cell(w, 5, str(val), border=1, align="C")
        pdf.ln()
    pdf.ln(1)


def render_panel(pdf, unit):
    coefs, anova = panels[unit]
    if coefs is None:
        pdf.subsection_title(
            f"{display_unit(unit)} -- skipped (no fit; <4 patients or missing data)"
        )
        return

    title = f"{display_unit(unit)} - Joint-bands GLMM"
    pdf.subsection_title(title)
    coef_table(pdf, coefs, unit)


def render_anova(pdf):
    pdf.section_title("Sequential model build (per region)")
    for u in UNITS:
        coefs, anova = panels[u]
        if anova is None:
            continue
        pdf.subsection_title(display_unit(u))
        headers = ["Model", "AIC", "BIC", "logLik", "Chisq", "Df", "p"]
        widths  = [28, 22, 22, 28, 22, 14, 22]
        pdf.set_font("Helvetica", "B", 8.5)
        for h, w in zip(headers, widths):
            pdf.cell(w, 5, h, border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8.5)
        for _, r in anova.iterrows():
            row = [
                str(r.get("Model", "")),
                fmt_fit(r.get("AIC", np.nan), 1),
                fmt_fit(r.get("BIC", np.nan), 1),
                fmt_fit(r.get("logLik", np.nan), 1),
                fmt_fit(r.get("Chisq", np.nan), 2),
                fmt_fit(r.get("Df", np.nan), 0),
                p_str(r.get("p.value", np.nan)),
            ]
            for val, w in zip(row, widths):
                pdf.cell(w, 5, str(val), border=1, align="C")
            pdf.ln()
        pdf.ln(1)


def build_pdf():
    pdf = APAReport()
    pdf.add_page()

    pdf.section_title("Method")
    pdf.body_text(
        "Per region/pair, fit:  Accuracy ~ theta_c + slow_gamma_c + StimCond "
        "+ theta_c:StimCond + slow_gamma_c:StimCond + (1 | Patient).  "
        "Predictors are raw band values; on convergence failure both bands "
        "are grand-mean centered as a numerical-stability fallback "
        "(see _glmm_centering_log.csv). Estimates are odds ratios. "
        "FDR-BH is applied separately to the two interaction families "
        "(theta:StimCond, slow_gamma:StimCond) across panels in this PDF; "
        "main effects are NOT FDR-corrected. PAC is excluded because it "
        "has only one band (slow_gamma)."
    )
    pdf.body_text(
        f"Family sizes: theta:StimCond = {len([p for p in theta_p if not pd.isna(p)])} "
        f"panels; slow_gamma:StimCond = {len([p for p in sg_p if not pd.isna(p)])} panels."
    )

    pdf.section_title("Per-region coefficients")
    for u in UNITS:
        render_panel(pdf, u)

    render_anova(pdf)

    pdf.output(OUTPUT_PDF)
    print(f"-> {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
