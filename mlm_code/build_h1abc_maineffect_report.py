#!/usr/bin/env python
"""
Build a single FDR-corrected main-effect GLMM PDF for one
(phase, measure, scope, contrast) combination produced by
run_h1abc_maineffect.R.

Family = (this PDF) x (band). Theta and slow gamma are corrected separately.

Model fit per region/pair:
  Accuracy ~ band_c + (1 | Patient)
Focal term for FDR-BH within band = band_c (the band slope on memory).

Contrasts (different trial filters in run_h1abc_maineffect.R):
  all     - all stim + nostim trials pooled
  nostim  - endogenous (no-stim) trials only
  stim    - stim trials only

Usage:
  python build_h1abc_maineffect_report.py <phase> <measure> <scope> <contrast>
    phase     encoding | retrieval
    measure   power | coherence | pac
    scope     BLAMTL | HPCrhinal | HippSubBLA | HippSubRhinal
    contrast  all | nostim | stim
"""

import os
import sys

import numpy as np
import pandas as pd
from fpdf import FPDF


if len(sys.argv) < 5:
    print("Usage: python build_h1abc_maineffect_report.py "
          "<phase> <measure> <scope> <contrast>")
    sys.exit(1)

PHASE    = sys.argv[1]
MEASURE  = sys.argv[2]
SCOPE    = sys.argv[3]
CONTRAST = sys.argv[4]
assert PHASE in ("encoding", "retrieval")
assert MEASURE in ("power", "coherence", "pac")
assert SCOPE in ("BLAMTL", "HPCrhinal", "HippSubBLA", "HippSubRhinal")
assert CONTRAST in ("all", "nostim", "stim")

REPO_ROOT = "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
PHASE_FOLDER = ("retrieval_memory_reports" if PHASE == "retrieval"
                else "encoding_memory_reports")
CSV_DIR = os.path.join(
    REPO_ROOT, "OUTPUTS", PHASE_FOLDER,
    "stats", f"h1abc_maineffect_{PHASE}_mlm",
    f"{MEASURE}_{SCOPE}_{CONTRAST}",
)
OUT_DIR = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER)
FAMILY_DIR = os.path.join(
    OUT_DIR, "stats",
    f"fdr_{MEASURE}_{SCOPE}_GLMM_MainEffect_{CONTRAST}_{PHASE}"
)
os.makedirs(FAMILY_DIR, exist_ok=True)

SCOPE_LABELS = {
    "BLAMTL":        "BLA-MTL (uses HPC)",
    "HPCrhinal":     "HPC-rhinal (uses HPC)",
    "HippSubBLA":    "Hippocampal subregions vs BLA",
    "HippSubRhinal": "Hippocampal subregions vs rhinal cortices",
}
CONTRAST_LABELS = {
    "all":    "All trials (stim + no-stim pooled)",
    "nostim": "Endogenous trials only (no-stim)",
    "stim":   "Stim trials only",
}
CONTRAST_SHORT = {
    "all":    "AllTrials",
    "nostim": "Endogenous",
    "stim":   "Stim",
}
PHASE_LABEL = "Retrieval" if PHASE == "retrieval" else "Encoding"

SCOPES = {
    "BLAMTL": dict(
        power_regions=["BLA", "ALLHPC", "EC", "PRC"],
        coh_pairs=["BLA_ALLHPC", "BLA_EC", "BLA_PRC"],
        pac_pairs=["BLA_ALLHPC", "BLA_EC", "BLA_PRC"],
    ),
    "HPCrhinal": dict(
        power_regions=["ALLHPC", "EC", "PRC"],
        coh_pairs=["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"],
        pac_pairs=["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"],
    ),
    "HippSubBLA": dict(
        power_regions=["BLA", "CA", "DG", "HPC"],
        coh_pairs=["BLA_CA", "BLA_DG", "BLA_HPC"],
        pac_pairs=["BLA_CA", "BLA_DG", "BLA_HPC"],
    ),
    "HippSubRhinal": dict(
        power_regions=["CA", "DG", "HPC"],
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
    f"GLMM Main Effect ({CONTRAST_LABELS[CONTRAST]}) - "
    f"{PHASE_LABEL} - ImBalanced Trials"
)
PDF_NAME = (f"{PHASE_LABEL}_{MEASURE.capitalize()}_{SCOPE}_GLMM_MainEffect_"
            f"{CONTRAST_SHORT[CONTRAST]}_FDRcorrected.pdf")
OUTPUT_PDF = os.path.join(OUT_DIR, PDF_NAME)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def display_unit(u):
    parts = []
    for p in str(u).split("_"):
        if p == "ALLHPC":
            parts.append("HPC")
        elif p == "HPC":
            parts.append("SUB")
        else:
            parts.append(p)
    return "-".join(parts)


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


# Per-panel centering log keyed by (contrast, modality, unit, band). Panels
# that hit a convergence fallback to grand-mean centering keep the "_c"
# suffix in their tables; non-centered panels render as plain "band".
def _load_centering_log():
    log_p = os.path.join(REPO_ROOT, "OUTPUTS", PHASE_FOLDER,
                         "stats", f"h1abc_maineffect_{PHASE}_mlm",
                         "_glmm_centering_log.csv")
    if not os.path.exists(log_p):
        return {}
    try:
        df = pd.read_csv(log_p)
    except Exception:
        return {}
    out = {}
    for _, r in df.iterrows():
        key = (str(r.get("contrast", CONTRAST)),
               str(r["modality"]), str(r["unit"]), str(r["band"]))
        out[key] = bool(r["centered"])
    return out


_CENTERING_LOG = _load_centering_log()


def panel_was_centered(unit, band):
    return bool(_CENTERING_LOG.get((CONTRAST, MEASURE, str(unit), str(band)),
                                   False))


def band_label(centered):
    return "band_c" if centered else "band"


def pretty_term(term, centered=False):
    bl = band_label(centered)
    if term == "(Intercept)":
        return "(Intercept)"
    if term == "band_c":
        return bl
    return term


def relabel_formula(formula_str, centered):
    if centered or not isinstance(formula_str, str):
        return formula_str
    return formula_str.replace("band_c", "band")


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
    coefs_p = os.path.join(CSV_DIR, f"{MEASURE}_{unit}_{band}_coefs.csv")
    anova_p = os.path.join(CSV_DIR, f"{MEASURE}_{unit}_{band}_anova.csv")
    if not (os.path.exists(coefs_p) and os.path.exists(anova_p)):
        return None, None
    return pd.read_csv(coefs_p), pd.read_csv(anova_p)


FOCAL_TERM = "band_c"


def get_term_p(coefs_df, term):
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
                self.line(x_start, self.get_y(),
                          x_start + total_w, self.get_y())
                self.add_page()
                self.set_font("Times", "I", 10)
                self.multi_cell(0, 5, title + " (continued)")
                self.ln(1)
                self.set_line_width(0.5)
                x_start = self.get_x()
                self.line(x_start, self.get_y(),
                          x_start + total_w, self.get_y())
                self.ln(1)
                self.set_font("Times", "B", 9)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 5, str(h), align="C")
                self.ln()
                self.set_line_width(0.3)
                self.line(x_start, self.get_y(),
                          x_start + total_w, self.get_y())
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


def compute_families():
    """{band: {unit: (p_raw, q_FDR)}} - one BH family per band."""
    out = {}
    for band in BANDS:
        rows = []
        for u in UNITS:
            coefs, _ = load_panel(u, band)
            rows.append({"unit": u, "p_raw": get_term_p(coefs, FOCAL_TERM)})
        df = pd.DataFrame(rows)
        df["q_FDR"] = bh_fdr(df["p_raw"].to_numpy())
        df["sig"] = df["q_FDR"].apply(sig_str)
        fam_csv = os.path.join(
            FAMILY_DIR,
            f"{MEASURE}_{SCOPE}_GLMM_MainEffect_{CONTRAST}_{band}_band_c.csv"
        )
        df.to_csv(fam_csv, index=False)
        out[band] = {r["unit"]: (r["p_raw"], r["q_FDR"])
                     for _, r in df.iterrows()}
    return out


def model_overview_text():
    return (
        f"Per region/pair, fit:  Accuracy ~ band + (1 | Patient). 'band' is "
        f"the raw band value; if a panel failed to converge it was "
        f"grand-mean-centered as a numerical-stability fallback, in which "
        f"case its table shows 'band_c' with a header note (see "
        f"_glmm_centering_log.csv). Estimates are odds ratios. FDR-BH is "
        f"applied to the band (main-effect) p-value across panels within "
        f"each band. Trial filter for this report: "
        f"{CONTRAST_LABELS[CONTRAST]}."
    )


def is_focal_term(term):
    return term == FOCAL_TERM


def render_panel(pdf, u, band, families_band):
    coefs, anova = load_panel(u, band)
    if coefs is None:
        pdf.subsection_title(
            f"{display_unit(u)} -- skipped (no fit; <4 patients or <30 trials)"
        )
        return

    centered = panel_was_centered(u, band)
    title_suffix = "  [band grand-mean-centered]" if centered else ""
    pdf.subsection_title(display_unit(u) + title_suffix)

    headers = ["Predictor", "OR", "95% CI", "z", "p", "q (FDR)", ""]
    col_widths = [78, 18, 38, 14, 14, 18, 8]

    coef_rows = []
    for _, r in coefs.iterrows():
        term = str(r["term"])
        q = np.nan
        if is_focal_term(term):
            tup = families_band.get(u)
            if tup is not None:
                q = tup[1]
        coef_rows.append([
            pretty_term(term, centered=centered),
            num_str(r["estimate"]),
            ci_str(r["conf.low"], r["conf.high"]),
            f"{r['statistic']:.3f}" if pd.notna(r.get("statistic")) else "",
            p_str(r["p.value"]),
            p_str(q) if pd.notna(q) else "-",
            sig_str(q) if pd.notna(q) else "",
        ])
    pdf.apa_table(
        "Table. Fixed-effect Coefficients (odds ratios)",
        headers, coef_rows, col_widths=col_widths,
        note=("q (FDR-BH) within this band is applied only on the band "
              "row; the (Intercept) row shows raw p only."),
    )

    anova_rows = []
    for _, r in anova.iterrows():
        anova_rows.append([
            str(r["Model"]),
            relabel_formula(str(r["Formula"]), centered=centered),
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
        col_widths=[16, 96, 10, 14, 14, 14, 14, 8, 14, 6],
        note="Sequential LR test vs previous row (m_full vs m0).",
    )


def panel_figure_paths():
    fig_root = os.path.join(OUT_DIR, "figures")
    scope_dir = os.path.join(
        fig_root, f"{SCOPE}_{MEASURE}_GLMM_MainEffect_{CONTRAST}"
    )
    items = []
    bands = (BANDS if MEASURE != "pac" else ["slow_gamma"])
    captions = {"theta": "Theta main effect",
                "slow_gamma": "Slow gamma main effect"}
    for band in bands:
        suffix = "slow_gamma" if band == "slow_gamma" else band
        if MEASURE == "pac":
            fname = (f"{SCOPE}_pac_MainEffect_{CONTRAST}_"
                     f"slow_gamma_maineffect.png")
        else:
            fname = (f"{SCOPE}_{MEASURE}_MainEffect_{CONTRAST}_"
                     f"{suffix}_maineffect.png")
        p = os.path.join(scope_dir, fname)
        if os.path.exists(p):
            items.append((captions[band], p))
    return items


def embed_panel_figures(pdf):
    items = panel_figure_paths()
    if not items:
        return False
    body = (
        f"GLMM-predicted P(remembered) as a function of {MEASURE} band value "
        f"({CONTRAST_LABELS[CONTRAST]}). One curve per panel; * marks panels "
        f"surviving FDR-BH within band (q < .05)."
    )
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    for i, (caption, fig_path) in enumerate(items):
        pdf.add_page()
        pdf.section_title(f"2{chr(ord('a') + i) if len(items) > 1 else ''}. "
                          f"{caption}")
        if i == 0:
            pdf.body_text(body)
        pdf.image(fig_path, x=pdf.l_margin, w=page_w)
        pdf.ln(4)
    return True


def render_report():
    pdf = APAReport()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(
        0, 9,
        f"{PHASE_LABEL} | {MEASURE.capitalize()} | {SCOPE_LABELS[SCOPE]}\n"
        f"GLMM main effect ({CONTRAST_LABELS[CONTRAST]})"
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6,
             "ImBalanced Trials | FDR-BH per band on band slope",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.section_title("1. Overview")
    pdf.body_text(model_overview_text())
    pdf.body_text(
        "Subjects BJH032 and BJH033 are folded to a single subject. Panels "
        "with < 4 unique patients or < 30 trials are dropped (do not enter "
        "FDR)."
    )

    has_fig = embed_panel_figures(pdf)

    families = compute_families()

    sec = 3 if has_fig else 2
    for band in BANDS:
        pdf.section_title(f"{sec}. {BAND_LABELS[band]}")
        sec += 1
        for u in UNITS:
            render_panel(pdf, u, band, families[band])

    pdf.output(OUTPUT_PDF)


render_report()

print(f"PDF -> {OUTPUT_PDF}")
print(f"Family CSVs -> {FAMILY_DIR}/")
