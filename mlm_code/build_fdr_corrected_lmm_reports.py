#!/usr/bin/env python
"""
FDR-corrected LMM reports (no StimCond): primary memory-modulation tests.

Two reports per phase:
  A: BLA-HPC subregions      (Power BLA/CA/DG/HPC; Coh/PAC BLA-CA, BLA-DG, BLA-HPC)
  B: MTL network             (Power HPC/CA/DG/EC/PRC;
                              Coh/PAC EC-HPC, HPC-PRC, CA-EC, CA-PRC)

Both moderators are tested with no StimCond:
  Continuous : feature ~ mem_mod_z + (1|Patient)
  Quad       : feature ~ QuadResponderGroup + (1|Patient)

FDR-BH applied per family; family = (report x measure x moderator-type x band).
Theta and slow gamma are corrected separately. PAC has only slow gamma.

Usage:
  python build_fdr_corrected_lmm_reports.py <retrieval|encoding> <A|B>
"""

import os
import sys

import numpy as np
import pandas as pd
from fpdf import FPDF


# ---------------------------------------------------------------------------
# CLI / paths
# ---------------------------------------------------------------------------

if len(sys.argv) < 3:
    print("Usage: python build_fdr_corrected_lmm_reports.py <retrieval|encoding> <A|B>")
    sys.exit(1)

PHASE = sys.argv[1]
REPORT = sys.argv[2].upper()
assert PHASE in ("retrieval", "encoding")
assert REPORT in ("A", "B")

REPO_ROOT = "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
OUT_DIR = os.path.join(REPO_ROOT, "OUTPUTS", "retrieval_memory_reports")
STATS_DIR = os.path.join(OUT_DIR, "stats")

if PHASE == "retrieval":
    CSV_DIR = os.path.join(
        REPO_ROOT, "OUTPUTS", "retrieval_memory_reports",
        "h1b_subregions_quad_lmm_imbalanced_retrieval_mlm_csvs",
    )
else:
    CSV_DIR = os.path.join(
        REPO_ROOT, "OUTPUTS", "encoding_memory_reports", "stats",
        "h2a_subregions_quad_lmm_imbalanced_encoding_mlm_csvs",
    )

# Sub-folder where this report's family-level FDR CSVs will be written.
FAMILY_DIR = os.path.join(STATS_DIR, f"fdr_report{REPORT}_{PHASE}_imbalanced")
os.makedirs(FAMILY_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Report definitions
# ---------------------------------------------------------------------------

REPORT_LABELS = {
    "A": "BLA-HPC Subregions",
    "B": "MTL Network with HPC Subregions",
}
PHASE_LABEL = "Retrieval" if PHASE == "retrieval" else "Encoding"
PHASE_HYPO = "Hypothesis 1b" if PHASE == "retrieval" else "Hypothesis 2a"

# Coverage tables. retrieval BLA-DG PAC dropped (only 4 patients in raw input
# -> primary models could not be fit at MIN_PATIENTS=5).
if REPORT == "A":
    POWER_REGIONS = ["BLA", "CA", "DG", "HPC"]
    COH_PAIRS = ["BLA_CA", "BLA_DG", "BLA_HPC"]
    if PHASE == "retrieval":
        PAC_PAIRS = ["BLA_CA", "BLA_HPC"]
    else:
        PAC_PAIRS = ["BLA_CA", "BLA_DG", "BLA_HPC"]
else:
    POWER_REGIONS = ["HPC", "CA", "DG", "EC", "PRC"]
    COH_PAIRS = ["EC_HPC", "HPC_PRC", "CA_EC", "CA_PRC"]
    PAC_PAIRS = ["EC_HPC", "HPC_PRC", "CA_EC", "CA_PRC"]

BANDS = {"theta": "Theta (4-8 Hz)", "slow_gamma": "Slow Gamma (30-55 Hz)"}
PAC_BAND = "slow_gamma"
PAC_BAND_LABEL = "SG PAC (30-50 Hz)"

REPORT_TITLE = (
    f"{PHASE_HYPO} - Quad Responder & Mem-Mod (LMM, FDR-BH) - "
    f"Report {REPORT}: {REPORT_LABELS[REPORT]} - {PHASE_LABEL} - ImBalanced Trials"
)
OUTPUT_PDF = os.path.join(
    OUT_DIR,
    f"{PHASE_HYPO.replace(' ', '_')}_Report{REPORT}_"
    f"{REPORT_LABELS[REPORT].replace(' ', '_').replace('-', '')}_"
    f"LMM_FDRcorrected_{PHASE_LABEL}_ImBalanced_Trials.pdf",
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def display_region(r):
    return r.replace("_", "-")


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
    if term.startswith("QuadResponderGroup"):
        return "Responder Status: " + term.replace("QuadResponderGroup", "")
    return term


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def csv_path(modality, unit, band, kind):
    """kind in {primaryMemMod, primaryRespGroup}, suffix is _coefs or _anova."""
    return os.path.join(CSV_DIR, f"{modality}_{unit}_{band}_{kind}")


def load_pair(modality, unit, band, kind):
    coefs_p = csv_path(modality, unit, band, f"{kind}_coefs.csv")
    anova_p = csv_path(modality, unit, band, f"{kind}_anova.csv")
    if not (os.path.exists(coefs_p) and os.path.exists(anova_p)):
        return None, None
    return pd.read_csv(coefs_p), pd.read_csv(anova_p)


def omnibus_p(anova_df):
    """Return the m1-vs-m0 LR test p-value (omnibus test of the predictor).

    Anova layout (sequential build): m0 row has NA p, m1 row has p vs m0.
    """
    if anova_df is None or anova_df.empty:
        return np.nan
    row = anova_df[anova_df["Model"] == "m1"]
    if row.empty:
        return np.nan
    return float(row["p.value"].iloc[0])


# ---------------------------------------------------------------------------
# FDR (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def bh_fdr(pvals):
    """Return BH-adjusted q-values aligned to the input order.

    NaNs are passed through unchanged and excluded from the correction.
    """
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
    # Enforce monotonicity from the largest rank down.
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_mono = np.minimum(q_mono, 1.0)
    q_full = np.empty(n, dtype=float)
    q_full[order] = q_mono
    out[valid] = q_full
    return out


# ---------------------------------------------------------------------------
# Family assembly
# ---------------------------------------------------------------------------

MOD_TYPES = [
    ("primaryMemMod",    "Continuous (mem_mod_z)"),
    ("primaryRespGroup", "Quad Responder"),
]


def assemble_family(modality, units, band, kind):
    """Collect omnibus p-values for one family across panels.

    Returns (rows, df) where df is a per-panel DataFrame with columns:
    [panel, modality, unit, band, kind, omnibus_p_raw, q_FDR, sig_FDR].
    """
    rows = []
    for unit in units:
        coefs, anova = load_pair(modality, unit, band, kind)
        if anova is None:
            rows.append({
                "panel": f"{modality} {display_region(unit)} {band}",
                "modality": modality,
                "unit": unit,
                "band": band,
                "kind": kind,
                "omnibus_p_raw": np.nan,
                "n_patients": np.nan,
            })
            continue
        # Best-effort patient count: pull from coefs std.error column count
        # not available; we leave n_patients blank (row count is trial-level).
        rows.append({
            "panel": f"{modality} {display_region(unit)} {band}",
            "modality": modality,
            "unit": unit,
            "band": band,
            "kind": kind,
            "omnibus_p_raw": omnibus_p(anova),
        })
    df = pd.DataFrame(rows)
    df["q_FDR"] = bh_fdr(df["omnibus_p_raw"].to_numpy())
    df["sig_FDR"] = df["q_FDR"].apply(
        lambda q: "" if pd.isna(q)
        else "***" if q < .001
        else "**" if q < .01
        else "*" if q < .05
        else "+" if q < .10
        else ""
    )
    return df


def family_id(modality, kind, band):
    mod_short = {"primaryMemMod": "Cont", "primaryRespGroup": "Quad"}[kind]
    return f"{modality}_{mod_short}_{band}"


def build_all_families():
    """Run FDR per family; return dict {family_id: family_df} and write CSVs."""
    families = {}
    for kind, _ in MOD_TYPES:
        # Power
        for band in BANDS:
            fid = family_id("power", kind, band)
            families[fid] = assemble_family("power", POWER_REGIONS, band, kind)
        # Coherence
        for band in BANDS:
            fid = family_id("coherence", kind, band)
            families[fid] = assemble_family("coherence", COH_PAIRS, band, kind)
        # PAC (slow gamma only)
        fid = family_id("pac", kind, PAC_BAND)
        families[fid] = assemble_family("pac", PAC_PAIRS, PAC_BAND, kind)

    # Persist family-level CSVs.
    for fid, df in families.items():
        out_path = os.path.join(FAMILY_DIR, f"{fid}.csv")
        df.to_csv(out_path, index=False)
    return families


# ---------------------------------------------------------------------------
# PDF rendering
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


def coef_table_rows(df):
    rows = []
    for _, r in df.iterrows():
        rows.append([
            pretty_term(str(r["term"])),
            num_str(r["estimate"]),
            ci_str(r["conf.low"], r["conf.high"]),
            f"{r['statistic']:.3f}" if pd.notna(r.get("statistic")) else "",
            p_str(r["p.value"]) if "p.value" in df.columns else "",
            sig_str(r["p.value"]) if "p.value" in df.columns else "",
        ])
    return rows


def anova_table_rows(df, q_fdr=None):
    """Render the sequential-build table.

    q_fdr is the FDR-corrected omnibus q-value to attach to the m1 row.
    """
    rows = []
    for _, r in df.iterrows():
        is_m1 = str(r["Model"]) == "m1"
        rows.append([
            str(r["Model"]),
            (str(r["Formula"])
                .replace("QuadResponderGroup", "Responder Status")
                .replace("mem_mod_z", "Memory Modulation Z")),
            f"{int(r['npar'])}" if not pd.isna(r["npar"]) else "-",
            fmt_fit(r["AIC"], 1),
            fmt_fit(r["BIC"], 1),
            fmt_fit(r["logLik"], 1),
            fmt_fit(r["Chisq"], 2),
            fmt_fit(r["Df"], 0),
            p_str(r["p.value"]) if not pd.isna(r["p.value"]) else "-",
            (p_str(q_fdr) if (is_m1 and q_fdr is not None) else "-"),
            sig_str(q_fdr) if (is_m1 and q_fdr is not None) else "",
        ])
    return rows


def render_panel(pdf, modality, unit, band, band_label, families, kind, mod_label):
    """Render one panel for one moderator type only."""
    coefs, anova = load_pair(modality, unit, band, kind)
    if coefs is None:
        return
    fid = family_id(modality, kind, band)
    fam_df = families[fid]
    row = fam_df[fam_df["unit"] == unit]
    q_val = row["q_FDR"].iloc[0] if not row.empty else np.nan
    p_raw = row["omnibus_p_raw"].iloc[0] if not row.empty else np.nan

    marker = sig_str(q_val)
    title_prefix = (f"{modality.capitalize()} - {display_region(unit)} - "
                    f"{band_label}")
    pdf.subsection_title(title_prefix)
    pdf.subsubsection_title(
        f"Moderator: {mod_label}    |    "
        f"omnibus LR p = {p_str(p_raw)}    "
        f"q (FDR-BH) = {p_str(q_val)} {marker}"
    )
    pdf.apa_table(
        f"Table. Fixed-effect Coefficients ({mod_label})",
        ["Predictor", "Estimate", "95% CI", "t", "p", ""],
        coef_table_rows(coefs),
        col_widths=[78, 18, 38, 14, 14, 8],
        note=("Reference: NonResp." if kind == "primaryRespGroup"
              else "mem_mod_z = z-scored avg_stim_dprime_diff (between-patient)."),
    )
    pdf.apa_table(
        f"Table. Sequential Model Build ({mod_label})",
        ["Model", "Formula", "npar", "AIC", "BIC", "logLik",
         "Chisq", "Df", "p", "q (FDR)", ""],
        anova_table_rows(anova, q_fdr=q_val),
        col_widths=[12, 80, 10, 13, 13, 13, 12, 8, 14, 14, 6],
        note=("Sequential model build (LR test vs previous row); the m1 row "
              "is the omnibus test of the moderator. q (FDR) is "
              "Benjamini-Hochberg-adjusted within the family "
              "(measure x moderator x band). "
              "* < .05, ** < .01, *** < .001, + < .10."),
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_report():
    families = build_all_families()
    pdf = APAReport()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(
        0, 9,
        f"{PHASE_HYPO} - Report {REPORT}: {REPORT_LABELS[REPORT]}\n"
        f"LMM (no StimCond) - FDR-BH corrected"
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"{PHASE_LABEL} Phase - ImBalanced Trials",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.section_title("1. Overview")
    pdf.body_text(
        "Per-panel LMMs test whether the neural feature varies with patient "
        "memory modulation, ignoring trial-level stim condition. Two "
        "moderators are run separately for every panel:"
    )
    pdf.body_text(
        "  Continuous: feature ~ mem_mod_z + (1 | Patient)\n"
        "  Quad      : feature ~ QuadResponderGroup + (1 | Patient)"
    )
    pdf.body_text(
        "mem_mod_z = z-scored avg_stim_dprime_diff (stim - no-stim d', "
        "patient-level). QuadResponderGroup is a 4-level factor with "
        "Non-responders as reference (Anti / Non=ref / Moderate / Strong). "
        "Subjects BJH032 and BJH033 are folded to a single subject (54 unique "
        "participants)."
    )
    pdf.body_text(
        "Models fit with REML; the sequential model build refits with ML for "
        "the m0->m1 likelihood-ratio test, which is the omnibus p-value "
        "subjected to FDR within each family."
    )

    pdf.subsection_title("FDR Family Definition")
    pdf.body_text(
        "Each family is corrected independently with Benjamini-Hochberg. "
        "Family = (report) x (measure: power / coherence / PAC) x "
        "(moderator type: continuous / quad) x (band: theta / slow gamma; "
        "PAC: slow gamma only). Theta and slow gamma are corrected separately "
        "to minimise the multiple-comparison penalty within each measure."
    )

    pdf.subsection_title("Coverage in this report")
    cov = []
    cov.append(f"Power regions (n = {len(POWER_REGIONS)}): "
               + ", ".join(POWER_REGIONS))
    cov.append(f"Coherence pairs (n = {len(COH_PAIRS)}): "
               + ", ".join(display_region(p) for p in COH_PAIRS))
    cov.append(f"PAC pairs (n = {len(PAC_PAIRS)}): "
               + ", ".join(display_region(p) for p in PAC_PAIRS))
    if REPORT == "B":
        cov.append("Note: PAC EC-HPC is used as the EC->HPC directional pair "
                   "(HPC->EC was not computed in upstream PAC pipelines).")
    if REPORT == "A" and PHASE == "retrieval":
        cov.append("Note: BLA-DG PAC is dropped from retrieval (only 4 patients "
                   "have BLA-DG PAC data; below MIN_PATIENTS=5 threshold).")
    for line in cov:
        pdf.body_text(line)

    # Two top-level sections by moderator: Continuous first, then Quad.
    sec_top = 1
    for kind, mod_label in MOD_TYPES:
        sec_top += 1
        title_word = ("Continuous Memory Modulation (mem_mod_z)"
                      if kind == "primaryMemMod"
                      else "Quad Responder Group")
        pdf.add_page()
        pdf.section_title(f"{sec_top}. {title_word}")

        sub = 0
        sub += 1
        pdf.subsection_title(f"{sec_top}.{sub} Power")
        for reg in POWER_REGIONS:
            for band, label in BANDS.items():
                render_panel(pdf, "power", reg, band, label,
                             families, kind, mod_label)

        sub += 1
        pdf.subsection_title(f"{sec_top}.{sub} Coherence")
        for pair in COH_PAIRS:
            for band, label in BANDS.items():
                render_panel(pdf, "coherence", pair, band, label,
                             families, kind, mod_label)

        sub += 1
        pdf.subsection_title(f"{sec_top}.{sub} PAC ({PAC_BAND_LABEL})")
        for pair in PAC_PAIRS:
            render_panel(pdf, "pac", pair, PAC_BAND, PAC_BAND_LABEL,
                         families, kind, mod_label)

    pdf.output(OUTPUT_PDF)
    print(f"PDF -> {OUTPUT_PDF}")
    print(f"Family CSVs -> {FAMILY_DIR}/")


if __name__ == "__main__":
    build_report()
