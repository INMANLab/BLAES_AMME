#!/usr/bin/env python
"""Compact APA-style sensitivity-summary table for trending main-effect
GLMMs in the stim and endogenous (no-stim) contrasts.

Reads from the post-hoc stability CSVs that already exist under
  OUTPUTS/encoding_memory_reports/{stim_effect,endogenous_memory_effect}
  /post_hoc_testing/
and assembles one row per trending panel with:
  - panel label (measure / pair-or-region / band)
  - contrast (stim or endogenous)
  - n trials / N subjects
  - focal raw p, q_FDR (from the existing trending_panels.csv)
  - LOSO Wald p range, % of LOSO refits below .05
  - Downsample median OR (or beta), % below .05
  - Subject-delta direction match (sign-matched fraction)

Output:
  OUTPUTS/encoding_memory_reports/main_effect_sensitivity_summary/
    Main_Effect_Sensitivity_Summary_StimAndEndogenous.pdf
    sensitivity_summary_long.csv  (the underlying tidy data)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF


REPO = Path(__file__).resolve().parent.parent
ENC = REPO / "OUTPUTS" / "encoding_memory_reports"

ANALYSES = [
    # (analysis_dir, label, contrast_token_in_fdr_family_paths)
    ("stim_effect",              "Stim only",            "stim"),
    ("endogenous_memory_effect", "Endogenous (no-stim)", "nostim"),
]


def lookup_q_fdr(measure, scope, unit, band, contrast):
    """Look up the BH-FDR-adjusted q for the focal band coefficient in the
    appropriate main-effect FDR family CSV."""
    fdr_csv = (ENC / "stats" /
               f"fdr_{measure}_{scope}_GLMM_MainEffect_{contrast}_encoding" /
               f"{measure}_{scope}_GLMM_MainEffect_{contrast}_{band}_band_c.csv")
    if not fdr_csv.exists():
        return float("nan")
    fam = pd.read_csv(fdr_csv)
    row = fam[fam["unit"] == unit]
    if row.empty:
        return float("nan")
    val = row["q_FDR"].iloc[0]
    return float(val) if pd.notna(val) else float("nan")

OUT_DIR = ENC / "main_effect_sensitivity_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Formatting helpers ────────────────────────────────────────────────────

def pretty(name):
    parts = []
    for p in str(name).split("_"):
        if p == "ALLHPC":
            parts.append("HPC")
        elif p == "HPC":
            parts.append("SUB")
        else:
            parts.append(p)
    return "-".join(parts)


def pretty_band(b):
    return "theta" if b == "theta" else "slow gamma"


def p_str(p):
    if pd.isna(p):
        return "-"
    if p < .001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def or_str(v):
    if pd.isna(v):
        return "-"
    if abs(v) > 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.2e}"
    return f"{v:.2f}"


def pct(num, denom):
    if denom == 0:
        return "-"
    return f"{num}/{denom} ({100 * num / denom:.0f}%)"


# ── Compute one row per trending panel ────────────────────────────────────

def summarize_panel(analysis_dir, contrast, trending_row):
    """Return a dict of summary stats for a single trending panel."""
    posthoc = ENC / analysis_dir / "post_hoc_testing"
    m = trending_row["measure"]
    u = trending_row["unit"]
    b = trending_row["band"]
    scope = trending_row["scope"]
    loso_p = posthoc / f"stability_{m}_{u}_{b}_loso.csv"
    ds_p   = posthoc / f"stability_{m}_{u}_{b}_downsample.csv"
    delta_p = posthoc / f"stability_{m}_{u}_{b}_deltas.csv"

    q_fdr = lookup_q_fdr(m, scope, u, b, contrast)

    summary = {
        "analysis_dir": analysis_dir,
        "measure": m,
        "scope": scope,
        "unit": u,
        "band": b,
        "estimate_OR": trending_row.get("estimate_OR", float("nan")),
        "p_value": trending_row["p_value"],
        "q_FDR": q_fdr,
        "n_trials": np.nan,
        "n_subjects": np.nan,
        "loso_p_min": np.nan,
        "loso_p_max": np.nan,
        "loso_n_below": 0,
        "loso_n_total": 0,
        "ds_n_below": 0,
        "ds_n_total": 0,
        "ds_median_OR": np.nan,
        "delta_n_match": 0,
        "delta_n_total": 0,
    }

    # LOSO
    if loso_p.exists():
        loso = pd.read_csv(loso_p)
        full = loso[loso["excluded"] == "<none>"].iloc[0]
        summary["n_trials"] = int(full["n_trials"])
        summary["n_subjects"] = int(full["n_patients"])
        rest = loso[loso["excluded"] != "<none>"].dropna(subset=["p"])
        if len(rest):
            summary["loso_p_min"] = float(rest["p"].min())
            summary["loso_p_max"] = float(rest["p"].max())
            summary["loso_n_below"] = int((rest["p"] < 0.05).sum())
            summary["loso_n_total"] = len(rest)

    # Downsample
    if ds_p.exists():
        ds = pd.read_csv(ds_p).dropna(subset=["OR", "p"])
        if len(ds):
            summary["ds_n_below"] = int((ds["p"] < 0.05).sum())
            summary["ds_n_total"] = len(ds)
            summary["ds_median_OR"] = float(ds["OR"].median())

    # Subject deltas: direction match = same sign as full-sample focal slope
    if delta_p.exists():
        deltas = pd.read_csv(delta_p)
        # For main effects, the relevant column is "delta"
        # (mean band | remembered  -  mean band | forgotten)
        col = "delta" if "delta" in deltas.columns else None
        if col is not None:
            d = deltas[col].dropna()
            if len(d) and not pd.isna(summary["estimate_OR"]):
                # Direction match: if OR > 1, expect positive deltas
                expect_positive = summary["estimate_OR"] > 1
                if expect_positive:
                    match = int((d > 0).sum())
                else:
                    match = int((d < 0).sum())
                summary["delta_n_match"] = match
                summary["delta_n_total"] = len(d)
    return summary


def is_bla_unit(unit):
    """Return True if the unit label refers to a BLA region or any pair
    that includes BLA. We split on '_' so 'BLA', 'BLA_DG', 'BLA_PRC',
    'BLA_ALLHPC', 'BLA_EC', 'BLA_HPC' all match; 'PRC' or 'DG' do not."""
    return "BLA" in str(unit).split("_")


def gather_all_summaries():
    rows = []
    for analysis_dir, _, contrast in ANALYSES:
        tp = ENC / analysis_dir / "post_hoc_testing" / "trending_panels.csv"
        if not tp.exists():
            continue
        trending = pd.read_csv(tp)
        trending = trending[~trending["unit"].apply(is_bla_unit)]
        for _, r in trending.iterrows():
            rows.append(summarize_panel(analysis_dir, contrast, r))
    return pd.DataFrame(rows)


# ── APA report class (mirrors the other PDFs) ─────────────────────────────

REPORT_TITLE = ("Encoding | Main-effect GLMM sensitivity summary | "
                "Stim + Endogenous (no-stim) contrasts")


class APAReport(FPDF):
    def __init__(self):
        super().__init__(orientation="L")  # landscape for the wide table
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, REPORT_TITLE, align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section(self, t):
        self.set_font("Helvetica", "B", 13)
        self.ln(3); self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def subsection(self, t):
        self.set_font("Helvetica", "B", 11)
        self.ln(2); self.cell(0, 7, t, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def body(self, t):
        self.set_font("Times", "", 10)
        self.multi_cell(0, 5.2, t); self.ln(1)

    def apa_table(self, title, headers, rows, col_widths, note=None,
                  bold_flags=None):
        """Render an APA-style table. bold_flags is an optional list of
        bools the same length as rows; True rows are rendered bold."""
        self.ln(2)
        self.set_font("Times", "I", 10)
        self.multi_cell(0, 5, title); self.ln(0.5)
        x = self.get_x(); total = sum(col_widths)
        self.set_line_width(0.5)
        self.line(x, self.get_y(), x + total, self.get_y()); self.ln(1)
        self.set_font("Times", "B", 8)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, str(h), align="C")
        self.ln()
        self.set_line_width(0.3)
        self.line(x, self.get_y(), x + total, self.get_y()); self.ln(1)
        for ridx, row in enumerate(rows):
            bold = bold_flags is not None and bold_flags[ridx]
            self.set_font("Times", "B" if bold else "", 8)
            for i, v in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 4.6, str(v), align=align)
            self.ln()
        self.set_line_width(0.5)
        self.line(x, self.get_y(), x + total, self.get_y()); self.ln(1)
        if note:
            self.set_font("Times", "I", 7)
            self.multi_cell(0, 3.6, f"Note. {note}"); self.ln(1)


# ── Build the PDF ─────────────────────────────────────────────────────────

def build_pdf(summaries):
    pdf = APAReport()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8,
        "Sensitivity analyses for trending main-effect GLMMs\n"
        "Encoding | Stim and Endogenous (no-stim) contrasts")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Compact summary across all panels with raw p < .10",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.section("Overview")
    pdf.body(
        "For every encoding main-effect GLMM panel that trended at raw "
        "p < .10 in the stim-only or endogenous (no-stim) contrast, we ran "
        "three sensitivity analyses: leave-one-subject-out (LOSO) refits, "
        "100 balanced-subsample refits (downsampling the majority class "
        "within each subject), and inspection of per-subject memory "
        "deltas. Panels involving BLA (BLA alone or any BLA-X pair) are "
        "excluded from this summary. The tables below compress the "
        "remaining analyses to one row per panel. Rows in bold mark "
        "panels that also survived FDR-BH correction within their "
        "band/scope/measure family at q < .05; all bold rows are the "
        "headline findings reported in detail in the main text. Non-bold "
        "rows are trending findings that did not survive FDR; their "
        "sensitivity profile is reported here for transparency."
    )

    # Build one table per contrast for readability.
    headers = [
        "Panel  (measure / pair-or-region / band)",
        "Scope",
        "n trials / N subj",
        "Focal OR",
        "Raw p",
        "q_FDR",
        "LOSO p range",
        "LOSO p<.05",
        "DS median OR",
        "DS p<.05",
        "Delta dir-match",
    ]
    col_widths = [72, 26, 22, 14, 12, 12, 26, 22, 22, 22, 22]
    note_text = (
        "LOSO p range = [min, max] of focal-coefficient Wald p across "
        "leave-one-subject-out refits (one refit per subject in the panel; "
        "lower values indicate stronger effect in the absence of an "
        "influential subject, upper values indicate fragility). "
        "LOSO p<.05 = fraction of LOSO refits whose focal p stays below "
        "0.05. DS = balanced-subsample refits "
        "(100 iterations; majority class within each subject downsampled "
        "to match the minority class); DS median OR summarizes the focal "
        "coefficient distribution under balancing. Delta dir-match = "
        "fraction of subjects whose within-subject delta "
        "(mean band|remembered - mean band|forgotten) matches the "
        "direction of the full-sample focal coefficient. "
        "Bold rows survived FDR-BH within their band/scope/measure "
        "family (q < .05)."
    )

    for analysis_dir, contrast_label, _ in ANALYSES:
        sub = summaries[summaries["analysis_dir"] == analysis_dir].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("p_value").reset_index(drop=True)

        rows = []; bold_flags = []
        for _, s in sub.iterrows():
            panel_label = (f"{s['measure']} / {pretty(s['unit'])} / "
                           f"{pretty_band(s['band'])}")
            n_str = (f"{int(s['n_trials'])}/{int(s['n_subjects'])}"
                     if not pd.isna(s["n_trials"]) else "-")
            loso_range = (f"[{s['loso_p_min']:.3f}, {s['loso_p_max']:.3f}]"
                          if not pd.isna(s["loso_p_min"]) else "-")
            rows.append([
                panel_label,
                s["scope"],
                n_str,
                or_str(s["estimate_OR"]),
                p_str(s["p_value"]),
                p_str(s["q_FDR"]),
                loso_range,
                pct(s["loso_n_below"], s["loso_n_total"]),
                or_str(s["ds_median_OR"]),
                pct(s["ds_n_below"], s["ds_n_total"]),
                pct(s["delta_n_match"], s["delta_n_total"]),
            ])
            bold_flags.append(
                (not pd.isna(s["q_FDR"])) and float(s["q_FDR"]) < 0.05)

        pdf.subsection(f"{contrast_label}  ({len(rows)} trending panels)")
        pdf.apa_table(
            f"Sensitivity summary for trending main-effect GLMMs - "
            f"{contrast_label}",
            headers, rows, col_widths=col_widths,
            note=note_text if analysis_dir == ANALYSES[0][0] else None,
            bold_flags=bold_flags,
        )

    out_pdf = OUT_DIR / "Main_Effect_Sensitivity_Summary_StimAndEndogenous.pdf"
    pdf.output(str(out_pdf))
    print(f"PDF -> {out_pdf}")


def main():
    summaries = gather_all_summaries()
    # Save the tidy long-form CSV alongside the PDF for paper-ready use.
    summaries.to_csv(OUT_DIR / "sensitivity_summary_long.csv", index=False)
    build_pdf(summaries)
    print(f"\nLong-form CSV -> {OUT_DIR / 'sensitivity_summary_long.csv'}")


if __name__ == "__main__":
    main()
