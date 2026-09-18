#!/usr/bin/env python
"""Build a PDF post-hoc report per analysis type. Reads the random-slope
comparison + trending panels + per-panel JN/stability CSVs and PNGs and
assembles them into a single PDF in each post_hoc_testing/ directory.

Outputs:
  OUTPUTS/encoding_memory_reports/{analysis_dir}/post_hoc_testing/
    {AnalysisDir}_PostHoc_Report.pdf
"""

from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF


REPO = Path(__file__).resolve().parent.parent
ENC = REPO / "OUTPUTS" / "encoding_memory_reports"

ANALYSES = [
    ("alltrials",                          False, "All trials (stim + no-stim pooled)"),
    ("endogenous_memory_effect",           False, "Endogenous (no-stim only)"),
    ("stim_effect",                        False, "Stim only"),
    ("GLMM_interactions_model_variations", True,  "band_c x StimCond interaction"),
]


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


def num(v, decimals=3):
    if pd.isna(v): return "-"
    if abs(v) > 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.2e}"
    return f"{v:.{decimals}f}"


def p_str(p):
    if pd.isna(p): return "-"
    if p < .001: return "< .001"
    return f"{p:.3f}".lstrip("0")


class APAReport(FPDF):
    def __init__(self, title):
        super().__init__()
        self.title_text = title
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, self.title_text, align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section(self, text):
        self.set_font("Helvetica", "B", 13)
        self.ln(4)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection(self, text):
        self.set_font("Helvetica", "B", 11)
        self.ln(2)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Times", "", 11)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def table(self, headers, rows, col_widths=None, title=None, note=None):
        self.ln(3)
        if title:
            self.set_font("Times", "I", 10)
            self.multi_cell(0, 5, title)
            self.ln(1)
        if col_widths is None:
            avail = self.w - self.l_margin - self.r_margin
            col_widths = [avail / len(headers)] * len(headers)
        x = self.get_x(); total = sum(col_widths)
        self.set_line_width(0.5)
        self.line(x, self.get_y(), x + total, self.get_y()); self.ln(1)
        self.set_font("Times", "B", 9)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, str(h), align="C")
        self.ln()
        self.set_line_width(0.3)
        self.line(x, self.get_y(), x + total, self.get_y()); self.ln(1)
        self.set_font("Times", "", 9)
        for row in rows:
            if self.get_y() > self.h - 35:
                self.set_line_width(0.5)
                self.line(x, self.get_y(), x + total, self.get_y())
                self.add_page()
                self.set_font("Times", "B", 9)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 5, str(h), align="C")
                self.ln()
                self.line(x, self.get_y(), x + total, self.get_y()); self.ln(1)
                self.set_font("Times", "", 9)
            for i, v in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 5, str(v), align=align)
            self.ln()
        self.set_line_width(0.5)
        self.line(x, self.get_y(), x + total, self.get_y())
        self.ln(1)
        if note:
            self.set_font("Times", "I", 8)
            self.multi_cell(0, 4, f"Note. {note}")
            self.ln(2)

    def image_full(self, path):
        if not Path(path).exists():
            self.body(f"[missing figure: {Path(path).name}]")
            return
        w = self.w - self.l_margin - self.r_margin
        self.image(str(path), x=self.l_margin, w=w)
        self.ln(4)


def build_report(analysis, is_interaction, contrast_label):
    out_dir = ENC / analysis / "post_hoc_testing"
    if not out_dir.exists():
        return None
    rs_csv = out_dir / "random_slope_comparison.csv"
    trending_csv = out_dir / "trending_panels.csv"

    title = f"Encoding post-hoc | {analysis} | {contrast_label}"
    pdf = APAReport(title)

    # Cover
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, f"Post-hoc analyses\n{analysis}")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Trial filter / model: {contrast_label}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Encoding phase only.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 1. Random-slope variant
    pdf.section("1. Random-slope variant comparison")
    if is_interaction:
        pdf.body(
            "RI baseline: Accuracy ~ band_c + StimCond + band_c:StimCond + (1|Patient).\n"
            "RS variant : Accuracy ~ band_c + StimCond + band_c:StimCond + "
            "(1+StimCond|Patient).\n"
            "Per-panel LRT (REML=FALSE) tests whether the random StimCond slope "
            "is justified."
        )
    else:
        pdf.body(
            "RI baseline: Accuracy ~ band_c + (1|Patient).\n"
            "RS variant : Accuracy ~ band_c + (1+band_c|Patient).\n"
            "Per-panel LRT tests whether random band_c slope is justified. "
            "Exploratory: random slopes can fail to converge or be singular "
            "with the small number of subjects we have per panel."
        )

    rs_png = out_dir / "random_slope_comparison.png"
    pdf.image_full(rs_png)

    if rs_csv.exists():
        rs = pd.read_csv(rs_csv)
        rows = []
        # Sort by LRT_p ascending (most-improved first)
        rs_sorted = rs.sort_values("LRT_p", na_position="last")
        for _, r in rs_sorted.iterrows():
            rows.append([
                f"{r['measure']} / {pretty(r['unit'])} / {r['band']}",
                f"{int(r['n_trials'])}/{int(r['n_patients'])}",
                num(r["ri_focal_p"]),
                num(r["rs_focal_p"]),
                num(r["dAIC"]),
                p_str(r["LRT_p"]),
                "Y" if r.get("rs_singular", False) is True else (
                    "N" if r.get("rs_singular", False) is False else "?"),
            ])
        pdf.table(
            ["Panel", "n/N", "RI focal p", "RS focal p", "dAIC", "LRT p", "RS sing?"],
            rows,
            col_widths=[70, 22, 22, 22, 20, 20, 18],
            title="All panels: RI vs RS comparison",
            note=("dAIC = AIC(RS) - AIC(RI); negative favors RS. LRT compares "
                  "the random-effects structure. RS sing = Y means lme4 flagged "
                  "the random-effect covariance matrix as near-singular."),
        )

    # 2. Trending panels
    pdf.add_page()
    pdf.section("2. Trending panels (raw p < .10) - focal coefficient")
    if not trending_csv.exists():
        pdf.body("No trending_panels.csv found.")
    else:
        trending = pd.read_csv(trending_csv)
        if trending.empty:
            pdf.body("No trending panels at p < .10.")
        else:
            rows = []
            for _, r in trending.iterrows():
                rows.append([
                    f"{r['measure']} / {pretty(r['unit'])} / {r['band']}",
                    r["scope"],
                    num(r["estimate_OR"]),
                    p_str(r["p_value"]),
                ])
            pdf.table(
                ["Panel", "Scope", "Focal OR", "Raw p"],
                rows,
                col_widths=[80, 50, 30, 30],
                title="Panels considered trending for stability testing",
            )

            # 3. Per-panel JN (if interactions) + stability diagnostic
            for _, r in trending.iterrows():
                u = r["unit"]; m = r["measure"]; b = r["band"]
                pdf.add_page()
                pdf.section(
                    f"3. {pretty(u)} {b.replace('_',' ')} {m}  "
                    f"(p_obs = {r['p_value']:.4f})"
                )

                if is_interaction:
                    pdf.subsection("Johnson-Neyman regions")
                    for mv, lbl in (("ri", "Random intercept only"),
                                    ("rs", "+ random StimCond slope")):
                        png = out_dir / f"jn_{m}_{u}_{b}_{mv}.png"
                        if png.exists():
                            pdf.body(f"{lbl}:")
                            pdf.image_full(png)

                pdf.subsection("Effect stability: LOSO + subject deltas + downsample")
                png = out_dir / f"stability_{m}_{u}_{b}_diagnostics.png"
                pdf.image_full(png)

    out_pdf = out_dir / f"{analysis}_PostHoc_Report.pdf"
    pdf.output(str(out_pdf))
    print(f"PDF -> {out_pdf}")
    return out_pdf


def main():
    for analysis, is_int, label in ANALYSES:
        build_report(analysis, is_int, label)


if __name__ == "__main__":
    main()
