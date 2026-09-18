#!/usr/bin/env python
"""Build the dedicated EC-PRC PAC random-slope-comparison report.

Reads CSVs produced by ec_prc_pac_rs_compare.R and the existing JN-v2 PNGs
from GLMM_interactions_model_variations/post_hoc_testing/, then assembles
a clean PDF with:
  1. Fixed-effect coefficient table (RI)
  2. Sequential model build (RI)
  3. Fixed-effect coefficient table (RS: + random StimCond slope)
  4. Sequential model build (RS)
  5. RI vs RS comparison: AIC/BIC, LRT, variance components
  6. Per-subject empirical signals figure (justifies RS-StimCond, not RS-band)
  7. Johnson-Neyman: RI vs RS panels with explanation
  8. Practical interpretation

Output:
  OUTPUTS/encoding_memory_reports/EC_PRC_PAC_random_slope_report/
    EC_PRC_PAC_RandomSlope_Report.pdf
    figures/ -- supporting PNGs
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "OUTPUTS" / "encoding_memory_reports" / "EC_PRC_PAC_random_slope_report"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# JN PNGs already produced by posthoc_johnson_neyman_v2.R
POSTHOC = REPO / "OUTPUTS" / "encoding_memory_reports" / \
    "GLMM_interactions_model_variations" / "post_hoc_testing"
JN_RI = POSTHOC / "jn_pac_EC_PRC_slow_gamma_ri.png"
JN_RS = POSTHOC / "jn_pac_EC_PRC_slow_gamma_rs.png"


# ── Formatting helpers (match build_h1abc_full_report.py) ──────────────────

def num(v, decimals=4):
    if pd.isna(v): return "-"
    if abs(v) > 1e4 or (abs(v) < 1e-4 and v != 0):
        return f"{v:.2e}"
    return f"{v:.{decimals}f}"


def ci_str(lo, hi):
    if pd.isna(lo) or pd.isna(hi): return ""
    return f"[{num(lo)}, {num(hi)}]"


def p_str(p):
    if isinstance(p, str): return p
    if pd.isna(p): return "-"
    if p < .001: return "< .001"
    return f"{p:.3f}".lstrip("0")


def sig_str(p):
    if isinstance(p, str) or pd.isna(p): return ""
    if p < .001: return "***"
    if p < .01: return "**"
    if p < .05: return "*"
    if p < .10: return "+"
    return ""


def fmt(v, decimals=2):
    if pd.isna(v): return "-"
    return f"{float(v):.{decimals}f}"


def pretty_term(term):
    if term == "(Intercept)":      return "(Intercept)"
    if term == "band_c":           return "band"
    if term == "StimCond_num":     return "StimCond (stim vs nostim)"
    if term == "band_c:StimCond_num": return "band:StimCond (stim vs nostim)"
    return term


def relabel_formula(formula):
    return (str(formula)
            .replace("StimCond_num", "StimCond"))


# ── PDF class (same style as the existing reports) ─────────────────────────

REPORT_TITLE = ("Encoding | EC-PRC PAC slow gamma | Random intercept vs "
                "+ random StimCond slope")


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

    def section(self, text):
        self.set_font("Helvetica", "B", 13)
        self.ln(4); self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection(self, text):
        self.set_font("Helvetica", "B", 11)
        self.ln(2); self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Times", "", 11)
        self.multi_cell(0, 5.5, text); self.ln(1)

    def apa_table(self, title, headers, rows, col_widths=None, note=None):
        self.ln(3)
        self.set_font("Times", "I", 10)
        self.multi_cell(0, 5, title); self.ln(1)
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
            for i, v in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 5, str(v), align=align)
            self.ln()
        self.set_line_width(0.5)
        self.line(x, self.get_y(), x + total, self.get_y()); self.ln(1)
        if note:
            self.set_font("Times", "I", 8)
            self.multi_cell(0, 4, f"Note. {note}"); self.ln(2)

    def image_full(self, path, max_h=None):
        if not Path(path).exists():
            self.body(f"[missing figure: {Path(path).name}]"); return
        w = self.w - self.l_margin - self.r_margin
        if max_h is not None:
            self.image(str(path), x=self.l_margin, w=w, h=max_h)
        else:
            self.image(str(path), x=self.l_margin, w=w)
        self.ln(4)


# ── Comparison figure: per-subject empirical signals ───────────────────────

def make_subject_figure(per_subj, vc):
    """Four-panel figure (2x2):
       A) Parallel-coordinates plot: one line per subject connecting
          nostim memory rate -> stim memory rate. Lines fan out / cross
          when subjects have heterogeneous stim responses.
       B) Caterpillar plot of the RS model's per-subject stim-effect BLUPs
          with 95% CIs (model-based, partial-pooled shrinkage estimates).
       C) Caterpillar plot of NAIVE per-subject stim effects (model-free,
          one GLM per subject, no shrinkage). Directly comparable to B;
          confirms the BLUP pattern isn't a random-effect artifact.
       D) Per-subject band-Accuracy correlations -- shows subjects do NOT
          differ in their band slope (justifies omitting a random slope on
          band).
    """
    blups = pd.read_csv(OUT_DIR / "per_subject_blups_rs.csv")
    naive = pd.read_csv(OUT_DIR / "per_subject_naive_stim_effect.csv")
    pop = pd.read_csv(OUT_DIR / "population_stim_effect_rs.csv").iloc[0]

    # We sort all caterpillars by the SAME subject ordering -- BLUP order --
    # so visual comparison across panels is straightforward.
    blups_sorted = blups.sort_values("subject_stim_effect").reset_index(drop=True)
    subj_order = blups_sorted["Patient"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    axes = axes.flatten()

    # ── Panel A: parallel-coordinates of memory rate ────────────────────
    ax = axes[0]
    ps = per_subj.copy()
    for _, r in ps.iterrows():
        delta = r["rem_rate_stim"] - r["rem_rate_nostim"]
        color = "#a0a0a0" if delta < 0 else "#2b2b2b"
        ax.plot([0, 1], [r["rem_rate_nostim"], r["rem_rate_stim"]],
                "-o", color=color, lw=1.6, ms=5, alpha=0.85)
        ax.annotate(r["Patient"], xy=(1.02, r["rem_rate_stim"]),
                    fontsize=7.5, color="0.35", va="center")
    nostim_mean = float(ps["rem_rate_nostim"].mean())
    stim_mean   = float(ps["rem_rate_stim"].mean())
    ax.plot([0, 1], [nostim_mean, stim_mean], "-o",
            color="#c0392b", lw=3, ms=9, label="Group mean", zorder=20)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No-stim", "Stim"], fontsize=11)
    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("P(remembered) within condition",
                  fontsize=11, fontweight="bold")
    n_up   = int((ps["rem_rate_stim"] > ps["rem_rate_nostim"]).sum())
    n_down = int((ps["rem_rate_stim"] < ps["rem_rate_nostim"]).sum())
    ax.set_title(
        f"A. Raw memory rate per subject\n"
        f"{n_up} stim>nostim;  {n_down} stim<nostim  -- lines fan out, not parallel",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2, axis="y")

    # ── Panel B: BLUP caterpillar (model-based, with shrinkage) ─────────
    ax = axes[1]
    b = blups.set_index("Patient").loc[subj_order].reset_index()
    y = np.arange(len(b))
    xerr_lo = b["subject_stim_effect"] - b["ci_lo"]
    xerr_hi = b["ci_hi"] - b["subject_stim_effect"]
    colors = ["#a0a0a0" if v < 0 else "#2b2b2b"
              for v in b["subject_stim_effect"]]
    for i, c in enumerate(colors):
        ax.errorbar(b["subject_stim_effect"].iloc[i], y[i],
                    xerr=[[xerr_lo.iloc[i]], [xerr_hi.iloc[i]]],
                    fmt="o", color=c, ecolor="0.55",
                    capsize=3, ms=6, lw=1.0)
    ax.axvline(0, color="0.4", lw=1, ls="-")
    ax.axvline(float(pop["fixed_b2"]),
               color="#c0392b", ls="--", lw=1.8,
               label=f"Population mean = {pop['fixed_b2']:+.2f}")
    sd = float(pop["sigma_StimCond"])
    ax.axvspan(pop["fixed_b2"] - sd, pop["fixed_b2"] + sd,
               color="#c0392b", alpha=0.10,
               label=f"+/- sigma_StimCond = {sd:.2f}")
    ax.set_yticks(y); ax.set_yticklabels(b["Patient"], fontsize=9)
    ax.set_xlabel("Per-subject stim effect on logit P(remembered)\n"
                  "BLUP from RS GLMM (partial-pooled, shrunk toward mean)",
                  fontsize=10, fontweight="bold")
    blup_emp_sd = float(blups["subject_stim_effect"].std(ddof=1))
    ax.set_title(
        f"B. Model-based per-subject stim effects (BLUPs)\n"
        f"BLUP empirical SD = {blup_emp_sd:.3f};  "
        f"model sigma_StimCond = {sd:.3f}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.25, axis="x")

    # ── Panel C: NAIVE per-subject stim effects (no shrinkage) ──────────
    ax = axes[2]
    nv = naive.set_index("Patient").loc[subj_order].reset_index()
    y = np.arange(len(nv))
    xerr_lo = nv["naive_stim_effect"] - nv["ci_lo"]
    xerr_hi = nv["ci_hi"] - nv["naive_stim_effect"]
    colors = ["#a0a0a0" if v < 0 else "#2b2b2b"
              for v in nv["naive_stim_effect"]]
    for i, c in enumerate(colors):
        ax.errorbar(nv["naive_stim_effect"].iloc[i], y[i],
                    xerr=[[xerr_lo.iloc[i]], [xerr_hi.iloc[i]]],
                    fmt="o", color=c, ecolor="0.55",
                    capsize=3, ms=6, lw=1.0)
    ax.axvline(0, color="0.4", lw=1, ls="-")
    naive_mean = float(nv["naive_stim_effect"].mean())
    naive_sd = float(nv["naive_stim_effect"].std(ddof=1))
    ax.axvline(naive_mean, color="#2980b9", ls="--", lw=1.8,
               label=f"Naive mean = {naive_mean:+.2f}")
    ax.set_yticks(y); ax.set_yticklabels(nv["Patient"], fontsize=9)
    ax.set_xlabel("Per-subject stim effect on logit P(remembered)\n"
                  "Naive: separate GLM per subject, no shrinkage",
                  fontsize=10, fontweight="bold")
    same_sign = int(np.sum(
        (b["subject_stim_effect"].to_numpy() * nv["naive_stim_effect"].to_numpy()) >= 0))
    ax.set_title(
        f"C. Model-free naive stim effects (one GLM per subject)\n"
        f"Naive SD = {naive_sd:.3f} (wider, no shrinkage);  "
        f"sign matches BLUP for {same_sign}/{len(nv)} subjects",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.25, axis="x")

    # ── Panel D: per-subject band slope (contrast) ──────────────────────
    ax = axes[3]
    ps2 = per_subj.sort_values("band_cor").reset_index(drop=True)
    y = np.arange(len(ps2))
    colors = ["#a0a0a0" if v < 0 else "#2b2b2b" for v in ps2["band_cor"]]
    ax.barh(y, ps2["band_cor"], color=colors)
    ax.axvline(0, color="0.4", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(ps2["Patient"], fontsize=9)
    ax.set_xlabel("Per-subject band x Accuracy correlation",
                  fontsize=10, fontweight="bold")
    band_sd = float(per_subj["band_cor"].std(ddof=1))
    ax.set_title(
        f"D. By contrast: subjects do NOT vary in band slope\n"
        f"Empirical SD = {band_sd:.3f}  -> RS-on-band would be singular",
        fontsize=11, fontweight="bold",
    )
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle(
        "Between-subject heterogeneity at EC-PRC PAC: present for StimCond "
        "(A-C), absent for band (D)",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    p = FIG_DIR / "per_subject_signals.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# ── Comparison figure: fit metrics + variance components ───────────────────

def make_fit_compare_figure(vc, lrt):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AIC / BIC bars
    ax = axes[0]
    labels = ["RI: (1|Patient)", "RS: (1+StimCond|Patient)"]
    aic_vals = vc["AIC"].tolist()
    bic_vals = vc["BIC"].tolist()
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, aic_vals, w, color="#2b2b2b", label="AIC")
    ax.bar(x + w / 2, bic_vals, w, color="#888888", label="BIC")
    for i, (a, b) in enumerate(zip(aic_vals, bic_vals)):
        ax.text(i - w / 2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Information criterion (lower = better)",
                  fontsize=11, fontweight="bold")
    dAIC = aic_vals[1] - aic_vals[0]
    dBIC = bic_vals[1] - bic_vals[0]
    lrt_p = float(lrt["Pr(>Chisq)"].iloc[1])
    ax.set_title(
        f"Fit comparison\ndAIC = {dAIC:+.2f}  dBIC = {dBIC:+.2f}  "
        f"LRT p = {p_str(lrt_p)}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left")

    # Right: variance components
    ax = axes[1]
    rs = vc[vc["model"] == "RS"].iloc[0]
    ri = vc[vc["model"] == "RI"].iloc[0]
    bars = ["sigma_intercept\n(RI)",
            "sigma_intercept\n(RS)",
            "sigma_StimCond\n(RS only)"]
    vals = [ri["sd_intercept"], rs["sd_intercept"], rs["sd_stim"]]
    colors = ["#888888", "#2b2b2b", "#c0392b"]
    ax.bar(range(len(bars)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels(bars, fontsize=10)
    ax.set_ylabel("Random-effect SD (logit scale)",
                  fontsize=11, fontweight="bold")
    rho = rs["cor_int_stim"]
    sing = bool(rs["singular"])
    ax.set_title(
        f"Variance components\n"
        f"RS correlation(int, slope) = {rho:+.2f};  RS singular = {sing}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("RI vs RS: model fit and random-effect structure",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = FIG_DIR / "fit_comparison.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# ── Main report build ──────────────────────────────────────────────────────

def main():
    coefs_ri = pd.read_csv(OUT_DIR / "coefs_RI.csv")
    coefs_rs = pd.read_csv(OUT_DIR / "coefs_RS.csv")
    anova_ri = pd.read_csv(OUT_DIR / "anova_RI.csv")
    anova_rs = pd.read_csv(OUT_DIR / "anova_RS.csv")
    lrt = pd.read_csv(OUT_DIR / "ri_vs_rs_lrt.csv")
    # R mangles column names on write.csv: Pr(>Chisq) -> Pr..Chisq.
    lrt = lrt.rename(columns={"Pr..Chisq.": "Pr(>Chisq)",
                              "rownames.cmp.": "rownames(cmp)"})
    vc  = pd.read_csv(OUT_DIR / "variance_components.csv")
    per_subj = pd.read_csv(OUT_DIR / "per_subject_summary.csv")
    fx = pd.read_csv(OUT_DIR / "fixed_effect_summary.csv")

    # Companion figures
    fit_png    = make_fit_compare_figure(vc, lrt)
    subj_png   = make_subject_figure(per_subj, vc)

    pdf = APAReport()

    # ── Cover ─────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(
        0, 9,
        "EC-PRC PAC slow gamma\n"
        "Encoding interaction model: random intercept vs "
        "+ random StimCond slope"
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"N = {int(per_subj['n_trials'].sum())} trials, "
             f"{len(per_subj)} patients", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Encoding phase only. All trials (stim + no-stim pooled).",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.section("Summary")
    ri_focal = fx[(fx["model"] == "RI") &
                  (fx["term"] == "band_c:StimCond_num")].iloc[0]
    rs_focal = fx[(fx["model"] == "RS") &
                  (fx["term"] == "band_c:StimCond_num")].iloc[0]
    rs_row = vc[vc["model"] == "RS"].iloc[0]
    pdf.body(
        f"This report compares two specifications of the encoding "
        f"interaction GLMM at EC-PRC slow gamma PAC. The fixed-effect "
        f"structure is identical: Accuracy ~ band + StimCond + band:StimCond. "
        f"The only difference is the random-effect structure.\n\n"
        f"RI:  (1 | Patient)                  -- every subject shares the "
        f"same stim effect.\n"
        f"RS:  (1 + StimCond | Patient)       -- each subject has their own "
        f"stim effect.\n\n"
        f"Focal interaction coefficient (band:StimCond):\n"
        f"  RI: OR = {num(ri_focal['or'])}  z = {ri_focal['z']:.2f}  "
        f"p = {p_str(ri_focal['p'])}\n"
        f"  RS: OR = {num(rs_focal['or'])}  z = {rs_focal['z']:.2f}  "
        f"p = {p_str(rs_focal['p'])}\n\n"
        f"The RS model successfully estimates between-subject variability "
        f"in the stim effect (sigma_StimCond = "
        f"{rs_row['sd_stim']:.3f} on the logit scale; "
        f"isSingular = {bool(rs_row['singular'])}). "
        f"By LRT and AIC the random slope is not justified, but its "
        f"successful estimation is informative about why the Johnson-Neyman "
        f"region disappears under RS."
    )

    # ── Analytic motivation (sits before the model tables) ──────────────
    pdf.add_page()
    pdf.section("0. Analytic motivation: why this report exists at all")
    pdf.body(
        "The encoding analytic pipeline at this panel proceeded in three "
        "stages, each making a different demand of the data:\n\n"
        "(1) LMM-cont:  band ~ avg_stim_dprime_diff (z-scored) + (1|Patient). "
        "Treats each subject's overall behavioral stim benefit as a "
        "between-subject moderator of the neural feature. Result at EC-PRC "
        "PAC slow gamma: not reliable. See "
        "Encoding_Pac_HPCrhinal_LMMcont_FDRcorrected_ImBalanced.pdf for "
        "the full LMM-cont family.\n\n"
        "(2) LMM-quad:  band ~ QuadResponderGroup (Non/Anti/Mod/Strong) + "
        "(1|Patient). Treats responder status as a categorical "
        "between-subject moderator. Same panel: not reliable. See "
        "Encoding_Pac_HPCrhinal_LMMquad_FDRcorrected_ImBalanced.pdf.\n\n"
        "(3) GLMM with band x StimCond:  Accuracy ~ band + StimCond + "
        "band:StimCond + (1|Patient). Switches the question from "
        "'do trait-level subject differences explain the neural feature?' "
        "to 'within subjects, does the band's relationship to trial-level "
        "memory differ between stim and no-stim?'. FDR-significant at this "
        "panel in the stim-only main-effect family (q = 0.027, see "
        "Encoding_Pac_HPCrhinal_GLMM_MainEffect_Stim_FDRcorrected.pdf).\n\n"
        "Why this matters for the heterogeneity story shown later in this "
        "report: because the trait-level approaches (steps 1-2) did not "
        "reliably detect effects on the neural feature, the between-subject "
        "variability captured by sigma_StimCond = "
        f"{rs_row['sd_stim']:.3f} in the random-slope variant is NOT "
        "reducible to overall behavioral responder status. It is residual "
        "heterogeneity beyond what trait-level phenotyping captures -- a "
        "finding worth reporting on its own terms, since it suggests "
        "that the local PAC modulation under stim is a separable "
        "individual-difference dimension from global behavioral response."
    )

    # ── RI model tables ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.section("Model 1. Random intercept only (RI)")
    pdf.body(
        "Accuracy ~ band + StimCond + band:StimCond + (1 | Patient)"
    )

    rows = []
    for _, r in coefs_ri.iterrows():
        rows.append([
            pretty_term(str(r["term"])),
            num(r["estimate"]),
            ci_str(r["conf.low"], r["conf.high"]),
            fmt(r["statistic"], 3),
            p_str(r["p.value"]),
        ])
    pdf.apa_table(
        "Table. Fixed-effect Coefficients (odds ratios)",
        ["Predictor", "OR", "95% CI", "z", "p"],
        rows,
        col_widths=[80, 30, 40, 18, 22],
        note=("Reference levels: StimCond = nostim. Coefficients are "
              "exponentiated; 95% CIs are Wald-based."),
    )

    rows = []
    for _, r in anova_ri.iterrows():
        rows.append([
            r["Model"],
            relabel_formula(r["Formula"]),
            f"{int(r['npar'])}" if pd.notna(r["npar"]) else "-",
            fmt(r["AIC"], 1),
            fmt(r["BIC"], 1),
            fmt(r["logLik"], 1),
            fmt(r["Chisq"], 2),
            fmt(r["Df"], 0),
            p_str(r["p.value"]) if pd.notna(r["p.value"]) else "-",
        ])
    pdf.apa_table(
        "Table. Sequential Model Build (RI)",
        ["Model", "Formula", "npar", "AIC", "BIC", "logLik",
         "Chisq", "Df", "p"],
        rows,
        col_widths=[14, 100, 10, 16, 16, 16, 14, 10, 14],
        note="Sequential LR test vs previous row.",
    )

    # ── RS model tables ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.section("Model 2. + Random StimCond slope (RS)")
    pdf.body(
        "Accuracy ~ band + StimCond + band:StimCond + "
        "(1 + StimCond | Patient)"
    )

    rows = []
    for _, r in coefs_rs.iterrows():
        rows.append([
            pretty_term(str(r["term"])),
            num(r["estimate"]),
            ci_str(r["conf.low"], r["conf.high"]),
            fmt(r["statistic"], 3),
            p_str(r["p.value"]),
        ])
    pdf.apa_table(
        "Table. Fixed-effect Coefficients (odds ratios)",
        ["Predictor", "OR", "95% CI", "z", "p"],
        rows,
        col_widths=[80, 30, 40, 18, 22],
        note=("Same fixed-effect structure as RI; only the random-effect "
              "term differs."),
    )

    rows = []
    for _, r in anova_rs.iterrows():
        rows.append([
            r["Model"],
            relabel_formula(r["Formula"]),
            f"{int(r['npar'])}" if pd.notna(r["npar"]) else "-",
            fmt(r["AIC"], 1),
            fmt(r["BIC"], 1),
            fmt(r["logLik"], 1),
            fmt(r["Chisq"], 2),
            fmt(r["Df"], 0),
            p_str(r["p.value"]) if pd.notna(r["p.value"]) else "-",
        ])
    pdf.apa_table(
        "Table. Sequential Model Build (RS)",
        ["Model", "Formula", "npar", "AIC", "BIC", "logLik",
         "Chisq", "Df", "p"],
        rows,
        col_widths=[14, 100, 10, 16, 16, 16, 14, 10, 14],
        note=("Each model uses the (1 + StimCond | Patient) random-effect "
              "structure; sequential LR test vs previous row."),
    )

    # Random-effect variance components table
    vc_rows = []
    for _, r in vc.iterrows():
        vc_rows.append([
            r["model"],
            fmt(r["sd_intercept"], 3),
            fmt(r["sd_stim"], 3) if pd.notna(r["sd_stim"]) else "-",
            fmt(r["cor_int_stim"], 3) if pd.notna(r["cor_int_stim"]) else "-",
            "yes" if bool(r["singular"]) else "no",
        ])
    pdf.apa_table(
        "Table. Random-effect variance components",
        ["Model", "sigma_intercept", "sigma_StimCond",
         "Corr(int, slope)", "Singular?"],
        vc_rows,
        col_widths=[30, 40, 40, 40, 30],
    )

    # ── Comparison page ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.section("Model fit comparison")
    pdf.image_full(fit_png)

    pdf.subsection("Cross-model LRT (RI vs RS):")
    lrt_rows = []
    for _, r in lrt.iterrows():
        lrt_rows.append([
            r.get("rownames(cmp)", "-"),
            fmt(r.get("npar", float("nan")), 0),
            fmt(r.get("AIC", float("nan")), 1),
            fmt(r.get("BIC", float("nan")), 1),
            fmt(r.get("logLik", float("nan")), 1),
            fmt(r.get("Chisq", float("nan")), 2),
            fmt(r.get("Df", float("nan")), 0),
            p_str(r.get("Pr(>Chisq)", float("nan"))),
        ])
    pdf.apa_table(
        "Table. Likelihood-ratio comparison of nested random-effect structures",
        ["Model", "npar", "AIC", "BIC", "logLik", "Chisq", "Df", "p"],
        lrt_rows,
        col_widths=[100, 14, 22, 22, 22, 16, 12, 22],
        note=("Both models share the same fixed-effect structure. The LRT "
              "tests whether the random StimCond slope adds enough fit to "
              "justify the 2 extra parameters (slope variance + correlation)."),
    )

    # ── Per-subject signal figure ─────────────────────────────────────────
    pdf.add_page()
    pdf.section("Why the data support a random StimCond slope (but not band)")
    pdf.image_full(subj_png)
    pdf.body(
        "Panel A (raw memory rate per subject): one line per subject "
        "connecting their no-stim memory rate to their stim memory rate. "
        "Lines fan out and cross -- the slopes are not parallel -- which is "
        "direct, model-free evidence that subjects respond differently to "
        "stim.\n\n"
        "Panel B (model-based BLUPs): each dot is one subject's "
        "per-subject stim-effect estimate from the random-slope GLMM (fixed "
        "b2 + their u_StimCond random effect). These are partial-pooled and "
        "shrunken toward the population mean. The pink band marks +/-1 "
        "sigma_StimCond -- the heterogeneity the random slope is estimating.\n\n"
        "Panel C (model-free naive estimates): the same quantity computed "
        "from a separate GLM per subject (no random effects, no pooling). "
        "Naive estimates are wider (no shrinkage), but should share the "
        "BLUPs' sign pattern. If they do, the heterogeneity in B is not an "
        "artifact of the random-effect parameterization -- it is in the raw "
        "subject-level data.\n\n"
        "Panel D (band slope control): each subject's raw band x Accuracy "
        "correlation. The between-subject spread is essentially zero -- "
        "subjects show the same shape of band-on-memory relationship. That "
        "is why the alternative variant (1 + band | Patient) produces a "
        "singular fit: there is no between-subject variability in band "
        "slope to model. The condition-level claim that the band slope "
        "differs between stim and no-stim is carried by the fixed-effect "
        "band:StimCond interaction, not by a random slope on band."
    )

    # ── Johnson-Neyman ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section("Johnson-Neyman: where does Stim vs No-stim differ along band?")
    pdf.body(
        "Each plot shows the simple slope of (Stim - No-stim) on "
        "logit P(remembered) as a function of band (the moderator). The "
        "cyan region marks band values where this slope is significantly "
        "non-zero (p < .05); the pink region marks values where it is not. "
        "The dashed teal line is the JN boundary; the thick black bar on "
        "y = 0 spans the range of band values that actually occur in the "
        "data."
    )

    pdf.subsection("JN: Random intercept only (RI)")
    pdf.image_full(JN_RI)

    pdf.add_page()
    pdf.subsection("JN: + Random StimCond slope (RS)")
    pdf.image_full(JN_RS)

    pdf.body(
        "The point estimates of the band:StimCond interaction are essentially "
        "identical between RI (OR = "
        f"{num(ri_focal['or'])}, p = {p_str(ri_focal['p'])}) and RS "
        f"(OR = {num(rs_focal['or'])}, p = {p_str(rs_focal['p'])}). What "
        "changes is the SE of the StimCond simple effect: in RI it reflects "
        "only within-subject sampling, while in RS it must account for "
        "between-subject variability in the stim effect (sigma_StimCond ~ "
        f"{rs_row['sd_stim']:.3f}). That inflation is enough that the cyan "
        "p < .05 region collapses outside the empirical band range under "
        "RS, even though the average interaction is the same.\n\n"
        "Interpretation: the RI-based JN is the upper estimate of how "
        "tightly we can localize the interaction in band space, conditional "
        "on assuming a homogeneous stim effect across subjects. The RS-based "
        "JN is the lower estimate once between-subject heterogeneity is "
        "acknowledged. With N = 11 patients, the population-level "
        "localization is genuinely uncertain."
    )

    # ── Practical interpretation ──────────────────────────────────────────
    pdf.add_page()
    pdf.section("Practical interpretation")
    pdf.body(
        "The interaction itself is the same under both models. What the "
        "RS comparison teaches us is about *generalization*: with this "
        "sample size, we can claim that the band:StimCond interaction "
        "exists on average, but we cannot claim a specific band value "
        "where stim vs no-stim differs reliably in a new patient.\n\n"
        "Recommended reporting strategy:\n"
        "  - Lead with the FDR-significant stim-only main effect of EC-PRC "
        "slow gamma PAC on memory (q_FDR = 0.027 in the encoding stim-only "
        "main-effect family).\n"
        "  - Present the band:StimCond interaction here as convergent "
        "supporting evidence (RI p = "
        f"{p_str(ri_focal['p'])}, q_FDR = .285 in the interaction family).\n"
        "  - Include the random-slope variant as a sensitivity / model-fit "
        "check, noting that it is not preferred on AIC/BIC/LRT grounds.\n"
        "  - Report the JN region from the RI model with the caveat that "
        "it widens (and ultimately disappears) once subject-level "
        "heterogeneity in the stim effect is modelled."
    )

    out_pdf = OUT_DIR / "EC_PRC_PAC_RandomSlope_Report.pdf"
    pdf.output(str(out_pdf))
    print(f"PDF -> {out_pdf}")


if __name__ == "__main__":
    main()
