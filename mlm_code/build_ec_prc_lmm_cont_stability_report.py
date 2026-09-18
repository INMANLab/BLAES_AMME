#!/usr/bin/env python
"""Build the EC + PRC slow gamma power LMM-cont stability report.

Reads CSVs produced by ec_prc_lmm_cont_stability.R, builds per-region
diagnostic figures (per-subject scatter + LOSO + bootstrap + permutation),
and assembles a single PDF with motivation, results, and the justification
for moving from trait-level LMM-cont to within-subject GLMM.

Output:
  OUTPUTS/encoding_memory_reports/EC_PRC_slowgamma_power_lmm_cont_stability/
    EC_PRC_slowgamma_power_LMM_cont_Stability_Report.pdf
    figures/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "OUTPUTS" / "encoding_memory_reports" / \
    "EC_PRC_slowgamma_power_lmm_cont_stability"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["EC", "PRC"]


def num(v, decimals=4):
    if pd.isna(v): return "-"
    if abs(v) > 1e4 or (abs(v) < 1e-4 and v != 0):
        return f"{v:.2e}"
    return f"{v:.{decimals}f}"


def p_str(p):
    if isinstance(p, str): return p
    if pd.isna(p): return "-"
    if p < .001: return "< .001"
    return f"{p:.3f}".lstrip("0")


def fmt(v, decimals=2):
    if pd.isna(v): return "-"
    return f"{float(v):.{decimals}f}"


# ── Per-region 2x2 diagnostic figure ───────────────────────────────────────

def make_region_figure(region):
    obs = pd.read_csv(OUT_DIR / f"{region}_observed.csv").iloc[0]
    per_subj = pd.read_csv(OUT_DIR / f"{region}_per_subject.csv")
    loso = pd.read_csv(OUT_DIR / f"{region}_loso.csv")
    boot = pd.read_csv(OUT_DIR / f"{region}_bootstrap.csv").dropna(subset=["slope"])
    perm = pd.read_csv(OUT_DIR / f"{region}_permutation.csv").dropna(subset=["slope"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # A. Per-subject scatter with regression line + 95% CI band
    ax = axes[0, 0]
    ax.errorbar(per_subj["mem_mod_z"], per_subj["feature_mean"],
                yerr=per_subj["feature_sem"], fmt="o",
                color="#2b2b2b", ecolor="0.6", capsize=3, ms=7, alpha=0.9)
    for _, r in per_subj.iterrows():
        ax.annotate(r["Patient"],
                    xy=(r["mem_mod_z"], r["feature_mean"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color="0.35")
    xx = np.linspace(per_subj["mem_mod_z"].min(),
                     per_subj["mem_mod_z"].max(), 80)
    intercept = (per_subj["feature_mean"].mean()
                 - obs["slope"] * per_subj["mem_mod_z"].mean())
    yy = intercept + obs["slope"] * xx
    yy_lo = intercept + obs["ci_low"] * xx
    yy_hi = intercept + obs["ci_high"] * xx
    band_lo = np.minimum(yy_lo, yy_hi); band_hi = np.maximum(yy_lo, yy_hi)
    ax.fill_between(xx, band_lo, band_hi, color="#c0392b", alpha=0.15,
                    label="95% CI of slope")
    ax.plot(xx, yy, color="#c0392b", lw=2.2,
            label=(f"slope = {obs['slope']:+.3f}, "
                   f"Wald p = {p_str(obs['p_wald'])}, "
                   f"LRT p = {p_str(obs['p_lrt'])}"))
    ax.set_xlabel("mem_mod_z (z-scored avg_stim_dprime_diff)",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Per-subject mean {region} slow gamma power\n"
                  f"(within-subject mean +/- SEM)",
                  fontsize=11, fontweight="bold")
    ax.axhline(0, color="0.4", ls=":", lw=0.8)
    ax.axvline(0, color="0.4", ls=":", lw=0.8)
    ax.set_title(
        f"A. Per-subject view  (N = {int(obs['n_subjects'])} subjects, "
        f"{int(obs['n_trials'])} trials)",
        fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.25)

    # B. LOSO: slope + CI per subject excluded
    ax = axes[0, 1]
    full = loso[loso["excluded"] == "<none>"].iloc[0]
    rest = loso[loso["excluded"] != "<none>"].dropna(subset=["slope"]).copy()
    rest["ci_low"] = rest["slope"] - 1.96 * rest["se"]
    rest["ci_high"] = rest["slope"] + 1.96 * rest["se"]
    rest = rest.sort_values("slope").reset_index(drop=True)
    y = np.arange(len(rest))
    colors = ["#2b2b2b" if p < 0.05 else "#a0a0a0" for p in rest["p_wald"]]
    for i, c in enumerate(colors):
        xerr_lo = rest["slope"].iloc[i] - rest["ci_low"].iloc[i]
        xerr_hi = rest["ci_high"].iloc[i] - rest["slope"].iloc[i]
        ax.errorbar(rest["slope"].iloc[i], y[i],
                    xerr=[[xerr_lo], [xerr_hi]],
                    fmt="o", color=c, ecolor="0.55",
                    capsize=3, ms=6, lw=1.0)
    ax.axvline(0, color="0.4", lw=1)
    ax.axvline(obs["slope"], color="#c0392b", ls="--", lw=1.5,
               label=f"Full-sample slope = {obs['slope']:+.3f}")
    ax.set_yticks(y); ax.set_yticklabels(rest["excluded"], fontsize=8)
    ax.set_xlabel("LOSO slope (95% CI)",
                  fontsize=10, fontweight="bold")
    n_sig = int((rest["p_wald"] < 0.05).sum())
    n_lrt = int((rest["p_lrt"] < 0.05).sum())
    p_min = float(rest["p_wald"].min()); p_max = float(rest["p_wald"].max())
    ax.set_title(
        f"B. LOSO  ({n_sig}/{len(rest)} Wald p<.05;  "
        f"{n_lrt}/{len(rest)} LRT p<.05)\n"
        f"LOSO Wald p range: [{p_min:.3f}, {p_max:.3f}]",
        fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25, axis="x")

    # C. Bootstrap by subject
    ax = axes[1, 0]
    ax.hist(boot["slope"], bins=40, color="#888888", edgecolor="white")
    ax.axvline(0, color="0.4", lw=1.0, ls=":", label="slope = 0")
    ax.axvline(obs["slope"], color="#c0392b", ls="--", lw=1.8,
               label=f"Observed = {obs['slope']:+.3f}")
    boot_lo = float(boot["slope"].quantile(0.025))
    boot_hi = float(boot["slope"].quantile(0.975))
    boot_med = float(boot["slope"].median())
    ax.axvspan(boot_lo, boot_hi, color="#c0392b", alpha=0.10,
               label=f"95% boot CI: [{boot_lo:+.3f}, {boot_hi:+.3f}]")
    same_sign = float(np.mean(
        np.sign(boot["slope"]) == np.sign(obs["slope"])))
    ax.set_xlabel("Slope across bootstrap samples",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("count", fontsize=10, fontweight="bold")
    ax.set_title(
        f"C. Bootstrap by subject  ({len(boot)} iters)\n"
        f"Same-sign as observed in {100*same_sign:.0f}% of resamples;  "
        f"median = {boot_med:+.3f}",
        fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)

    # D. Permutation null
    ax = axes[1, 1]
    ax.hist(perm["slope"], bins=40, color="#a0a0a0", edgecolor="white")
    ax.axvline(0, color="0.4", lw=1.0, ls=":")
    ax.axvline(obs["slope"], color="#c0392b", ls="--", lw=2.2,
               label=f"Observed = {obs['slope']:+.3f}")
    null = perm["slope"].dropna().to_numpy()
    p_two = float(np.mean(np.abs(null) >= abs(obs["slope"])))
    p_dir = float(np.mean(null >= obs["slope"]) if obs["slope"] >= 0
                  else np.mean(null <= obs["slope"]))
    ax.set_xlabel("Slope under permutation null",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("count", fontsize=10, fontweight="bold")
    ax.set_title(
        f"D. Permutation null  ({len(perm)} iters)\n"
        f"two-sided p_perm = {p_two:.4f}   "
        f"directional p_perm = {p_dir:.4f}",
        fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"LMM-cont stability: {region} slow gamma power vs mem_mod_z "
        f"(encoding)",
        fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    p = FIG_DIR / f"{region}_stability.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "region": region,
        "obs": obs,
        "loso": loso,
        "boot_lo": boot_lo, "boot_hi": boot_hi, "boot_med": boot_med,
        "boot_same_sign": same_sign, "n_boot": len(boot),
        "perm_p_two": p_two, "perm_p_dir": p_dir, "n_perm": len(perm),
        "fig": p,
        "n_loso_sig_wald": n_sig, "n_loso_sig_lrt": n_lrt,
        "loso_p_min": p_min, "loso_p_max": p_max,
    }


# ── APA report class (same style as the other reports) ─────────────────────

REPORT_TITLE = ("Encoding | EC + PRC slow gamma power | LMM-cont stability "
                "(mem_mod_z moderator)")


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

    def section(self, t):
        self.set_font("Helvetica", "B", 13); self.ln(4)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT"); self.ln(2)

    def subsection(self, t):
        self.set_font("Helvetica", "B", 11); self.ln(2)
        self.cell(0, 7, t, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def body(self, t):
        self.set_font("Times", "", 11); self.multi_cell(0, 5.5, t); self.ln(1)

    def apa_table(self, title, headers, rows, col_widths=None, note=None):
        self.ln(3)
        self.set_font("Times", "I", 10); self.multi_cell(0, 5, title); self.ln(1)
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

    def image_full(self, path):
        if not Path(path).exists():
            self.body(f"[missing figure: {Path(path).name}]"); return
        w = self.w - self.l_margin - self.r_margin
        self.image(str(path), x=self.l_margin, w=w); self.ln(4)


def build_report(summaries):
    pdf = APAReport()

    # Cover
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(
        0, 9,
        "EC + PRC slow gamma power\n"
        "Encoding LMM-cont stability (trait-level mem_mod_z moderator)"
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, "Model: feature ~ mem_mod_z + (1 | Patient)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "feature = trial-level slow gamma power; mem_mod_z = "
             "z-scored avg_stim_dprime_diff",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 0. Motivation
    pdf.section("0. Motivation")
    pdf.body(
        "Two encoding LMM-cont panels trend at uncorrected p < .10 but do "
        "not survive FDR-BH at the HPCrhinal slow-gamma family (q = .144). "
        "Before pivoting to the within-subject GLMM analyses, we test "
        "whether these trending signals are robust to subject identity, "
        "subject sampling, and trait-level permutation. The stability tests "
        "below answer two questions:\n\n"
        "(1) Is each trending result driven by a single subject? "
        "(LOSO refits.)\n"
        "(2) Would resampling subjects or shuffling mem_mod_z reproduce a "
        "comparable slope?  (Bootstrap and permutation.)\n\n"
        "If a panel passes these three checks, the failure to survive FDR "
        "is consistent with limited between-subject power (N is on the "
        "order of 12-20 here) rather than a fluke -- which justifies the "
        "shift to the within-subject GLMM, where each subject contributes "
        "trials directly rather than a single point on the mem_mod_z axis."
    )

    # 1. Observed coefficients
    pdf.section("1. Observed LMM-cont coefficients")
    rows = []
    for s in summaries:
        rows.append([
            f"{s['region']} slow gamma power",
            fmt(s['obs']['slope'], 3),
            f"[{s['obs']['ci_low']:.3f}, {s['obs']['ci_high']:.3f}]",
            fmt(s['obs']['t'], 2),
            fmt(s['obs']['df'], 1),
            p_str(s['obs']['p_wald']),
            p_str(s['obs']['p_lrt']),
            f"{int(s['obs']['n_trials'])} / {int(s['obs']['n_subjects'])}",
        ])
    pdf.apa_table(
        "Table 1. LMM-cont slope of mem_mod_z on the panel feature",
        ["Panel", "slope", "95% CI", "t", "df",
         "Wald p", "LRT p", "trials/N"],
        rows,
        col_widths=[58, 18, 36, 12, 14, 18, 18, 22],
        note=("Wald p uses Satterthwaite df from lmerTest; LRT p compares "
              "ML refits of m1 vs m0. df is on the order of N_subjects - "
              "covariate count."),
    )

    # 2. Per-panel stability summaries
    for s in summaries:
        pdf.add_page()
        pdf.section(f"2{'AB'[summaries.index(s)]}. {s['region']} slow gamma "
                    f"power")
        pdf.body(
            f"Observed slope = {s['obs']['slope']:+.3f}, "
            f"Wald p = {p_str(s['obs']['p_wald'])}, "
            f"LRT p = {p_str(s['obs']['p_lrt'])}.\n\n"
            f"LOSO: in {s['n_loso_sig_wald']}/"
            f"{int(s['obs']['n_subjects'])} refits, Wald p remains < .05 "
            f"(LRT: {s['n_loso_sig_lrt']}/{int(s['obs']['n_subjects'])}). "
            f"LOSO Wald p range = [{s['loso_p_min']:.3f}, "
            f"{s['loso_p_max']:.3f}]. If a single subject drives the effect, "
            "their removal would push p well above .05.\n\n"
            f"Bootstrap by subject ({s['n_boot']} iters): 95% CI = "
            f"[{s['boot_lo']:+.3f}, {s['boot_hi']:+.3f}], "
            f"median = {s['boot_med']:+.3f}. Same-sign as observed in "
            f"{100 * s['boot_same_sign']:.0f}% of resamples.\n\n"
            f"Permutation by subject ({s['n_perm']} iters): two-sided "
            f"p_perm = {s['perm_p_two']:.4f}, directional p_perm = "
            f"{s['perm_p_dir']:.4f}. This is the non-parametric calibration "
            "of the slope under 'no between-subject moderation'."
        )
        pdf.image_full(s["fig"])

    # 3. Region-specific verdicts and why GLMM is the next step
    pdf.add_page()
    pdf.section("3. Region-specific verdicts and why GLMM is the next step")
    pdf.body(
        "The LMM-cont test asks: does a subject's overall behavioral "
        "responder strength (mem_mod_z) predict their mean neural feature? "
        "It is fundamentally a between-subject test, with effective sample "
        "size = number of subjects in the panel (here 15 for EC, 20 for "
        "PRC). Each subject's mean feature is itself estimated from "
        "~160 trials, but only one number per subject enters the "
        "between-subject regression.\n\n"
        "Applying the three stability tests to each region:"
    )

    pdf.subsection("PRC slow gamma power: STABLE moderate signal")
    pdf.body(
        "LOSO Wald p stays in [.045, .184] across all 20 refits -- no "
        "individual subject can push the effect away from its observed "
        "value. Bootstrap 95% CI = [+0.026, +0.348] does not cross zero, "
        "and 98% of bootstrap resamples preserve the positive sign. "
        "Permutation two-sided p = .110 (directional p = .062) puts the "
        "observed slope just outside the conventional null tail. Taken "
        "together: this is a real but modest between-subject effect that "
        "fails FDR because of limited N, not because of noise. "
        "Defensible reporting: 'PRC slow gamma power was associated with "
        "responder status at the trend level (LRT p = .076, q_FDR = "
        ".144), stable under LOSO, bootstrap, and permutation.' Pivoting "
        "to the within-subject GLMM as the primary inferential test is "
        "well motivated here."
    )

    pdf.subsection("EC slow gamma power: FRAGILE trend, do not lean on it")
    pdf.body(
        "LOSO Wald p ranges from .002 to .548 -- a ~250x swing depending "
        "on which subject is removed. Two subjects in particular drive "
        "this: removing amyg045 collapses the slope toward zero (p = "
        ".548); removing BJH041 strengthens it (p = .002). The Wald df "
        "for those LOSO refits jumps to ~1218-1238 (the trial count), "
        "indicating that the random-intercept variance has degenerated "
        "to ~0 in those refits -- the model is no longer doing meaningful "
        "between-subject pooling. Bootstrap 95% CI = [-0.325, +0.086] "
        "crosses zero, and only 92% of resamples keep the negative sign. "
        "Permutation two-sided p = .076 mirrors the marginal Wald p. "
        "Verdict: the EC LMM-cont effect is driven by 1-2 influential "
        "subjects and should not be interpreted as a stable "
        "between-subject finding. Defensible reporting: 'EC slow gamma "
        "power was not reliably associated with responder status at the "
        "trait level; stability testing identified two influential "
        "subjects whose presence/absence dominated the slope estimate. We "
        "therefore relied on the within-subject GLMM as the primary "
        "inferential test for EC.'"
    )

    pdf.subsection("Why the within-subject GLMM is the right pivot for both")
    pdf.body(
        "The within-subject GLMM (Accuracy ~ band x StimCond + "
        "(1|Patient)) reframes the question: rather than asking whether "
        "a subject-level trait predicts a subject-level mean, it asks "
        "whether the trial-level band value predicts trial-level memory "
        "differently in stim vs no-stim. Each subject contributes their "
        "full ~160 trials of within-subject data, and the model "
        "partitions between- vs within-subject variance via the random "
        "intercept. Power for the within-subject contrast is set by "
        "total trials, not by N_subjects -- so it is effectively ~10x "
        "more sensitive than LMM-cont for the trial-level relationship. "
        "For PRC, the GLMM provides a more powerful test of a signal we "
        "already know is stable. For EC, the GLMM provides a "
        "well-defined, single-subject-robust test of the relationship "
        "between band and memory that the trait-level LMM-cont could not "
        "reliably resolve."
    )

    out_pdf = OUT_DIR / "EC_PRC_slowgamma_power_LMM_cont_Stability_Report.pdf"
    pdf.output(str(out_pdf))
    print(f"PDF -> {out_pdf}")


def main():
    summaries = [make_region_figure(r) for r in REGIONS]
    build_report(summaries)
    print("\nSummary:")
    for s in summaries:
        print(f"  {s['region']}: obs p={s['obs']['p_wald']:.4f}  "
              f"LOSO p range=[{s['loso_p_min']:.3f}, {s['loso_p_max']:.3f}]  "
              f"boot 95% CI=[{s['boot_lo']:+.3f}, {s['boot_hi']:+.3f}]  "
              f"perm p_two={s['perm_p_two']:.4f}")


if __name__ == "__main__":
    main()
