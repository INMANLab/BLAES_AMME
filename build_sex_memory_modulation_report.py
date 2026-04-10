#!/usr/bin/env python3
"""
Patient-level linear model of memory modulation (avg_stim_dprime_diff) by sex.

This is a one-row-per-patient analysis -- memory modulation is a single value
per patient, so the correct level of analysis is a plain linear model (or
equivalently, a two-sample t-test). A patient-level random intercept cannot
be fit on this outcome because the outcome does not vary within patient.

Outputs a single APA-formatted PDF to outputs/stim_sex_memory reports/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "stim_sex_memory reports"
OUTPUT_PDF = OUTPUT_DIR / "Sex_Memory_Modulation_LM_Report.pdf"
SWARM_FIGURE = SCRIPT_DIR / "behavioral figures" / "AMMEBLAES_sex_swarmplot_stats.png"

BEH_CANDIDATES = [
    SCRIPT_DIR.parent / "AMMEBLAES_includedpts_firstsession_behavioral.csv",
    SCRIPT_DIR / "behavioral figures" / "AMMEBLAES_includedpts_firstsession_behavioral.csv",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_behavior() -> pd.DataFrame:
    for path in BEH_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            break
    else:
        raise FileNotFoundError("Could not locate behavioral summary CSV")

    df = df.rename(columns={"avg_stim_dprime_diff": "memory_modulation"})
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df = df[df["sex"].isin(["male", "female"])].copy()
    df["memory_modulation"] = pd.to_numeric(df["memory_modulation"], errors="coerce")
    df = df[np.isfinite(df["memory_modulation"])].copy()
    return df[["Patient", "sex", "memory_modulation"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sex")["memory_modulation"]
    out = pd.DataFrame({
        "N": g.size(),
        "Mean": g.mean(),
        "SD": g.std(ddof=1),
        "SE": g.std(ddof=1) / np.sqrt(g.size()),
        "Median": g.median(),
        "Min": g.min(),
        "Max": g.max(),
    })
    return out.reindex(["female", "male"])


def welch_ttest(female, male):
    t_stat, p_val = stats.ttest_ind(male, female, equal_var=False)
    nf, nm = len(female), len(male)
    vf = female.var(ddof=1)
    vm = male.var(ddof=1)
    df_num = (vf / nf + vm / nm) ** 2
    df_den = (vf ** 2) / ((nf ** 2) * (nf - 1)) + (vm ** 2) / ((nm ** 2) * (nm - 1))
    df_welch = df_num / df_den
    mean_diff = male.mean() - female.mean()
    se_diff = np.sqrt(vf / nf + vm / nm)
    t_crit = stats.t.ppf(0.975, df_welch)
    ci_lo = mean_diff - t_crit * se_diff
    ci_hi = mean_diff + t_crit * se_diff
    return {
        "t": t_stat, "df": df_welch, "p": p_val,
        "mean_diff": mean_diff, "se_diff": se_diff,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def student_ttest(female, male):
    t_stat, p_val = stats.ttest_ind(male, female, equal_var=True)
    nf, nm = len(female), len(male)
    sp2 = ((nf - 1) * female.var(ddof=1) + (nm - 1) * male.var(ddof=1)) / (nf + nm - 2)
    se_diff = np.sqrt(sp2 * (1 / nf + 1 / nm))
    df_pooled = nf + nm - 2
    mean_diff = male.mean() - female.mean()
    t_crit = stats.t.ppf(0.975, df_pooled)
    return {
        "t": t_stat, "df": df_pooled, "p": p_val,
        "mean_diff": mean_diff, "se_diff": se_diff,
        "ci_lo": mean_diff - t_crit * se_diff,
        "ci_hi": mean_diff + t_crit * se_diff,
        "sp2": sp2,
    }


def cohens_d(female, male):
    nf, nm = len(female), len(male)
    sp2 = ((nf - 1) * female.var(ddof=1) + (nm - 1) * male.var(ddof=1)) / (nf + nm - 2)
    sp = np.sqrt(sp2)
    d = (male.mean() - female.mean()) / sp
    J = 1 - 3 / (4 * (nf + nm) - 9)
    g = J * d
    se_d = np.sqrt((nf + nm) / (nf * nm) + (d ** 2) / (2 * (nf + nm)))
    ci_lo = d - 1.96 * se_d
    ci_hi = d + 1.96 * se_d
    return {"d": d, "hedges_g": g, "ci_lo": ci_lo, "ci_hi": ci_hi}


def ols_model(df: pd.DataFrame) -> dict:
    """Fit memory_modulation ~ sex (female = reference).

    With a single binary predictor and OLS, the slope equals the mean difference
    and the slope t-test is mathematically identical to Student's t-test.
    """
    x = (df["sex"] == "male").astype(float).values
    y = df["memory_modulation"].values
    n = len(y)

    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    sse = (resid ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    dof = n - 2
    mse = sse / dof
    cov = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_vals = beta / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), dof))
    t_crit = stats.t.ppf(0.975, dof)
    ci_lo = beta - t_crit * se
    ci_hi = beta + t_crit * se

    r2 = 1 - sse / sst
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
    f_stat = (r2 / 1) / ((1 - r2) / dof)
    f_p = 1 - stats.f.cdf(f_stat, 1, dof)

    return {
        "terms": ["(Intercept) [female]", "Sex [male]"],
        "estimate": beta,
        "se": se,
        "t": t_vals,
        "p": p_vals,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "df_resid": dof,
        "r2": r2,
        "adj_r2": adj_r2,
        "f": f_stat,
        "f_p": f_p,
        "n": n,
        "sigma": np.sqrt(mse),
    }


def assumption_checks(female, male):
    sw_f = stats.shapiro(female)
    sw_m = stats.shapiro(male)
    lev = stats.levene(female, male, center="median")
    return {
        "shapiro_f": sw_f,
        "shapiro_m": sw_m,
        "levene": lev,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def p_str(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}".lstrip("0")


def sig_str(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "+"
    return ""


def fmt(value, decimals=3):
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def ci_str(lo, hi, decimals=3):
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{fmt(lo, decimals)}, {fmt(hi, decimals)}]"


# ---------------------------------------------------------------------------
# APA PDF class (matches build_stim_sex_retrieval_report.py)
# ---------------------------------------------------------------------------
class APAReport(FPDF):
    def __init__(self, report_title):
        super().__init__()
        self._report_title = report_title
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, self._report_title, align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def big_title(self, text):
        self.set_font("Helvetica", "B", 16)
        self.multi_cell(0, 9, text)
        self.ln(2)

    def subtitle(self, text):
        self.set_font("Helvetica", "", 11)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(6)

    def section_title(self, text):
        self.set_font("Helvetica", "B", 13)
        self.ln(4)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.ln(2)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Times", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_font("Times", "B", 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def italic_text(self, text):
        self.set_font("Times", "I", 9)
        self.multi_cell(0, 4.5, text)
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
# Report writer
# ---------------------------------------------------------------------------
def write_report():
    df = load_behavior()
    desc = descriptives(df)

    female = df.loc[df["sex"] == "female", "memory_modulation"].values
    male = df.loc[df["sex"] == "male", "memory_modulation"].values

    welch = welch_ttest(female, male)
    student = student_ttest(female, male)
    effect = cohens_d(female, male)
    model = ols_model(df)
    checks = assumption_checks(female, male)

    report_title = "Sex Differences in Memory Modulation - Patient-Level LM"
    pdf = APAReport(report_title)
    pdf.add_page()

    # ====================================================================
    # 1. Title + Overview
    # ====================================================================
    pdf.big_title(
        "Do Males and Females Differ in\n"
        "Stimulation-Induced Memory Modulation?")
    pdf.subtitle("Patient-Level Linear Model - One Row Per Patient")

    pdf.section_title("1. Overview")
    pdf.body_text(
        "This report tests whether stimulation-induced memory modulation "
        "(renamed here from avg_stim_dprime_diff) differs between male and "
        "female patients. Memory modulation is defined as the d-prime "
        "difference between stimulation and no-stimulation trials at the "
        "one-day delay, computed once per patient.")
    pdf.body_text(
        "Because memory modulation is a single value per patient, the "
        "appropriate level of analysis is a patient-level linear model "
        "(equivalently, a two-sample t-test). A mixed-effects model with a "
        "patient random intercept cannot be fit on this outcome because the "
        "outcome does not vary within patient -- the random intercept would "
        "absorb all variance and leave no residual for fixed effects (ICC = "
        "1.000). This is why prior patient-level MLM attempts on this "
        "outcome produced uniformly null p-values.")

    pdf.subsection_title("Purpose")
    pdf.body_text(
        "The goal of this analysis is to inform a modeling decision for "
        "downstream neural-behavioral models: should biological sex be "
        "entered as a covariate, or should it be tested as a factor via "
        "Sex x Stimulation interactions? A clear main effect of sex on the "
        "memory modulation outcome would support either choice; the absence "
        "of a main effect would argue for treating sex as a nuisance "
        "covariate rather than a substantive factor.")

    pdf.subsection_title("Statistical Approach")
    pdf.body_text(
        "Primary test: Welch's two-sample t-test (does not assume equal "
        "variance). Linear model: memory_modulation ~ sex, with female as "
        "the reference category. Student's t-test and Cohen's d (with "
        "Hedges' g small-sample correction) are reported alongside. "
        "Normality is assessed with Shapiro-Wilk tests within each group, "
        "and homogeneity of variance with Levene's test (median-centered).")

    # ====================================================================
    # 2. Descriptives
    # ====================================================================
    pdf.section_title("2. Descriptive Statistics")
    desc_rows = []
    for sex_label in ["female", "male"]:
        r = desc.loc[sex_label]
        desc_rows.append([
            sex_label.capitalize(),
            f"{int(r['N'])}",
            fmt(r["Mean"]),
            fmt(r["SD"]),
            fmt(r["SE"]),
            fmt(r["Median"]),
            fmt(r["Min"]),
            fmt(r["Max"]),
        ])
    pdf.apa_table(
        "Table 1. Memory Modulation by Sex",
        ["Sex", "N", "M", "SD", "SE", "Mdn", "Min", "Max"],
        desc_rows,
        col_widths=[22, 14, 22, 22, 22, 22, 22, 22],
        note="Memory modulation = avg_stim_dprime_diff (stim minus no-stim "
             "d-prime at one-day delay). Positive values indicate memory "
             "enhancement by stimulation; negative values indicate impairment.",
    )

    # ====================================================================
    # 3. Group comparison
    # ====================================================================
    pdf.section_title("3. Group Comparison")

    test_rows = [
        [
            "Welch's t (unequal variance)",
            fmt(welch["mean_diff"]),
            fmt(welch["se_diff"]),
            ci_str(welch["ci_lo"], welch["ci_hi"]),
            fmt(welch["t"]),
            fmt(welch["df"], 2),
            p_str(welch["p"]),
            sig_str(welch["p"]),
        ],
        [
            "Student's t (equal variance)",
            fmt(student["mean_diff"]),
            fmt(student["se_diff"]),
            ci_str(student["ci_lo"], student["ci_hi"]),
            fmt(student["t"]),
            fmt(student["df"], 0),
            p_str(student["p"]),
            sig_str(student["p"]),
        ],
    ]
    pdf.apa_table(
        "Table 2. Two-Sample Tests of Memory Modulation by Sex",
        ["Test", "M diff", "SE", "95% CI", "t", "df", "p", ""],
        test_rows,
        col_widths=[58, 18, 16, 36, 14, 18, 18, 8],
        note="Mean difference is male minus female. Welch's test does not "
             "assume equal variances and is reported as the primary test. "
             "* p < .05, ** p < .01, *** p < .001, + p < .10.",
    )

    effect_rows = [[
        fmt(effect["d"]),
        fmt(effect["hedges_g"]),
        ci_str(effect["ci_lo"], effect["ci_hi"]),
    ]]
    pdf.apa_table(
        "Table 3. Effect Size",
        ["Cohen's d", "Hedges' g", "95% CI (d)"],
        effect_rows,
        col_widths=[40, 40, 60],
        note="Positive values indicate higher memory modulation in males. "
             "Hedges' g applies a small-sample correction to Cohen's d. "
             "Conventional benchmarks: |d| = 0.2 small, 0.5 medium, 0.8 large.",
    )

    # ====================================================================
    # 4. Linear model
    # ====================================================================
    pdf.section_title("4. Linear Model")
    pdf.body_text(
        "Equivalent parameterization as a linear model with sex as the sole "
        "predictor. The slope on Sex [male] equals the mean difference "
        "(male - female), and its t-test is mathematically identical to "
        "Student's t. This framing is convenient for reporting alongside "
        "other regression-based analyses.")

    lm_rows = []
    for i, term in enumerate(model["terms"]):
        lm_rows.append([
            term,
            fmt(model["estimate"][i]),
            fmt(model["se"][i]),
            ci_str(model["ci_lo"][i], model["ci_hi"][i]),
            fmt(model["t"][i]),
            p_str(model["p"][i]),
            sig_str(model["p"][i]),
        ])
    pdf.apa_table(
        f"Table 4. Linear Model: memory_modulation ~ sex (N = {model['n']})",
        ["Predictor", "Estimate", "SE", "95% CI", "t", "p", ""],
        lm_rows,
        col_widths=[58, 20, 20, 38, 16, 20, 8],
        note=f"Reference category: female. Residual df = {int(model['df_resid'])}. "
             f"Residual SD = {fmt(model['sigma'])}. "
             f"R2 = {fmt(model['r2'])}, adjusted R2 = {fmt(model['adj_r2'])}. "
             f"Model F(1, {int(model['df_resid'])}) = {fmt(model['f'], 2)}, "
             f"p = {p_str(model['f_p'])}.")

    # ====================================================================
    # 5. Assumption checks
    # ====================================================================
    pdf.section_title("5. Assumption Checks")
    sw_f = checks["shapiro_f"]
    sw_m = checks["shapiro_m"]
    lev = checks["levene"]
    assum_rows = [
        ["Shapiro-Wilk (female)", fmt(sw_f.statistic), "--", p_str(sw_f.pvalue), sig_str(sw_f.pvalue)],
        ["Shapiro-Wilk (male)", fmt(sw_m.statistic), "--", p_str(sw_m.pvalue), sig_str(sw_m.pvalue)],
        ["Levene (median)", fmt(lev.statistic), f"1, {len(female) + len(male) - 2}", p_str(lev.pvalue), sig_str(lev.pvalue)],
    ]
    pdf.apa_table(
        "Table 5. Normality and Homogeneity of Variance",
        ["Test", "Statistic", "df", "p", ""],
        assum_rows,
        col_widths=[60, 34, 30, 30, 10],
        note="Shapiro-Wilk tests normality within each sex. Levene's test "
             "(median-centered, robust) tests equality of variances. "
             "Non-significant p-values indicate assumptions are met. "
             "Welch's t is the primary test and is robust to variance "
             "inequality regardless.",
    )

    # ====================================================================
    # 6. Figure
    # ====================================================================
    if SWARM_FIGURE.exists():
        pdf.section_title("6. Figure")
        pdf.body_text(
            "Distribution of memory modulation (avg_stim_dprime_diff) across "
            "female and male patients. Dashed line at 0 indicates no stim-induced "
            "modulation (equal d-prime for stim and no-stim trials).")
        pdf.ln(2)
        try:
            pdf.image(str(SWARM_FIGURE), x=35, w=140)
        except Exception as e:
            pdf.italic_text(f"(Figure could not be embedded: {e})")
        pdf.ln(2)

    # ====================================================================
    # 7. Interpretation
    # ====================================================================
    pdf.section_title("7. Interpretation")

    primary_p = welch["p"]
    direction = "higher" if welch["mean_diff"] > 0 else "lower"
    effect_size_label = (
        "small" if abs(effect["d"]) < 0.5 else
        "medium" if abs(effect["d"]) < 0.8 else
        "large"
    )

    pdf.bold_text("Primary result")
    pdf.body_text(
        f"Welch's t({fmt(welch['df'], 2)}) = {fmt(welch['t'])}, "
        f"p = {p_str(primary_p)}. Males showed {direction} memory "
        f"modulation than females by a mean of {fmt(welch['mean_diff'])} "
        f"d-prime units (95% CI {ci_str(welch['ci_lo'], welch['ci_hi'])}). "
        f"The standardized effect is Cohen's d = {fmt(effect['d'])} "
        f"(Hedges' g = {fmt(effect['hedges_g'])}), a {effect_size_label} "
        "effect by conventional benchmarks.")

    pdf.bold_text("Covariate vs. interaction-test decision")
    if primary_p < 0.05:
        pdf.body_text(
            "There IS a significant sex difference in memory modulation at "
            "the patient level. This supports treating sex as a substantive "
            "factor in downstream neural-behavioral models -- i.e., testing "
            "Sex x Stimulation interactions (which has already been done in "
            "the stim x sex GLMMs) rather than simply partialling sex out as "
            "a nuisance covariate. The GLMM interaction framework is the "
            "correct follow-up because it tests whether the neural "
            "mechanisms supporting the memory difference also differ by sex.")
        pdf.body_text(
            "However, if the interaction GLMMs already run did not find "
            "Sex x Stim interactions in the neural predictors of interest, "
            "that is also informative: the sex difference may operate "
            "through pathways other than the specific Stim x Neural "
            "interactions tested, and sex can reasonably be included as a "
            "main-effect covariate in those follow-up models to control for "
            "the overall behavioral difference without further interaction "
            "tests.")
    elif primary_p < 0.10:
        pdf.body_text(
            "The sex difference in memory modulation is MARGINAL "
            f"(p = {p_str(primary_p)}). This is a borderline case. With "
            f"N = {len(female)} female and N = {len(male)} male patients, "
            "the analysis is moderately powered for a medium effect. "
            "Including sex as a main-effect covariate in downstream models "
            "is a defensible conservative choice; formally testing Sex x "
            "Stimulation interactions is also justifiable if theoretical "
            "expectations warrant it.")
    else:
        pdf.body_text(
            "There is NO statistically significant sex difference in memory "
            f"modulation at the patient level (p = {p_str(primary_p)}). "
            "Given the absence of a detectable main effect on the primary "
            "outcome, treating sex as a main-effect covariate in downstream "
            "neural-behavioral models (rather than formally testing Sex x "
            "Stimulation interactions) is the more parsimonious choice. "
            "This controls for any residual between-subject variance "
            "associated with sex without multiplying the number of "
            "interaction tests.")

    pdf.bold_text("Caveats")
    pdf.body_text(
        f"(1) Sample is modest and unbalanced (N = {len(female)} female, "
        f"N = {len(male)} male). (2) Memory modulation is a single "
        "patient-level summary; within-patient variability across sessions "
        "or regions is not modeled here by design. (3) This test does not "
        "adjust for age, IQ, baseline memory ability, or stimulation-site "
        "laterality, any of which may confound a raw sex comparison. "
        "(4) The question of whether to test Sex x Stimulation "
        "interactions is ultimately a theoretical/pre-registration "
        "decision, not solely a statistical one; this report provides "
        "the empirical anchor but does not replace that judgment.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PDF))
    print(f"Wrote {OUTPUT_PDF}")

    # Also write the underlying stats to a CSV for record-keeping.
    stats_csv = OUTPUT_DIR / "Sex_Memory_Modulation_LM_stats.csv"
    stats_df = pd.DataFrame({
        "metric": [
            "n_female", "n_male",
            "mean_female", "mean_male",
            "sd_female", "sd_male",
            "mean_diff", "welch_t", "welch_df", "welch_p",
            "welch_ci_lo", "welch_ci_hi",
            "student_t", "student_df", "student_p",
            "cohens_d", "hedges_g", "d_ci_lo", "d_ci_hi",
            "lm_intercept", "lm_sex_male_beta", "lm_sex_male_se",
            "lm_sex_male_t", "lm_sex_male_p",
            "lm_R2", "lm_adj_R2", "lm_F", "lm_F_p",
            "shapiro_female_W", "shapiro_female_p",
            "shapiro_male_W", "shapiro_male_p",
            "levene_stat", "levene_p",
        ],
        "value": [
            len(female), len(male),
            female.mean(), male.mean(),
            female.std(ddof=1), male.std(ddof=1),
            welch["mean_diff"], welch["t"], welch["df"], welch["p"],
            welch["ci_lo"], welch["ci_hi"],
            student["t"], student["df"], student["p"],
            effect["d"], effect["hedges_g"], effect["ci_lo"], effect["ci_hi"],
            model["estimate"][0], model["estimate"][1], model["se"][1],
            model["t"][1], model["p"][1],
            model["r2"], model["adj_r2"], model["f"], model["f_p"],
            sw_f.statistic, sw_f.pvalue,
            sw_m.statistic, sw_m.pvalue,
            lev.statistic, lev.pvalue,
        ],
    })
    stats_df.to_csv(stats_csv, index=False)
    print(f"Wrote {stats_csv}")


def main():
    write_report()


if __name__ == "__main__":
    main()
