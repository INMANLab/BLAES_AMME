#!/usr/bin/env python3
"""Effect sizes for the regression-based (GLMM / LMM) memory models.

Companion to permutation_code/compute_cluster_effect_sizes.py. The cluster file
holds Cohen's d_z for cluster-based permutation tests; this file holds the
effect sizes that belong with the mixed-effects models:

  * GLMM (logistic) interaction terms    -> odds ratio (OR) + 95% CI, plus
                                            nested-model AIC/BIC change (LRT).
  * LMM continuous predictor             -> standardized beta (b / SD_outcome),
                                            and the patient-level Pearson r
                                            (the appropriate between-patient unit).
  * LMM responder-group contrasts        -> standardized contrast (b / SD_outcome)
                                            and the patient-level Cohen's d /
                                            Hedges g vs Non-responders.

Coefficient values are read from the saved report CSVs; patient-level statistics
and outcome SDs are recomputed from the trial-level input CSVs so the file is
reproducible. Writes:
  OUTPUTS/effect_sizes/model_effect_sizes.csv
  OUTPUTS/effect_sizes/model_effect_sizes.md
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "OUTPUTS" / "csvs"
OUT_DIR = ROOT / "OUTPUTS" / "effect_sizes"
RET_FULL = ROOT / "OUTPUTS" / "retrieval_memory_reports" / "stats" / "h1abc_full_retrieval_mlm"
RET_NB = ROOT / "OUTPUTS" / "retrieval memory reports permutation test based" / "stats" / "narrowband_h1abc_retrieval_mlm"

SLOW_GAMMA = (30.0, 54.9999)
THETA = (4.0, 8.0)


def band_mean_series(df, lo, hi):
    fc = [c for c in df.columns if c.startswith("diff_Freq_")]
    fr = np.array([float(re.sub(r"[^0-9.]", "", c)) for c in fc])
    win = [c for c, f in zip(fc, fr) if lo - 1e-6 <= f <= hi + 1e-6]
    return df[win].mean(axis=1)


def outcome_sd(modality_csv, region, band):
    df = pd.read_csv(CSV_DIR / modality_csv)
    df = df[df["Region"] == region]
    vals = band_mean_series(df, *band).values
    return float(np.nanstd(vals, ddof=1))


def patient_means(modality_csv, region, band):
    df = pd.read_csv(CSV_DIR / modality_csv)
    df = df[df["Region"] == region].copy()
    df["val"] = band_mean_series(df, *band)
    return df.groupby("Patient")["val"].mean().reset_index()


def coef_row(path, term):
    d = pd.read_csv(path)
    return d[d["term"] == term].iloc[0]


def aic_bic_delta(anova_path):
    """GLMM nested step: additive m2 vs interaction m_full."""
    a = pd.read_csv(anova_path).set_index("Model")
    return dict(
        delta_aic=round(a.loc["m_full", "AIC"] - a.loc["m2", "AIC"], 3),
        delta_bic=round(a.loc["m_full", "BIC"] - a.loc["m2", "BIC"], 3),
        lrt_chisq=round(a.loc["m_full", "Chisq"], 3),
        lrt_df=int(a.loc["m_full", "Df"]),
        lrt_p=round(a.loc["m_full", "p.value"], 5),
    )


def aic_bic_delta_lmm(anova_path):
    """LMM single-step: intercept-only m0 vs m0 + predictor m1."""
    a = pd.read_csv(anova_path).set_index("Model")
    return dict(
        delta_aic=round(a.loc["m1", "AIC"] - a.loc["m0", "AIC"], 3),
        delta_bic=round(a.loc["m1", "BIC"] - a.loc["m0", "BIC"], 3),
        lrt_chisq=round(a.loc["m1", "Chisq"], 3),
        lrt_df=int(a.loc["m1", "Df"]),
        lrt_p=round(a.loc["m1", "p.value"], 5),
    )


def responder_groups():
    resp = pd.read_csv(CSV_DIR / "AMMEBLAES_responder_status.csv")
    resp["Patient"] = resp["Patient"].replace("BJH033", "BJH032")
    resp = resp.drop_duplicates("Patient")
    scol = [c for c in resp.columns if "responder" in c.lower() and "status" in c.lower()][0]
    gmap = {"Non-responders": "NonResp", "Anti-responders": "AntiResp",
            "Moderate responders": "Moderate", "Strong responders": "Strong"}
    resp["grp"] = resp[scol].map(gmap)
    return resp[["Patient", "grp"]]


def cohen_d_vs_ref(x, ref):
    n1, n2 = len(x), len(ref)
    sp = np.sqrt(((n1 - 1) * x.std(ddof=1) ** 2 + (n2 - 1) * ref.std(ddof=1) ** 2) / (n1 + n2 - 2))
    d = (x.mean() - ref.mean()) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    return d, d * J, n1


def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return tuple(np.tanh([z - 1.96 * se, z + 1.96 * se]))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---------- GLMM interaction terms: odds ratios + AIC/BIC ----------
    # PRC theta power x stim (narrowband HippSubRhinal GLMM)
    c = coef_row(RET_NB / "power_HippSubRhinal_GLMM" / "power_PRC_narrow_theta_coefs.csv", "band_c:StimCondstim")
    ab = aic_bic_delta(RET_NB / "power_HippSubRhinal_GLMM" / "power_PRC_narrow_theta_anova.csv")
    rows.append(dict(figure="2.11", table_ref="Table 4", analysis="GLMM interaction (memory ~ power x stim)",
                     region="PRC", modality="power", band="theta", model="HippSubRhinal GLMM (narrowband)",
                     term="theta x stim", unit_level="trial (GLMM)", effect_type="odds_ratio",
                     estimate=round(c["estimate"], 3), ci_low=round(c["conf.low"], 3), ci_high=round(c["conf.high"], 3),
                     statistic=round(c["statistic"], 2), stat_type="z", p_value=round(c["p.value"], 4), fdr_q=0.014,
                     n_patients=20, **ab, notes="OR per 1 unit centered theta power"))

    # CA1-PRC and EC-subiculum slow gamma coherence x stim (full HippSubRhinal GLMM)
    for region, pair, fdr_q, npat in [("CA1-PRC", "CA_PRC", 0.013, 15), ("EC-subiculum", "EC_HPC", 0.020, 14)]:
        base = RET_FULL / "coherence_HippSubRhinal_GLMM"
        c = coef_row(base / f"coherence_{pair}_slow_gamma_coefs.csv", "band_c:StimCondstim")
        ab = aic_bic_delta(base / f"coherence_{pair}_slow_gamma_anova.csv")
        rows.append(dict(figure="2.11", table_ref="Table 3", analysis="GLMM interaction (memory ~ coherence x stim)",
                         region=region, modality="coherence", band="slow gamma", model="HippSubRhinal GLMM (full)",
                         term="coherence x stim", unit_level="trial (GLMM)", effect_type="odds_ratio",
                         estimate=round(c["estimate"], 3), ci_low=round(c["conf.low"], 3), ci_high=round(c["conf.high"], 3),
                         statistic=round(c["statistic"], 2), stat_type="z", p_value=round(c["p.value"], 4), fdr_q=fdr_q,
                         n_patients=npat, **ab, notes="OR per 1 unit centered coherence; CI wide (predictor scale)"))

    # ---------- LMM continuous: EC-PRC coherence ~ memory modulation ----------
    for band_name, band, fdr_q in [("theta", THETA, 0.045), ("slow gamma", SLOW_GAMMA, None)]:
        sub = "coherence_HPCrhinal_LMMcont" / Path(f"coherence_EC_PRC_{band_name.replace(' ', '_')}")
        c = coef_row(RET_FULL / f"{sub}_coefs.csv", "mem_mod_z")
        sd_y = outcome_sd("combined_retrieval_coherence_all_mlmr_input.csv", "EC_PRC", band)
        beta = c["estimate"] / sd_y
        fit = aic_bic_delta_lmm(RET_FULL / f"{sub}_anova.csv")
        rows.append(dict(figure="2.11", table_ref="Table 4", analysis="LMM (coherence ~ memory modulation)",
                         region="EC-PRC", modality="coherence", band=band_name, model="HPCrhinal LMMcont (full)",
                         term="mem_mod_z", unit_level="trial (LMM)", effect_type="std_beta",
                         estimate=round(beta, 3), ci_low=round(c["conf.low"] / sd_y, 3), ci_high=round(c["conf.high"] / sd_y, 3),
                         statistic=round(c["statistic"], 2), stat_type="t", p_value=round(c["p.value"], 4), fdr_q=fdr_q,
                         n_patients=10, **fit,
                         notes=f"std beta = b/SD_outcome (SD={sd_y:.3f}); b={c['estimate']:.4f}, SE={c['std.error']:.4f}; "
                               "AIC/BIC = m0 (intercept) vs m1 (+mem_mod_z)"))

    # patient-level Pearson r (EC-PRC theta) -- the appropriate between-patient unit
    pm = patient_means("combined_retrieval_coherence_all_mlmr_input.csv", "EC_PRC", THETA)
    resp = responder_groups()
    mm = pd.read_csv(CSV_DIR / "AMMEBLAES_responder_status.csv")
    mm["Patient"] = mm["Patient"].replace("BJH033", "BJH032")
    mm = mm.drop_duplicates("Patient")
    mm["mem_mod_z"] = (mm["avg_stim_dprime_diff"] - mm["avg_stim_dprime_diff"].mean()) / mm["avg_stim_dprime_diff"].std(ddof=0)
    j = pm.merge(mm[["Patient", "mem_mod_z"]], on="Patient")
    r, p = stats.pearsonr(j["mem_mod_z"], j["val"]); ci = fisher_ci(r, len(j))
    rows.append(dict(figure="2.11", table_ref="Table 4", analysis="LMM (coherence ~ memory modulation)",
                     region="EC-PRC", modality="coherence", band="theta", model="patient-level correlation",
                     term="mem_mod_z", unit_level="patient", effect_type="pearson_r",
                     estimate=round(r, 3), ci_low=round(ci[0], 3), ci_high=round(ci[1], 3),
                     statistic="", stat_type="", p_value=round(p, 4), fdr_q=None, n_patients=len(j),
                     notes=f"between-patient r; r^2={r**2:.2f}"))

    # ---------- LMM responder groups: BLA-PRC slow gamma coherence ----------
    pm_bp = patient_means("combined_retrieval_coherence_all_mlmr_input.csv", "BLA_PRC", SLOW_GAMMA).merge(resp, on="Patient")
    sd_y = outcome_sd("combined_retrieval_coherence_all_mlmr_input.csv", "BLA_PRC", SLOW_GAMMA)
    ref = pm_bp[pm_bp["grp"] == "NonResp"]["val"]
    coefs = RET_FULL / "coherence_BLAMTL_LMMquad" / "coherence_BLA_PRC_slow_gamma_coefs.csv"
    fit_bp = aic_bic_delta_lmm(RET_FULL / "coherence_BLAMTL_LMMquad" / "coherence_BLA_PRC_slow_gamma_anova.csv")
    for grp, term, fdr_q in [("AntiResp", "QuadResponderGroupAntiResp", 0.012),
                             ("Strong", "QuadResponderGroupStrong", 0.026),
                             ("Moderate", "QuadResponderGroupModerate", 0.062)]:
        c = coef_row(coefs, term)
        x = pm_bp[pm_bp["grp"] == grp]["val"]
        d, g, n1 = cohen_d_vs_ref(x, ref)
        # trial-level standardized contrast (fit change is the omnibus responder-group factor, m0 vs m1)
        rows.append(dict(figure="2.13", table_ref="Table 5", analysis="LMM (coherence ~ responder group)",
                         region="BLA-PRC", modality="coherence", band="slow gamma", model="BLAMTL LMMquad (full)",
                         term=f"{grp} vs NonResp", unit_level="trial (LMM)", effect_type="std_beta",
                         estimate=round(c["estimate"] / sd_y, 3), ci_low=round(c["conf.low"] / sd_y, 3),
                         ci_high=round(c["conf.high"] / sd_y, 3), statistic=round(c["statistic"], 2), stat_type="t",
                         p_value=round(c["p.value"], 4), fdr_q=fdr_q, n_patients=int(n1), **fit_bp,
                         notes=f"std contrast = b/SD_outcome (SD={sd_y:.3f}); b={c['estimate']:.4f}, SE={c['std.error']:.4f}; "
                               "AIC/BIC/LRT = omnibus responder-group factor (m0 vs m1)"))
        # patient-level Cohen's d / Hedges g
        rows.append(dict(figure="2.13", table_ref="Table 5", analysis="LMM (coherence ~ responder group)",
                         region="BLA-PRC", modality="coherence", band="slow gamma", model="patient-level group means",
                         term=f"{grp} vs NonResp", unit_level="patient", effect_type="cohens_d",
                         estimate=round(d, 3), ci_low=None, ci_high=None, statistic=round(g, 3), stat_type="hedges_g",
                         p_value=None, fdr_q=fdr_q, n_patients=int(n1),
                         notes=f"vs NonResp (n={len(ref)}); SMALL n -> imprecise; Hedges g in statistic col"))

    # ---------- LMM responder groups: EC-PRC PAC slow gamma (theta phase -> slow gamma amp) ----------
    # Only 8 patients total (~2 per responder group) -> patient-level Cohen's d is NOT computed
    # (cannot estimate a within-group SD from 2 points); report trial-level std beta with caveat.
    sd_pac = outcome_sd("combined_retrieval_pac_all_mlmr_input.csv", "EC_PRC", SLOW_GAMMA)
    pac_coefs = RET_FULL / "pac_HPCrhinal_LMMquad" / "pac_EC_PRC_slow_gamma_coefs.csv"
    fit_pac = aic_bic_delta_lmm(RET_FULL / "pac_HPCrhinal_LMMquad" / "pac_EC_PRC_slow_gamma_anova.csv")
    for grp, term, fdr_q in [("AntiResp", "QuadResponderGroupAntiResp", 0.036),
                             ("Strong", "QuadResponderGroupStrong", 0.049)]:
        c = coef_row(pac_coefs, term)
        rows.append(dict(figure="2.14", table_ref="Table 6", analysis="LMM (PAC ~ responder group)",
                         region="EC-PRC", modality="pac", band="theta-slow gamma", model="HPCrhinal LMMquad (full)",
                         term=f"{grp} vs NonResp", unit_level="trial (LMM)", effect_type="std_beta",
                         estimate=round(c["estimate"] / sd_pac, 3), ci_low=round(c["conf.low"] / sd_pac, 3),
                         ci_high=round(c["conf.high"] / sd_pac, 3), statistic=round(c["statistic"], 2), stat_type="t",
                         p_value=round(c["p.value"], 4), fdr_q=fdr_q, n_patients=8, **fit_pac,
                         notes=f"std contrast = b/SD_outcome (SD={sd_pac:.4f}); b={c['estimate']:.5f}, SE={c['std.error']:.5f}; "
                               "AIC/BIC/LRT = omnibus responder-group factor (m0 vs m1); "
                               "<4 patients/group -> patient-level d not computed (insufficient)"))

    cols = ["figure", "table_ref", "analysis", "region", "modality", "band", "model", "term", "unit_level",
            "effect_type", "estimate", "ci_low", "ci_high", "statistic", "stat_type", "p_value", "fdr_q",
            "delta_aic", "delta_bic", "lrt_chisq", "lrt_df", "lrt_p", "n_patients", "notes"]
    out = pd.DataFrame(rows)
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[cols]
    out.to_csv(OUT_DIR / "model_effect_sizes.csv", index=False)

    # Markdown
    lines = ["# Model-based effect sizes (GLMM / LMM)\n",
             "Companion to cluster_effect_sizes.csv. Effect sizes here belong with the mixed models: "
             "odds ratios (logistic GLMM), standardized betas (LMM), and the patient-level correlations / "
             "Cohen's d that put patient-level predictors on their correct unit of analysis.\n",
             "**Caveat carried by every patient-level predictor here:** memory modulation and responder status "
             "are patient-level, but the LMMs are fit on trials, so the reported model df overstates precision "
             "(pseudoreplication). The patient-level r / d rows are the appropriate-unit effect sizes; the small "
             "group ns (3-5 per responder group; n=10 for EC-PRC) make those estimates imprecise.\n",
             "| Fig | Region | Modality/Band | Term | Unit | Effect | Estimate | 95% CI | stat | p | FDR q | ΔAIC | ΔBIC |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = "" if r.get("ci_low") is None or (isinstance(r.get("ci_low"), float) and np.isnan(r.get("ci_low"))) else f"[{r['ci_low']}, {r['ci_high']}]"
        stat = "" if r.get("statistic") in (None, "") else f"{r['statistic']} ({r['stat_type']})"
        p = "" if r.get("p_value") is None else r["p_value"]
        q = "" if r.get("fdr_q") is None else r["fdr_q"]
        da = r.get("delta_aic", ""); db = r.get("delta_bic", "")
        da = "" if da is None or (isinstance(da, float) and np.isnan(da)) else da
        db = "" if db is None or (isinstance(db, float) and np.isnan(db)) else db
        lines.append(f"| {r['figure']} | {r['region']} | {r['modality']}/{r['band']} | {r['term']} | "
                     f"{r['unit_level']} | {r['effect_type']} | {r['estimate']} | {ci} | {stat} | {p} | {q} | {da} | {db} |")
    (OUT_DIR / "model_effect_sizes.md").write_text("\n".join(lines) + "\n")

    pd.set_option("display.width", 240, "display.max_columns", 40)
    print(out.to_string(index=False))
    print(f"\nWrote: {OUT_DIR / 'model_effect_sizes.csv'}")
    print(f"Wrote: {OUT_DIR / 'model_effect_sizes.md'}")


if __name__ == "__main__":
    main()
