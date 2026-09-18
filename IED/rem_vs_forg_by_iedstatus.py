"""Remembered vs forgotten within IED and no-IED trials (encoding & retrieval).

Builds per-patient remembered/forgotten counts for four groups:
  encoding IED, encoding no-IED, retrieval IED, retrieval no-IED
then fits an intercept-only mixed-effects logistic regression to each
(rem_vs_forg_by_iedstatus.R, lme4::glmer) and writes a formatted stats file with
effect sizes (odds ratio, Cohen's g) and reportable sentences.

IED counts come straight from the IED detection files (authoritative). no-IED =
per-patient total (reconstructed from raw LFP files) - IED, the same subtraction
used by the ratio figure (avoids the BLAES trial-id mismatch in the raw<->IED join).

Outputs -> OUTPUTS/updated_IED_figures/
"""

from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

import ied_memory_ratio_by_iedstatus as core

IED_DIR = Path(__file__).resolve().parent
DATA_DIR = IED_DIR / "outputs" / "rem_vs_forg_iedstatus"
OUT_DIR = IED_DIR.parent / "OUTPUTS" / "updated_IED_figures"
R_SCRIPT = IED_DIR / "rem_vs_forg_by_iedstatus.R"


def ied_per_patient(csv):
    d = pd.read_csv(csv)
    d = d[d["MemoryOutcome"].isin(["remembered", "forgotten"])]
    t = d.groupby(["Patient", "Trial"])["MemoryOutcome"].first().reset_index()
    g = t.groupby("Patient")["MemoryOutcome"].agg(
        rem=lambda s: (s == "remembered").sum(),
        forg=lambda s: (s == "forgotten").sum()).reset_index()
    g.columns = ["patient", "rem", "forg"]
    return g


def total_per_patient(phase, patients):
    recon = core.reconstruct(phase, patients)
    recon = recon[recon["memory"].isin(["remembered", "forgotten"])]
    g = recon.groupby("patient")["memory"].agg(
        rem=lambda s: (s == "remembered").sum(),
        forg=lambda s: (s == "forgotten").sum()).reset_index()
    return g


def build_datasets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    enc_ied_full = pd.read_csv(core.IED_ENC)
    ret_ied_full = pd.read_csv(core.IED_RET)
    enc_pts = sorted(enc_ied_full["Patient"].unique())
    ret_pts = sorted(ret_ied_full[ret_ied_full["MemoryOutcome"].isin(
        ["remembered", "forgotten"])]["Patient"].unique())

    for phase, ied_csv, pts, tag in [
        ("phase1", core.IED_ENC, enc_pts, "enc"),
        ("phase3", core.IED_RET, ret_pts, "ret"),
    ]:
        ied = ied_per_patient(ied_csv)
        tot = total_per_patient(phase, pts)
        merged = tot.merge(ied, on="patient", how="left", suffixes=("_tot", "_ied")).fillna(0)
        merged["rem"] = (merged["rem_tot"] - merged["rem_ied"]).clip(lower=0).astype(int)
        merged["forg"] = (merged["forg_tot"] - merged["forg_ied"]).clip(lower=0).astype(int)
        ied.to_csv(DATA_DIR / f"{tag}_ied.csv", index=False)
        merged[["patient", "rem", "forg"]].to_csv(DATA_DIR / f"{tag}_noied.csv", index=False)


def fmt_p(p):
    return "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".")


def g_label(g):
    a = abs(g)
    return "large" if a >= 0.25 else ("medium" if a >= 0.15 else "small")


def main():
    build_datasets()
    res_csv = DATA_DIR / "glmm_results.csv"
    subprocess.run(["Rscript", str(R_SCRIPT), str(DATA_DIR), str(res_csv)], check=True)
    res = pd.read_csv(res_csv)

    lines = ["Remembered vs forgotten within IED and no-IED trials",
             "Intercept-only mixed-effects logistic regression: memory ~ 1 + (1|patient)",
             "(lme4::glmer, binomial/logit; same framework as the IED multiwindow GLMMs)",
             "Effect size: OR = odds of remembering (exp intercept); Cohen's g = P(rem) - 0.5.",
             "Model fit: LRT of the fixed intercept (~ 1 vs ~ 0, log-odds fixed at chance),",
             "both with the patient random intercept; dAIC/dBIC = full - null (negative favors",
             "the fitted model) and chi-square(df) on 1 df.",
             "=" * 78, ""]
    sentences = []
    for _, r in res.iterrows():
        OR = np.exp(r["intercept"]); lo = np.exp(r["intercept"] - 1.96 * r["se"])
        hi = np.exp(r["intercept"] + 1.96 * r["se"])
        prop = r["rem"] / r["n"]; g = prop - 0.5
        dAIC = r["aic_full"] - r["aic_null"]; dBIC = r["bic_full"] - r["bic_null"]
        lines.append(f"{r['phase']} | {r['group']} trials")
        lines.append(f"  N = {int(r['n'])} ({int(r['rem'])} remembered, {int(r['forg'])} forgotten), "
                     f"{int(r['npat'])} patients | {100*prop:.1f}% remembered")
        lines.append(f"  intercept = {r['intercept']:.3f} (SE {r['se']:.3f}), z = {r['z']:.2f}, p {fmt_p(r['p'])}")
        lines.append(f"  OR(remember) = {OR:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] | "
                     f"odds of forgetting = {np.exp(-r['intercept']):.2f} | "
                     f"Cohen's g = {g:.2f} ({g_label(g)})")
        lines.append(f"  model fit: "
                     f"AIC {r['aic_full']:.1f} (fitted intercept) vs {r['aic_null']:.1f} (chance), "
                     f"dAIC {dAIC:+.1f}; "
                     f"BIC {r['bic_full']:.1f} (fitted intercept) vs {r['bic_null']:.1f} (chance), "
                     f"dBIC {dBIC:+.1f}; "
                     f"LRT chi-square({int(r['lrt_df'])}) = {r['lrt_chisq']:.2f}, p {fmt_p(r['lrt_p'])}")
        lines.append("")
        sentences.append(
            f"{r['phase']} ({r['group']}): items were remembered significantly more often than "
            f"forgotten ({int(r['rem'])} [{100*prop:.1f}%] vs. {int(r['forg'])}; N = {int(r['n'])}, "
            f"{int(r['npat'])} patients), intercept-only mixed-effects logistic regression "
            f"OR = {OR:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], z = {r['z']:.2f}, p {fmt_p(r['p'])}, "
            f"Cohen's g = {g:.2f} (vs. a chance-level model: dAIC = {dAIC:+.1f}, dBIC = {dBIC:+.1f}, "
            f"chi-square({int(r['lrt_df'])}) = {r['lrt_chisq']:.2f}, p {fmt_p(r['lrt_p'])}).")

    lines.append("-" * 78)
    lines.append("REPORTABLE SENTENCES")
    lines.append("-" * 78)
    lines.extend("- " + s for s in sentences)

    out = OUT_DIR / "rem_vs_forg_by_iedstatus_stats.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
