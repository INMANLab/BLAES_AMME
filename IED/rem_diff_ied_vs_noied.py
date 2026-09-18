"""Difference in odds of remembering: IED vs no-IED trials, encoding & retrieval.

GLMM  memory ~ ied + (1|patient)  (lme4::glmer, binomial/logit), ied = 1 (IED)
vs 0 (no-IED, reference). The ied coefficient is the odds ratio of remembering
for with-IED vs without-IED trials.

Reuses the per-patient counts built by rem_vs_forg_by_iedstatus.py.
Outputs -> OUTPUTS/updated_IED_figures/rem_diff_ied_vs_noied_stats.txt
"""

from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

import rem_vs_forg_by_iedstatus as base

IED_DIR = Path(__file__).resolve().parent
DATA_DIR = base.DATA_DIR
OUT_DIR = IED_DIR.parent / "OUTPUTS" / "updated_IED_figures"
R_SCRIPT = IED_DIR / "rem_diff_ied_vs_noied.R"


def build_long():
    base.build_datasets()  # ensures the per-patient ied / no-IED CSVs exist
    for tag in ("enc", "ret"):
        ied = pd.read_csv(DATA_DIR / f"{tag}_ied.csv").assign(ied=1)
        noied = pd.read_csv(DATA_DIR / f"{tag}_noied.csv").assign(ied=0)
        long = pd.concat([ied, noied], ignore_index=True)[["patient", "ied", "rem", "forg"]]
        long.to_csv(DATA_DIR / f"{tag}_long.csv", index=False)


def fmt_p(p):
    return "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".")


def main():
    build_long()
    res_csv = DATA_DIR / "glmm_diff_results.csv"
    subprocess.run(["Rscript", str(R_SCRIPT), str(DATA_DIR), str(res_csv)], check=True)
    res = pd.read_csv(res_csv)

    phase_name = {"enc": "Encoding", "ret": "Retrieval"}
    lines = ["Difference in odds of REMEMBERING: trials WITH IEDs vs WITHOUT IEDs",
             "GLMM: memory ~ ied + (1|patient)  (lme4::glmer, binomial/logit)",
             "ied = 1 (IED) vs 0 (no-IED, reference); OR > 1 => IED trials remembered MORE.",
             "Model fit: LRT of the ied term (~ ied vs ~ 1), both with the patient random",
             "intercept; dAIC/dBIC = full - null (negative favors adding ied), chi-square on 1 df.",
             "=" * 78, ""]
    sentences = []
    for _, r in res.iterrows():
        ph = phase_name[r["phase"]]
        long = pd.read_csv(DATA_DIR / f"{r['phase']}_long.csv")
        ied_rem, ied_n = long[long.ied == 1]["rem"].sum(), long[long.ied == 1][["rem", "forg"]].sum().sum()
        no_rem, no_n = long[long.ied == 0]["rem"].sum(), long[long.ied == 0][["rem", "forg"]].sum().sum()
        OR = np.exp(r["ied_coef"]); lo = np.exp(r["ied_coef"] - 1.96 * r["ied_se"])
        hi = np.exp(r["ied_coef"] + 1.96 * r["ied_se"])
        sig = "p" + fmt_p(r["ied_p"])
        dAIC = r["aic_full"] - r["aic_null"]; dBIC = r["bic_full"] - r["bic_null"]
        fit = (f"dAIC = {dAIC:+.1f}, dBIC = {dBIC:+.1f}, "
               f"chi-square({int(r['lrt_df'])}) = {r['lrt_chisq']:.2f}, p {fmt_p(r['lrt_p'])}")
        direction = ("no significant difference" if r["ied_p"] >= 0.05
                     else ("lower" if OR < 1 else "higher"))
        lines.append(f"{ph}:  with-IED {100*ied_rem/ied_n:.1f}% remembered ({int(ied_n)} trials)  "
                     f"vs  no-IED {100*no_rem/no_n:.1f}% ({int(no_n)} trials)")
        lines.append(f"  IED vs no-IED: OR(remember) = {OR:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], "
                     f"z = {r['ied_z']:.2f}, {sig}")
        lines.append(f"  model fit (adding ied): "
                     f"AIC {r['aic_full']:.1f} (+ied) vs {r['aic_null']:.1f} (intercept-only), "
                     f"BIC {r['bic_full']:.1f} (+ied) vs {r['bic_null']:.1f} (intercept-only), {fit}")
        lines.append("")
        if r["ied_p"] >= 0.05:
            sentences.append(
                f"{ph}: the odds of remembering did not differ between trials with and without IEDs "
                f"(with-IED {100*ied_rem/ied_n:.1f}% vs. no-IED {100*no_rem/no_n:.1f}% remembered; "
                f"GLMM memory ~ ied + (1|patient): OR = {OR:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], "
                f"z = {r['ied_z']:.2f}, p {fmt_p(r['ied_p'])}; adding ied did not improve fit over the "
                f"intercept-only model, {fit}).")
        else:
            sentences.append(
                f"{ph}: trials with IEDs were remembered at {direction} odds than trials without IEDs "
                f"(with-IED {100*ied_rem/ied_n:.1f}% vs. no-IED {100*no_rem/no_n:.1f}%; "
                f"GLMM: OR = {OR:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], z = {r['ied_z']:.2f}, "
                f"p {fmt_p(r['ied_p'])}; {fit}).")

    lines.append("-" * 78)
    lines.append("REPORTABLE SENTENCES")
    lines.append("-" * 78)
    lines.extend("- " + s for s in sentences)

    out = OUT_DIR / "rem_diff_ied_vs_noied_stats.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
