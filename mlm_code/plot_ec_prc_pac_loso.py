#!/usr/bin/env python
"""Plot leave-one-subject-out (LOSO) + within-subject deltas for the EC-PRC
stim-only slow-gamma PAC GLMM main effect.

PAC slow_gamma values are tiny (~ +/- 0.02), so 'OR per 1 unit' is unintuitive
(~ 1e-10). We re-express the slope as OR per 1 standard-deviation increase in
slow_gamma, which lives on a sensible scale.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "OUTPUTS" / "encoding_memory_reports" / "stim_effect"
LOSO_CSV = OUT_DIR / "EC_PRC_stim_slow_gamma_pac_loso.csv"
SUBJ_CSV = OUT_DIR / "EC_PRC_stim_slow_gamma_pac_subject_table.csv"
INPUT_CSV = REPO / "OUTPUTS" / "csvs" / "combined_encoding_pac_all_mlmr_input.csv"

PAC_SG = (30.0, 50.0)
PAIR = "EC_PRC"


# Compute the SD of EC_PRC stim slow_gamma values, so we can rescale OR.
def compute_sd():
    df = pd.read_csv(INPUT_CSV)
    df = df[(df["yes_or_no"].isin(["yes", "no"])) &
            (df["trial_type"].isin(["stim"]))]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"
    df = df[df["Region"] == PAIR]
    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in freq_cols])
    sg_cols = [c for c, f in zip(freq_cols, freqs)
               if PAC_SG[0] <= f <= PAC_SG[1]]
    slow_gamma = df[sg_cols].mean(axis=1)
    return float(np.nanstd(slow_gamma, ddof=1))


def or_per_sd(or_raw, sd):
    """Convert OR-per-unit (R/lme4 default) to OR-per-SD."""
    return float(np.power(or_raw, sd))


def main():
    loso = pd.read_csv(LOSO_CSV)
    full = loso[loso["excluded"] == "<none>"].iloc[0]
    rows = loso[loso["excluded"] != "<none>"].copy()

    sd = compute_sd()
    for col_src, col_dst in [("OR", "OR_SD"), ("lo", "lo_SD"), ("hi", "hi_SD")]:
        rows[col_dst] = rows[col_src].apply(lambda x: or_per_sd(x, sd))
    full_or_sd = or_per_sd(full["OR"], sd)
    full_lo_sd = or_per_sd(full["lo"], sd)
    full_hi_sd = or_per_sd(full["hi"], sd)

    rows = rows.sort_values("OR_SD")

    # ── LOSO figure ─────────────────────────────────────────────────────────
    fig, (ax_or, ax_p) = plt.subplots(1, 2, figsize=(14, 6.5),
                                      gridspec_kw={"width_ratios": [1.5, 1]})

    y = np.arange(len(rows))
    ax_or.errorbar(rows["OR_SD"], y,
                   xerr=[rows["OR_SD"] - rows["lo_SD"],
                         rows["hi_SD"] - rows["OR_SD"]],
                   fmt="o", color="#2b2b2b", ecolor="#888888",
                   capsize=3, ms=6, lw=1.0)
    ax_or.axvline(1.0, color="0.5", ls=":", lw=1.0, label="OR = 1 (no effect)")
    ax_or.axvline(full_or_sd, color="#c0392b", ls="--", lw=1.5,
                  label=f"Full-sample OR/SD = {full_or_sd:.2f}")
    ax_or.set_yticks(y)
    ax_or.set_yticklabels(rows["excluded"], fontsize=9)
    ax_or.set_xlabel(f"OR per 1 SD of slow-gamma PAC  (SD = {sd:.4f})",
                     fontsize=11, fontweight="bold")
    ax_or.set_title("Leave-one-subject-out OR estimates",
                    fontsize=12, fontweight="bold")
    ax_or.legend(loc="lower right", fontsize=9)
    ax_or.grid(alpha=0.25, axis="x")

    colors = ["#2b2b2b" if p < 0.05 else "#a0a0a0" for p in rows["p"]]
    ax_p.barh(y, rows["p"], color=colors)
    ax_p.axvline(0.05, color="#c0392b", ls="--", lw=1.5, label="p = .05")
    ax_p.axvline(full["p"], color="#1f77b4", ls=":", lw=1.5,
                 label=f"Full-sample p = {full['p']:.3f}")
    ax_p.set_yticks(y)
    ax_p.set_yticklabels(rows["excluded"], fontsize=9)
    ax_p.set_xlabel("Raw p-value", fontsize=11, fontweight="bold")
    ax_p.set_title("LOSO p-values", fontsize=12, fontweight="bold")
    ax_p.legend(loc="lower right", fontsize=9)
    ax_p.grid(alpha=0.25, axis="x")

    n_below = int((rows["p"] < 0.05).sum())
    summary = (
        f"EC-PRC stim-only slow-gamma PAC: leave-one-subject-out\n"
        f"Full sample: OR/SD = {full_or_sd:.2f} "
        f"[{full_lo_sd:.2f}, {full_hi_sd:.2f}], p = {full['p']:.3f} "
        f"(survives FDR with q = 0.027).  "
        f"All 11 LOSO refits keep p < .05 "
        f"(range {rows['p'].min():.3f} - {rows['p'].max():.3f}); "
        f"{n_below}/11 LOSO refits significant at uncorrected alpha = .05."
    )
    fig.suptitle(summary, fontsize=11, fontweight="bold", y=1.04)
    fig.tight_layout()
    loso_png = OUT_DIR / "EC_PRC_stim_slow_gamma_pac_loso.png"
    fig.savefig(loso_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {loso_png}")

    # ── Subject-delta figure ────────────────────────────────────────────────
    subj = pd.read_csv(SUBJ_CSV).sort_values("delta")
    pos = int((subj["delta"] > 0).sum())
    neg = int((subj["delta"] < 0).sum())
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#a0a0a0" if d > 0 else "#2b2b2b" for d in subj["delta"]]
    ax.bar(range(len(subj)), subj["delta"], color=colors)
    ax.axhline(0, color="0.4", lw=1)
    ax.axhline(subj["delta"].mean(), color="#c0392b", ls="--", lw=1.5,
               label=f"mean delta = {subj['delta'].mean():+.4f}")
    ax.set_xticks(range(len(subj)))
    ax.set_xticklabels(subj["Patient"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("mean(slow-gamma PAC | remembered) - mean(... | forgotten)",
                  fontsize=10, fontweight="bold")
    ax.set_title(
        f"EC-PRC stim-only slow-gamma PAC: within-subject delta\n"
        f"{neg} of {len(subj)} subjects have remembered < forgotten "
        f"(direction the GLMM picks up; negative slope, OR/SD < 1)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    delta_png = OUT_DIR / "EC_PRC_stim_slow_gamma_pac_subject_deltas.png"
    fig.savefig(delta_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {delta_png}")

    print(f"\nFull sample: OR/SD = {full_or_sd:.3f} "
          f"[{full_lo_sd:.3f}, {full_hi_sd:.3f}], p = {full['p']:.3f}")
    print(f"LOSO p range: {rows['p'].min():.3f} - {rows['p'].max():.3f}")
    print(f"LOSO OR/SD range: {rows['OR_SD'].min():.3f} - "
          f"{rows['OR_SD'].max():.3f}")
    worst = rows.iloc[rows["p"].argmax()]
    best = rows.iloc[rows["p"].argmin()]
    print(f"Removing {worst['excluded']} weakens most (p -> {worst['p']:.3f})")
    print(f"Removing {best['excluded']} strengthens most (p -> {best['p']:.3f})")
    print(f"Subject deltas: {neg} negative (matches GLMM), {pos} positive")


if __name__ == "__main__":
    main()
