#!/usr/bin/env python
"""Diagnostic: what does the GLMM 'see' for HPC-PRC (ALLHPC_PRC) slow-gamma
coherence on stim-only encoding trials?

Three companion views to the existing raw-spectrum plot:
  1. Per-subject delta = mean(slow_gamma | remembered) - mean(slow_gamma | forgotten),
     one bar per subject. Visualizes the within-subject signal the GLMM picks up.
  2. Per-subject memory rate vs per-subject mean slow_gamma coherence
     (between-subject view; not the GLMM's actual focal axis).
  3. Trial-level scatter of slow_gamma value vs memory outcome with binned
     P(remembered) curve, overlaid with the GLMM-predicted curve from the
     existing fit (the actual focal axis of the model).

Output: OUTPUTS/encoding_memory_reports/stim_effect/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_IN = REPO_ROOT / "OUTPUTS" / "csvs" / "combined_encoding_coherence_all_mlmr_input.csv"
PRED_IN = (REPO_ROOT / "OUTPUTS" / "encoding_memory_reports" / "stats" /
           "h1abc_maineffect_encoding_mlm" / "coherence_HPCrhinal_stim" /
           "coherence_ALLHPC_PRC_slow_gamma_predictions.csv")
COEF_IN = (REPO_ROOT / "OUTPUTS" / "encoding_memory_reports" / "stats" /
           "h1abc_maineffect_encoding_mlm" / "coherence_HPCrhinal_stim" /
           "coherence_ALLHPC_PRC_slow_gamma_coefs.csv")
OUT_DIR = REPO_ROOT / "OUTPUTS" / "encoding_memory_reports" / "stim_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SG_LO, SG_HI = 30.27, 54.69
ALLHPC_PRC_SOURCES = ("HPC_PRC", "CA_PRC", "DG_PRC")


def build_allhpc_prc(df, freq_cols):
    sub = df[df["Region"].isin(ALLHPC_PRC_SOURCES)].copy()
    sub["trial_id"] = sub.groupby(["Patient", "Region"]).cumcount()
    grp_keys = ["Patient", "trial_id", "trial_type", "yes_or_no"]
    agg = (sub.groupby(grp_keys, as_index=False)[list(freq_cols)]
              .mean(numeric_only=True))
    agg["Region"] = "ALLHPC_PRC"
    return agg.drop(columns=["trial_id"])


def main():
    df = pd.read_csv(CSV_IN)
    df = df[(df["yes_or_no"].isin(["yes", "no"])) &
            (df["trial_type"].isin(["nostim", "stim"]))]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"

    freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in freq_cols])
    sg_cols = [c for c, f in zip(freq_cols, freqs) if SG_LO <= f <= SG_HI]

    allhpc = build_allhpc_prc(df, freq_cols)
    stim = allhpc[allhpc["trial_type"] == "stim"].copy()
    stim["slow_gamma"] = stim[sg_cols].mean(axis=1)
    stim["Accuracy"] = (stim["yes_or_no"] == "yes").astype(int)

    # ── Per-subject delta (remembered - forgotten in slow gamma) ────────────
    per_subj = (stim.groupby(["Patient", "yes_or_no"])["slow_gamma"]
                    .mean().unstack("yes_or_no"))
    per_subj = per_subj.rename(columns={"yes": "remembered", "no": "forgotten"})
    per_subj = per_subj.dropna(subset=["remembered", "forgotten"])
    per_subj["delta"] = per_subj["remembered"] - per_subj["forgotten"]
    per_subj = per_subj.sort_values("delta")
    pos = (per_subj["delta"] > 0).sum()
    neg = (per_subj["delta"] < 0).sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#a0a0a0" if d < 0 else "#2b2b2b" for d in per_subj["delta"]]
    ax.bar(range(len(per_subj)), per_subj["delta"], color=colors)
    ax.axhline(0, color="0.4", lw=1)
    ax.axhline(per_subj["delta"].mean(), color="#c0392b", ls="--", lw=1.5,
               label=f"mean delta = {per_subj['delta'].mean():+.4f}")
    ax.set_xticks(range(len(per_subj)))
    ax.set_xticklabels(per_subj.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("mean(slow gamma | remembered) - mean(slow gamma | forgotten)",
                  fontsize=10, fontweight="bold")
    ax.set_title(
        f"HPC-PRC stim-only slow-gamma coherence: within-subject delta\n"
        f"{pos} of {len(per_subj)} subjects have remembered > forgotten",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    out1 = OUT_DIR / "HPC_PRC_stim_slow_gamma_subject_deltas.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Trial-level scatter + binned curve + GLMM prediction ───────────────
    pred = pd.read_csv(PRED_IN)
    coefs = pd.read_csv(COEF_IN)
    band_c_row = coefs[coefs["term"] == "band_c"].iloc[0]
    or_str = (f"OR = {band_c_row['estimate']:.2f}  "
              f"[{band_c_row['conf.low']:.2f}, {band_c_row['conf.high']:.2f}],  "
              f"p = {band_c_row['p.value']:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.04, 0.04, size=len(stim))
    ax.scatter(stim["slow_gamma"], stim["Accuracy"] + jitter,
               s=10, alpha=0.18, color="#666666",
               label=f"{len(stim)} stim trials (jittered)")

    bins = pd.qcut(stim["slow_gamma"], 10, duplicates="drop")
    agg = stim.groupby(bins, observed=True).agg(
        x=("slow_gamma", "mean"),
        acc=("Accuracy", "mean"),
        n=("Accuracy", "size"),
    )
    agg["se"] = np.sqrt(agg["acc"] * (1 - agg["acc"]) / agg["n"])
    ax.errorbar(agg["x"], agg["acc"], yerr=agg["se"],
                fmt="o", color="#2b2b2b", ms=8, capsize=4, lw=1.5,
                label="Decile bins (mean +/- binomial SE)")

    ax.fill_between(pred["band"], pred["p_lo"], pred["p_hi"],
                    color="#c0392b", alpha=0.20, lw=0)
    ax.plot(pred["band"], pred["p_hat"], color="#c0392b", lw=2.4,
            label=f"GLMM P(remembered) +/- 95% CI  ({or_str})")

    ax.axhline(0.5, ls=":", color="0.5", lw=0.8)
    ax.set_xlabel("HPC-PRC slow-gamma coherence (per-trial mean over 30-55 Hz)",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("P(remembered)", fontsize=11, fontweight="bold")
    ax.set_title(
        "HPC-PRC stim-only slow gamma coherence vs memory: what the GLMM sees\n"
        "(trial-level data, deciles, and model fit)",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(-0.08, 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=1, fontsize=9, framealpha=0.92)
    fig.tight_layout()
    out2 = OUT_DIR / "HPC_PRC_stim_slow_gamma_trial_level.png"
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    print(f"\nSubject delta summary: pos={pos}, neg={neg}, "
          f"mean delta = {per_subj['delta'].mean():+.4f}")
    print(per_subj[["forgotten", "remembered", "delta"]].round(4))


if __name__ == "__main__":
    main()
