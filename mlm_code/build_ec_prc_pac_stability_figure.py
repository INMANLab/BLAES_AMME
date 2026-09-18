#!/usr/bin/env python
"""Build a 3x2 stability figure for the EC-PRC PAC slow-gamma stim-only
main-effect GLMM (FDR-significant, q = .027).

Panels:
  (1,1) Stim trials: PAC spectrum, remembered (dark red) vs forgotten
        (light red), group mean +/- SEM across subjects.
  (1,2) Subject-level variability: parallel-coordinates plot of each
        subject's mean PAC for forgotten -> remembered stim trials;
        group mean line overlaid.
  (2,1) Leave-one-subject-out (LOSO): per-LOSO-refit OR per 1 SD of
        slow-gamma PAC, with 95% CI.
  (2,2) Balanced trials: distribution of OR/SD across 100
        balanced-subsample refits, with observed and 95% percentile
        range.
  (3,1) Within-subject permutation null: histogram of focal-coefficient z
        under shuffled Accuracy labels; observed marked.
  (3,2) Per-subject deltas: bar chart of within-subject
        mean(PAC|remembered) - mean(PAC|forgotten).

Output:
  OUTPUTS/encoding_memory_reports/stim_effect/post_hoc_testing/
    EC_PRC_PAC_stability_summary.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
STIM_DIR = REPO / "OUTPUTS" / "encoding_memory_reports" / "stim_effect"
POSTHOC = STIM_DIR / "post_hoc_testing"

LOSO_CSV       = POSTHOC / "stability_pac_EC_PRC_slow_gamma_loso.csv"
DELTA_CSV      = POSTHOC / "stability_pac_EC_PRC_slow_gamma_deltas.csv"
DS_CSV         = POSTHOC / "stability_pac_EC_PRC_slow_gamma_downsample.csv"
SD_CSV         = POSTHOC / "EC_PRC_pac_stim_band_sd.csv"
PERM_CSV       = (POSTHOC / "effect_stability_testing" /
                  "EC_PRC_pac_permutation_null_z.csv")
SPECTRUM_CSV   = (POSTHOC / "effect_stability_testing" /
                  "EC_PRC_encoding_pac_stim_remembered_vs_forgotten.csv")

REM_COLOR  = "#8b0000"   # dark red
FORG_COLOR = "#e89d9d"   # light red / salmon


def main():
    # ── Load all data ─────────────────────────────────────────────────────
    loso     = pd.read_csv(LOSO_CSV)
    deltas   = pd.read_csv(DELTA_CSV)
    ds       = pd.read_csv(DS_CSV).dropna(subset=["OR", "p"])
    sd_band  = float(pd.read_csv(SD_CSV)["sd_band_c"].iloc[0])
    perm     = pd.read_csv(PERM_CSV)
    spec     = pd.read_csv(SPECTRUM_CSV)

    full = loso[loso["excluded"] == "<none>"].iloc[0]
    obs_OR    = float(full["OR"])
    obs_p     = float(full["p"])
    obs_OR_SD = float(np.power(obs_OR, sd_band))

    fig, axes = plt.subplots(3, 2, figsize=(17, 17))

    # ── (1,1) Spectrum: stim remembered vs forgotten ──────────────────────
    ax = axes[0, 0]
    ax.fill_between(spec["freq_hz"],
                    spec["forgotten_mean"] - spec["forgotten_sem"],
                    spec["forgotten_mean"] + spec["forgotten_sem"],
                    color=FORG_COLOR, alpha=0.30, lw=0)
    ax.plot(spec["freq_hz"], spec["forgotten_mean"], color=FORG_COLOR,
            lw=2.2, label="Stim forgotten")
    ax.fill_between(spec["freq_hz"],
                    spec["remembered_mean"] - spec["remembered_sem"],
                    spec["remembered_mean"] + spec["remembered_sem"],
                    color=REM_COLOR, alpha=0.25, lw=0)
    ax.plot(spec["freq_hz"], spec["remembered_mean"], color=REM_COLOR,
            lw=2.2, label="Stim remembered")
    ax.axhline(0, color="0.5", ls=":", lw=0.8)
    ax.set_xlim(spec["freq_hz"].min(), spec["freq_hz"].max())
    ax.set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Encoding PAC (stim - baseline)",
                  fontsize=11, fontweight="bold")
    ax.set_title(
        "A. EC-PRC PAC spectrum, stim trials\n"
        "remembered vs forgotten (group mean +/- SEM)",
        fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.25)

    # ── (1,2) Subject-level variability (parallel coordinates) ────────────
    ax = axes[0, 1]
    d_clean = deltas.dropna(subset=["delta",
                                    "band_c_mean_yes", "band_c_mean_no"])
    for _, r in d_clean.iterrows():
        # Effect direction is NEGATIVE (OR<1): remembered PAC LOWER than forgotten.
        # Dark = direction matches the GLMM (rem < forg)
        direction_match = r["band_c_mean_yes"] < r["band_c_mean_no"]
        color = "#2b2b2b" if direction_match else "#a0a0a0"
        ax.plot([0, 1], [r["band_c_mean_no"], r["band_c_mean_yes"]],
                "-o", color=color, lw=1.6, ms=5, alpha=0.85)
        ax.annotate(r["Patient"], xy=(1.02, r["band_c_mean_yes"]),
                    fontsize=7.5, color="0.35", va="center")
    forg_mean = float(d_clean["band_c_mean_no"].mean())
    rem_mean  = float(d_clean["band_c_mean_yes"].mean())
    ax.plot([0, 1], [forg_mean, rem_mean], "-o",
            color=REM_COLOR, lw=3, ms=9, label="Group mean", zorder=20)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Forgotten", "Remembered"],
                                              fontsize=11)
    ax.set_xlim(-0.15, 1.25)
    ax.set_ylabel("Per-subject mean slow-gamma PAC\n(within stim trials)",
                  fontsize=11, fontweight="bold")
    n_match = int((d_clean["band_c_mean_yes"]
                   < d_clean["band_c_mean_no"]).sum())
    ax.set_title(
        f"B. Subject-level variability\n"
        f"{n_match} of {len(d_clean)} subjects show remembered < forgotten "
        f"(GLMM direction)",
        fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.2, axis="y")

    # ── (2,1) LOSO with OR per SD + 95% CI ────────────────────────────────
    ax = axes[1, 0]
    rest = loso[loso["excluded"] != "<none>"].dropna(subset=["OR"]).copy()
    rest["OR_SD"] = np.power(rest["OR"], sd_band)
    rest["lo_SD"] = np.power(rest["lo"], sd_band)
    rest["hi_SD"] = np.power(rest["hi"], sd_band)
    rest = rest.sort_values("OR_SD").reset_index(drop=True)
    y = np.arange(len(rest))
    colors = ["#2b2b2b" if p < 0.05 else "#a0a0a0" for p in rest["p"]]
    for i, c in enumerate(colors):
        xerr_lo = rest["OR_SD"].iloc[i] - rest["lo_SD"].iloc[i]
        xerr_hi = rest["hi_SD"].iloc[i] - rest["OR_SD"].iloc[i]
        ax.errorbar(rest["OR_SD"].iloc[i], y[i],
                    xerr=[[xerr_lo], [xerr_hi]],
                    fmt="o", color=c, ecolor="0.55",
                    capsize=3, ms=6, lw=1.0)
    ax.axvline(1.0, color="0.4", lw=1, ls=":")
    ax.axvline(obs_OR_SD, color=REM_COLOR, ls="--", lw=1.5,
               label=f"Full-sample OR/SD = {obs_OR_SD:.2f}")
    ax.set_yticks(y); ax.set_yticklabels(rest["excluded"], fontsize=9)
    ax.set_xlabel(f"LOSO OR per 1 SD of slow-gamma PAC (95% CI)\n"
                  f"SD = {sd_band:.4f}",
                  fontsize=10, fontweight="bold")
    n_sig = int((rest["p"] < 0.05).sum())
    p_min = float(rest["p"].min()); p_max = float(rest["p"].max())
    ax.set_title(
        f"C. Leave-one-subject-out  ({n_sig}/{len(rest)} Wald p<.05)\n"
        f"LOSO Wald p range: [{p_min:.3f}, {p_max:.3f}]",
        fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25, axis="x")

    # ── (2,2) Balanced trials (downsample) ────────────────────────────────
    ax = axes[1, 1]
    # Convert downsample OR to OR per SD
    ds["OR_SD"] = np.power(ds["OR"], sd_band)
    ds_lo = float(ds["OR_SD"].quantile(0.025))
    ds_hi = float(ds["OR_SD"].quantile(0.975))
    ds_med = float(ds["OR_SD"].median())
    ax.hist(ds["OR_SD"], bins=30, color="#888888", edgecolor="white")
    ax.axvline(1.0, color="0.4", lw=1, ls=":", label="OR/SD = 1")
    ax.axvline(obs_OR_SD, color=REM_COLOR, ls="--", lw=1.8,
               label=f"Observed = {obs_OR_SD:.2f}")
    ax.axvspan(ds_lo, ds_hi, color=REM_COLOR, alpha=0.10,
               label=f"95% range: [{ds_lo:.2f}, {ds_hi:.2f}]")
    n_below = int((ds["p"] < 0.05).sum())
    ax.set_xlabel("OR per 1 SD across balanced-subsample refits",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("count", fontsize=10, fontweight="bold")
    ax.set_title(
        f"D. Balanced trials  ({len(ds)} refits)\n"
        f"{n_below}/{len(ds)} ({100*n_below/len(ds):.0f}%) with p<.05;  "
        f"median OR/SD = {ds_med:.2f}",
        fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)

    # ── (3,1) Permutation null ────────────────────────────────────────────
    ax = axes[2, 0]
    obs_perm_z = float(perm["observed_z"].iloc[0])
    null = perm["z_null"].dropna().to_numpy()
    p_two = float(np.mean(np.abs(null) >= abs(obs_perm_z)))
    p_dir = (float(np.mean(null >= obs_perm_z)) if obs_perm_z >= 0
             else float(np.mean(null <= obs_perm_z)))
    ax.hist(null, bins=40, color="#a0a0a0", edgecolor="white")
    ax.axvline(0, color="0.4", lw=1, ls=":")
    ax.axvline(obs_perm_z, color=REM_COLOR, ls="--", lw=2.2,
               label=f"Observed z = {obs_perm_z:+.2f}")
    ax.set_xlabel("z under within-subject permutation null",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("count", fontsize=10, fontweight="bold")
    ax.set_title(
        f"E. Within-subject permutation  ({len(null)} iters)\n"
        f"two-sided p_perm = {p_two:.4f}   "
        f"directional p_perm = {p_dir:.4f}",
        fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)

    # ── (3,2) Per-subject deltas ──────────────────────────────────────────
    ax = axes[2, 1]
    d2 = deltas.dropna(subset=["delta"]).copy().sort_values("delta")
    d2 = d2.reset_index(drop=True)
    # Dark = direction matches GLMM (negative delta = rem < forg)
    colors = ["#2b2b2b" if v < 0 else "#a0a0a0" for v in d2["delta"]]
    ax.bar(range(len(d2)), d2["delta"], color=colors)
    ax.axhline(0, color="0.4", lw=1)
    mean_delta = float(d2["delta"].mean())
    ax.axhline(mean_delta, color=REM_COLOR, ls="--", lw=1.5,
               label=f"mean = {mean_delta:+.4g}")
    ax.set_xticks(range(len(d2)))
    ax.set_xticklabels(d2["Patient"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(
        "mean(slow-gamma PAC | remembered) - mean(... | forgotten)\n"
        "within stim trials, per subject",
        fontsize=9, fontweight="bold")
    n_neg = int((d2["delta"] < 0).sum())
    ax.set_title(
        f"F. Per-subject deltas  ({n_neg}/{len(d2)} match GLMM direction)",
        fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"Stability: EC-PRC slow-gamma PAC vs memory (stim-only, encoding)\n"
        f"Full sample: OR/SD = {obs_OR_SD:.2f}, "
        f"Wald p = {obs_p:.4f} (FDR q = .027)",
        fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()

    out_png = POSTHOC / "EC_PRC_PAC_stability_summary.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
