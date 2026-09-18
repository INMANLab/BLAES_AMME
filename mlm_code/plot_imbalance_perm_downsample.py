#!/usr/bin/env python
"""Plot permutation null and downsample sensitivity for the two encoding
stim-only slow-gamma main effects.

Reads CSVs written by diag_imbalance_perm_downsample.R and writes a side-by-side
permutation-vs-downsample summary PNG per effect to:
  OUTPUTS/encoding_memory_reports/stim_effect/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "OUTPUTS" / "encoding_memory_reports" / "stim_effect"

EFFECTS = [
    {
        "label": "HPC_PRC_coherence",
        "title": "HPC-PRC stim-only slow-gamma coherence (q_FDR = 0.058, marginal)",
        "or_panel_xlim": (0, 12),  # OR plotted in linear units
        "or_scale": "raw",
    },
    {
        "label": "EC_PRC_pac",
        "title": "EC-PRC stim-only slow-gamma PAC (q_FDR = 0.027, FDR-survivor)",
        # PAC ORs are extreme on the raw scale; we'll plot OR per 1 SD instead
        "or_panel_xlim": (0.5, 1.2),
        "or_scale": "per_sd",
        "sd": 0.0095,  # matches plot_ec_prc_pac_loso.py SD of slow-gamma PAC
    },
]


def two_sided_p(observed, null):
    null = null[~np.isnan(null)]
    if len(null) == 0:
        return np.nan
    return float(np.mean(np.abs(null) >= abs(observed)))


def directional_p(observed, null):
    null = null[~np.isnan(null)]
    if len(null) == 0:
        return np.nan
    if observed >= 0:
        return float(np.mean(null >= observed))
    return float(np.mean(null <= observed))


def render(effect):
    label = effect["label"]
    perm_csv = OUT / f"{label}_permutation_null_z.csv"
    ds_csv   = OUT / f"{label}_downsample_estimates.csv"
    if not perm_csv.exists() or not ds_csv.exists():
        print(f"  SKIP {label}: input CSVs not found.")
        return

    perm = pd.read_csv(perm_csv)
    ds = pd.read_csv(ds_csv)

    obs_z = float(perm["observed_z"].iloc[0])
    obs_or = float(perm["observed_or"].iloc[0])
    obs_p = float(perm["observed_p"].iloc[0])
    n_perm = len(perm)

    null_z = perm["z_null"].to_numpy(dtype=float)
    p_two = two_sided_p(obs_z, null_z)
    p_dir = directional_p(obs_z, null_z)

    # Downsample summary
    ds_clean = ds.dropna(subset=["or", "p"])
    n_ds = len(ds_clean)
    if effect["or_scale"] == "per_sd":
        sd = effect["sd"]
        or_vals = np.power(ds_clean["or"].to_numpy(), sd)
        obs_or_panel = float(np.power(obs_or, sd))
    else:
        or_vals = ds_clean["or"].to_numpy()
        obs_or_panel = obs_or

    med_or = float(np.median(or_vals)) if n_ds else np.nan
    lo_or  = float(np.quantile(or_vals, 0.025)) if n_ds else np.nan
    hi_or  = float(np.quantile(or_vals, 0.975)) if n_ds else np.nan
    p_sig_frac = float(np.mean(ds_clean["p"] < 0.05)) if n_ds else np.nan
    p_med = float(np.median(ds_clean["p"])) if n_ds else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: permutation null histogram of z-statistic
    ax = axes[0]
    valid_z = null_z[~np.isnan(null_z)]
    ax.hist(valid_z, bins=50, color="#a0a0a0", edgecolor="white")
    ax.axvline(obs_z, color="#c0392b", lw=2.2,
               label=f"Observed z = {obs_z:+.2f}")
    ax.axvline(0, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel("band_c z-statistic under null", fontsize=11, fontweight="bold")
    ax.set_ylabel("count", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Permutation null (n = {len(valid_z)} valid of {n_perm})\n"
        f"two-sided p_perm = {p_two:.4f}   directional p_perm = {p_dir:.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25)

    # Panel 2: downsample OR distribution
    ax = axes[1]
    ax.hist(or_vals, bins=30, color="#2b2b2b", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="0.5", ls=":", lw=1.0, label="OR = 1")
    ax.axvline(obs_or_panel, color="#c0392b", lw=2.2,
               label=(f"Full-sample OR/SD = {obs_or_panel:.2f}"
                      if effect["or_scale"] == "per_sd"
                      else f"Full-sample OR = {obs_or_panel:.2f}"))
    ax.axvline(med_or, color="#2980b9", ls="--", lw=1.8,
               label=f"DS median = {med_or:.2f}  [{lo_or:.2f}, {hi_or:.2f}]")
    xlim = effect["or_panel_xlim"]
    ax.set_xlim(*xlim)
    ax.set_xlabel(("OR per 1 SD" if effect["or_scale"] == "per_sd"
                   else "OR per 1 unit"), fontsize=11, fontweight="bold")
    ax.set_ylabel("count", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Balanced-subsample OR (n = {n_ds} refits)\n"
        f"95% range: [{lo_or:.2f}, {hi_or:.2f}]",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    # Panel 3: downsample p-value distribution
    ax = axes[2]
    pvals = ds_clean["p"].to_numpy()
    ax.hist(pvals, bins=np.linspace(0, 1, 41),
            color="#a0a0a0", edgecolor="white")
    ax.axvline(0.05, color="#c0392b", ls="--", lw=1.8, label="p = .05")
    ax.axvline(obs_p, color="#1f77b4", ls=":", lw=1.8,
               label=f"Full-sample p = {obs_p:.3f}")
    ax.axvline(p_med, color="#2980b9", ls="-.", lw=1.5,
               label=f"DS median p = {p_med:.3f}")
    ax.set_xlabel("p-value", fontsize=11, fontweight="bold")
    ax.set_ylabel("count", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Balanced-subsample p-values\n"
        f"{int(p_sig_frac * n_ds)}/{n_ds} ({p_sig_frac * 100:.1f}%) below .05",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(effect["title"], fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    png = OUT / f"{label}_imbalance_diagnostics.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"  perm p_two = {p_two:.4f}  perm p_dir = {p_dir:.4f}")
    print(f"  DS OR median = {med_or:.3f}  [{lo_or:.3f}, {hi_or:.3f}]")
    print(f"  DS p<.05 in {int(p_sig_frac * n_ds)}/{n_ds}")


def main():
    for eff in EFFECTS:
        render(eff)


if __name__ == "__main__":
    main()
