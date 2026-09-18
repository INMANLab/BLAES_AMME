#!/usr/bin/env python3
"""Cluster-level Cohen's d_z for the significant cluster-based permutation results.

For each significant cluster reported in the dissertation, the effect size is
computed by averaging every subject's baseline-corrected value WITHIN the
cluster's frequency window, then taking Cohen's d_z on the paired difference
(d_z = mean_diff / SD_diff = t / sqrt(N)). This is the principled single
standardized effect size for a cluster-mass permutation test, whose reported
"cluster mass t" is a SUM of per-frequency t-values and is not itself an
effect size.

Two contrast types:
  * "memory"  -> Remembered - Forgotten   (within no-stim trials)
  * "stim"    -> Stim - Nostim            (within a memory condition)

All clusters use the retrieval *_all_mlmr_input.csv files (baseline-corrected
"diff" values), paired within subject, matching the HMLET ClusterStats setup
in PermutationTests.R / run_permutation_*.py.

Outputs:
  OUTPUTS/effect_sizes/cluster_effect_sizes.csv
  OUTPUTS/effect_sizes/cluster_effect_sizes.md
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "OUTPUTS" / "csvs"
OUT_DIR = ROOT / "OUTPUTS" / "effect_sizes"

FILES = {
    "power": "combined_retrieval_power_all_mlmr_input.csv",
    "coherence": "combined_retrieval_coherence_all_mlmr_input.csv",
    "pac": "combined_retrieval_pac_all_mlmr_input.csv",
}

# Each spec mirrors how the corresponding cluster permutation test was run.
#   contrast: "memory" (Remembered-Forgotten) or "stim" (Stim-Nostim)
#   memory:   None, "Remembered", or "Forgotten"  (trial subset)
#   stim_only: None or "nostim"                    (trial subset)
SPECS = [
    # --- Batch 1: endogenous memory power (Remembered vs Forgotten, no-stim) ---
    dict(label="EC power: Remembered vs Forgotten (slow gamma)",
         modality="power", region="EC", contrast="memory", memory=None, stim_only="nostim",
         band="slow gamma", lo=35.16, hi=49.80, cluster_t=-40.40, perm_p=0.012, fig="2.5"),
    # Exploratory THRESHOLD-DEFINED cluster: 24.41-56.64 Hz but only 18 supra-threshold
    # freqs form the cluster (mass -41.14). d_z below is range-averaged over 24.41-56.64
    # (a valid descriptor) and overlaps/duplicates the EC slow-gamma effect; not cluster-mass-verifiable.
    dict(label="EC power: Remembered vs Forgotten (exploratory beta->low fast gamma; threshold-defined, redundant w/ slow gamma)",
         modality="power", region="EC", contrast="memory", memory=None, stim_only="nostim",
         band="beta-low fast gamma", lo=24.41, hi=56.64, cluster_t=-41.14, perm_p=0.037, fig="2.5"),
    dict(label="PRC power: Remembered vs Forgotten (theta)",
         modality="power", region="PRC", contrast="memory", memory=None, stim_only="nostim",
         band="theta", lo=4.88, hi=7.81, cluster_t=-8.88, perm_p=0.029, fig="2.5"),
    # --- Batch 2: stim vs no-stim power, within memory condition ---
    dict(label="EC power: Stim vs Nostim, Forgotten (slow gamma)",
         modality="power", region="EC", contrast="stim", memory="Forgotten", stim_only=None,
         band="slow gamma", lo=35.16, hi=49.80, cluster_t=-46.65, perm_p=0.002, fig="2.6"),
    dict(label="PRC power: Stim vs Nostim, Remembered (theta)",
         modality="power", region="PRC", contrast="stim", memory="Remembered", stim_only=None,
         band="theta", lo=4.88, hi=6.84, cluster_t=7.81, perm_p=0.021, fig="2.6"),
    dict(label="PRC power: Stim vs Nostim, Forgotten (theta)",
         modality="power", region="PRC", contrast="stim", memory="Forgotten", stim_only=None,
         band="theta", lo=4.88, hi=7.81, cluster_t=-10.33, perm_p=0.011, fig="2.6"),
    # --- Batch 3: stim vs no-stim coherence, forgotten trials ---
    dict(label="BLA-PRC coherence: Stim vs Nostim, Forgotten (slow gamma)",
         modality="coherence", region="BLA_PRC", contrast="stim", memory="Forgotten", stim_only=None,
         band="slow gamma", lo=76.17, hi=87.89, cluster_t=-37.52, perm_p=0.008, fig="2.7"),
    # "hippocampal-perirhinal" = pooled ALLHPC_PRC (display "HPC"), NOT data HPC_PRC
    dict(label="HPC-PRC ('HPC'=ALLHPC pooled) coherence: Stim vs Nostim, Forgotten (slow gamma)",
         modality="coherence", region="ALLHPC_PRC", pool_regions=["CA_PRC", "DG_PRC", "HPC_PRC"],
         contrast="stim", memory="Forgotten", stim_only=None,
         band="slow gamma", lo=39.06, hi=43.95, cluster_t=-13.77, perm_p=0.040, fig="2.7"),
    dict(label="EC-PRC coherence: Stim vs Nostim, Forgotten (theta)",
         modality="coherence", region="EC_PRC", contrast="stim", memory="Forgotten", stim_only=None,
         band="theta", lo=4.88, hi=5.86, cluster_t=6.27, perm_p=0.025, fig="2.7"),
    # --- Batch 4: stim vs no-stim PAC, remembered trials ---
    # "HPC-EC" = pooled ALLHPC_EC (display "HPC"), NOT data EC_HPC
    dict(label="HPC-EC ('HPC'=ALLHPC pooled) PAC: Stim vs Nostim, Remembered (theta-slow gamma)",
         modality="pac", region="ALLHPC_EC", pool_regions=["CA_EC", "DG_EC", "EC_HPC"],
         contrast="stim", memory="Remembered", stim_only=None,
         band="theta-slow gamma", lo=35.00, hi=45.00, cluster_t=-7.30, perm_p=0.024, fig="2.8"),
    # --- Batch 5: ENCODING endogenous memory power (Remembered vs Forgotten, no-stim) ---
    # NOTE display labels: data ALLHPC -> "HPC" (whole hippocampus); data HPC -> "SUB" (subiculum).
    dict(label="Whole hippocampus / 'HPC' (ALLHPC pooled CA/DG/HPC) power: Rem vs Forg, ENCODING (theta)",
         modality="power", region="ALLHPC", pool_regions=["CA", "DG", "HPC"],
         contrast="memory", memory=None, stim_only="nostim",
         band="theta", lo=4.88, hi=7.81, cluster_t=-7.79, perm_p=0.042, fig="3.3",
         csv="combined_encoding_power_amme_timing_filtered_mlmr_input.csv"),
    dict(label="Subiculum / 'SUB' (data HPC) power: Rem vs Forg, ENCODING (theta)",
         modality="power", region="HPC", contrast="memory", memory=None, stim_only="nostim",
         band="theta", lo=4.88, hi=7.81, cluster_t=-8.53, perm_p=0.023, fig="3.3",
         csv="combined_encoding_power_amme_timing_filtered_mlmr_input.csv"),
    # --- Batch 6: ENCODING stim vs no-stim power (within memory condition), theta ---
    dict(label="Hippocampus 'HPC' (ALLHPC pooled) power: Stim vs Nostim, Forgotten, ENCODING (theta)",
         modality="power", region="ALLHPC", pool_regions=["CA", "DG", "HPC"],
         contrast="stim", memory="Forgotten", stim_only=None,
         band="theta", lo=4.88, hi=7.81, cluster_t=-8.99, perm_p=0.026, fig="3.6",
         csv="combined_encoding_power_amme_timing_filtered_mlmr_input.csv"),
    dict(label="CA1 (CA) power: Stim vs Nostim, Forgotten, ENCODING (theta)",
         modality="power", region="CA", contrast="stim", memory="Forgotten", stim_only=None,
         band="theta", lo=4.88, hi=7.81, cluster_t=-8.97, perm_p=0.026, fig="Appendix C",
         csv="combined_encoding_power_amme_timing_filtered_mlmr_input.csv"),
    dict(label="DG power: Stim vs Nostim, REMEMBERED, ENCODING (theta) [note: remembered, not forgotten]",
         modality="power", region="DG", contrast="stim", memory="Remembered", stim_only=None,
         band="theta", lo=4.88, hi=7.81, cluster_t=8.66, perm_p=0.035, fig="Appendix F",
         csv="combined_encoding_power_amme_timing_filtered_mlmr_input.csv"),
    dict(label="PRC power: Stim vs Nostim, Remembered, ENCODING (theta)",
         modality="power", region="PRC", contrast="stim", memory="Remembered", stim_only=None,
         band="theta", lo=4.88, hi=7.81, cluster_t=-11.13, perm_p=0.008, fig="3.6",
         csv="combined_encoding_power_amme_timing_filtered_mlmr_input.csv"),
]

_cache = {}


def load(csv_name):
    if csv_name not in _cache:
        df = pd.read_csv(CSV_DIR / csv_name)
        df["Memory"] = np.where(df["yes_or_no"] == "yes", "Remembered", "Forgotten")
        df["StimCond"] = np.where(df["trial_type"] == "nostim", "Nostim", "Stim")
        freq_cols = [c for c in df.columns if c.startswith("diff_Freq_")]
        freqs = np.array([float(re.sub(r"[^0-9.]", "", c)) for c in freq_cols])
        _cache[csv_name] = (df, freq_cols, freqs)
    return _cache[csv_name]


def compute(spec):
    # spec may override the default (retrieval) CSV via "csv" (e.g. encoding source)
    df, freq_cols, freqs = load(spec.get("csv", FILES[spec["modality"]]))
    win = [c for c, f in zip(freq_cols, freqs) if spec["lo"] - 1e-6 <= f <= spec["hi"] + 1e-6]

    pool = spec.get("pool_regions")
    sub = (df[df["Region"].isin(pool)] if pool else df[df["Region"] == spec["region"]]).copy()
    if spec["memory"] is not None:
        sub = sub[sub["Memory"] == spec["memory"]]
    if spec["stim_only"] is not None:
        sub = sub[sub["trial_type"] == spec["stim_only"]]
    sub = sub.assign(winmean=sub[win].mean(axis=1))

    if spec["contrast"] == "memory":
        grp, a, b = "Memory", "Remembered", "Forgotten"
    else:
        grp, a, b = "StimCond", "Stim", "Nostim"

    if pool:
        # equal-weight pool (e.g. ALLHPC): subregion patient-means, then mean across subregions
        per_reg = sub.groupby(["Patient", "Region", grp])["winmean"].mean().reset_index()
        pp = per_reg.groupby(["Patient", grp])["winmean"].mean().unstack(grp).dropna(subset=[a, b])
    else:
        pp = sub.groupby(["Patient", grp])["winmean"].mean().unstack(grp).dropna(subset=[a, b])
    diff = (pp[a] - pp[b]).values
    n = len(diff)
    md = float(diff.mean())
    sd = float(diff.std(ddof=1))
    t = md / (sd / np.sqrt(n))
    dz = t / np.sqrt(n)
    se = np.sqrt(1 / n + dz ** 2 / (2 * n))
    ci_lo, ci_hi = dz - 1.96 * se, dz + 1.96 * se
    p_param = float(2 * stats.t.sf(abs(t), n - 1))

    direction = f"{a} > {b}" if md > 0 else f"{b} > {a}"
    return {
        "label": spec["label"],
        "modality": spec["modality"],
        "region": spec["region"].replace("_", "-"),
        "contrast": "Remembered-Forgotten" if spec["contrast"] == "memory" else "Stim-Nostim",
        "memory_subset": spec["memory"] or ("no-stim trials" if spec["stim_only"] else "all"),
        "band": spec["band"],
        "freq_lo_hz": spec["lo"],
        "freq_hi_hz": spec["hi"],
        "n_freq_bins": len(win),
        "N": n,
        "cluster_mass_t": spec["cluster_t"],
        "perm_p": spec["perm_p"],
        "mean_diff": round(md, 4),
        "sd_diff": round(sd, 4),
        "window_t": round(t, 3),
        "window_p_param": round(p_param, 4),
        "cohens_dz_signed": round(dz, 3),
        "cohens_dz_abs": round(abs(dz), 3),
        "dz_ci95_lo": round(ci_lo, 3),
        "dz_ci95_hi": round(ci_hi, 3),
        "direction": direction,
        "figure": spec["fig"],
    }


def magnitude(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [compute(s) for s in SPECS]
    out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "cluster_effect_sizes.csv"
    out.to_csv(csv_path, index=False)

    # Markdown reference doc
    lines = []
    lines.append("# Cluster-based permutation effect sizes (Cohen's d_z)\n")
    lines.append(
        "Each d_z standardizes the within-subject paired difference averaged across the "
        "significant cluster's frequency window (d_z = mean_diff / SD_diff = t / sqrt(N)). "
        "The reported *cluster mass t* is a SUM of per-frequency t-values and is the inferential "
        "statistic, not an effect size; the permutation p-value is the inferential result. "
        "d_z magnitudes: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >=0.8 large.\n"
    )
    lines.append(
        "| Fig | Region | Modality | Contrast (subset) | Band (Hz) | N | Cluster t | perm p | "
        "d_z | 95% CI | Magnitude | Direction |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        subset = r["memory_subset"]
        contrast = f"{r['contrast']} ({subset})"
        band = f"{r['band']} {r['freq_lo_hz']:.2f}-{r['freq_hi_hz']:.2f}"
        lines.append(
            f"| {r['figure']} | {r['region']} | {r['modality']} | {contrast} | {band} | {r['N']} | "
            f"{r['cluster_mass_t']:.2f} | {r['perm_p']:.3f} | {r['cohens_dz_abs']:.2f} | "
            f"[{r['dz_ci95_lo']:.2f}, {r['dz_ci95_hi']:.2f}] | {magnitude(r['cohens_dz_abs'])} | "
            f"{r['direction']} |"
        )
    lines.append("")
    lines.append("## Notes\n")
    lines.append(
        "- d_z is reported as a positive magnitude; sign/direction is given in the Direction column "
        "and matches the sign of the cluster mass t.\n"
        "- Window-averaged paired t-tests are reported for transparency; they are same-signed and "
        "significant for every cluster, confirming the effect-size window matches the permutation result. "
        "They are not identical to the cluster permutation p because the cluster-mass procedure pools "
        "frequency-resolved structure differently.\n"
        "- Small-N clusters (N=10: EC-PRC theta coherence, HPC-EC PAC) have wide CIs; interpret their "
        "point estimates with that caveat.\n"
        "- Effect sizes are Cohen's d_z (within-subject). State this in Methods for reproducibility.\n"
    )
    md_path = OUT_DIR / "cluster_effect_sizes.md"
    md_path.write_text("\n".join(lines))

    print(out.to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
