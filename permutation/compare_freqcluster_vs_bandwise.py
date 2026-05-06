#!/usr/bin/env python
"""Compare freq-cluster permutation results vs band-wise permutation results.

Covers all 4 contrasts (retrieval/encoding x stim_nostim/endogenous_memory),
all 3 modalities. For stim_nostim contrasts, the band-wise CSVs include a
'memory' column (Rem/Forg). For endogenous_memory contrasts, there is no
memory split.

For each (region, memory if applicable, band) cell, report:
  - Freq-cluster: did any significant cluster overlap this band's freq range?
  - Band-wise: t-stat and p-value for this band.
  - Status: 'agree_sig', 'agree_ns', 'gain' (only band sig), 'lose'
    (only freq-cluster sig).

Outputs CSV + summary to stdout:
  outputs/PermutationOutputsAlireza/comparison_freqcluster_vs_bandwise.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
BASE = ROOT / "outputs" / "PermutationOutputsAlireza"

BANDS = {
    "delta":      (1.95, 3.0),
    "theta":      (4.0,  8.0),
    "alpha":      (9.0,  13.0),
    "beta":       (14.0, 29.0),
    "slow_gamma": (30.0, 55.0),
    "HFA":        (70.0, 100.0),
}
CONFIRMATORY = {"theta", "slow_gamma"}


CONTRASTS = [
    # (phase_branch_freq, phase_branch_band, contrast_dir, has_memory)
    ("retreival",  "retreival_bandwise",  "stim_nostim",        True),
    ("encoding",   "encoding_bandwise",   "stim_nostim",        True),
    ("retreival",  "retreival_bandwise",  "endogenous_memory",  False),
    ("encoding",   "encoding_bandwise",   "endogenous_memory",  False),
]


def overlapping_clusters(fc, region, memory, band_lo, band_hi, has_memory):
    sub = fc[fc["region"] == region]
    if has_memory and "memory" in fc.columns:
        sub = sub[sub["memory"] == memory]
    overlap = sub[(sub["freq_lo_hz"] <= band_hi)
                  & (sub["freq_hi_hz"] >= band_lo)]
    return overlap


def compare(phase_freq, phase_band, contrast, has_memory, modality):
    fc_path = BASE / phase_freq / contrast / modality / f"{modality}_cluster_stats.csv"
    bw_path = BASE / phase_band / contrast / modality / f"{modality}_band_stats.csv"
    if not (fc_path.exists() and bw_path.exists()):
        print(f"  [{phase_freq}/{contrast}/{modality}] missing input(s); skip")
        return pd.DataFrame()
    fc = pd.read_csv(fc_path)
    bw = pd.read_csv(bw_path)

    rows = []
    for _, b in bw.iterrows():
        region, band = b["region"], b["band"]
        memory = b["memory"] if has_memory and "memory" in bw.columns else ""
        band_lo, band_hi = BANDS[band]
        bw_p = float(b["p_value"])
        bw_sig = bool(b["significant"])
        bw_t = float(b["t_stat"])

        clusters = overlapping_clusters(fc, region, memory, band_lo, band_hi,
                                        has_memory)
        if clusters.empty:
            fc_p = float("nan")
            fc_sig = False
            fc_overlap_freq = ""
        else:
            min_idx = clusters["p_value"].idxmin()
            fc_p = float(clusters.loc[min_idx, "p_value"])
            fc_sig = bool(fc_p < 0.05)
            fc_overlap_freq = (
                f'{clusters.loc[min_idx, "freq_lo_hz"]:.1f}-'
                f'{clusters.loc[min_idx, "freq_hi_hz"]:.1f}'
            )
        if bw_sig and fc_sig:
            status = "agree_sig"
        elif (not bw_sig) and (not fc_sig):
            status = "agree_ns"
        elif bw_sig and not fc_sig:
            status = "gain"
        else:
            status = "lose"
        rows.append({
            "phase": phase_freq,
            "contrast": contrast,
            "modality": modality,
            "region": region,
            "memory": memory,
            "band": band,
            "family": "confirmatory" if band in CONFIRMATORY else "exploratory",
            "band_freq_range": f"{band_lo}-{band_hi} Hz",
            "freqcluster_p": fc_p,
            "freqcluster_sig": fc_sig,
            "freqcluster_overlap_range": fc_overlap_freq,
            "bandwise_t": bw_t,
            "bandwise_p": bw_p,
            "bandwise_sig": bw_sig,
            "status": status,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_rows = []
    for phase_freq, phase_band, contrast, has_memory in CONTRASTS:
        print(f"\n=== {phase_freq} / {contrast} ===")
        for m in ("power", "coherence", "pac"):
            df = compare(phase_freq, phase_band, contrast, has_memory, m)
            if not df.empty:
                all_rows.append(df)
    full = pd.concat(all_rows, ignore_index=True)
    out_path = BASE / "comparison_freqcluster_vs_bandwise.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    print(f"\nSaved comparison: {out_path}")

    print("\n=== STATUS COUNTS PER (phase, contrast, modality) ===")
    print(full.groupby(["phase", "contrast", "modality", "status"])
              .size().unstack(fill_value=0))

    print("\n=== ALL CHANGES (gain = only band sig; lose = only freq-cluster sig) ===")
    changes = full[full["status"].isin(["gain", "lose"])].copy()
    if changes.empty:
        print("(none)")
    else:
        for c in ("freqcluster_p", "bandwise_t", "bandwise_p"):
            changes[c] = changes[c].round(4)
        print(changes[["phase", "contrast", "modality", "region", "memory",
                       "band", "family",
                       "freqcluster_p", "freqcluster_overlap_range",
                       "bandwise_t", "bandwise_p", "status"]]
              .to_string(index=False))

    print("\n=== AGREEMENT (significant in both) ===")
    agree = full[full["status"] == "agree_sig"].copy()
    if agree.empty:
        print("(none)")
    else:
        for c in ("freqcluster_p", "bandwise_t", "bandwise_p"):
            agree[c] = agree[c].round(4)
        print(agree[["phase", "contrast", "modality", "region", "memory",
                     "band", "family",
                     "freqcluster_p", "freqcluster_overlap_range",
                     "bandwise_t", "bandwise_p"]]
              .to_string(index=False))
