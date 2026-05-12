#!/usr/bin/env python
"""Export significant cluster-permutation findings (unfiltered, no balanced)
in a flat CSV ready for LMM/GLMM downstream use.

Sources (vanilla branches only):
  outputs/PermutationOutputsAlireza/encoding/{stim_nostim, endogenous_memory}/<m>/<m>_cluster_stats.csv
  outputs/PermutationOutputsAlireza/retreival/{stim_nostim, endogenous_memory}/<m>/<m>_cluster_stats.csv

Output:
  outputs/PermutationOutputsAlireza/significant_clusters_for_lmm.csv

Each row = one significant cluster with everything an LMM/GLMM script
needs: phase, contrast, modality, region, memory (if applicable),
freq_lo_hz, freq_hi_hz, n_freqs, n_subjects, t_threshold, cluster_mass_t,
direction, p_value, family (per-freq FWE family of the cluster), and
canonical_band_overlap (which canonical band(s) the cluster's freq range
overlaps).
"""
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
BASE = ROOT / "outputs" / "PermutationOutputsAlireza"

# Canonical band ranges (for the human-readable "which bands does this
# cluster span" column). These are independent of the per-freq FWE
# family the cluster was tested under.
CANONICAL_BANDS = [
    ("delta",      1.95, 3.0),
    ("theta",      4.0,  8.0),
    ("alpha",      9.0,  13.0),
    ("beta",       14.0, 29.0),
    ("slow_gamma", 30.0, 55.0),
    ("HFA",        70.0, 100.0),
]


def cluster_band_overlap(lo, hi):
    hits = [name for name, b_lo, b_hi in CANONICAL_BANDS
            if (lo <= b_hi) and (hi >= b_lo)]
    return "+".join(hits) if hits else "none"


def collect():
    rows = []
    for phase in ("encoding", "retreival"):
        for contrast in ("stim_nostim", "endogenous_memory"):
            for modality in ("power", "coherence", "pac"):
                csv = (BASE / phase / contrast / modality
                       / f"{modality}_cluster_stats.csv")
                if not csv.exists():
                    continue
                df = pd.read_csv(csv)
                if df.empty:
                    continue
                sig = df[df["significant"] == True].copy()
                if sig.empty:
                    continue
                sig["phase"] = phase
                sig["contrast"] = contrast
                sig["modality"] = modality
                if "memory" not in sig.columns:
                    sig["memory"] = ""
                sig["canonical_band_overlap"] = sig.apply(
                    lambda r: cluster_band_overlap(r["freq_lo_hz"],
                                                   r["freq_hi_hz"]),
                    axis=1)
                rows.append(sig)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = ["phase", "contrast", "modality", "region", "memory",
            "n_subjects", "family", "confirmatory",
            "freq_lo_hz", "freq_hi_hz", "n_freqs",
            "canonical_band_overlap",
            "direction", "cluster_mass_t", "t_threshold",
            "p_value", "significant"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]
    out = out.sort_values(["phase", "contrast", "modality", "region",
                           "memory", "freq_lo_hz"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    out_df = collect()
    out_path = BASE / "significant_clusters_for_lmm.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Rows: {len(out_df)}")
    print()
    print(out_df.to_string(index=False))
