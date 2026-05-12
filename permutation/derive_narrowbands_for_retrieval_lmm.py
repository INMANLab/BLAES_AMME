#!/usr/bin/env python
"""Derive per-modality 'narrow band' freq ranges from the retrieval stim_nostim
significant clusters export.

Per (modality, family in {theta, slow_gamma}):
  - Collect all significant clusters in that modality x family across all
    regions / memory subsets.
  - Set the narrow band freq range = MEDIAN of cluster bounds:
      lo = median(freq_lo_hz across sig clusters)
      hi = median(freq_hi_hz across sig clusters)
    For a single cluster this reduces to that cluster's range.
  - Skip exploratory-family clusters (HFA etc).

The resulting narrow band is applied DOWNSTREAM to every region in every
scope, even regions without their own significant cluster -- so the hypothesis
test covers the full scope.

Inputs:
  outputs/PermutationOutputsAlireza/significant_clusters_for_lmm.csv

Outputs:
  OUTPUTS/retrieval memory reports permutation test based/narrow_bands.csv
    columns: modality, band, family, narrow_lo_hz, narrow_hi_hz,
             n_source_clusters, source_summary
"""
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
SRC = ROOT / "outputs" / "PermutationOutputsAlireza" / "significant_clusters_for_lmm.csv"
OUT_DIR = ROOT / "OUTPUTS" / "retrieval memory reports permutation test based"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def derive():
    df = pd.read_csv(SRC)
    df = df[
        (df["phase"] == "retreival")
        & (df["contrast"] == "stim_nostim")
        & (df["family"].isin(["theta", "slow_gamma"]))
        & df["significant"]
    ].copy()

    rows = []
    for (modality, family), grp in df.groupby(["modality", "family"]):
        lo_med = float(grp["freq_lo_hz"].median())
        hi_med = float(grp["freq_hi_hz"].median())
        sources = ", ".join(
            f"{r.region}/{r.memory}({r.freq_lo_hz:.2f}-{r.freq_hi_hz:.2f})"
            for _, r in grp.iterrows()
        )
        rows.append({
            "modality": modality,
            "band": f"narrow_{family}",
            "family": family,
            "narrow_lo_hz": round(lo_med, 3),
            "narrow_hi_hz": round(hi_med, 3),
            "n_source_clusters": len(grp),
            "source_summary": sources,
        })
    out = pd.DataFrame(rows).sort_values(["modality", "family"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    out = derive()
    out_path = OUT_DIR / "narrow_bands.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Rows: {len(out)}")
    print()
    print(out.to_string(index=False))
