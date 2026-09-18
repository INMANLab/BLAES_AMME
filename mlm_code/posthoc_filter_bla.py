#!/usr/bin/env python
"""Filter BLA-related panels out of stim_effect and GLMM_interactions_model_variations
post-hoc artifacts (encoding only).

Removes:
  - scope in {BLAMTL, HippSubBLA}
  - any unit containing 'BLA' (BLA, BLA_X, X_BLA pairs)

Preserves the original CSVs as *_with_BLA_orig.csv. Overwrites the working
trending_panels.csv, all_panels.csv, and random_slope_comparison.csv so the
plotter and PDF builder downstream produce HPCrhinal + HippSubRhinal only.

Also deletes BLA-related per-panel artifacts (stability/jn/rs_coefs PNGs and
CSVs) so the directory listing is clean.
"""

from pathlib import Path
import pandas as pd
import shutil

REPO = Path(__file__).resolve().parent.parent
ENC = REPO / "OUTPUTS" / "encoding_memory_reports"

TARGET_ANALYSES = ("stim_effect", "GLMM_interactions_model_variations")
BAD_SCOPES = {"BLAMTL", "HippSubBLA"}


def is_bla_panel(scope, unit):
    return scope in BAD_SCOPES or "BLA" in str(unit)


def filter_csv(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"scope", "unit"}.issubset(df.columns):
        return None
    backup = path.with_name(path.stem + "_with_BLA_orig.csv")
    if not backup.exists():
        shutil.copy(path, backup)
    keep = ~df.apply(lambda r: is_bla_panel(r["scope"], r["unit"]), axis=1)
    df_clean = df[keep].reset_index(drop=True)
    df_clean.to_csv(path, index=False)
    return len(df) - len(df_clean)


def delete_bla_artifacts(d):
    deleted = 0
    for csv in d.glob("rs_coefs_*.csv"):
        # rs_coefs_{measure}_{unit}_{band}.csv -- unit is between measure and band
        # We can't always tell measure/unit/band split unambiguously from name,
        # so we read the CSV; the unit is encoded in the filename pattern.
        stem = csv.stem
        if "BLA" in stem:
            csv.unlink(); deleted += 1
    for pat in ("stability_*", "jn_*"):
        for f in d.glob(pat):
            if "BLA" in f.stem:
                f.unlink(); deleted += 1
    return deleted


def main():
    for analysis in TARGET_ANALYSES:
        d = ENC / analysis / "post_hoc_testing"
        print(f"\n=== {analysis} ===")
        for fname in ("trending_panels.csv", "all_panels.csv",
                      "random_slope_comparison.csv", "jn_summary.csv",
                      "jn_summary_v2.csv"):
            n_removed = filter_csv(d / fname)
            if n_removed is not None:
                print(f"  {fname}: removed {n_removed} BLA-related rows")
        n_files = delete_bla_artifacts(d)
        print(f"  deleted {n_files} BLA-related per-panel files")


if __name__ == "__main__":
    main()
