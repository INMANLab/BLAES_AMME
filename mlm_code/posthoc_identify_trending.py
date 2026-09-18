#!/usr/bin/env python
"""Scan existing encoding GLMM results to identify panels trending at raw
p < 0.10. Writes a manifest per analysis type so downstream scripts know
which panels to refit / probe further. Encoding phase only.

Analysis types and their focal terms:
  alltrials    : Accuracy ~ band_c + (1|Patient)              focal = band_c
  endogenous   : same, no-stim trials only                    focal = band_c
  stim         : same, stim trials only                       focal = band_c
  interactions : Accuracy ~ band_c * StimCond + (1|Patient)   focal = band_c:StimCondstim

Outputs:
  OUTPUTS/encoding_memory_reports/{dir}/post_hoc_testing/trending_panels.csv
where {dir} in {alltrials, endogenous_memory_effect, stim_effect,
                GLMM_interactions_model_variations}.
"""

from pathlib import Path
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
ENC_DIR = REPO / "OUTPUTS" / "encoding_memory_reports"

ANALYSES = [
    # (output_dir_name, source_subdir, contrast_token, focal_term)
    ("alltrials",                     "h1abc_maineffect_encoding_mlm", "all",    "band_c"),
    ("endogenous_memory_effect",      "h1abc_maineffect_encoding_mlm", "nostim", "band_c"),
    ("stim_effect",                   "h1abc_maineffect_encoding_mlm", "stim",   "band_c"),
    ("GLMM_interactions_model_variations",
                                      "h1abc_full_encoding_mlm",       None,     "band_c:StimCondstim"),
]

SCOPES = ("BLAMTL", "HPCrhinal", "HippSubBLA", "HippSubRhinal")
MEASURES = ("power", "coherence", "pac")
PAC_BANDS = ("slow_gamma",)
DEFAULT_BANDS = ("theta", "slow_gamma")


def stats_dir_for(source_subdir, measure, scope, contrast):
    base = ENC_DIR / "stats" / source_subdir
    if contrast is None:
        return base / f"{measure}_{scope}_GLMM"
    return base / f"{measure}_{scope}_{contrast}"


def scan_one(source_subdir, contrast, focal_term):
    rows = []
    for measure in MEASURES:
        bands = PAC_BANDS if measure == "pac" else DEFAULT_BANDS
        for scope in SCOPES:
            d = stats_dir_for(source_subdir, measure, scope, contrast)
            if not d.exists():
                continue
            for coef_csv in sorted(d.glob(f"{measure}_*_coefs.csv")):
                stem = coef_csv.stem  # e.g. coherence_CA_EC_theta_coefs
                tokens = stem[:-len("_coefs")].split("_")
                # tokens[0] = measure; last token = band
                band = None
                for b in bands:
                    if stem.endswith(f"_{b}_coefs"):
                        band = b
                        break
                if band is None:
                    continue
                unit = stem[len(measure) + 1: -(len(band) + len("_coefs") + 1)]
                df = pd.read_csv(coef_csv)
                rr = df[df["term"] == focal_term]
                if rr.empty:
                    continue
                p = float(rr["p.value"].iloc[0])
                est = float(rr["estimate"].iloc[0])
                conf_low = float(rr.get("conf.low", pd.Series([float("nan")])).iloc[0]) \
                    if "conf.low" in rr.columns else float("nan")
                conf_high = float(rr.get("conf.high", pd.Series([float("nan")])).iloc[0]) \
                    if "conf.high" in rr.columns else float("nan")
                rows.append({
                    "measure": measure,
                    "scope": scope,
                    "unit": unit,
                    "band": band,
                    "term": focal_term,
                    "estimate_OR": est,
                    "conf_low": conf_low,
                    "conf_high": conf_high,
                    "p_value": p,
                })
    return pd.DataFrame(rows)


def main():
    summary = {}
    for out_dir_name, source_subdir, contrast, focal_term in ANALYSES:
        out_dir = ENC_DIR / out_dir_name / "post_hoc_testing"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = scan_one(source_subdir, contrast, focal_term)
        df["trending"] = df["p_value"] < 0.10
        df["sig_uncorrected"] = df["p_value"] < 0.05
        df = df.sort_values("p_value").reset_index(drop=True)
        all_path = out_dir / "all_panels.csv"
        trending_path = out_dir / "trending_panels.csv"
        df.to_csv(all_path, index=False)
        df[df["trending"]].to_csv(trending_path, index=False)
        summary[out_dir_name] = {
            "n_panels": len(df),
            "n_trending_lt_10": int(df["trending"].sum()),
            "n_sig_lt_05": int(df["sig_uncorrected"].sum()),
            "trending_path": str(trending_path),
        }
        print(f"\n== {out_dir_name} ==")
        print(f"  n panels: {len(df)}, n trending p<.10: {int(df['trending'].sum())}, "
              f"n uncorrected p<.05: {int(df['sig_uncorrected'].sum())}")
        if df["trending"].any():
            print("  top trending:")
            for _, r in df[df["trending"]].head(8).iterrows():
                print(f"    {r['measure']:10s} {r['scope']:14s} "
                      f"{r['unit']:14s} {r['band']:10s}  "
                      f"p={r['p_value']:.4f}  OR={r['estimate_OR']:.3g}")
    print("\nWrote trending_panels.csv to each post_hoc_testing/ directory.")


if __name__ == "__main__":
    main()
