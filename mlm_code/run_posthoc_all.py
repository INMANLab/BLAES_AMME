#!/usr/bin/env python
"""Orchestrator for the full encoding post-hoc pipeline. Steps:
  0. Identify trending panels (already done if trending_panels.csv exists,
     but re-runs the scan to keep it current).
  1. Random-slope variant refits on all 70 panels per analysis type.
     - alltrials, endogenous_memory_effect, stim_effect: (1 + band_c | Patient)
     - GLMM_interactions_model_variations: (1 + StimCond | Patient)
  2. Johnson-Neyman for trending interaction panels (both RI and RS).
  3. Stability tests (LOSO + deltas + balanced downsample) on trending panels.
  4. Generate diagnostic figures.
  5. Build PDF report per analysis type.

Stim_effect random-slope refits may already exist - if so they are NOT skipped
(re-running is cheap relative to the rest); pass --skip-rs to disable.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent

ANALYSES = [
    "alltrials",
    "endogenous_memory_effect",
    "stim_effect",
    "GLMM_interactions_model_variations",
]


def run(cmd, label):
    print(f"\n>>> {label}")
    print(f"    $ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=REPO)
    if res.returncode != 0:
        sys.exit(f"FAILED ({res.returncode}): {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rs", action="store_true",
                    help="Skip random-slope refits (use existing CSVs).")
    ap.add_argument("--only", choices=ANALYSES,
                    help="Restrict to a single analysis type.")
    args = ap.parse_args()

    analyses = [args.only] if args.only else ANALYSES

    # Step 0: refresh trending manifests
    run([sys.executable, str(SCRIPT_DIR / "posthoc_identify_trending.py")],
        "[0/5] Re-scan trending panels")

    # Step 1: random-slope refits
    if not args.skip_rs:
        for a in analyses:
            run(["Rscript",
                 str(SCRIPT_DIR / "posthoc_random_slope_refits.R"), a],
                f"[1/5] Random-slope refits: {a}")

    # Step 2: Johnson-Neyman (only meaningful for interaction analysis)
    if "GLMM_interactions_model_variations" in analyses:
        run(["Rscript", str(SCRIPT_DIR / "posthoc_johnson_neyman.R")],
            "[2/5] Johnson-Neyman: GLMM_interactions_model_variations")

    # Step 3: stability tests on trending panels
    for a in analyses:
        run(["Rscript",
             str(SCRIPT_DIR / "posthoc_stability_tests.R"), a],
            f"[3/5] Stability tests: {a}")

    # Step 4: figures
    run([sys.executable, str(SCRIPT_DIR / "posthoc_plot_results.py")],
        "[4/5] Plot results")

    # Step 5: PDF reports
    run([sys.executable, str(SCRIPT_DIR / "posthoc_build_reports.py")],
        "[5/5] Build PDF reports")

    print("\n===== POST-HOC PIPELINE DONE =====")


if __name__ == "__main__":
    main()
