#!/usr/bin/env python
"""Orchestrator for the h1abc main-effect GLMM pipeline.

Runs the four steps in order:
  1. Rscript run_h1abc_maineffect.R           (fits + predictions)
  2. build_h1abc_maineffect_report.py (1st)   (writes FDR family CSVs)
  3. plot_h1abc_maineffect_glmm.py            (writes panel PNGs)
  4. build_h1abc_maineffect_report.py (2nd)   (embeds PNGs into PDFs)

Args optionally narrow the sweep:
  python run_h1abc_maineffect_all.py
    -> full sweep over 2 phases x 3 measures x 4 scopes x 3 contrasts
  python run_h1abc_maineffect_all.py encoding
    -> all measures/scopes/contrasts for encoding
  python run_h1abc_maineffect_all.py encoding coherence HippSubRhinal all
    -> a single combination

Stats are written under
  OUTPUTS/<phase>_memory_reports/stats/h1abc_maineffect_<phase>_mlm/
PDFs under
  OUTPUTS/<phase>_memory_reports/<Phase>_<Measure>_<Scope>_GLMM_MainEffect_
  <Contrast>_FDRcorrected.pdf
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

PHASES = ("encoding", "retrieval")
MEASURES = ("power", "coherence", "pac")
SCOPES = ("BLAMTL", "HPCrhinal", "HippSubBLA", "HippSubRhinal")
CONTRASTS = ("all", "nostim", "stim")


def parse_args(argv):
    phases = list(PHASES)
    measures = list(MEASURES)
    scopes = list(SCOPES)
    contrasts = list(CONTRASTS)

    args = list(argv)
    if args and args[0] in PHASES:
        phases = [args.pop(0)]
    if len(args) >= 3:
        measures = [args[0]]
        scopes = [args[1]]
        contrasts = [args[2]]
    return phases, measures, scopes, contrasts


def run(cmd, label):
    print(f"\n>>> {label}")
    print(f"    $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    if res.returncode != 0:
        sys.exit(f"FAILED ({res.returncode}): {label}")


def main():
    phases, measures, scopes, contrasts = parse_args(sys.argv[1:])

    # 1. R fits + predictions
    for phase in phases:
        if len(measures) == 1 and len(scopes) == 1 and len(contrasts) == 1:
            run([
                "Rscript", str(SCRIPT_DIR / "run_h1abc_maineffect.R"),
                phase, measures[0], scopes[0], contrasts[0],
            ], f"[1/4] R fits: {phase} {measures[0]} {scopes[0]} {contrasts[0]}")
        else:
            # Sweep this phase across all measure/scope/contrast (R script
            # iterates internally when fewer than 4 combo args provided).
            run([
                "Rscript", str(SCRIPT_DIR / "run_h1abc_maineffect.R"),
                phase,
            ], f"[1/4] R fits: {phase} (full sweep)")

    pdf_script = str(SCRIPT_DIR / "build_h1abc_maineffect_report.py")
    plot_script = str(SCRIPT_DIR / "plot_h1abc_maineffect_glmm.py")

    # 2. First PDF pass -> writes FDR family CSVs (plotter needs them)
    for phase in phases:
        for m in measures:
            for s in scopes:
                for c in contrasts:
                    run([
                        sys.executable, pdf_script, phase, m, s, c,
                    ], f"[2/4] FDR + PDF (pre-figures): "
                       f"{phase} {m} {s} {c}")

    # 3. Plotter
    for phase in phases:
        if len(measures) == 1 and len(scopes) == 1 and len(contrasts) == 1:
            run([
                sys.executable, plot_script,
                phase, measures[0], scopes[0], contrasts[0],
            ], f"[3/4] Plot: {phase} {measures[0]} {scopes[0]} {contrasts[0]}")
        else:
            run([
                sys.executable, plot_script, phase,
            ], f"[3/4] Plot: {phase} (full sweep)")

    # 4. Second PDF pass -> embeds PNGs
    for phase in phases:
        for m in measures:
            for s in scopes:
                for c in contrasts:
                    run([
                        sys.executable, pdf_script, phase, m, s, c,
                    ], f"[4/4] PDF with figures: {phase} {m} {s} {c}")

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()
