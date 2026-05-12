#!/usr/bin/env python
"""Combined 3x4 LMM figure for BLAMTL + HPCrhinal coherence (theta + slow
gamma), in the panel order from the user's reference figure. Reuses
plot_lmmcont_panel / plot_lmmquad_panel from build_h1abc_hypothesis_figures
so the per-panel style stays in sync.

Panel layout (matches the reference):
  Row 1: BLA-ALLHPC theta | BLA-EC theta    | BLA-ALLHPC slow_gamma | BLA-EC slow_gamma
  Row 2: BLA-PRC theta    | EC-PRC theta    | BLA-PRC slow_gamma    | EC-PRC slow_gamma
  Row 3: ALLHPC-EC theta  | ALLHPC-PRC theta| ALLHPC-EC slow_gamma  | ALLHPC-PRC slow_gamma

Usage:
  python build_combined_blamtl_hpcrhinal_coh_lmm.py [retrieval|encoding] LMMcont
  python build_combined_blamtl_hpcrhinal_coh_lmm.py [retrieval|encoding] LMMquad
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_h1abc_hypothesis_figures as h1f


# (unit, band, scope) — order matches the reference figure exactly.
# Cols 1-2: theta. Cols 3-4: slow gamma. Same pair appears in matching column
# positions across the two band halves (col 1 <-> col 3, col 2 <-> col 4).
PANEL_ORDER = [
    ("BLA_ALLHPC", "theta",       "BLAMTL"),
    ("BLA_EC",     "theta",       "BLAMTL"),
    ("BLA_ALLHPC", "slow_gamma",  "BLAMTL"),
    ("BLA_EC",     "slow_gamma",  "BLAMTL"),

    ("BLA_PRC",    "theta",       "BLAMTL"),
    ("EC_PRC",     "theta",       "HPCrhinal"),
    ("BLA_PRC",    "slow_gamma",  "BLAMTL"),
    ("EC_PRC",     "slow_gamma",  "HPCrhinal"),

    ("ALLHPC_EC",  "theta",       "HPCrhinal"),
    ("ALLHPC_PRC", "theta",       "HPCrhinal"),
    ("ALLHPC_EC",  "slow_gamma",  "HPCrhinal"),
    ("ALLHPC_PRC", "slow_gamma",  "HPCrhinal"),
]

MODELTYPE_TITLES = {
    "LMMcont": "Memory modulation status (continuous) predicting coherence",
    "LMMquad": "Responder group status (quad) predicting coherence",
}
MODELTYPE_XLABEL = {
    "LMMcont": "Z-scored memory modulation status",
    "LMMquad": "Responder group",
}


def main():
    args = list(sys.argv[1:])
    phase = "retrieval"
    if args and args[0] in ("retrieval", "encoding"):
        phase = args.pop(0)
    h1f._set_phase(phase)

    if not args or args[0] not in ("LMMcont", "LMMquad"):
        sys.exit("usage: ... [retrieval|encoding] LMMcont|LMMquad")
    modeltype = args[0]

    measure = "coherence"
    responder = h1f.load_responder()
    blamtl_units    = ["BLA_ALLHPC", "BLA_EC", "BLA_PRC"]
    hpcrhinal_units = ["ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"]

    # patient_means uses a single scope's needs_allhpc flag, but since both
    # BLAMTL and HPCrhinal have needs_allhpc=True, we can pull theta/SG once
    # per scope and select within.
    blamtl_th = h1f.patient_means(measure, "BLAMTL",    blamtl_units,    "theta",      responder)
    blamtl_sg = h1f.patient_means(measure, "BLAMTL",    blamtl_units,    "slow_gamma", responder)
    hpcr_th   = h1f.patient_means(measure, "HPCrhinal", hpcrhinal_units, "theta",      responder)
    hpcr_sg   = h1f.patient_means(measure, "HPCrhinal", hpcrhinal_units, "slow_gamma", responder)
    data_lookup = {
        ("BLAMTL", "theta"):       blamtl_th,
        ("BLAMTL", "slow_gamma"):  blamtl_sg,
        ("HPCrhinal", "theta"):    hpcr_th,
        ("HPCrhinal", "slow_gamma"): hpcr_sg,
    }

    panel_fn = (h1f.plot_lmmcont_panel if modeltype == "LMMcont"
                else h1f.plot_lmmquad_panel)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (unit, band, scope) in zip(axes, PANEL_ORDER):
        data = data_lookup[(scope, band)]
        panel_fn(ax, measure, scope, unit, band, data)

    fig.suptitle(
        f"BLAMTL + HPCrhinal - Coherence - {MODELTYPE_TITLES[modeltype]} at {phase}",
        fontsize=13,
    )
    fig.supxlabel(MODELTYPE_XLABEL[modeltype], fontsize=20, fontweight="bold")
    fig.supylabel("Band-averaged coherence", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.96])

    out_dir = os.path.join(h1f.OUT_FIG_ROOT, "BLAMTL_HPCrhinal_coherence_combined")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"BLAMTL_HPCrhinal_coherence_{modeltype}_combined.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
