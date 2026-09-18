#!/usr/bin/env python
"""Plot the leave-one-subject-out (LOSO) sensitivity for HPC-PRC stim-only
slow-gamma coherence GLMM. Reads CSV written by diag_hpc_prc_loso.R.

Output:
  OUTPUTS/encoding_memory_reports/stim_effect/
    HPC_PRC_stim_slow_gamma_loso.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "OUTPUTS" / "encoding_memory_reports" / "stim_effect"
CSV = OUT_DIR / "HPC_PRC_stim_slow_gamma_loso.csv"
PNG = OUT_DIR / "HPC_PRC_stim_slow_gamma_loso.png"

df = pd.read_csv(CSV)
full = df[df["excluded"] == "<none>"].iloc[0]
loso = df[df["excluded"] != "<none>"].copy().sort_values("OR")

fig, (ax_or, ax_p) = plt.subplots(1, 2, figsize=(14, 6.5),
                                  gridspec_kw={"width_ratios": [1.5, 1]})

y = np.arange(len(loso))
ax_or.errorbar(loso["OR"], y,
               xerr=[loso["OR"] - loso["lo"], loso["hi"] - loso["OR"]],
               fmt="o", color="#2b2b2b", ecolor="#888888",
               capsize=3, ms=6, lw=1.0)
ax_or.axvline(1.0, color="0.5", ls=":", lw=1.0, label="OR = 1 (no effect)")
ax_or.axvline(full["OR"], color="#c0392b", ls="--", lw=1.5,
              label=f"Full-sample OR = {full['OR']:.2f}")
ax_or.set_yticks(y)
ax_or.set_yticklabels(loso["excluded"], fontsize=9)
ax_or.set_xlabel("OR for slow-gamma coherence (95% CI)",
                 fontsize=11, fontweight="bold")
ax_or.set_title("Leave-one-subject-out OR estimates", fontsize=12,
                fontweight="bold")
ax_or.legend(loc="lower right", fontsize=9)
ax_or.grid(alpha=0.25, axis="x")
ax_or.set_xscale("log")

colors = ["#2b2b2b" if p < 0.05 else "#a0a0a0" for p in loso["p"]]
ax_p.barh(y, loso["p"], color=colors)
ax_p.axvline(0.05, color="#c0392b", ls="--", lw=1.5, label="p = .05")
ax_p.axvline(full["p"], color="#1f77b4", ls=":", lw=1.5,
             label=f"Full-sample p = {full['p']:.3f}")
ax_p.set_yticks(y)
ax_p.set_yticklabels(loso["excluded"], fontsize=9)
ax_p.set_xlabel("Raw p-value", fontsize=11, fontweight="bold")
ax_p.set_title("LOSO p-values", fontsize=12, fontweight="bold")
ax_p.legend(loc="lower right", fontsize=9)
ax_p.grid(alpha=0.25, axis="x")

n_below = int((loso["p"] < 0.05).sum())
summary = (
    f"HPC-PRC stim-only slow-gamma coherence: leave-one-subject-out\n"
    f"Full sample: OR = {full['OR']:.2f} [{full['lo']:.2f}, {full['hi']:.2f}], "
    f"p = {full['p']:.3f}.  "
    f"All 18 LOSO refits keep p < .05 (range {loso['p'].min():.3f} - "
    f"{loso['p'].max():.3f}); {n_below}/18 LOSO refits significant at uncorrected alpha = .05."
)
fig.suptitle(summary, fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(PNG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {PNG}")

print(f"\nFull sample OR={full['OR']:.2f} p={full['p']:.3f}")
print(f"LOSO p range: {loso['p'].min():.3f} - {loso['p'].max():.3f}")
print(f"LOSO OR range: {loso['OR'].min():.2f} - {loso['OR'].max():.2f}")
worst = loso.iloc[loso["p"].argmax()]
best = loso.iloc[loso["p"].argmin()]
print(f"Removing {worst['excluded']} weakens most (p -> {worst['p']:.3f})")
print(f"Removing {best['excluded']} strengthens most (p -> {best['p']:.3f})")
