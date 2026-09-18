"""Descriptive count of remembered vs forgotten target (old-item) trials,
encoding and retrieval. No IED split, no test -- pure descriptives.

Counts reconstructed from raw LFP files (targets only, lures excluded) via
ied_memory_ratio_by_iedstatus.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ied_memory_ratio_by_iedstatus as core

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "Helvetica"

REM_COLOR = "#FFD43B"   # yellow
FORG_COLOR = "#9C7AC9"  # purple
OUT_DIR = Path(__file__).resolve().parent.parent / "OUTPUTS" / "updated_IED_figures"


def compute():
    # IED target trials only -- identical to the "with IEDs" column of
    # ied_memory_ratio_ied_vs_noied.png
    ied_e = pd.read_csv(core.IED_ENC)
    ied_r = pd.read_csv(core.IED_RET)
    e_rem, e_forg = core.ied_counts(core.IED_ENC)
    r_rem, r_forg = core.ied_counts(core.IED_RET)
    enc_pts = ied_e["Patient"].nunique()
    ret_pts = ied_r[ied_r["MemoryOutcome"].isin(["remembered", "forgotten"])]["Patient"].nunique()
    return {
        "Encoding": {"rem": e_rem, "forg": e_forg, "pts": enc_pts},
        "Retrieval": {"rem": r_rem, "forg": r_forg, "pts": ret_pts},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = compute()

    phases = ["Encoding", "Retrieval"]
    x = np.arange(len(phases))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for i, ph in enumerate(phases):
        d = data[ph]
        n = d["rem"] + d["forg"]
        br = ax.bar(i - width / 2, d["rem"], width, color=REM_COLOR, edgecolor="black", linewidth=1.0,
                    label="Remembered" if i == 0 else None, zorder=2)
        bf = ax.bar(i + width / 2, d["forg"], width, color=FORG_COLOR, edgecolor="black", linewidth=1.0,
                    label="Forgotten" if i == 0 else None, zorder=2)
        ax.bar_label(br, labels=[f"{d['rem']}\n({d['rem']/n:.0%})"], padding=3, fontsize=10, fontweight="bold")
        ax.bar_label(bf, labels=[f"{d['forg']}\n({d['forg']/n:.0%})"], padding=3, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ph}\n{data[ph]['pts']} patients · {data[ph]['rem']+data[ph]['forg']:,} targets"
                        for ph in phases], fontsize=15, fontweight="bold")
    ax.set_ylabel("Number of target trials", fontsize=14, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    ax.set_title("Remembered vs forgotten — IED target trials",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", prop={"weight": "bold", "size": 13})
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(data["Encoding"]["rem"], data["Retrieval"]["rem"]) * 1.18)
    ax.margins(x=0.15)

    fig.tight_layout()
    png = OUT_DIR / "target_memory_descriptive_counts.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"Saved:\n  {png}")


if __name__ == "__main__":
    main()
