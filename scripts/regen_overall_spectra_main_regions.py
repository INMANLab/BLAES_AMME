#!/usr/bin/env python
"""Overall (grand-mean) power and coherence spectra restricted to main regions.

Power panel: ALLHPC, BLA, EC, PRC.
Coherence panel: all pairs whose two ends are within {ALLHPC, BLA, EC, PRC}.

Mirrors the existing `plot_power_by_roi` / `plot_coherence_by_roi_core` look:
each ROI line is the across-patient mean spectrum (after collapsing across
stim/no-stim and remembered/forgotten within each patient); shading is +/-1 SD
across patients. ALLHPC and ALLHPC pair composites are formed within-patient
before the across-patient average, matching the rest of the pipeline.

Outputs (new files; existing figures untouched):
  OUTPUTS/<phase>_power/all/GroupPower_byROI_<phase>_mainregions.png
  OUTPUTS/<phase>_coherence/all/GroupCoherency_byROI_<phase>_mainregions.png
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/"
            "InmanLab/BLAES_data/dissertation/AMME_BLAES")
OUTPUTS = ROOT / "OUTPUTS"

MAIN_REGIONS = {"ALLHPC", "BLA", "EC", "PRC"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _merge_region_subject(*dicts):
    merged = {}
    for d in dicts:
        for region, sd in (d or {}).items():
            target = merged.setdefault(region, {})
            for subj, vec in sd.items():
                target[str(subj)] = np.asarray(vec, dtype=np.float64)
    return merged


def load_combined(measure: str, phase: str):
    if measure == "power":
        path = ROOT / f"{phase}_code" / f"combined_{phase}_power.py"
    else:
        path = ROOT / f"{phase}_code" / f"combined_{phase}_coherence.py"
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT))
    mod = _load_module(f"_{measure}_{phase}", path)

    if phase == "retrieval":
        loader_b = mod.load_blaes_retrieval
        loader_a = mod.load_amme_retrieval
    else:
        loader_b = mod.load_blaes_encoding
        loader_a = mod.load_amme_encoding

    if measure == "power":
        blaes = mod.augment_with_power_composites(loader_b())
        amme = mod.augment_with_power_composites(loader_a())
        gap = _merge_region_subject(
            blaes.get("group_all_power", {}),
            amme.get("group_all_power", {}),
        )
    else:
        blaes = mod.augment_with_allhpc_pair_composites(loader_b())
        amme = mod.augment_with_allhpc_pair_composites(loader_a())
        # BLA_ALLHPC (and BLA_MTL) are built by a separate function that
        # returns its own dict — merge its group_all_power back in.
        blaes_bla = mod.build_bla_composite_data(blaes)
        amme_bla = mod.build_bla_composite_data(amme)
        gap = _merge_region_subject(
            blaes.get("group_all_power", {}),
            amme.get("group_all_power", {}),
            blaes_bla.get("group_all_power", {}),
            amme_bla.get("group_all_power", {}),
        )
    fp_b = blaes.get("freqs_post"); fp_a = amme.get("freqs_post")
    freqs = fp_b if fp_b is not None else fp_a
    return mod, gap, freqs


def _is_main_power(region: str) -> bool:
    return region in MAIN_REGIONS


def _is_main_pair(region: str) -> bool:
    parts = str(region).split("_")
    return len(parts) == 2 and parts[0] in MAIN_REGIONS and parts[1] in MAIN_REGIONS


def plot_overall(mod, gap, freqs, measure: str, phase: str):
    if not gap or freqs is None:
        print(f"  [{phase} {measure}] no data, skipping.")
        return None

    keep_fn = _is_main_power if measure == "power" else _is_main_pair
    rois = sorted(r for r in gap.keys() if keep_fn(r))
    if not rois:
        print(f"  [{phase} {measure}] no main regions present, skipping.")
        return None

    if measure == "power":
        palette = {r: mod.ROI_COLORS_SPEC.get(r, "#808080") for r in rois}
    else:
        palette = mod.make_roi_color_map(rois)

    fig, ax = plt.subplots(figsize=(12, 8))
    for roi in rois:
        mat = np.array(list(gap[roi].values()), dtype=np.float64)
        if mat.size == 0:
            continue
        mean = mat.mean(0)
        std = mat.std(0)
        c = palette[roi]
        ax.plot(freqs, mean, color=c, label=f"{roi} ({mat.shape[0]})", lw=1.6)
        ax.fill_between(freqs, mean - std, mean + std, alpha=0.2, color=c)

    ax.set_xlabel("Frequency (Hz)", fontsize=18, fontweight="bold")
    if measure == "power":
        ax.set_ylabel("Power (dB)", fontsize=18, fontweight="bold")
        title_measure = "Power"
    else:
        ax.set_ylabel("Coherence (Fisher Z)", fontsize=18, fontweight="bold")
        title_measure = "Coherency"
    ax.set_title(
        f"{phase.capitalize()} Group {title_measure} by ROI (Main Regions)",
        fontsize=20, fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left",
              prop={"weight": "bold", "size": 12})

    out_dir = OUTPUTS / f"{phase}_{'power' if measure == 'power' else 'coherence'}" / "all"
    out_dir.mkdir(parents=True, exist_ok=True)
    if measure == "power":
        fname = f"GroupPower_byROI_{phase}_mainregions.png"
    else:
        fname = f"GroupCoherency_byROI_{phase}_mainregions.png"
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{phase} {measure}] Saved: {out_dir / fname}")
    return out_dir / fname


if __name__ == "__main__":
    for measure in ("power", "coherence"):
        for phase in ("retrieval", "encoding"):
            print(f"\n=== {phase} {measure}: overall spectra (main regions) ===")
            mod, gap, freqs = load_combined(measure, phase)
            plot_overall(mod, gap, freqs, measure, phase)
    print("\nDone.")
