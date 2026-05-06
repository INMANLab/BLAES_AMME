#!/usr/bin/env python
"""Exploratory paired permutation tests for endogenous memory at retrieval.

Same paired sign-flip permutation procedure as the confirmatory endogenous
memory analysis, but covering frequency bands OUTSIDE Theta and Slow gamma:
    Delta = 1-3 Hz
    Alpha = 9-12 Hz
    Beta  = 13-20 Hz
    Gamma = 51-69 Hz
    HFA   = 70-100 Hz

Bands without spectral samples in a modality (e.g., Delta in PAC TG) are
omitted automatically.

Outputs:
  outputs/parametric retrieval test/endogenous memory/exploratory permutation testing/
    power/permutation.pdf, permutation.csv
    coherence/permutation.pdf, permutation.csv
    PAC/permutation.pdf, permutation.csv
"""

from pathlib import Path

import numpy as np

import endogenous_memory_confirmatory_permutation_testing as conf
from endogenous_memory_confirmatory_permutation_testing import (
    PAC_KEYS,
    POWER_COH_KEYS,
    load_coherence_data,
    load_pac_data,
    load_power_data,
    run_permutation_for_modality,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = SCRIPT_DIR / "outputs" / "parametric retrieval test" / "endogenous memory" / "exploratory permutation testing"

EXPLORATORY_BAND_SPECS = {
    "Delta": (1.0, 3.0),
    "Alpha": (9.0, 12.0),
    "Beta": (13.0, 20.0),
    "Gamma": (51.0, 69.0),
    "HFA": (70.0, 100.0),
}


def main():
    # Swap module-level constants in the confirmatory module so its writers
    # operate on the exploratory bands and write to the exploratory folder.
    original_band_specs = conf.BAND_SPECS
    original_output_base = conf.OUTPUT_BASE
    conf.BAND_SPECS = EXPLORATORY_BAND_SPECS
    conf.OUTPUT_BASE = OUTPUT_BASE

    band_order = list(EXPLORATORY_BAND_SPECS.keys())
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    try:
        print("[1/3] Power...")
        power_data = load_power_data()
        power_stats = run_permutation_for_modality(
            "Power", power_data, POWER_COH_KEYS, "freqs_diff", EXPLORATORY_BAND_SPECS,
        )
        conf.write_modality_outputs(
            "Power", "power", np.asarray(power_data["freqs_diff"], dtype=float),
            power_stats, band_order,
        )

        print("[2/3] Coherence...")
        coh_data = load_coherence_data()
        coh_stats = run_permutation_for_modality(
            "Coherence", coh_data, POWER_COH_KEYS, "freqs_diff", EXPLORATORY_BAND_SPECS,
        )
        conf.write_modality_outputs(
            "Coherence", "coherence", np.asarray(coh_data["freqs_diff"], dtype=float),
            coh_stats, band_order,
        )

        print("[3/3] PAC...")
        pac_data = load_pac_data()
        pac_stats = run_permutation_for_modality(
            "PAC", pac_data, PAC_KEYS, "freqs_diff", EXPLORATORY_BAND_SPECS,
        )
        conf.write_modality_outputs(
            "PAC", "PAC", np.asarray(pac_data["freqs_diff"], dtype=float),
            pac_stats, band_order,
        )
    finally:
        conf.BAND_SPECS = original_band_specs
        conf.OUTPUT_BASE = original_output_base

    print("Done.")


if __name__ == "__main__":
    main()
