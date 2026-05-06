#!/usr/bin/env python
"""Exploratory paired permutation tests for retrieval power, coherence, and PAC.

Same paired sign-flip permutation procedure as the confirmatory analysis, but
covering frequency bands OUTSIDE Theta and Slow gamma:
    Delta = 1-3 Hz
    Alpha = 9-12 Hz
    Beta  = 13-20 Hz
    Gamma = 51-69 Hz
    HFA   = 70-100 Hz

Bands without any spectral samples in a modality (e.g., Delta in PAC TG) are
omitted automatically.

For each modality (power, coherence, PAC) and each memory condition
(remembered, forgotten):

1. Per subject, compute the within-subject mean difference (stim - no stim) of
   the baseline-corrected value averaged within each band.
2. Compute the observed group-level mean difference and t-statistic per
   (region, band).
3. Run 1000 sign-flip permutations: each iteration flips the sign of
   (stim - no stim) for a random subset of subjects (Bernoulli 1/2 per
   subject) and recomputes the group t.
4. Family-wise correction across all (region x band) tests within a memory
   condition uses max-statistic permutation: the max |t| across the family at
   each iteration forms the null against which observed |t| is compared.

Outputs (one PDF + one CSV per modality):
  outputs/parametric retrieval test/exploratory permutation testing/
    power/permutation.pdf, permutation.csv
    coherence/permutation.pdf, permutation.csv
    PAC/permutation.pdf, permutation.csv

Regions containing "PNAS" are excluded.
"""

from pathlib import Path

import numpy as np

from retrieval_confirmatory_permutation_testing import (
    MEMORY_KEYS_PAC,
    MEMORY_KEYS_POWER_COH,
    N_PERMUTATIONS,
    RNG_SEED,
    load_coherence_data,
    load_pac_data,
    load_power_data,
    run_permutation_for_modality,
    write_modality_outputs,
)
import retrieval_confirmatory_permutation_testing as conf


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = SCRIPT_DIR / "outputs" / "parametric retrieval test" / "stim vs no stim" / "exploratory permutation testing"

EXPLORATORY_BAND_SPECS = {
    "Delta": (1.0, 3.0),
    "Alpha": (9.0, 12.0),
    "Beta": (13.0, 20.0),
    "Gamma": (51.0, 69.0),
    "HFA": (70.0, 100.0),
}


def _swap_in_exploratory_bands():
    """Override module-level state in the confirmatory module so its helpers
    (band loops, write paths, table band ordering) operate on the exploratory
    band set and write into the exploratory output folder.
    """
    conf.BAND_SPECS = EXPLORATORY_BAND_SPECS
    conf.OUTPUT_BASE = OUTPUT_BASE
    return list(EXPLORATORY_BAND_SPECS.keys())


def main():
    band_order = _swap_in_exploratory_bands()
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Patch the band order used inside write_modality_outputs's PDF loop.
    # write_modality_outputs hard-codes ["Theta", "Slow gamma"]; we monkey-patch
    # the function to iterate over our exploratory bands instead.
    original_write = conf.write_modality_outputs

    def patched_write(modality_label, modality_subdir, freqs, stats_df):
        # Reuse the confirmatory builder code, but with exploratory band order.
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        out_dir = conf.OUTPUT_BASE / modality_subdir
        conf.reset_dir(out_dir)
        csv_path = out_dir / "permutation.csv"
        pdf_path = out_dir / "permutation.pdf"

        stats_df = stats_df.sort_values(["Memory", "Region", "Band"]).reset_index(drop=True)
        stats_df.to_csv(csv_path, index=False)

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(conf.make_title_page(modality_label, conf.N_PERMUTATIONS, freqs))
            plt.close("all")

            for memory_label in ["Remembered", "Forgotten"]:
                memory_df = stats_df[stats_df["Memory"] == memory_label].copy()
                for band_label in band_order:
                    band_df = memory_df[memory_df["Band"] == band_label].copy()
                    if band_df.empty:
                        continue
                    chunks = conf.chunk_dataframe(band_df, chunk_size=24)
                    total = len(chunks)
                    for idx, chunk in enumerate(chunks, start=1):
                        title = f"{modality_label} | {memory_label} Trials | {band_label}"
                        if total > 1:
                            title += f" ({idx}/{total})"
                        pdf.savefig(conf.make_table_page(title, chunk))
                        plt.close("all")

            pdf.savefig(conf.make_summary_page(stats_df))
            plt.close("all")

        print(f"  -> {pdf_path}")
        print(f"  -> {csv_path}")

    conf.write_modality_outputs = patched_write

    try:
        print("[1/3] Loading power data...")
        power_data = load_power_data()
        print("[1/3] Running exploratory paired sign-flip permutation tests for power...")
        power_stats = run_permutation_for_modality(
            "Power", power_data, MEMORY_KEYS_POWER_COH, "freqs_diff",
            n_perm=N_PERMUTATIONS, seed=RNG_SEED,
        )
        conf.write_modality_outputs("Power", "power", np.asarray(power_data["freqs_diff"], dtype=float), power_stats)

        print("[2/3] Loading coherence data...")
        coh_data = load_coherence_data()
        print("[2/3] Running exploratory paired sign-flip permutation tests for coherence...")
        coh_stats = run_permutation_for_modality(
            "Coherence", coh_data, MEMORY_KEYS_POWER_COH, "freqs_diff",
            n_perm=N_PERMUTATIONS, seed=RNG_SEED,
        )
        conf.write_modality_outputs("Coherence", "coherence", np.asarray(coh_data["freqs_diff"], dtype=float), coh_stats)

        print("[3/3] Loading PAC data...")
        pac_data = load_pac_data()
        print("[3/3] Running exploratory paired sign-flip permutation tests for PAC...")
        pac_stats = run_permutation_for_modality(
            "PAC", pac_data, MEMORY_KEYS_PAC, "freqs_diff",
            n_perm=N_PERMUTATIONS, seed=RNG_SEED,
        )
        conf.write_modality_outputs("PAC", "PAC", np.asarray(pac_data["freqs_diff"], dtype=float), pac_stats)
    finally:
        conf.write_modality_outputs = original_write

    print("Done.")


if __name__ == "__main__":
    main()
