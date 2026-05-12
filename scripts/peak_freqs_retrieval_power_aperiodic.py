"""Detrend 1/f from the GroupPower core-regions retrieval spectra and report
oscillatory peaks plus contiguous spans of elevated activity.

For each ROI:
  1. Take mean log10(power) across patients
  2. Fit aperiodic background as a straight line in log10(freq) vs log10(power)
  3. Compute residual (data - fit) - this is the oscillatory component in dB units
  4. Report peak of residual in theta (4-8 Hz) and slow gamma (30-55 Hz)
  5. Report contiguous spans where residual > 0.5 dB above the 1/f line

Note: input is already in dB (10*log10), so the y-axis is already log-power.
We fit y = a*log10(f) + b  (i.e., a line in log-freq vs dB).
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'retrieval_code'))

import combined_retrieval_power as P

CORE = ['BLA', 'CA', 'DG', 'EC', 'HPC', 'PRC']
THETA = (4.0, 8.0)
SLOW_GAMMA = (30.0, 55.0)
# Multitaper parameter boundary creates a spurious bump around 15-18 Hz - mask it
ARTIFACT_BAND = (12.0, 22.0)
SPAN_THRESHOLD_DB = 0.5  # contiguous freqs where residual exceeds this count as a "span"


def merge_all(blaes, amme):
    keys = ['group_all_power', 'stim_rem', 'stim_forg', 'nostim_rem', 'nostim_forg']
    out = {k: P.merge_dicts(blaes.get(k, {}), amme.get(k, {})) for k in keys}
    out['freqs_post'] = blaes['freqs_post'] if blaes['freqs_post'] is not None else amme['freqs_post']
    out['use_memory_collapsed_overall'] = True
    return out


def fit_aperiodic(freqs, spectrum_db):
    """Fit dB = a*log10(f) + b. Exclude theta, slow gamma, and the multitaper
    boundary artifact band so the fit is driven by clean aperiodic regions."""
    log_f = np.log10(freqs)
    excl = (
        ((freqs >= 3.0) & (freqs <= 9.0))
        | ((freqs >= ARTIFACT_BAND[0]) & (freqs <= ARTIFACT_BAND[1]))
        | ((freqs >= 25.0) & (freqs <= 60.0))
    )
    keep = ~excl & np.isfinite(spectrum_db)
    a, b = np.polyfit(log_f[keep], spectrum_db[keep], 1)
    return a * log_f + b


def peak_in_band(freqs, residual, band):
    f0, f1 = band
    mask = (freqs >= f0) & (freqs <= f1)
    if not np.any(mask):
        return np.nan, np.nan
    sub_f, sub_r = freqs[mask], residual[mask]
    idx = int(np.nanargmax(sub_r))
    return float(sub_f[idx]), float(sub_r[idx])


def find_spans(freqs, residual, threshold):
    """Contiguous frequency ranges where residual > threshold, with the
    multitaper artifact band masked out (treated as if not above threshold)."""
    artifact_mask = (freqs >= ARTIFACT_BAND[0]) & (freqs <= ARTIFACT_BAND[1])
    above = (residual > threshold) & ~artifact_mask
    spans = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            f_lo, f_hi = float(freqs[i]), float(freqs[j])
            peak_idx = i + int(np.argmax(residual[i:j+1]))
            spans.append((f_lo, f_hi, float(freqs[peak_idx]), float(residual[peak_idx])))
            i = j + 1
        else:
            i += 1
    return spans


def main():
    print("Loading BLAES retrieval power...")
    blaes = P.load_blaes_retrieval()
    print("Loading AMME retrieval power...")
    amme = P.load_amme_retrieval()
    data = merge_all(blaes, amme)

    gap = P.get_overall_power_for_plot(data)
    freqs = np.asarray(data['freqs_post'], dtype=np.float64)

    print(f"\nPower spectra: {freqs.min():.1f}-{freqs.max():.1f} Hz, "
          f"{len(freqs)} bins\n")
    print(f"Aperiodic fit: linear in log10(freq) vs dB, excluding 3-9 Hz, "
          f"{ARTIFACT_BAND[0]:.0f}-{ARTIFACT_BAND[1]:.0f} Hz (multitaper boundary artifact), "
          f"and 25-60 Hz from the fit")
    print(f"Span threshold: residual > {SPAN_THRESHOLD_DB} dB above 1/f line "
          f"(spans inside {ARTIFACT_BAND[0]:.0f}-{ARTIFACT_BAND[1]:.0f} Hz are masked)\n")

    print("=" * 100)
    print(f"{'ROI':<6} {'N':>3}  {'Theta peak (4-8 Hz)':<22}  {'Slow-gamma peak (30-55 Hz)':<28}  Spans above 1/f")
    print("=" * 100)

    roi_residuals = {}
    for roi in CORE:
        if roi not in gap:
            continue
        mat = np.array(list(gap[roi].values()), dtype=np.float64)
        mat[~np.isfinite(mat)] = np.nan
        n = mat.shape[0]
        mean_db = np.nanmean(mat, axis=0)
        baseline = fit_aperiodic(freqs, mean_db)
        residual = mean_db - baseline
        roi_residuals[roi] = residual

        th_f, th_v = peak_in_band(freqs, residual, THETA)
        sg_f, sg_v = peak_in_band(freqs, residual, SLOW_GAMMA)
        spans = find_spans(freqs, residual, SPAN_THRESHOLD_DB)
        span_str = '; '.join(
            f"{f0:.1f}-{f1:.1f} Hz (max {pk:.1f} Hz, +{val:.2f} dB)"
            for f0, f1, pk, val in spans
        ) if spans else 'none'

        print(f"{roi:<6} {n:>3}  "
              f"{th_f:>5.1f} Hz / {th_v:+.2f} dB     "
              f"{sg_f:>5.1f} Hz / {sg_v:+.2f} dB           "
              f"{span_str}")

    print('-' * 100)
    grand = np.nanmean(np.vstack(list(roi_residuals.values())), axis=0)
    th_f, th_v = peak_in_band(freqs, grand, THETA)
    sg_f, sg_v = peak_in_band(freqs, grand, SLOW_GAMMA)
    spans = find_spans(freqs, grand, SPAN_THRESHOLD_DB)
    span_str = '; '.join(
        f"{f0:.1f}-{f1:.1f} Hz (max {pk:.1f} Hz, +{val:.2f} dB)"
        for f0, f1, pk, val in spans
    ) if spans else 'none'
    print(f"{'MEAN':<6} {len(roi_residuals):>3}  "
          f"{th_f:>5.1f} Hz / {th_v:+.2f} dB     "
          f"{sg_f:>5.1f} Hz / {sg_v:+.2f} dB           "
          f"{span_str}")


if __name__ == '__main__':
    main()
