"""Compute peak frequencies for the GroupPower / GroupCoherency *core_regions*
encoding figures (combined BLAES + AMME, all stim and memory conditions).

Mirrors scripts/peak_freqs_retrieval_core_regions.py and
peak_freqs_retrieval_power_aperiodic.py but for ENCODING data.

For power: also does 1/f detrending and reports peaks plus spans, with the
multitaper boundary artifact (~12-22 Hz) masked.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'encoding_code'))

import combined_encoding_power as P
import combined_encoding_coherence as C

CORE = ['BLA', 'CA', 'DG', 'EC', 'HPC', 'PRC']
THETA = (4.0, 8.0)
SLOW_GAMMA = (30.0, 55.0)
ARTIFACT_BAND = (12.0, 22.0)
SPAN_THRESHOLD_DB = 0.5


def merge_all(blaes, amme, module):
    """Replicate the all_data merge from the encoding __main__ blocks.
    Note: encoding sets use_memory_collapsed_overall=False for the merged
    'all' (so group_all_power is used directly)."""
    keys = [
        'group_all_power', 'stim_power', 'nostim_power',
        'bc_stim', 'bc_nostim',
        'stim_rem', 'stim_forg', 'nostim_rem', 'nostim_forg',
        'bc_stim_rem', 'bc_stim_forg', 'bc_nostim_rem', 'bc_nostim_forg',
    ]
    out = {k: module.merge_dicts(blaes.get(k, {}), amme.get(k, {})) for k in keys}
    out['freqs_post'] = blaes.get('freqs_post') if blaes.get('freqs_post') is not None else amme.get('freqs_post')
    out['freqs_diff'] = blaes.get('freqs_diff') if blaes.get('freqs_diff') is not None else amme.get('freqs_diff')
    out['use_memory_collapsed_overall'] = False
    return out


def is_core_pair(roi):
    parts = [p.strip() for p in str(roi).split('_') if p.strip()]
    return len(parts) == 2 and all(p in CORE for p in parts)


def peak_in_band(freqs, vals, band):
    f0, f1 = band
    mask = (freqs >= f0) & (freqs <= f1)
    if not np.any(mask):
        return np.nan, np.nan
    sub_f, sub_v = freqs[mask], vals[mask]
    idx = int(np.nanargmax(sub_v))
    return float(sub_f[idx]), float(sub_v[idx])


def fit_aperiodic(freqs, spectrum_db):
    log_f = np.log10(freqs)
    excl = (
        ((freqs >= 3.0) & (freqs <= 9.0))
        | ((freqs >= ARTIFACT_BAND[0]) & (freqs <= ARTIFACT_BAND[1]))
        | ((freqs >= 25.0) & (freqs <= 60.0))
    )
    keep = ~excl & np.isfinite(spectrum_db)
    a, b = np.polyfit(log_f[keep], spectrum_db[keep], 1)
    return a * log_f + b


def find_spans(freqs, residual, threshold):
    artifact_mask = (freqs >= ARTIFACT_BAND[0]) & (freqs <= ARTIFACT_BAND[1])
    above = (residual > threshold) & ~artifact_mask
    spans = []
    i, n = 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            peak_idx = i + int(np.argmax(residual[i:j+1]))
            spans.append((float(freqs[i]), float(freqs[j]),
                          float(freqs[peak_idx]), float(residual[peak_idx])))
            i = j + 1
        else:
            i += 1
    return spans


def report_raw_argmax(label, data, module, roi_filter, unit):
    gap = module.get_overall_power_for_plot(data)
    freqs = np.asarray(data['freqs_post'], dtype=np.float64)
    rois = sorted(roi for roi in gap if roi_filter(roi))
    if not rois:
        print(f"\n[{label}] no ROIs")
        return None, None

    print(f"\n{'='*100}")
    print(f"{label} - raw argmax peak frequencies (no detrending)")
    print(f"{'='*100}")
    print(f"Plotted range: {freqs.min():.1f}-{freqs.max():.1f} Hz")
    header = f"{'ROI':<10} {'N':>3}  {'Peak(full)':>16}  {'Theta(4-8)':>20}  {'SlowGamma(30-55)':>22}"
    print(header)
    print('-' * len(header))

    roi_means = {}
    for roi in rois:
        mat = np.array(list(gap[roi].values()), dtype=np.float64)
        mat[~np.isfinite(mat)] = np.nan
        n = mat.shape[0]
        mean = np.nanmean(mat, axis=0)
        roi_means[roi] = mean
        full_idx = int(np.nanargmax(mean))
        full_f, full_v = float(freqs[full_idx]), float(mean[full_idx])
        th_f, th_v = peak_in_band(freqs, mean, THETA)
        sg_f, sg_v = peak_in_band(freqs, mean, SLOW_GAMMA)
        print(f"{roi:<10} {n:>3}  "
              f"{full_f:>5.1f}Hz {full_v:>+6.2f}{unit}  "
              f"{th_f:>5.1f}Hz {th_v:>+6.2f}{unit}    "
              f"{sg_f:>5.1f}Hz {sg_v:>+6.2f}{unit}")

    grand = np.nanmean(np.vstack(list(roi_means.values())), axis=0)
    full_idx = int(np.nanargmax(grand))
    full_f, full_v = float(freqs[full_idx]), float(grand[full_idx])
    th_f, th_v = peak_in_band(freqs, grand, THETA)
    sg_f, sg_v = peak_in_band(freqs, grand, SLOW_GAMMA)
    print('-' * len(header))
    print(f"{'MEAN':<10} {len(rois):>3}  "
          f"{full_f:>5.1f}Hz {full_v:>+6.2f}{unit}  "
          f"{th_f:>5.1f}Hz {th_v:>+6.2f}{unit}    "
          f"{sg_f:>5.1f}Hz {sg_v:>+6.2f}{unit}")
    return freqs, roi_means


def report_detrended_power(freqs, roi_means):
    print(f"\n{'='*100}")
    print(f"POWER (dB) - 1/f-detrended peak frequencies "
          f"(artifact band {ARTIFACT_BAND[0]:.0f}-{ARTIFACT_BAND[1]:.0f} Hz masked)")
    print(f"{'='*100}")
    print(f"{'ROI':<6} {'Theta peak':<22}  {'Slow-gamma peak':<26}  Spans above 1/f")
    print('-' * 100)

    roi_residuals = {}
    for roi in CORE:
        if roi not in roi_means:
            continue
        mean_db = roi_means[roi]
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
        print(f"{roi:<6} {th_f:>5.1f} Hz / {th_v:+.2f} dB     "
              f"{sg_f:>5.1f} Hz / {sg_v:+.2f} dB        "
              f"{span_str}")

    grand = np.nanmean(np.vstack(list(roi_residuals.values())), axis=0)
    th_f, th_v = peak_in_band(freqs, grand, THETA)
    sg_f, sg_v = peak_in_band(freqs, grand, SLOW_GAMMA)
    spans = find_spans(freqs, grand, SPAN_THRESHOLD_DB)
    span_str = '; '.join(
        f"{f0:.1f}-{f1:.1f} Hz (max {pk:.1f} Hz, +{val:.2f} dB)"
        for f0, f1, pk, val in spans
    ) if spans else 'none'
    print('-' * 100)
    print(f"{'MEAN':<6} {th_f:>5.1f} Hz / {th_v:+.2f} dB     "
          f"{sg_f:>5.1f} Hz / {sg_v:+.2f} dB        "
          f"{span_str}")


def main():
    print("Loading BLAES encoding power...")
    blaes_p = P.load_blaes_encoding()
    print("Loading AMME encoding power...")
    amme_p = P.load_amme_encoding()
    all_p = merge_all(blaes_p, amme_p, P)

    print("\nLoading BLAES encoding coherence...")
    blaes_c = C.augment_with_allhpc_pair_composites(C.load_blaes_encoding())
    print("Loading AMME encoding coherence...")
    amme_c = C.augment_with_allhpc_pair_composites(C.load_amme_encoding())
    all_c = merge_all(blaes_c, amme_c, C)
    all_c = C.augment_with_allhpc_pair_composites(C.canonicalize_coherence_data(all_c))

    freqs_p, roi_means_p = report_raw_argmax(
        'POWER (dB)', all_p, P, lambda r: r in CORE, 'dB')
    report_raw_argmax(
        'COHERENCE (Fisher Z)', all_c, C, is_core_pair, 'Z')
    if freqs_p is not None:
        report_detrended_power(freqs_p, roi_means_p)


if __name__ == '__main__':
    main()
