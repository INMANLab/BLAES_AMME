"""Compute peak frequencies for the GroupPower / GroupCoherency *core_regions*
retrieval figures (combined BLAES + AMME, all stim and memory conditions).

Reproduces the data path used by `plot_power_by_roi_core` and
`plot_coherence_by_roi_core` in retrieval_code/, then reports the argmax
frequency in (a) the full plotted range, (b) theta (4-8 Hz), and (c) slow
gamma (30-55 Hz) for each ROI plus a cross-ROI mean.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'retrieval_code'))

import combined_retrieval_power as P
import combined_retrieval_coherence as C

CORE = {'BLA', 'CA', 'DG', 'EC', 'HPC', 'PRC'}
THETA = (4.0, 8.0)
SLOW_GAMMA = (30.0, 55.0)


def merge_all(blaes, amme, module):
    """Replicate the all_data merge from the __main__ blocks."""
    keys_dicts = [
        'group_all_power', 'stim_power', 'nostim_power',
        'bc_stim', 'bc_nostim',
        'stim_rem', 'stim_forg', 'nostim_rem', 'nostim_forg',
        'bc_stim_rem', 'bc_stim_forg', 'bc_nostim_rem', 'bc_nostim_forg',
    ]
    out = {k: module.merge_dicts(blaes.get(k, {}), amme.get(k, {})) for k in keys_dicts}
    fp_b, fp_a = blaes.get('freqs_post'), amme.get('freqs_post')
    out['freqs_post'] = fp_b if fp_b is not None else fp_a
    out['freqs_diff'] = blaes.get('freqs_diff') if blaes.get('freqs_diff') is not None else amme.get('freqs_diff')
    out['has_memory'] = True
    out['use_memory_collapsed_overall'] = True
    return out


def is_core_single(roi):
    return roi in CORE


def is_core_pair(roi):
    parts = [p.strip() for p in str(roi).split('_') if p.strip()]
    return len(parts) == 2 and all(p in CORE for p in parts)


def peak_in_band(freqs, mean, band):
    f0, f1 = band
    mask = (freqs >= f0) & (freqs <= f1)
    if not np.any(mask):
        return np.nan, np.nan
    sub_f = freqs[mask]
    sub_m = mean[mask]
    idx = int(np.nanargmax(sub_m))
    return float(sub_f[idx]), float(sub_m[idx])


def report(measure_label, data, roi_filter, unit):
    gap = data['module'].get_overall_power_for_plot(data['data'])
    freqs = np.asarray(data['data']['freqs_post'], dtype=np.float64)
    rois = sorted(roi for roi in gap if roi_filter(roi))
    if not rois:
        print(f"\n[{measure_label}] no ROIs found")
        return

    print(f"\n{'='*78}")
    print(f"{measure_label} - peak frequencies by ROI (mean across patients, "
          f"collapsed over stim/no-stim and remembered/forgotten)")
    print(f"{'='*78}")
    print(f"Plotted range: {freqs.min():.1f}-{freqs.max():.1f} Hz")
    header = f"{'ROI':<10} {'N':>3}  {'PeakFull':>11}  {'Theta(4-8Hz)':>20}  {'SlowGamma(30-55Hz)':>22}"
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
              f"{full_f:>5.1f}Hz {full_v:>+5.2f}{unit}  "
              f"{th_f:>5.1f}Hz {th_v:>+5.2f}{unit}      "
              f"{sg_f:>5.1f}Hz {sg_v:>+5.2f}{unit}")

    grand = np.nanmean(np.vstack(list(roi_means.values())), axis=0)
    full_idx = int(np.nanargmax(grand))
    full_f, full_v = float(freqs[full_idx]), float(grand[full_idx])
    th_f, th_v = peak_in_band(freqs, grand, THETA)
    sg_f, sg_v = peak_in_band(freqs, grand, SLOW_GAMMA)
    print('-' * len(header))
    print(f"{'MEAN(ROIs)':<10} {len(rois):>3}  "
          f"{full_f:>5.1f}Hz {full_v:>+5.2f}{unit}  "
          f"{th_f:>5.1f}Hz {th_v:>+5.2f}{unit}      "
          f"{sg_f:>5.1f}Hz {sg_v:>+5.2f}{unit}")


def main():
    print("Loading BLAES retrieval power...")
    blaes_p = P.load_blaes_retrieval()
    print("Loading AMME retrieval power...")
    amme_p = P.load_amme_retrieval()
    all_p = merge_all(blaes_p, amme_p, P)

    print("\nLoading BLAES retrieval coherence...")
    blaes_c = C.augment_with_allhpc_pair_composites(C.load_blaes_retrieval())
    print("Loading AMME retrieval coherence...")
    amme_c = C.augment_with_allhpc_pair_composites(C.load_amme_retrieval())
    all_c = merge_all(blaes_c, amme_c, C)
    all_c = C.augment_with_allhpc_pair_composites(C.canonicalize_coherence_data(all_c))

    report('POWER (dB)',
           {'data': all_p, 'module': P},
           is_core_single, 'dB')
    report('COHERENCE (Fisher Z)',
           {'data': all_c, 'module': C},
           is_core_pair, 'Z')


if __name__ == '__main__':
    main()
