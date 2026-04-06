"""
Simulate bandpass + notch filtered iEEG with distinct IED morphologies.

Channels : LHipp1-3, RAmy1-3, REnt1-3
IED types (all POSITIVE / upward deflection):
  A (REnt1-3)   – brief sharp upward spike
  E (RAmy1 only) – single large upward spike + long perfectly smooth slow wave
  J (LHipp2 only) – 2-5 variable-height polyspikes + perfectly smooth slow wave

Slow waves are injected AFTER filtering to guarantee smoothness.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

np.random.seed(42)
fs = 2000
duration = 10.0
n_channels = 9
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
ch_labels = ['LHipp1', 'LHipp2', 'LHipp3',
             'RAmy1',  'RAmy2',  'RAmy3',
             'REnt1',  'REnt2',  'REnt3']
CH = {name: i for i, name in enumerate(ch_labels)}

bp_low, bp_high = 0.5, 200
notch_q = 30
n_samples = int(fs * duration)
t = np.arange(n_samples) / fs


# ── helpers ─────────────────────────────────────────────────────────
def pink_noise(n):
    white = np.random.randn(n)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    freqs[0] = 1.0
    spec = np.fft.rfft(white) / np.sqrt(freqs)
    return np.fft.irfft(spec, n=n)


def add_osc(sig, freq, bw, amp):
    b, a = butter(2, [max(freq - bw/2, 0.5), freq + bw/2], 'band', fs=fs)
    return sig + filtfilt(b, a, np.random.randn(len(sig))) * amp


def smooth_bridge(start_val, end_val, n):
    x = np.linspace(0, 1, n, endpoint=False)
    return start_val + (end_val - start_val) * (3 * x**2 - 2 * x**3)


# ── IED waveform generators ────────────────────────────────────────
# These return (spike_waveform, slow_waveform, spike_len)
# Spike goes through the filter; slow wave is added post-filter.

def ied_type_a(amp=500):
    """Type A: brief symmetric sharp upward spike. No slow wave."""
    dur_ms = np.random.uniform(30, 50)
    n = int(dur_ms / 1000 * fs)
    spike = amp * np.sin(np.pi * np.arange(n) / n)
    tl = min(6, len(spike) // 4)
    spike[:tl] *= np.linspace(0, 1, tl)
    spike[-tl:] *= np.linspace(1, 0, tl)
    return spike, None, len(spike)


def ied_type_e(amp=650):
    """Type E: single large upward spike + long perfectly smooth slow wave."""
    # spike (~25-35 ms)
    n_spike = int(np.random.uniform(0.025, 0.035) * fs)
    spike = amp * np.sin(np.pi * np.arange(n_spike) / n_spike)
    tl = min(6, n_spike // 3)
    spike[:tl] *= np.linspace(0, 1, tl)

    # slow wave: raised cosine, perfectly smooth (~200-300 ms)
    slow_dur = np.random.uniform(0.20, 0.30)
    n_slow = int(slow_dur * fs)
    # single downward arch: -(1 - cos(2pi*x/n))/2
    slow = -amp * 0.45 * (1 - np.cos(2 * np.pi * np.arange(n_slow) / n_slow)) / 2
    # taper end
    tl_s = min(20, n_slow // 4)
    slow[-tl_s:] *= np.linspace(1, 0, tl_s)

    return spike, slow, len(spike)


def ied_type_j(amp=500):
    """Type J: 2-5 polyspikes with variable heights. Slow wave separate."""
    n_spikes = np.random.randint(2, 6)
    spike_dur_ms = np.random.uniform(12, 22)
    n_per_spike = int(spike_dur_ms / 1000 * fs)

    pieces = []
    for k in range(n_spikes):
        scale = np.random.uniform(0.4, 1.0)
        sp = amp * scale * np.sin(np.pi * np.arange(n_per_spike) / n_per_spike)
        pieces.append(sp)
        if k < n_spikes - 1:
            gap_ms = np.random.uniform(10, 30)
            pieces.append(np.zeros(int(gap_ms / 1000 * fs)))

    polyspike = np.concatenate(pieces)
    tl = min(6, len(polyspike) // 6)
    polyspike[:tl] *= np.linspace(0, 1, tl)

    # slow wave: raised cosine (~150-250 ms)
    slow_dur = np.random.uniform(0.15, 0.25)
    n_slow = int(slow_dur * fs)
    slow = -amp * 0.50 * (1 - np.cos(2 * np.pi * np.arange(n_slow) / n_slow)) / 2
    tl_s = min(20, n_slow // 4)
    slow[-tl_s:] *= np.linspace(1, 0, tl_s)

    return polyspike, slow, len(polyspike)


# ── build background ───────────────────────────────────────────────
data = np.zeros((n_channels, n_samples))
for ch in range(n_channels):
    sig = pink_noise(n_samples)
    sig = add_osc(sig, 6, 3, 0.4)
    sig = add_osc(sig, 10, 4, 0.3)
    sig = add_osc(sig, 20, 8, 0.15)
    sig += 0.25 * np.sin(2 * np.pi * 60 * t + np.random.uniform(0, 2*np.pi))
    sig = sig / np.std(sig) * 80
    data[ch] = sig


# ── IED schedule ────────────────────────────────────────────────────
sync_times_hipp_amy = [1.0, 3.0, 5.2, 7.5]
sync_times_rent = [2.0, 4.5, 6.8]

ied_schedule = []
for ts in sync_times_hipp_amy:
    ied_schedule.append((ts, 'LHipp2', 'J', np.random.uniform(470, 560)))
    ied_schedule.append((ts, 'RAmy1',  'E', np.random.uniform(620, 710)))
for ts in sync_times_rent:
    ied_schedule.append((ts, 'REnt1', 'A', np.random.uniform(470, 560)))
    ied_schedule.append((ts, 'REnt2', 'A', np.random.uniform(470, 560)))
    ied_schedule.append((ts, 'REnt3', 'A', np.random.uniform(470, 560)))

ied_generators = {'A': ied_type_a, 'E': ied_type_e, 'J': ied_type_j}

# ── inject ONLY spike portions into data (pre-filter) ──────────────
# Store slow wave info for post-filter injection
spike_windows = []       # (ch_idx, onset, end) — for red highlight
slow_wave_inject = []    # (ch_idx, onset_sample, slow_waveform)

for evt_time, ch_name, ied_type, amp in ied_schedule:
    gen = ied_generators[ied_type]
    spike_wf, slow_wf, spike_len = gen(amp=amp)
    ch_idx = CH[ch_name]
    onset = int(evt_time * fs)
    spike_end = onset + spike_len

    # inject spike into pre-filter data
    if 0 <= onset and spike_end <= n_samples:
        data[ch_idx, onset:spike_end] += spike_wf
        spike_windows.append((ch_idx, onset, spike_end))

        # queue slow wave for post-filter injection
        if slow_wf is not None:
            slow_onset = spike_end
            slow_end = slow_onset + len(slow_wf)
            if slow_end <= n_samples:
                slow_wave_inject.append((ch_idx, slow_onset, slow_wf))
                # extend the highlight window to include the slow wave
                spike_windows[-1] = (ch_idx, onset, slow_end)


# ── apply filters (with padding to avoid edge artifacts) ───────────
pad_len = int(1.0 * fs)
b_bp, a_bp = butter(4, [bp_low, bp_high], 'band', fs=fs)
notch_bas = []
for nf in [60, 120, 180]:
    if nf < fs / 2:
        bn, an = iirnotch(nf, notch_q, fs)
        notch_bas.append((bn, an))

filtered = np.zeros_like(data)
for ch in range(n_channels):
    padded = np.concatenate([data[ch, pad_len:0:-1],
                             data[ch],
                             data[ch, -2:-pad_len-2:-1]])
    sig = filtfilt(b_bp, a_bp, padded)
    for bn, an in notch_bas:
        sig = filtfilt(bn, an, sig)
    filtered[ch] = sig[pad_len:pad_len + n_samples]


# ── replace post-spike segments with smooth slow waves ──────────────
for ch_idx, slow_onset, slow_wf in slow_wave_inject:
    slow_end = slow_onset + len(slow_wf)
    start_val = filtered[ch_idx, slow_onset - 1] if slow_onset > 0 else filtered[ch_idx, slow_onset]
    end_val = filtered[ch_idx, slow_end] if slow_end < n_samples else filtered[ch_idx, slow_end - 1]
    baseline = smooth_bridge(start_val, end_val, len(slow_wf))
    filtered[ch_idx, slow_onset:slow_end] = baseline + slow_wf


# ── compute per-channel 3 SD threshold for highlighting ────────────
ch_std = np.array([np.std(filtered[ch]) for ch in range(n_channels)])


# ── figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
spacing = 400

for ch in range(n_channels):
    offset = (n_channels - 1 - ch) * spacing
    trace = filtered[ch]
    ax.plot(t, trace + offset, color='k', linewidth=0.45)

    for w_ch, w_s, w_e in spike_windows:
        if w_ch != ch:
            continue
        w_s = max(w_s, 0)
        w_e = min(w_e, n_samples)
        segment = trace[w_s:w_e]
        if np.max(np.abs(segment)) > 3 * ch_std[ch]:
            mask = np.zeros(n_samples, dtype=bool)
            mask[w_s:w_e] = True
            ax.plot(t[mask], trace[mask] + offset, color='red', linewidth=0.85)

ax.set_yticks([i * spacing for i in range(n_channels)])
ax.set_yticklabels(list(reversed(ch_labels)), fontsize=9, fontfamily='monospace')
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_title('Simulated iEEG  |  BP 0.5–200 Hz  •  Notch 60/120/180 Hz\n'
             'Type A → REnt  •  Type E → RAmy1  •  Type J → LHipp2   '
             '(red = > 3 SD)',
             fontsize=11)
ax.set_xlim(0, duration)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# scale bar
sb_x = duration + 0.08
sb_y = -spacing * 0.5
ax.plot([sb_x, sb_x], [sb_y, sb_y + 500], 'k-', lw=2, clip_on=False)
ax.text(sb_x + 0.1, sb_y + 250, '500 µV', va='center', fontsize=8)
ax.plot([sb_x, sb_x + 0.5], [sb_y, sb_y], 'k-', lw=2, clip_on=False)
ax.text(sb_x + 0.25, sb_y - 90, '0.5 s', ha='center', fontsize=8)

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'simulated_ieeg_ieds.png'), dpi=200, bbox_inches='tight')
plt.show()


# ── figure 2: single channel raw trace with scale bars only ────────
fig2, ax2 = plt.subplots(figsize=(8, 2.5))
# just one channel of raw filtered data
single_ch = filtered[0]  # LHipp1
ax2.plot(t, single_ch, color='k', linewidth=0.6)
ax2.set_xlim(0, duration)

# remove all axes — clean like the reference
ax2.axis('off')

# scale bars: 100 µV vertical, 1 s horizontal (bottom-right corner)
ylims = ax2.get_ylim()
sb_x = duration - 1.5
sb_y = ylims[0] + (ylims[1] - ylims[0]) * 0.1
# vertical bar
ax2.plot([sb_x, sb_x], [sb_y, sb_y + 100], 'k-', lw=2)
ax2.text(sb_x - 0.15, sb_y + 50, '100 µV', va='center', ha='right',
         fontsize=9, fontweight='bold')
# horizontal bar
ax2.plot([sb_x, sb_x + 1.0], [sb_y, sb_y], 'k-', lw=2)
ax2.text(sb_x + 0.5, sb_y - 25, '1 s', ha='center', fontsize=9,
         fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'simulated_ieeg_single_channel.png'),
            dpi=200, bbox_inches='tight')
plt.show()

print("Done - saved outputs/simulated_ieeg_ieds.png, "
      "outputs/simulated_ieeg_single_channel.png")
