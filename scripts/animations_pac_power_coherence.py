"""Animated illustrations of three neural concepts.

1. Phase-amplitude coupling (PAC): theta phase in region 1 modulating gamma
   amplitude in region 2.
2. 1/f LFP power spectrum with a theta bump that is larger in region 1 than
   region 2.
3. Theta-band coherence between region 1 and region 2.

All outputs are written as GIFs to outputs/animations/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

OUTDIR = Path(
    "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/"
    "dissertation/AMME_BLAES/outputs/animations"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

R1_COLOR = "#008080"  # teal
R2_COLOR = "#e07b00"  # orange
FPS = 30
SLOW_FPS = 12  # for PAC and coherence playback
LABEL_FS = 18
TICK_FS = 14
LEGEND_FS = 14


def animate_pac():
    """Theta phase (region 1) drives gamma amplitude (region 2)."""
    fs = 1000.0
    window_s = 2.0
    t = np.arange(0, window_s, 1.0 / fs)
    theta_f = 6.0
    gamma_f = 70.0

    # Total animation duration in seconds of "neural time"
    total_s = 6.0
    step_s = 1.0 / FPS
    n_frames = int(total_s / step_s)

    full_t = np.arange(0, total_s + window_s, 1.0 / fs)
    theta = np.sin(2 * np.pi * theta_f * full_t)
    # Modulate gamma amplitude so its envelope peaks at theta peaks
    # (sin peaks at phase pi/2 → align cosine-shifted envelope to that phase).
    mod = 0.5 * (1 + np.cos(2 * np.pi * theta_f * full_t - np.pi / 2))
    gamma = (0.2 + 0.8 * mod) * np.sin(2 * np.pi * gamma_f * full_t)
    rng = np.random.default_rng(0)
    gamma = gamma + 0.05 * rng.standard_normal(gamma.size)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1.2]},
    )

    line1, = ax1.plot([], [], color=R1_COLOR, lw=2)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_ylabel("Region 1\ntheta", color=R1_COLOR, fontsize=LABEL_FS)
    ax1.tick_params(axis="both", labelsize=TICK_FS)
    ax1.axhline(0, color="gray", lw=0.5)

    line2, = ax2.plot([], [], color=R2_COLOR, lw=1)
    env_pos, = ax2.plot([], [], color="black", lw=1.5)
    env_neg, = ax2.plot([], [], color="black", lw=1.5)
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_ylabel("Region 2\ngamma", color=R2_COLOR, fontsize=LABEL_FS)
    ax2.tick_params(axis="both", labelsize=TICK_FS)
    ax2.axhline(0, color="gray", lw=0.5)

    line3a, = ax3.plot([], [], color=R1_COLOR, lw=2, label="Region 1 theta")
    line3b, = ax3.plot([], [], color=R2_COLOR, lw=1, alpha=0.6,
                       label="Region 2 gamma")
    line3env, = ax3.plot([], [], color="black", lw=1.5,
                        label="Gamma envelope")
    ax3.set_ylim(-1.4, 1.4)
    ax3.set_ylabel("Overlay", fontsize=LABEL_FS)
    ax3.set_xlabel("Time (s)", fontsize=LABEL_FS)
    ax3.tick_params(axis="both", labelsize=TICK_FS)
    ax3.legend(loc="upper right", fontsize=LEGEND_FS, ncol=3)
    ax3.set_xlim(0, window_s)
    fig.tight_layout()

    def init():
        for ln in (line1, line2, env_pos, env_neg, line3a, line3b, line3env):
            ln.set_data([], [])
        return line1, line2, env_pos, env_neg, line3a, line3b, line3env

    def update(frame):
        start = frame * step_s
        idx0 = int(start * fs)
        idx1 = idx0 + t.size
        seg_t = full_t[idx0:idx1] - full_t[idx0]
        seg_theta = theta[idx0:idx1]
        seg_gamma = gamma[idx0:idx1]
        seg_mod = mod[idx0:idx1]
        envelope = 0.2 + 0.8 * seg_mod

        line1.set_data(seg_t, seg_theta)
        line2.set_data(seg_t, seg_gamma)
        env_pos.set_data(seg_t, envelope)
        env_neg.set_data(seg_t, -envelope)
        line3a.set_data(seg_t, seg_theta)
        line3b.set_data(seg_t, seg_gamma)
        line3env.set_data(seg_t, envelope)
        return line1, line2, env_pos, env_neg, line3a, line3b, line3env

    anim = FuncAnimation(
        fig, update, frames=n_frames, init_func=init, interval=1000 / SLOW_FPS,
        blit=True,
    )
    out = OUTDIR / "pac_theta_gamma.gif"
    anim.save(out, writer=PillowWriter(fps=SLOW_FPS))
    plt.close(fig)
    print(f"wrote {out}")


def animate_power():
    """1/f power spectrum with a larger theta bump in region 1 than region 2."""
    freqs = np.logspace(np.log10(1), np.log10(100), 400)

    def spectrum(theta_amp):
        base = 1.0 / freqs
        bump = theta_amp * np.exp(-0.5 * ((np.log(freqs) - np.log(6.5)) / 0.18) ** 2)
        return base + bump

    n_frames = 120
    # Region 1 stays high; region 2 lower; a slow oscillation makes the
    # difference visible while emphasising the theta band.
    phase = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    r1_amp = 0.55 + 0.10 * np.sin(phase)
    r2_amp = 0.18 + 0.05 * np.sin(phase + np.pi / 3)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_FS)
    ax.set_ylabel("Power", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)

    ax.plot(freqs, 1.0 / freqs, color="gray", lw=1, ls="--", label="1/f baseline")
    ax.axvspan(4, 8, color="lightgray", alpha=0.5, label="theta (4–8 Hz)")

    line1, = ax.plot([], [], color=R1_COLOR, lw=2.2, label="Region 1")
    line2, = ax.plot([], [], color=R2_COLOR, lw=2.2, label="Region 2")
    marker1, = ax.plot([], [], "o", color=R1_COLOR, ms=8)
    marker2, = ax.plot([], [], "o", color=R2_COLOR, ms=8)
    ax.legend(loc="lower left", fontsize=LEGEND_FS)
    fig.tight_layout()

    ax.set_xlim(freqs.min(), freqs.max())
    ax.set_ylim(1e-3, 5)

    theta_idx = np.argmin(np.abs(freqs - 6.5))

    def init():
        for ln in (line1, line2, marker1, marker2):
            ln.set_data([], [])
        return line1, line2, marker1, marker2

    def update(frame):
        s1 = spectrum(r1_amp[frame])
        s2 = spectrum(r2_amp[frame])
        line1.set_data(freqs, s1)
        line2.set_data(freqs, s2)
        marker1.set_data([freqs[theta_idx]], [s1[theta_idx]])
        marker2.set_data([freqs[theta_idx]], [s2[theta_idx]])
        return line1, line2, marker1, marker2

    anim = FuncAnimation(
        fig, update, frames=n_frames, init_func=init, interval=1000 / FPS,
        blit=True,
    )
    out = OUTDIR / "lfp_power_1overf_theta.gif"
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {out}")


def animate_coherence():
    """Theta-band coherence between two regions, growing then decaying."""
    fs = 1000.0
    window_s = 2.0
    t = np.arange(0, window_s, 1.0 / fs)
    theta_f = 6.0

    total_s = 6.0
    step_s = 1.0 / FPS
    n_frames = int(total_s / step_s)
    full_t = np.arange(0, total_s + window_s, 1.0 / fs)

    rng = np.random.default_rng(1)
    # Shared theta component plus independent noise per region
    shared = np.sin(2 * np.pi * theta_f * full_t)
    n1 = rng.standard_normal(full_t.size)
    n2 = rng.standard_normal(full_t.size)

    # Coherence ramps up across the animation
    coh_curve = np.clip(np.linspace(0.05, 0.95, n_frames), 0, 1)

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6.5),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_top, ax_bot = axes

    line1, = ax_top.plot([], [], color=R1_COLOR, lw=1.8, label="Region 1 theta")
    line2, = ax_top.plot([], [], color=R2_COLOR, lw=1.8, label="Region 2 theta")
    ax_top.set_xlim(0, window_s)
    ax_top.set_ylim(-3, 3)
    ax_top.set_ylabel("Filtered LFP", fontsize=LABEL_FS)
    ax_top.set_xlabel("Time (s)", fontsize=LABEL_FS)
    ax_top.tick_params(axis="both", labelsize=TICK_FS)
    ax_top.legend(loc="upper right", fontsize=LEGEND_FS)
    coh_text = ax_top.text(
        0.02, 0.95, "", transform=ax_top.transAxes,
        ha="left", va="top", fontsize=LEGEND_FS,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
    )

    bar = ax_bot.barh([0], [0], color="#444444", height=0.5)
    ax_bot.set_xlim(0, 1)
    ax_bot.set_yticks([0])
    ax_bot.set_yticklabels(["Theta coherence"], fontsize=LABEL_FS)
    ax_bot.set_xlabel("Coherence (0–1)", fontsize=LABEL_FS)
    ax_bot.tick_params(axis="x", labelsize=TICK_FS)
    for x in (0.25, 0.5, 0.75):
        ax_bot.axvline(x, color="gray", lw=0.5, ls=":")

    plt.tight_layout()

    def make_signals(c):
        # Mix shared signal with independent noise so that effective coherence
        # roughly equals c.
        a = np.sqrt(c)
        b = np.sqrt(1 - c)
        s1 = a * shared + b * n1 * 0.7
        s2 = a * shared + b * n2 * 0.7
        return s1, s2

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        coh_text.set_text("")
        bar[0].set_width(0)
        return line1, line2, coh_text, bar[0]

    def update(frame):
        c = coh_curve[frame]
        s1, s2 = make_signals(c)
        idx0 = int(frame * step_s * fs)
        idx1 = idx0 + t.size
        seg = full_t[idx0:idx1] - full_t[idx0]
        line1.set_data(seg, s1[idx0:idx1])
        line2.set_data(seg, s2[idx0:idx1])
        coh_text.set_text(f"theta coherence ≈ {c:.2f}")
        bar[0].set_width(c)
        return line1, line2, coh_text, bar[0]

    anim = FuncAnimation(
        fig, update, frames=n_frames, init_func=init, interval=1000 / SLOW_FPS,
        blit=False,
    )
    out = OUTDIR / "theta_coherence.gif"
    anim.save(out, writer=PillowWriter(fps=SLOW_FPS))
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    animate_pac()
    animate_power()
    animate_coherence()
