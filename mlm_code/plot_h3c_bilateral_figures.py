#!/usr/bin/env python
"""
Bar figures for the Hypothesis 3c bilateral-IED report.

Figure 1: Bilateral vs unilateral IED -> % forgetting (encoding).
Figure 2: Within bilateral-IED trials, % forgetting by timing window
          (During-Stim vs After-Image), IED present vs absent.

Palette: dark purple (bilateral / IED present / stim) + lavender (unilateral /
IED absent / no-stim). Bold axis labels.
Saves PNGs to OUTPUTS/updated_IED_figures/.
"""

import os
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'IED', 'ied_timing_memory'))
FIG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'OUTPUTS', 'updated_IED_figures'))

# Bilateral = dark purple; unilateral = lavender (figure A only).
PURPLE = '#4B2E83'      # bilateral IED (present)
LAVENDER = '#C9B8E8'    # unilateral IED
# Within-bilateral window contrast: absent-in-window uses gray (NOT lavender, so it
# does not read as "unilateral").
WINDOW_ABSENT = '#9E9E9E'
# Stimulation contrast
STIM_RED = '#C0392B'
NOSTIM_BLUE = '#2C6FBB'
TEAL = PURPLE       # alias: bilateral / primary category
SAND = LAVENDER     # alias: unilateral
INK = '#333333'

plt.rcParams.update({
    'font.size': 12,
    'axes.edgecolor': INK,
    'axes.labelcolor': INK,
    'axes.labelweight': 'bold',
    'text.color': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _is_dark(color):
    r, g, b = mcolors.to_rgb(color)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.55


def pct_forget(sub):
    n = len(sub)
    f = int((sub['memory'] == 0).sum())
    return n, f, (100 * f / n if n else 0.0)


def se_prop(f, n):
    """Standard error of a proportion, on the percentage scale."""
    if n == 0:
        return 0.0
    p = f / n
    return 100 * math.sqrt(p * (1 - p) / n)


def label_bars(ax, bars, ns, errs=None):
    for i, (bar, n) in enumerate(zip(bars, ns)):
        h = bar.get_height()
        top = h + (errs[i] if errs is not None else 0)
        ax.text(bar.get_x() + bar.get_width() / 2, top + 1.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        tc = 'white' if _is_dark(bar.get_facecolor()) else INK
        ax.text(bar.get_x() + bar.get_width() / 2, 2.5,
                f'n={n}', ha='center', va='bottom', fontsize=9, color=tc)


def main():
    enc = pd.read_csv(os.path.join(DATA_DIR, 'h3c_encoding_bilateral.csv'))
    bil = enc[enc['trial_hemisphere'] == 'Bilateral']
    uni = enc[enc['trial_hemisphere'].isin(['L', 'R'])]

    # ── Figure 1: Bilateral vs Unilateral -> % forgetting ────────────────────
    nb, fbc, fb = pct_forget(bil)
    nu, fuc, fu = pct_forget(uni)
    errs = [se_prop(fbc, nb), se_prop(fuc, nu)]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    bars = ax.bar(['Bilateral IED', 'Unilateral IED'], [fb, fu],
                  color=[TEAL, SAND], width=0.62, edgecolor=INK, linewidth=0.8,
                  yerr=errs, capsize=5, ecolor=INK, error_kw={'elinewidth': 1.3})
    label_bars(ax, bars, [nb, nu], errs)
    ax.set_ylabel('% Forgotten')
    ax.set_ylim(0, 100)
    ax.set_title('Encoding', fontweight='bold', fontsize=14)
    fig.tight_layout()
    out1 = os.path.join(FIG_DIR, 'h3c_bilateral_vs_unilateral_forgetting.png')
    fig.savefig(out1, dpi=300)
    plt.close(fig)

    # ── Figure 2: Timing window within bilateral trials -> % forgetting ──────
    windows = [('Before\nImage', 'ied_before_image'),
               ('During\nImage', 'ied_during_image'),
               ('After\nImage', 'ied_after_image'),
               ('During\nStim', 'ied_during_stim')]
    present, absent, ns_pres, ns_abs, se_pres, se_abs = [], [], [], [], [], []
    for _, col in windows:
        np_, fpc, fp = pct_forget(bil[bil[col] == 1])
        na_, fac, fa = pct_forget(bil[bil[col] == 0])
        present.append(fp); ns_pres.append(np_); se_pres.append(se_prop(fpc, np_))
        absent.append(fa); ns_abs.append(na_); se_abs.append(se_prop(fac, na_))

    x = range(len(windows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], present, w, color=PURPLE,
                edgecolor=INK, linewidth=0.8, label='Bilateral IED present in window',
                yerr=se_pres, capsize=4, ecolor=INK, error_kw={'elinewidth': 1.2})
    b2 = ax.bar([i + w / 2 for i in x], absent, w, color=WINDOW_ABSENT,
                edgecolor=INK, linewidth=0.8, label='Bilateral IED absent in window',
                yerr=se_abs, capsize=4, ecolor=INK, error_kw={'elinewidth': 1.2})
    label_bars(ax, b1, ns_pres, se_pres)
    label_bars(ax, b2, ns_abs, se_abs)
    ax.set_xticks(list(x))
    ax.set_xticklabels([w0 for w0, _ in windows])
    ax.set_ylabel('% Forgotten')
    ax.set_ylim(0, 100)
    ax.set_title('Encoding', fontweight='bold', fontsize=14)
    ax.legend(frameon=False, loc='upper left', fontsize=10)
    fig.tight_layout()
    out2 = os.path.join(FIG_DIR, 'h3c_bilateral_window_forgetting.png')
    fig.savefig(out2, dpi=300)
    plt.close(fig)

    # ── Figure 3: Count of bilateral-IED trials by timing window ─────────────
    all_windows = [('Before\nImage', 'ied_before_image'),
                   ('During\nImage', 'ied_during_image'),
                   ('After\nImage', 'ied_after_image'),
                   ('During\nStim', 'ied_during_stim')]
    counts = [int((bil[c] == 1).sum()) for _, c in all_windows]
    nb = len(bil)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bars = ax.bar([w0 for w0, _ in all_windows], counts,
                  color=TEAL, width=0.66, edgecolor=INK, linewidth=0.8)
    for bar, c in zip(bars, counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f'{c}\n({100 * c / nb:.0f}%)', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel('Bilateral-IED trials with an IED')
    ax.set_ylim(0, max(counts) + 14)
    ax.set_title('Encoding', fontweight='bold', fontsize=14)
    fig.tight_layout()
    out3 = os.path.join(FIG_DIR, 'h3c_bilateral_window_counts.png')
    fig.savefig(out3, dpi=300)
    plt.close(fig)

    # ── Figure 4: Bilateral-IED trials, forgetting by stimulation ────────────
    stim = bil[bil['stim'] == 1]
    nostim = bil[bil['stim'] == 0]
    ns_s, fsc, fs = pct_forget(stim)
    ns_n, fnc, fn = pct_forget(nostim)
    errs = [se_prop(fsc, ns_s), se_prop(fnc, ns_n)]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    bars = ax.bar(['Stim', 'No-stim'], [fs, fn],
                  color=[STIM_RED, NOSTIM_BLUE], width=0.62, edgecolor=INK, linewidth=0.8,
                  yerr=errs, capsize=5, ecolor=INK, error_kw={'elinewidth': 1.3})
    label_bars(ax, bars, [ns_s, ns_n], errs)
    ax.set_ylabel('% Forgotten')
    ax.set_ylim(0, 100)
    ax.set_title('Encoding', fontweight='bold', fontsize=14)
    fig.tight_layout()
    out4 = os.path.join(FIG_DIR, 'h3c_bilateral_stim_forgetting.png')
    fig.savefig(out4, dpi=300)
    plt.close(fig)

    print(f'Saved: {out1}')
    print(f'Saved: {out2}')
    print(f'Saved: {out3}')
    print(f'Saved: {out4}')

    # ── Retrieval figures ────────────────────────────────────────────────────
    # Retrieval has only two timing windows (Before Image, During Image) and no
    # stimulation, so it gets the bilateral-vs-unilateral, window-forgetting, and
    # window-count figures (no stim contrast).
    ret = pd.read_csv(os.path.join(DATA_DIR, 'h3c_retrieval_bilateral.csv'))
    rbil = ret[ret['trial_hemisphere'] == 'Bilateral']
    runi = ret[ret['trial_hemisphere'].isin(['L', 'R'])]

    # ── Retrieval Figure 1: Bilateral vs Unilateral -> % forgetting ──────────
    nb, fbc, fb = pct_forget(rbil)
    nu, fuc, fu = pct_forget(runi)
    errs = [se_prop(fbc, nb), se_prop(fuc, nu)]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    bars = ax.bar(['Bilateral IED', 'Unilateral IED'], [fb, fu],
                  color=[TEAL, SAND], width=0.62, edgecolor=INK, linewidth=0.8,
                  yerr=errs, capsize=5, ecolor=INK, error_kw={'elinewidth': 1.3})
    label_bars(ax, bars, [nb, nu], errs)
    ax.set_ylabel('% Forgotten')
    ax.set_ylim(0, 100)
    ax.set_title('Retrieval', fontweight='bold', fontsize=14)
    fig.tight_layout()
    rout1 = os.path.join(FIG_DIR, 'h3c_retrieval_bilateral_vs_unilateral_forgetting.png')
    fig.savefig(rout1, dpi=300)
    plt.close(fig)

    # ── Retrieval Figure 2: Timing window within bilateral trials -> forgetting ─
    ret_windows = [('Before\nImage', 'ied_before_image'),
                   ('During\nImage', 'ied_during_image')]
    present, absent, ns_pres, ns_abs, se_pres, se_abs = [], [], [], [], [], []
    for _, col in ret_windows:
        np_, fpc, fp = pct_forget(rbil[rbil[col] == 1])
        na_, fac, fa = pct_forget(rbil[rbil[col] == 0])
        present.append(fp); ns_pres.append(np_); se_pres.append(se_prop(fpc, np_))
        absent.append(fa); ns_abs.append(na_); se_abs.append(se_prop(fac, na_))

    x = range(len(ret_windows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], present, w, color=PURPLE,
                edgecolor=INK, linewidth=0.8, label='Bilateral IED present in window',
                yerr=se_pres, capsize=4, ecolor=INK, error_kw={'elinewidth': 1.2})
    b2 = ax.bar([i + w / 2 for i in x], absent, w, color=WINDOW_ABSENT,
                edgecolor=INK, linewidth=0.8, label='Bilateral IED absent in window',
                yerr=se_abs, capsize=4, ecolor=INK, error_kw={'elinewidth': 1.2})
    label_bars(ax, b1, ns_pres, se_pres)
    label_bars(ax, b2, ns_abs, se_abs)
    ax.set_xticks(list(x))
    ax.set_xticklabels([w0 for w0, _ in ret_windows])
    ax.set_ylabel('% Forgotten')
    ax.set_ylim(0, 100)
    ax.set_title('Retrieval', fontweight='bold', fontsize=14)
    ax.legend(frameon=False, loc='upper left', fontsize=10)
    fig.tight_layout()
    rout2 = os.path.join(FIG_DIR, 'h3c_retrieval_bilateral_window_forgetting.png')
    fig.savefig(rout2, dpi=300)
    plt.close(fig)

    # ── Retrieval Figure 3: Count of bilateral-IED trials by timing window ────
    counts = [int((rbil[c] == 1).sum()) for _, c in ret_windows]
    nb = len(rbil)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    bars = ax.bar([w0 for w0, _ in ret_windows], counts,
                  color=TEAL, width=0.6, edgecolor=INK, linewidth=0.8)
    for bar, c in zip(bars, counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f'{c}\n({100 * c / nb:.0f}%)', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel('Bilateral-IED trials with an IED')
    ax.set_ylim(0, max(counts) + 8)
    ax.set_title('Retrieval', fontweight='bold', fontsize=14)
    fig.tight_layout()
    rout3 = os.path.join(FIG_DIR, 'h3c_retrieval_bilateral_window_counts.png')
    fig.savefig(rout3, dpi=300)
    plt.close(fig)

    print(f'Saved: {rout1}')
    print(f'Saved: {rout2}')
    print(f'Saved: {rout3}')


if __name__ == '__main__':
    main()
