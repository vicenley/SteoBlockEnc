#!/usr/bin/env python3
"""
Generate publication-quality figures from pre-computed simulation data.

All figures are designed for two-column (REVTeX reprint) layout:
  - Figures 1-4: full-width (figure*), ~7 inches wide
  - Figure 5: full-width, 2 panels

Reads .npz files from data/ (produced by run_simulations.py).

Usage:
  python scripts/generate_figures_from_data.py           # all figures
  python scripts/generate_figures_from_data.py --fig 1 3 # specific figures
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from numpy.polynomial.chebyshev import chebval

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stereo_block_enc.numerical.qsp_phases import (
    decoded_function, stereo_qsp_product, qsp_product, r_to_a, cot_base_function,
)

# ── Publication style ───────────────────────────────────────────────────────
rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.linewidth': 0.6,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.fontsize': 7.5,
    'legend.handlelength': 1.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'lines.linewidth': 1.0,
})

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'ms', 'figures')
DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Colorblind-friendly palette (Okabe-Ito)
C = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'cyan':    '#56B4E9',
    'yellow':  '#F0E442',
    'black':   '#000000',
    'gray':    '#999999',
}
PAL = [C['blue'], C['orange'], C['green'], C['red'], C['purple'], C['cyan']]

# Two-column textwidth in REVTeX reprint ≈ 7.08 in;  columnwidth ≈ 3.4 in
TEXTWIDTH = 7.08
COLWIDTH = 3.4


def _save(fig, name):
    fig.savefig(os.path.join(FIGDIR, f'{name}.pdf'))
    fig.savefig(os.path.join(FIGDIR, f'{name}.png'))
    plt.close(fig)


def load(name):
    path = os.path.join(DATADIR, name)
    if not os.path.exists(path):
        print(f'    WARNING: {path} not found. Run simulations first.')
        return None
    return np.load(path, allow_pickle=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def TB(k, r):
    a = r / np.sqrt(1 + r**2)
    return chebval(a, np.eye(k + 1)[k])

def SB(k, r):
    a = r / np.sqrt(1 + r**2)
    s = 1.0 / np.sqrt(1 + r**2)
    theta = np.arccos(np.clip(a, -1, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        Uk = np.where(np.abs(np.sin(theta)) > 1e-15,
                      np.sin(k * theta) / np.sin(theta),
                      k * np.cos(k * theta) / np.cos(theta))
    return Uk * s

def _panel_label(ax, text, x=-0.12, y=1.06):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Rational Chebyshev basis and decoded cotangent functions
# ═══════════════════════════════════════════════════════════════════════════

def figure_1():
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5))
    r = np.linspace(-10, 10, 1000)
    r_pos = np.linspace(0.02, 10, 1000)

    # (a) TB_k
    ax = axes[0]
    _panel_label(ax, '(a)')
    styles = ['-', '-', '-', '--', '--']
    for k in range(5):
        ax.plot(r, TB(k, r), color=PAL[k], linewidth=1.0, linestyle=styles[k],
                label=f'$k={k}$')
    ax.set_xlabel('$r$')
    ax.set_ylabel('$\\mathrm{TB}_k(r)$')
    ax.set_ylim(-1.35, 1.35)
    ax.axhline(1, color=C['gray'], linewidth=0.3, ls=':')
    ax.axhline(-1, color=C['gray'], linewidth=0.3, ls=':')
    ax.axhline(0, color=C['gray'], linewidth=0.3)
    ax.axvline(0, color=C['gray'], linewidth=0.3)
    ax.legend(ncol=3, loc='lower center', frameon=False, fontsize=7,
              columnspacing=0.8, handletextpad=0.4)

    # (b) SB_k
    ax = axes[1]
    _panel_label(ax, '(b)')
    for k in range(1, 5):
        ax.plot(r, SB(k, r), color=PAL[k], linewidth=1.0, label=f'$k={k}$')
    ax.set_xlabel('$r$')
    ax.set_ylabel('$\\mathrm{SB}_k(r)$')
    ax.set_ylim(-1.35, 1.35)
    ax.axhline(1, color=C['gray'], linewidth=0.3, ls=':')
    ax.axhline(-1, color=C['gray'], linewidth=0.3, ls=':')
    ax.axhline(0, color=C['gray'], linewidth=0.3)
    ax.axvline(0, color=C['gray'], linewidth=0.3)
    ax.legend(ncol=2, loc='lower center', frameon=False, fontsize=7,
              columnspacing=0.8, handletextpad=0.4)

    # (c) Decoded cot_k — the unbounded rational functions
    ax = axes[2]
    _panel_label(ax, '(c)')
    for i, k in enumerate([2, 3, 4, 5]):
        y = cot_base_function(k, r_pos)
        y_clip = np.where(np.abs(y) < 14, y, np.nan)
        ax.plot(r_pos, y_clip, color=PAL[i], linewidth=1.0, label=f'$k={k}$')
    # Shade the unbounded region to emphasize it
    ax.axhspan(-14, -1, alpha=0.03, color=C['red'])
    ax.axhspan(1, 14, alpha=0.03, color=C['red'])
    ax.axhline(0, color=C['gray'], linewidth=0.3)
    ax.set_xlabel('$r$')
    ax.set_ylabel('$z_k(r) = \\cot(k\\arctan(1/r))$')
    ax.set_ylim(-14, 14)
    ax.set_xlim(0, 10)
    ax.legend(loc='upper right', frameon=False, fontsize=7)

    fig.tight_layout(w_pad=1.8)
    _save(fig, 'rational_chebyshev')
    print('    saved: rational_chebyshev.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Phase-finding and the bounded-to-unbounded mechanism
# ═══════════════════════════════════════════════════════════════════════════

def figure_2():
    data1 = load('sim1_base_cases.npz')

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5))
    r_fine = np.linspace(0.08, 12, 600)

    # (a) Base-case recovery k=2 (near machine precision)
    ax = axes[0]
    _panel_label(ax, '(a)')
    if data1 is not None:
        phis = data1['k2_phis']
        target = cot_base_function(2, r_fine)
        recovered = decoded_function(phis, r_fine).real
        t_clip = np.where(np.abs(target) < 14, target, np.nan)
        r_clip = np.where(np.abs(recovered) < 14, recovered, np.nan)
        ax.plot(r_fine, t_clip, '-', color=PAL[0], linewidth=1.2, label='target $z_2(r)$')
        ax.plot(r_fine[::4], r_clip[::4], 'o', color=PAL[1], markersize=2.5,
                markeredgewidth=0, alpha=0.8, label='QSP recovered')
        cost_val = float(data1['k2_cost'])
        ax.text(0.97, 0.05, f'cost $= {cost_val:.0e}$',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C['gray'], alpha=0.8))
    ax.set_xlabel('$r$')
    ax.set_ylabel('$f(r)$')
    ax.set_ylim(-14, 14)
    ax.axhline(0, color=C['gray'], linewidth=0.3)
    ax.legend(loc='upper left', frameon=False, fontsize=7)

    # (b) Higher-degree base cases k=3, 5, 7
    ax = axes[1]
    _panel_label(ax, '(b)')
    if data1 is not None:
        for k, col in [(3, PAL[0]), (5, PAL[2]), (7, PAL[3])]:
            key = f'k{k}_phis'
            if key not in data1:
                continue
            phis = data1[key]
            target = cot_base_function(k, r_fine)
            t_clip = np.where(np.abs(target) < 14, target, np.nan)
            ax.plot(r_fine, t_clip, '-', color=col, linewidth=1.0, alpha=0.35)
            f_found = decoded_function(phis, r_fine).real
            f_clip = np.where(np.abs(f_found) < 14, f_found, np.nan)
            ax.plot(r_fine[::5], f_clip[::5], '.', color=col, markersize=2,
                    alpha=0.85, label=f'$k={k}$')
    ax.set_xlabel('$r$')
    ax.set_ylabel('$z_k(r)$')
    ax.set_ylim(-14, 14)
    ax.axhline(0, color=C['gray'], linewidth=0.3)
    ax.legend(loc='upper left', frameon=False, fontsize=7,
              title='solid: target, dots: recovered', title_fontsize=6)

    # (c) Bounded |P| -> unbounded P/Q (the key insight)
    ax = axes[2]
    _panel_label(ax, '(c)')
    if data1 is not None and 'k5_phis' in data1:
        phis = data1['k5_phis']
        r_demo = np.linspace(0.08, 12, 500)
        a_demo = r_to_a(r_demo)
        W = qsp_product(phis, a_demo)
        P_vals = np.abs(W[:, 0, 0])
        f_decoded = decoded_function(phis, r_demo).real
        f_clip = np.where(np.abs(f_decoded) < 18, f_decoded, np.nan)

        # Shaded unitarity region
        ax.fill_between(a_demo, 0, 1, alpha=0.08, color=PAL[0])
        ax.plot(a_demo, P_vals, '-', color=PAL[0], linewidth=1.3,
                label='$|P(\\tilde{r})| \\leq 1$')
        ax.axhline(1, color=C['gray'], linewidth=0.4, ls=':')

        ax2 = ax.twinx()
        ax2.plot(a_demo, f_clip, '-', color=PAL[1], linewidth=1.0,
                 label='decoded $P/Q$')
        ax2.set_ylabel('decoded $P/Q$', color=PAL[1], fontsize=9)
        ax2.tick_params(axis='y', labelcolor=PAL[1])
        ax2.set_ylim(-18, 18)

        # Arrow annotation showing divergence
        ax2.annotate('unbounded', xy=(0.96, 12), fontsize=7, color=PAL[1],
                     ha='right', style='italic')

        ax.set_xlabel('$\\tilde{r} = r/\\sqrt{1+r^2}$')
        ax.set_ylabel('$|P(\\tilde{r})|$', color=PAL[0], fontsize=9)
        ax.tick_params(axis='y', labelcolor=PAL[0])
        ax.set_ylim(0, 1.18)
        ax.set_xlim(a_demo[0], a_demo[-1])

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center left',
                  frameon=False, fontsize=7)
    else:
        ax.text(0.5, 0.5, 'Run simulations first', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color=C['gray'])

    fig.tight_layout(w_pad=1.5)
    _save(fig, 'phase_finding')
    print('    saved: phase_finding.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Chebyshev convergence: rational vs standard
# ═══════════════════════════════════════════════════════════════════════════

def figure_3():
    data = load('sim4_convergence.npz')
    data3 = load('sim3_various_targets.npz')

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.8))

    if data is not None:
        K_max = int(data['K_max'])
        ks = np.arange(K_max)

        # (a) Coefficient magnitudes for 1/(1+r^4)
        ax = axes[0]
        _panel_label(ax, '(a)')
        name = 'inv_1pr4'
        c_rat = np.abs(data[f'{name}_c_rat']) + 1e-18
        c_std = np.abs(data[f'{name}_c_std']) + 1e-18
        ax.semilogy(ks, c_rat, 'o-', color=PAL[0], markersize=3, linewidth=0.8,
                    markeredgewidth=0, label='Rational Chebyshev')
        ax.semilogy(ks, c_std, 's-', color=PAL[1], markersize=3, linewidth=0.8,
                    markeredgewidth=0, label='Standard Chebyshev')
        ax.set_xlabel('Expansion index $k$')
        ax.set_ylabel('$|c_k|$')
        ax.set_ylim(1e-17, 3)
        ax.set_xlim(-0.5, K_max - 0.5)
        ax.legend(frameon=False, fontsize=7)

        # (b) Truncation error — multiple targets
        ax = axes[1]
        _panel_label(ax, '(b)')
        Ks = np.arange(1, K_max + 1)

        # Primary target
        ax.semilogy(Ks, data[f'{name}_err_rat'], 'o-', color=PAL[0],
                    markersize=3, linewidth=0.8, markeredgewidth=0,
                    label='$1/(1+r^4)$ rational')
        ax.semilogy(Ks, data[f'{name}_err_std'], 's-', color=PAL[1],
                    markersize=3, linewidth=0.8, markeredgewidth=0,
                    label='$1/(1+r^4)$ standard')

        # Additional targets (lighter)
        extras = [
            ('sech', '$\\mathrm{sech}(r)$', PAL[2], '^'),
            ('inv_1pr6', '$1/(1+r^6)$', PAL[3], 'v'),
            ('gaussian', '$e^{-r^2}$', PAL[4], 'D'),
        ]
        for ename, elabel, ecol, emk in extras:
            key = f'{ename}_err_rat'
            if key in data:
                ax.semilogy(Ks, data[key], f'{emk}-', color=ecol,
                            markersize=2.5, linewidth=0.6, markeredgewidth=0,
                            alpha=0.7, label=f'{elabel} rational')

        ax.set_xlabel('Truncation degree $K$')
        ax.set_ylabel('$\\|f - f_K\\|_\\infty$')
        ax.set_ylim(1e-17, 3)
        ax.set_xlim(0.5, K_max + 0.5)
        ax.legend(frameon=False, fontsize=6.5, ncol=2, loc='upper right',
                  columnspacing=0.8)
    else:
        for ax in axes[:2]:
            ax.text(0.5, 0.5, 'Run simulations first', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color=C['gray'])

    # (c) QSP phase-finding cost vs degree for various targets
    ax = axes[2]
    _panel_label(ax, '(c)')
    if data3 is not None:
        # Select targets that show a clear range of behaviors
        qsp_targets = [
            ('lorentzian',    '$1/(1{+}r^2)$',        PAL[0], 'o'),
            ('gaussian',      '$e^{-r^2}$',            PAL[1], 's'),
            ('inv_sqrt_1pr2', '$1/\\sqrt{1{+}r^2}$',  PAL[2], '^'),
            ('step_at_2',     'step at $r{=}2$',       PAL[3], 'v'),
            ('sign_r',        'sign$(r)$',             PAL[4], 'D'),
        ]
        for tname, tlabel, tcol, tmk in qsp_targets:
            dlist_key = f'{tname}_d_list'
            if dlist_key not in data3:
                continue
            d_list = data3[dlist_key]
            costs = [float(data3[f'{tname}_d{d}_cost']) for d in d_list]
            ax.semilogy(d_list, costs, f'{tmk}-', color=tcol,
                        markersize=3.5, linewidth=0.8, markeredgewidth=0,
                        label=tlabel)
        ax.set_xlabel('QSP degree $d$')
        ax.set_ylabel('Phase-finding cost')
        ax.legend(frameon=False, fontsize=6.5, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Run simulations first', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color=C['gray'])

    fig.tight_layout(w_pad=1.5)
    _save(fig, 'convergence')
    print('    saved: convergence.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Eigenvalue inversion example
# ═══════════════════════════════════════════════════════════════════════════

def figure_4():
    data = load('sim2_inversion.npz')

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5))

    if data is not None:
        r_fine = data['r_fine']
        a_fine = data['a_fine']
        d_show = [d for d in [2, 4, 8, 14] if f'd{d}_phis' in data]

        # (a) Internal |P(a)| bounded by 1
        ax = axes[0]
        _panel_label(ax, '(a)')
        a_plot = np.linspace(-0.999, 0.999, 500)
        ax.fill_between(a_plot, 0, 1, alpha=0.06, color=PAL[0])
        ax.plot(a_plot, np.sqrt(1 - a_plot**2), color=C['black'], ls='--',
                linewidth=0.8, alpha=0.5, label='ideal $\\sqrt{1-a^2}$')
        for i, d in enumerate(d_show):
            P = data[f'd{d}_P_vals']
            ax.plot(a_fine, np.abs(P), color=PAL[i], linewidth=1.0,
                    label=f'$d={d}$')
        ax.axhline(1, color=C['gray'], linewidth=0.4, ls=':')
        ax.set_xlabel('$a = r/\\sqrt{1+r^2}$')
        ax.set_ylabel('$|P(a)|$')
        ax.set_ylim(0, 1.15)
        ax.legend(frameon=False, fontsize=6.5, ncol=2)

        # (b) Decoded f(r) ≈ 1/r
        ax = axes[1]
        _panel_label(ax, '(b)')
        ax.plot(r_fine, 1.0 / r_fine, color=C['black'], ls='--', linewidth=0.8,
                alpha=0.5, label='$1/r$')
        for i, d in enumerate(d_show):
            f = data[f'd{d}_f_found']
            ax.plot(r_fine, f, color=PAL[i], linewidth=1.0, label=f'$d={d}$')
        ax.set_xlabel('$r$')
        ax.set_ylabel('$f(r)$')
        ax.set_ylim(-0.5, 5)
        ax.set_xlim(0, 15)
        ax.legend(frameon=False, fontsize=6.5, ncol=2)

        # (c) Pointwise error + inset convergence
        ax = axes[2]
        _panel_label(ax, '(c)')
        for i, d in enumerate(d_show):
            err = data[f'd{d}_abs_err']
            ax.semilogy(r_fine, err + 1e-16, color=PAL[i], linewidth=1.0,
                        label=f'$d={d}$')
        ax.set_xlabel('$r$')
        ax.set_ylabel('$|f(r) - 1/r|$')
        ax.set_xlim(0, 15)
        ax.set_ylim(1e-4, 50)
        ax.legend(frameon=False, fontsize=6.5)

        # Convergence inset: L2 error vs degree
        d_values = data['d_values']
        l2_errs = []
        d_plot = []
        for d in d_values:
            key = f'd{d}_l2_err'
            if key in data:
                l2_errs.append(float(data[key]))
                d_plot.append(d)
        if l2_errs:
            ax_in = ax.inset_axes([0.42, 0.52, 0.53, 0.43])
            ax_in.semilogy(d_plot, l2_errs, 'o-', color=C['black'],
                           markersize=3, linewidth=0.8, markeredgewidth=0)
            ax_in.set_xlabel('degree $d$', fontsize=6.5)
            ax_in.set_ylabel('$L^2$ error', fontsize=6.5)
            ax_in.tick_params(labelsize=6)
            ax_in.set_ylim(min(l2_errs) * 0.5, max(l2_errs) * 2)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'Run simulations first', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color=C['gray'])

    fig.tight_layout(w_pad=1.8)
    _save(fig, 'inversion_example')
    print('    saved: inversion_example.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Error analysis: amplification and measurement cost
# ═══════════════════════════════════════════════════════════════════════════

def figure_5():
    data = load('sim5_error_analysis.npz')

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.8))

    if data is not None:
        r_cont = data['r_cont']

        # (a) Error amplification with MC validation
        ax = axes[0]
        _panel_label(ax, '(a)')
        ax.semilogy(r_cont, data['amp_abs'], '-', color=PAL[0], linewidth=1.5,
                    label='Absolute: $(1{+}r^2)(1{+}r)/2$')
        ax.semilogy(r_cont, data['amp_rel'], '--', color=PAL[1], linewidth=1.5,
                    label='Relative: $(1{+}r^2)(1{+}r)/(2r)$')

        # MC validation overlay (delta=0.001 is where Taylor bound is tight)
        r_vals = data['r_values']
        for delta, mk, lab, col in [
            (0.001, 'v', '$\\delta=10^{-3}$ (99th)', PAL[2]),
            (0.01,  '^', '$\\delta=10^{-2}$ (99th)', PAL[3]),
        ]:
            key99 = f'delta{delta}_99th_err'
            if key99 in data:
                mc_99 = data[key99]
                ax.semilogy(r_vals, mc_99 / delta, mk, color=col,
                            markersize=4.5, markeredgewidth=0.4,
                            markeredgecolor='white', label=f'MC {lab}')

        ax.set_xlabel('$r$ (signal magnitude)')
        ax.set_ylabel('Error amplification factor')
        ax.set_xlim(0, 22)
        ax.set_ylim(0.5, 1e6)
        ax.legend(frameon=False, fontsize=6.5, loc='upper left')

        # (b) Measurement shots
        ax = axes[1]
        _panel_label(ax, '(b)')
        styles = [('-', 1.5), ('--', 1.5), ('-.', 1.5)]
        for (eps, col), (ls, lw) in zip(
            [(0.1, PAL[0]), (0.01, PAL[1]), (0.001, PAL[2])], styles
        ):
            shots = data[f'shots_eps{eps}']
            ax.semilogy(r_cont, shots, ls, color=col, linewidth=lw,
                        label=f'$\\epsilon = {eps}$')
        ax.set_xlabel('$r$ (signal magnitude)')
        ax.set_ylabel('Measurement shots $N$')
        ax.set_xlim(0, 22)
        ax.legend(frameon=False, fontsize=7, loc='lower right')

        # Annotate scaling
        ax.text(18, 1e14, '$\\sim r^6/\\epsilon^2$', fontsize=8,
                color=C['gray'], ha='center', style='italic')
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'Run simulations first', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color=C['gray'])

    fig.tight_layout(w_pad=2.0)
    _save(fig, 'error_analysis')
    print('    saved: error_analysis.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — Worked example pipeline (not in main text but available)
# ═══════════════════════════════════════════════════════════════════════════

def figure_6():
    data1 = load('sim1_base_cases.npz')
    data2 = load('sim2_inversion.npz')

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5))
    r = np.linspace(0.01, 10, 500)

    # (a) Pipeline: r -> r_tilde -> T_3 -> cot_3
    ax = axes[0]
    _panel_label(ax, '(a)')
    r_tilde = r / np.sqrt(1 + r**2)
    T3 = np.cos(3 * np.arccos(r_tilde))
    cot3 = cot_base_function(3, r)
    cot3_clip = np.where(np.abs(cot3) < 6, cot3, np.nan)

    ax.plot(r, r_tilde, '-', color=PAL[0], linewidth=1.2,
            label='$\\tilde{r} = r/\\sqrt{1{+}r^2}$')
    ax.plot(r, T3, '--', color=PAL[1], linewidth=1.0,
            label='$T_3(\\tilde{r})$')
    ax.plot(r, cot3_clip, '-', color=PAL[2], linewidth=1.2,
            label='$\\cot(3\\arctan(1/r))$')
    ax.axhline(1, color=C['gray'], linewidth=0.3, ls=':')
    ax.axhline(-1, color=C['gray'], linewidth=0.3, ls=':')
    ax.axhline(0, color=C['gray'], linewidth=0.3)
    ax.set_xlabel('$r$')
    ax.set_ylabel('value')
    ax.set_ylim(-6, 5)
    ax.legend(frameon=False, fontsize=6.5, loc='lower right')

    # (b) Random decoded functions
    ax = axes[1]
    _panel_label(ax, '(b)')
    np.random.seed(42)
    for i in range(5):
        phis = np.random.uniform(-np.pi, np.pi, 5)
        f = decoded_function(phis, r).real
        f_clip = np.where(np.abs(f) < 8, f, np.nan)
        ax.plot(r, f_clip, color=PAL[i], linewidth=0.9, alpha=0.85)
    ax.set_xlabel('$r$')
    ax.set_ylabel('$f(r) = P/Q$')
    ax.set_ylim(-8, 8)
    ax.axhline(0, color=C['gray'], linewidth=0.3)
    ax.text(0.95, 0.95, 'unbounded\nrational\nfunctions', transform=ax.transAxes,
            fontsize=6.5, ha='right', va='top', color=C['gray'], style='italic')

    # (c) 1/r approximation
    ax = axes[2]
    _panel_label(ax, '(c)')
    if data2 is not None:
        r_fine = data2['r_fine']
        ax.plot(r_fine, 1.0 / r_fine, color=C['black'], ls='--', linewidth=0.8,
                alpha=0.5, label='$1/r$')
        for d, col in [(3, PAL[0]), (5, PAL[1]), (9, PAL[2])]:
            key = f'd{d}_f_found'
            if key in data2:
                ax.plot(r_fine, data2[key], color=col, linewidth=1.0,
                        label=f'$d={d}$')
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 4)
    ax.set_xlabel('$r$')
    ax.set_ylabel('$f(r)$')
    ax.legend(frameon=False, fontsize=6.5)

    fig.tight_layout(w_pad=1.5)
    _save(fig, 'worked_example')
    print('    saved: worked_example.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

ALL_FIGS = {
    1: ('Rational Chebyshev basis', figure_1),
    2: ('Phase-finding demonstration', figure_2),
    3: ('Convergence comparison', figure_3),
    4: ('Inversion example', figure_4),
    5: ('Error analysis', figure_5),
    6: ('Worked example pipeline', figure_6),
}

def main():
    parser = argparse.ArgumentParser(description='Generate figures from simulation data')
    parser.add_argument('--fig', type=int, nargs='*', default=None,
                        help='Which figures to generate (1-6). Default: all.')
    args = parser.parse_args()
    figs_to_gen = args.fig if args.fig else list(ALL_FIGS.keys())

    print('Generating figures from simulation data...')
    for fig_id in figs_to_gen:
        if fig_id not in ALL_FIGS:
            print(f'  WARNING: Unknown figure {fig_id}')
            continue
        name, func = ALL_FIGS[fig_id]
        print(f'  Fig {fig_id}: {name}')
        try:
            func()
        except Exception as e:
            print(f'    ERROR: {e}')
            import traceback; traceback.print_exc()
    print('Done.')

if __name__ == '__main__':
    main()
