#!/usr/bin/env python3
"""
Generate the Heisenberg end-to-end demo figure from saved data.
Reads data/sim7_heisenberg_demo.npz, writes ms/figures/heisenberg_demo.pdf.

Publication-quality 3-panel figure:
  (a) Eigenvalue spectrum (original vs shifted)
  (b) Decoded f(r)=1/r vs exact, with eigenvalue markers
  (c) Error at eigenvalues vs QSP degree, with predicted O(r^6) scaling
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
    'legend.fontsize': 7,
    'legend.handlelength': 1.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'lines.linewidth': 1.0,
})

# Okabe-Ito palette
OI = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00', 'purple': '#CC79A7', 'cyan': '#56B4E9',
    'yellow': '#F0E442', 'black': '#000000',
}

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'ms', 'figures')


def main():
    datapath = os.path.join(DATADIR, 'sim7_heisenberg_demo.npz')
    if not os.path.exists(datapath):
        print(f'ERROR: {datapath} not found. Run heisenberg_demo_pub.py first.')
        sys.exit(1)

    data = np.load(datapath, allow_pickle=True)

    eigenvalues = data['eigenvalues']
    eig_shifted = data['eigenvalues_shifted']
    shift = float(data['shift'])
    degrees = data['degrees']
    r_fine = data['r_fine']
    f_exact_fine = data['f_exact_fine']

    colors_eig = [OI['blue'], OI['orange'], OI['green'], OI['red']]
    n_eig = len(eigenvalues)

    # Select up to 4 degrees for the continuous plot (avoid clutter)
    if len(degrees) > 4:
        plot_degrees = [degrees[0], degrees[len(degrees)//3],
                        degrees[2*len(degrees)//3], degrees[-1]]
    else:
        plot_degrees = list(degrees)

    deg_colors_list = [OI['blue'], OI['orange'], OI['green'], OI['red'],
                       OI['purple'], OI['cyan']]
    deg_colors = {d: deg_colors_list[i % len(deg_colors_list)]
                  for i, d in enumerate(plot_degrees)}

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    # Panel (a): Spectrum
    ax = axes[0]
    for j in range(n_eig):
        ax.plot([0, 1], [eigenvalues[j], eig_shifted[j]],
                'o-', color=colors_eig[j], ms=4.5, lw=1.0,
                label=f'$\\lambda_{j+1}={eigenvalues[j]:.1f}$')
    ax.axhline(0, color='gray', ls='--', lw=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Original', 'Shifted'], fontsize=7.5)
    ax.set_ylabel('Eigenvalue')
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9)
    ax.text(0.03, 0.97, r'$\mathbf{(a)}$', transform=ax.transAxes,
            fontsize=10, va='top', ha='left')

    # Panel (b): Decoded function
    ax = axes[1]
    ax.plot(r_fine, f_exact_fine, 'k--', lw=1.3, label='$1/r$ (exact)',
            zorder=10)
    for d in plot_degrees:
        key = f'd{d}_f_fine'
        if key in data:
            ax.plot(r_fine, data[key], color=deg_colors[d],
                    lw=1.0, label=f'$d={d}$', alpha=0.85)
    # Eigenvalue markers
    for j in range(n_eig):
        ax.axvline(eig_shifted[j], color=colors_eig[j], ls=':', lw=0.5,
                   alpha=0.4)
    ax.set_xlabel('$r$ (shifted eigenvalue)')
    ax.set_ylabel('$f(r)$')
    ymax = min(6, 1.2 / min(eig_shifted) + 0.5)
    ax.set_ylim(-0.3, ymax)
    ax.legend(fontsize=6, ncol=2)
    ax.text(0.03, 0.97, r'$\mathbf{(b)}$', transform=ax.transAxes,
            fontsize=10, va='top', ha='left')

    # Panel (c): Error at eigenvalues vs degree
    ax = axes[2]
    all_degrees = sorted(degrees)

    for j in range(n_eig):
        errs = []
        degs_plot = []
        for d in all_degrees:
            key = f'd{d}_errs_eig'
            if key in data:
                err_arr = data[key]
                if j < len(err_arr) and err_arr[j] > 0:
                    errs.append(err_arr[j])
                    degs_plot.append(d)
        if errs:
            ax.semilogy(degs_plot, errs, 'o-', color=colors_eig[j],
                        ms=4, lw=0.9,
                        label=f"$\\lambda'={eig_shifted[j]:.1f}$")

    ax.set_xlabel('QSP degree $d$')
    ax.set_ylabel('$|f_{\\mathrm{stereo}} - 1/\\lambda|$')
    ax.legend(fontsize=6, loc='upper right')
    ax.text(0.03, 0.97, r'$\mathbf{(c)}$', transform=ax.transAxes,
            fontsize=10, va='top', ha='left')

    plt.tight_layout(w_pad=1.0)

    os.makedirs(FIGDIR, exist_ok=True)
    outpath = os.path.join(FIGDIR, 'heisenberg_demo.pdf')
    fig.savefig(outpath)
    plt.close()
    print(f'Figure saved to {outpath}')


if __name__ == '__main__':
    main()
