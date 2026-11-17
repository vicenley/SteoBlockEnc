#!/usr/bin/env python3
"""Figure: cost d/F against 1/eps at equal L2(R) error, cotangent grid versus QSVT pipeline.

Writes data/like_for_like.npz and ms-10p/figures/like_for_like.pdf.
Pass --plot-only to redraw from the saved data.
"""
import os
import sys

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, HERE)
from like4like import ours, qsvt as qsvt_lower  # noqa: E402
from qsvt_bounded import qsvt_upper  # noqa: E402

DATA = os.path.join(ROOT, 'data', 'like_for_like.npz')
FIG = os.path.join(ROOT, 'ms-10p', 'figures', 'like_for_like.pdf')


def best(gen):
    vals = [max(d, 1) / F for d, F in gen if d is not None]
    return min(vals) if vals else np.nan


def compute():
    out = {}
    eps = np.logspace(-1.5, -6, 10)
    out['eps'] = eps
    for name, f, amax in [('sech', lambda y: 1 / np.cosh(y), 60),
                          ('gauss', lambda y: np.exp(-y**2 / 2), 30)]:
        norm2 = 2 * integrate.quad(lambda y: f(y)**2, 0, np.inf)[0]
        fixed, tuned, lo, up = [], [], [], []
        for e in eps:
            fixed.append(best([ours(f, 1.0, e)]))
            tuned.append(best(ours(f, L, e) for L in np.geomspace(0.5, 12, 60)))
            lo.append(best(qsvt_lower(f, a, e, norm2) for a in np.geomspace(1.5, amax, 50)))
            ups = [qsvt_upper(f, a, e, norm2) for a in np.geomspace(1.5, amax, 50)]
            up.append(min(d / F for d, F in (u for u in ups if u)))
        out.update({f'{name}_fixed': fixed, f'{name}_tuned': tuned,
                    f'{name}_lo': lo, f'{name}_up': up})
    for m, epsm in [(1, np.logspace(-0.7, -2, 6)), (2, np.logspace(-1, -4, 7))]:
        f = (lambda y, m=m: (1 + y**2)**(-m / 2))
        norm2 = 2 * integrate.quad(lambda y: f(y)**2, 0, np.inf)[0]
        out[f'alg{m}_eps'] = epsm
        out[f'alg{m}_lo'] = [best(qsvt_lower(f, a, e, norm2) for a in np.geomspace(1.5, 3e5, 70))
                             for e in epsm]
    np.savez(DATA, **{k: np.asarray(v, float) for k, v in out.items()})


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
        'font.family': 'serif', 'font.size': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
        'xtick.direction': 'in', 'ytick.direction': 'in',
    })
    # Blue is the cotangent grid and orange the uniform grid (QSVT) in every panel.
    BLUE, ORANGE, DARKORANGE, GREY = '#0072B2', '#E69F00', '#B87A00', '#666666'
    D = np.load(DATA)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)
    ax[1].sharey(ax[0])
    for a, name, title in [(ax[0], 'sech', r'(a) $\mathrm{sech}\,y$'),
                           (ax[1], 'gauss', r'(b) $e^{-y^2/2}$')]:
        x = 1 / D['eps']
        a.fill_between(x, D[f'{name}_lo'], D[f'{name}_up'], color=ORANGE, alpha=0.4, lw=0,
                       label='uniform grid (QSVT)')
        a.plot(x, D[f'{name}_fixed'], '--', color=BLUE, label=r'cotangent grid, $L=1$')
        a.plot(x, D[f'{name}_tuned'], '-o', ms=2.5, color=BLUE,
               label=r'cotangent grid, best $L$')
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel(r'$1/\varepsilon$')
        a.set_title(title, fontsize=9)
    ax[0].set_ylabel(r'cost $d/\mathcal{F}$')
    ax[0].legend(frameon=False, fontsize=7, loc='upper left', handlelength=1.8,
                 borderaxespad=0.3)
    ax[1].tick_params(labelleft=False)
    a = ax[2]
    for m, mk, slope in [(1, 'o', 3), (2, 's', 1)]:
        e = D[f'alg{m}_eps']
        ref = D[f'alg{m}_lo'][-1] * (e[-1] / e)**slope
        a.plot(1 / e, ref, ':', color=GREY, lw=0.8)
        a.plot(1 / e, D[f'alg{m}_lo'], '-', marker=mk, ms=3, color=ORANGE, mfc='white',
               mew=0.9)
    for c in (1.0, np.sqrt(2)):
        a.axhline(c, color=BLUE)
    a.text(140, 4e5, r'$(1+y^2)^{-1/2}$', fontsize=7, ha='left', va='center')
    a.text(1500, 500, r'$(1+y^2)^{-1}$', fontsize=7, ha='left', va='center')
    a.text(20, 1.2e5, r'$\propto\varepsilon^{-3}$', color=GREY, fontsize=7, ha='right',
           va='center')
    a.text(400, 1.6e4, r'$\propto\varepsilon^{-1}$', color=GREY, fontsize=7, ha='right',
           va='center')
    a.text(0.97, 0.27, 'uniform grid (QSVT)', color=DARKORANGE, fontsize=7, ha='right',
           transform=a.transAxes)
    a.text(0.97, 0.12, r'cotangent grid: $1$ and $\sqrt{2}$', color=BLUE, fontsize=7,
           ha='right', transform=a.transAxes)
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_ylim(bottom=0.4)
    a.set_xlabel(r'$1/\varepsilon$')
    a.set_title('(c) algebraic tails', fontsize=9)
    fig.savefig(FIG, dpi=300)
    print('wrote', os.path.relpath(FIG, ROOT))


if __name__ == '__main__':
    if '--plot-only' not in sys.argv:
        compute()
    plot()
