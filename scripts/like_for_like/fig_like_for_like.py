#!/usr/bin/env python3
"""Figure: L2(R) error against cost d/F, cotangent grid versus QSVT pipeline.

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
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'mathtext.fontset': 'cm',
                         'axes.linewidth': 0.6, 'lines.linewidth': 1.2})
    BLUE, ORANGE, GREEN, GREY = '#0072B2', '#E69F00', '#009E73', '#666666'
    D = np.load(DATA)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True)
    for a, name, title in [(ax[0], 'sech', r'(a) $\mathrm{sech}\,y$'),
                           (ax[1], 'gauss', r'(b) $e^{-y^2/2}$')]:
        e = D['eps']
        a.fill_betweenx(e, D[f'{name}_lo'], D[f'{name}_up'], color=ORANGE, alpha=0.35, lw=0,
                        label='QSVT pipeline')
        a.plot(D[f'{name}_fixed'], e, '--', color=BLUE, label=r'this work, $L=1$')
        a.plot(D[f'{name}_tuned'], e, '-', color=BLUE, label=r'this work, best $L$')
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel(r'cost $d/\mathcal{F}$')
        a.set_title(title, fontsize=9)
    ax[0].set_ylabel(r'$L^2(\mathbb{R})$ error $\varepsilon$')
    ax[0].set_xlim(left=0.5)
    ax[0].legend(frameon=False, fontsize=7, loc='lower left', handlelength=1.5)
    a = ax[2]
    for m, col, lab, ours_c, slope in [(1, GREEN, r'$(1+y^2)^{-1/2}$', 1.0, 3),
                                       (2, ORANGE, r'$(1+y^2)^{-1}$', np.sqrt(2), 1)]:
        e = D[f'alg{m}_eps']
        a.plot(1 / e, D[f'alg{m}_lo'], 'o-', ms=3, color=col, label='QSVT, ' + lab)
        a.axhline(ours_c, color=col, ls='--')
        ref = D[f'alg{m}_lo'][-1] * (e[-1] / e)**slope
        a.plot(1 / e, ref, ':', color=GREY, lw=0.8)
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel(r'$1/\varepsilon$')
    a.set_ylabel(r'cost $d/\mathcal{F}$')
    a.set_title('(c) algebraic tails', fontsize=9)
    a.legend(frameon=False, fontsize=7, loc=(0.36, 0.1))
    fig.savefig(FIG, dpi=300)
    print('wrote', os.path.relpath(FIG, ROOT))


if __name__ == '__main__':
    if '--plot-only' not in sys.argv:
        compute()
    plot()
