"""Degree law at the best L: min over L of the L2 error of the degree-d truncation of G.

Prints, for each d, the best L and error, then fits log(1/err) ~ c d^s and L* ~ d^t,
and the cost proxy C = d/F against log(1/err).
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np

M = 1 << 14
th = 2 * np.pi * (np.arange(M) + 0.5) / M
half = slice(0, M // 2)


def err_F(f, L, ds):
    with np.errstate(all='ignore'):
        G = np.nan_to_num(f(L / np.tan(th)) / np.sin(th))
    F = np.sqrt(np.mean(G[half]**2)) / np.abs(G[half]).max()
    p = (np.abs(np.fft.rfft(G)) / M)**2
    p[1:] *= 2
    tail = np.sqrt(np.cumsum(p[::-1])[::-1] / p.sum())      # tail[k] = error keeping < k
    return np.array([tail[d + 1] for d in ds]), F


for name, f, Ls in [('sech', lambda y: 1 / np.cosh(y), np.geomspace(0.5, 20, 160)),
                    ('gauss', lambda y: np.exp(-y**2 / 2), np.geomspace(0.5, 20, 160))]:
    ds = np.arange(5, 201, 2)                               # odd harmonics
    E = np.empty((len(Ls), len(ds)))
    Fs = np.empty(len(Ls))
    for i, L in enumerate(Ls):
        E[i], Fs[i] = err_F(f, L, ds)
    ib = E.argmin(axis=0)
    eb = E[ib, np.arange(len(ds))]
    ok = eb > 1e-12
    d, e, Lb, Fb = ds[ok], eb[ok], Ls[ib[ok]], Fs[ib[ok]]
    print(f"== {name}: d range {d[0]}..{d[-1]}, err {e[0]:.1e}..{e[-1]:.1e}")
    for j in range(0, len(d), max(1, len(d) // 8)):
        print(f"   d={d[j]:4d} L*={Lb[j]:6.2f} F={Fb[j]:.3f} err={e[j]:.2e}")
    lo = len(d) // 3                                        # fit on the upper two thirds
    s = np.polyfit(np.log(d[lo:]), np.log(np.log(1 / e[lo:])), 1)[0]
    t = np.polyfit(np.log(d[lo:]), np.log(Lb[lo:]), 1)[0]
    c = np.polyfit(np.log(np.log(1 / e[lo:])), np.log(d[lo:] / Fb[lo:]), 1)[0]
    print(f"   log(1/err) ~ d^{s:.2f};  L* ~ d^{t:.2f};  d/F ~ log(1/err)^{c:.2f}  (error-optimal L, not cost-optimal)")
