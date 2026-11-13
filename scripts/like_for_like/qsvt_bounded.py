"""QSVT pipeline cost with the |h| <= 1 on [-1,1] constraint enforced.

Upper bound (constructive): h(s) = f(a arcsin s) is bounded by max f on all of
[-1,1]; truncate its Chebyshev series on the FULL interval at degree d, rescale by
max|h_d| on [-1,1].  Error measured as L2(R) distance of normalized functions:
window part on y in [-a,a] (y = a arcsin s, s in [-sin1, sin1]) plus discarded tail.
Lower bound: the earlier proxy (fit on the window only, no outside constraint).
"""
import os, sys
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np
from numpy.polynomial import chebyshev as Ch
from scipy.fft import dct
from scipy import integrate
sys.path.insert(0, os.path.dirname(__file__))
from like4like import ours, qsvt as qsvt_lower

def qsvt_upper(f, a, eps, norm2, M=1 << 15, dmax=6000):
    tail2 = 2 * integrate.quad(lambda y: f(y)**2, a, np.inf, limit=200)[0]
    if tail2 / norm2 >= eps**2:
        return None
    s_nodes = np.cos(np.pi * (np.arange(M) + 0.5) / M)
    c = dct(f(a * np.arcsin(s_nodes)), type=2) / M
    c[0] /= 2
    t = (np.arange(4000) + 0.5) / 4000 * 2 - 1           # y/a on the window
    sw = np.sin(t)
    fw = f(a * t)
    sfull = np.cos(np.pi * (np.arange(4000) + 0.5) / 4000)
    def stats(d):
        cd = c[: d + 1]
        P = Ch.chebval(sw, cd)
        # normalized L2(R) distance: represented function is P on window, 0 outside
        nP2 = a * 2 * np.mean(P**2)
        ov = a * 2 * np.mean(P * fw)
        err2 = 2 - 2 * ov / np.sqrt(nP2 * norm2)
        F = np.sqrt(np.mean(P**2)) / np.abs(Ch.chebval(sfull, cd)).max()
        return err2, F
    lo, hi = 1, 2
    while stats(hi)[0] > eps**2:
        hi *= 2
        if hi > dmax:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if stats(mid)[0] <= eps**2:
            hi = mid
        else:
            lo = mid
    return hi, stats(hi)[1]

T = {'lorentz': (lambda y: 1 / (1 + y**2), [1e-2, 1e-3], 3e3),
     'sech': (lambda y: 1 / np.cosh(y), [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 60),
     'gauss': (lambda y: np.exp(-y**2 / 2), [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 30)}
for name, (f, epss, amax) in T.items():
    norm2 = 2 * integrate.quad(lambda y: f(y)**2, 0, np.inf)[0]
    print(f"== {name}:  eps | ours C | QSVT lower C | QSVT upper C (a*, d, F)")
    for eps in epss:
        bo = min((max(d, 1) / F for d, F in (ours(f, L, eps) for L in np.geomspace(0.5, 12, 60)) if d is not None))
        bl = min((max(d, 1) / F for d, F in (qsvt_lower(f, a, eps, norm2) for a in np.geomspace(1.5, amax, 50)) if d is not None))
        bu = None
        for a in np.geomspace(1.5, amax, 50):
            r = qsvt_upper(f, a, eps, norm2)
            if r and (bu is None or r[0] / r[1] < bu[0]):
                bu = (r[0] / r[1], a, r[0], r[1])
        print(f"  {eps:.0e} | {bo:7.1f} | {bl:9.1f} | {bu[0]:9.1f}  ({bu[1]:.1f}, {bu[2]}, {bu[3]:.3f})")
