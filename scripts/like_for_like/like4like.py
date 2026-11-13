"""Like-for-like cost proxies C = d / F (gates per qubit, up to constants).

Common metric: L2(R) distance between the normalized continuum target and the
function represented by the prepared state.
  ours : weighted cotangent state, G(theta) = f(L cot theta)/sin(theta), free L
  qsvt : uniform grid on [-a, a], polynomial in s = sin(y/a) (McArdle et al.)
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np
from scipy.fft import dct
from scipy import integrate

TARGETS = {
    'lorentz p=2': (lambda y: 1 / (1 + y**2), 'alg'),
    'cauchy-amp p=1': (lambda y: 1 / np.sqrt(1 + y**2), 'alg'),
    'sech': (lambda y: 1 / np.cosh(y), 'exp'),
    'gauss': (lambda y: np.exp(-y**2 / 2), 'exp'),
}


def ours(f, L, eps, M=1 << 15):
    th = 2 * np.pi * (np.arange(M) + 0.5) / M
    with np.errstate(all='ignore'):
        G = f(L / np.tan(th)) / np.sin(th)          # antiperiodic extension
    G = np.nan_to_num(G)
    half = G[: M // 2]
    F = np.sqrt(np.mean(half**2)) / np.abs(half).max()
    best = None
    for ext in (G, G * np.sign(np.sin(th))):        # odd- or even-harmonic extension
        c = np.abs(np.fft.rfft(ext)) / M
        p = c**2
        p[1:] *= 2
        tail = np.sqrt(np.maximum(np.cumsum(p[::-1])[::-1], 0) / p.sum())  # tail[d] = err keeping k<d
        idx = np.argmax(tail <= eps) if np.any(tail <= eps) else None
        if idx is not None:
            d = max(idx - 1, 0)
            best = d if best is None else min(best, d)
    return best, F


def qsvt(f, a, eps, norm2, M=1 << 18):
    # tail fraction outside [-a, a]
    tail2 = 2 * integrate.quad(lambda y: f(y)**2, a, np.inf, limit=200)[0] / norm2
    if tail2 >= eps**2:
        return None, None
    ep = np.sqrt(eps**2 - tail2)
    u = np.cos(np.pi * (np.arange(M) + 0.5) / M)
    H = f(a * np.arcsin(u * np.sin(1.0)))           # h(s) on s in [-sin 1, sin 1]
    c = dct(H, type=2) / M
    c[0] /= 2
    p = c**2
    p[0] *= 2
    tail = np.sqrt(np.maximum(np.cumsum(p[::-1])[::-1], 0) / p.sum())
    if not np.any(tail <= ep):
        return None, None
    d = max(int(np.argmax(tail <= ep)) - 1, 0)
    t = np.linspace(-1, 1, 20001)
    F = np.sqrt(np.mean(f(a * t)**2)) / np.abs(f(a * t)).max()
    return d, F


if __name__ == '__main__':
    for name, (f, kind) in TARGETS.items():
        norm2 = 2 * integrate.quad(lambda y: f(y)**2, 0, np.inf)[0]
        epss = [1e-1, 3e-2, 1e-2, 3e-3] if name.startswith('cauchy') else \
               [1e-2, 1e-3, 1e-4] if kind == 'alg' else [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        print(f"\n== {name}")
        print(f"{'eps':>8} | {'L*':>6} {'d':>6} {'F':>6} {'C=d/F':>9} | {'a*':>9} {'d':>7} {'F':>6} {'C=d/F':>10}")
        for eps in epss:
            bo = None
            for L in np.geomspace(0.5, 12, 60):
                d, F = ours(f, L, eps)
                if d is None:
                    continue
                C = max(d, 1) / F
                if bo is None or C < bo[0]:
                    bo = (C, L, d, F)
            bq = None
            amax = 3e5 if kind == 'alg' else 60
            for a in np.geomspace(1.5, amax, 70):
                d, F = qsvt(f, a, eps, norm2)
                if d is None:
                    continue
                C = max(d, 1) / F
                if bq is None or C < bq[0]:
                    bq = (C, a, d, F)
            so = f"{bo[1]:6.2f} {bo[2]:6d} {bo[3]:6.3f} {bo[0]:9.1f}" if bo else "   -"
            sq = f"{bq[1]:9.1f} {bq[2]:7d} {bq[3]:6.3f} {bq[0]:10.1f}" if bq else "   - (out of range)"
            print(f"{eps:8.0e} | {so} | {sq}")
