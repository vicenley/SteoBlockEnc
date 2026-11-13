"""Compare Fourier decay of g = f(L cot) (even harmonics) and G = g/sin (odd harmonics)."""
import numpy as np
M = 1 << 16
th = 2 * np.pi * (np.arange(M) + 0.5) / M
for name, f, sig in [('sech', lambda y: 1 / np.cosh(y), 0.5), ('gauss', lambda y: np.exp(-y**2 / 2), 2 / 3)]:
    with np.errstate(all='ignore'):
        g = np.nan_to_num(f(1 / np.tan(th)))
        G = np.nan_to_num(g / np.sin(th))
    cg = np.abs(np.fft.rfft(g)) / M
    cG = np.abs(np.fft.rfft(G)) / M
    for lab, c, ks in [('g', cg, np.arange(20, 400, 2)), ('G', cG, np.arange(21, 400, 2))]:
        v = c[ks]
        ok = v > 1e-13
        # envelope fit: log|c_k| = A - q k^sig  (use local maxima to skip oscillation zeros)
        k, lv = ks[ok], np.log(v[ok])
        env = [i for i in range(1, len(k) - 1) if lv[i] >= lv[i - 1] and lv[i] >= lv[i + 1]] or list(range(len(k)))
        q = -np.polyfit(k[env]**sig, lv[env], 1)[0]
        # free exponent fit
        best = min(((np.polyfit(k[env]**s, lv[env], 1, full=True)[1][0], s) for s in np.linspace(0.3, 1, 71)))
        print(f"{name:5s} {lab}: q at sigma={sig:.3f}: {q:.3f}   best-fit sigma={best[1]:.2f}   (k up to {k[-1]})")
