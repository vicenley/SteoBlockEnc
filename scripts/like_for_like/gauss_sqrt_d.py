"""Gaussian on the cotangent grid with L = lam*sqrt(d): is the L2 error geometric in d?

High-precision (mpmath) sine coefficients of G = exp(-L^2 cot^2(theta)/2)/sin(theta),
error from Parseval.  Heuristic: balancing the endpoint decay exp(-(3/4)(L d)^(2/3))
against the band-limit decay exp(-d^2/(2 L^2)) gives lam = (2/3)^(3/8) = 0.859 and
rate 1/(2 lam^2) = 0.678 per degree (up to prefactors polynomial in d).
"""
import sys
from mpmath import mp, mpf, sin, tan, exp, pi, log, sqrt
mp.dps = 260
lams = [0.7, 0.8, 0.86, 0.95]
ds = [int(a) for a in sys.argv[1:]] or [41, 81, 161, 321]
for d in ds:
    M = 4 * d
    th = [pi * (2 * j + 1) / (2 * M) for j in range(M)]
    best = None
    for lam in lams:
        L = lam * sqrt(d)
        G = [exp(-(L / tan(t))**2 / 2) / sin(t) for t in th]
        nrm = sum(g * g for g in G) / M
        s = mpf(0)
        for k in range(1, d + 1, 2):
            b = 2 * sum(g * sin(k * t) for g, t in zip(G, th)) / M
            s += b * b / 2
        err = sqrt(abs(1 - s / nrm))
        F = sqrt(nrm) / max(G)
        r = float(-log(err) / d)
        print(f"d={d:4d} lam={lam:.2f} err={float(err):.2e} -log(err)/d={r:.3f} F*L^0.5={float(F*sqrt(L)):.3f}", flush=True)
        if best is None or r > best[0]:
            best = (r, lam)
    print(f"   best lam={best[1]}  rate={best[0]:.3f}")
