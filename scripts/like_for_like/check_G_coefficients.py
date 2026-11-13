"""Check: coefficient of e^{i(2j+1)theta} in G = g/sin equals 2i * sum_{m>j} a_m (j >= 0),
where g = sum_m a_m e^{2 i m theta} and g(0) = 0."""
import numpy as np

M = 1 << 14
th = 2 * np.pi * (np.arange(M) + 0.5) / M
for name, f in [('sech', lambda y: 1 / np.cosh(y)), ('gauss', lambda y: np.exp(-y**2 / 2)),
                ('lorentz', lambda y: 1 / (1 + y**2))]:
    with np.errstate(all='ignore'):
        g = np.nan_to_num(f(1 / np.tan(th)))
        G = np.nan_to_num(g / np.sin(th))
    shift = np.exp(-1j * np.arange(M) * np.pi / M)          # midpoint-grid phase
    cg = np.fft.fft(g) / M * shift                          # coefficient of e^{i k theta}
    cG = np.fft.fft(G) / M * shift
    a = lambda m: cg[(2 * m) % M]
    worst = 0
    for j in range(0, 40):
        tail = sum(a(m) for m in range(j + 1, M // 8))
        worst = max(worst, abs(cG[2 * j + 1] - 2j * tail))
    print(f"{name:8s} max |c_(2j+1) - 2i sum_(m>j) a_m| over j<40: {worst:.1e}")
