"""Checks for the weighted-state theorem."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np
from scipy import integrate

rng = np.random.default_rng(0)

def grid(N):
    return np.pi * (2 * np.arange(N) + 1) / (2 * N)

def poly(c, ks, th):
    return sum(ck * np.exp(1j * k * th) for ck, k in zip(c, ks))

# (1) discrete norm = continuum norm for pure parity, d <= N-1; fails otherwise
N = 16
fine = np.pi * (np.arange(200000) + 0.5) / 200000
for label, ks in [('odd, d=N-1', np.arange(-(N - 1), N, 2)),
                  ('even, d=N-2', np.arange(-(N - 2), N - 1, 2)),
                  ('even, d=N', np.arange(-N, N + 1, 2)),
                  ('mixed, d=3', np.arange(-3, 4))]:
    c = rng.normal(size=len(ks)) + 1j * rng.normal(size=len(ks))
    disc = np.pi / N * np.sum(np.abs(poly(c, ks, grid(N)))**2)
    cont = np.pi * np.mean(np.abs(poly(c, ks, fine))**2)
    print(f"{label:12s} discrete/continuum - 1 = {disc / cont - 1:+.2e}")

# (2) end to end: f -> G -> odd/even-harmonic truncation p -> state on N points -> function on R
def run(name, f, L, d, n, parity):
    N = 2**n
    M = 1 << 17
    th = np.pi * (np.arange(M) + 0.5) / M                 # fine grid on (0, pi)
    G = f(L / np.tan(th)) / np.sin(th)
    ks = np.arange(-d, d + 1, 2)
    assert (d % 2) == parity
    # L2(0,pi) projection onto e^{ik th}, k = d mod 2 (orthogonal on (0,pi))
    c = np.array([np.mean(G * np.exp(-1j * k * th)) for k in ks])
    p = lambda t: poly(c, ks, t)
    pmax = np.abs(p(th)).max()
    # state on the N-point grid and its exactly known success amplitude
    amp = p(grid(N)) / pmax
    a_disc = np.sqrt(np.mean(np.abs(amp)**2))
    a_cont = np.sqrt(np.mean(np.abs(p(th))**2)) / pmax
    psi = amp / np.linalg.norm(amp)
    # represented function on R: (T^-1 p)(y) = p(theta) sin(theta) / sqrt(L), theta = arccot(y/L)
    nf = np.sqrt(integrate.quad(lambda y: f(y)**2, -np.inf, np.inf)[0])
    npn = np.sqrt(L * np.pi * np.mean(np.abs(p(th))**2))
    def err2(y):
        t = np.arctan2(L, y)
        return np.abs(f(y) / nf - p(t) * np.sin(t) / npn)**2
    eR = np.sqrt(sum(integrate.quad(err2, a, b, limit=400, epsabs=1e-15, epsrel=1e-10)[0] for a, b in [(-np.inf, -30), (-30, -3), (-3, 0), (0, 3), (3, 30), (30, np.inf)]))
    eT = np.sqrt(np.pi * np.mean(np.abs(G / np.sqrt(np.pi * np.mean(G**2)) - p(th) / np.sqrt(np.pi * np.mean(np.abs(p(th))**2)))**2))
    # interpolant of the N-point state reproduces p/||p||
    Kn = np.arange(-(N - 1), N, 2) if parity else np.arange(-(N - 2), N - 1, 2)
    E = np.exp(1j * np.outer(grid(N), Kn))
    coef = np.linalg.lstsq(E, psi * np.sqrt(N / np.pi), rcond=None)[0]
    rec = E @ coef
    interp = poly(coef, Kn, th)
    eI = np.abs(interp - p(th) / np.sqrt(np.pi * np.mean(np.abs(p(th))**2))).max()
    print(f"{name:10s} L={L} d={d} n={n}: amp disc={a_disc:.12f} cont={a_cont:.12f} | "
          f"L2(R) err={eR:.3e}  L2(0,pi) err={eT:.3e} | interpolant-p sup diff={eI:.1e}")

run('lorentz', lambda y: 1 / (1 + y**2), 1.0, 1, 2, 1)
run('cauchy', lambda y: 1 / np.sqrt(1 + y**2), 1.0, 0, 2, 0)
run('sech', lambda y: 1 / np.cosh(y), 2.5, 21, 5, 1)
run('sech', lambda y: 1 / np.cosh(y), 2.5, 20, 5, 0)
run('gauss', lambda y: np.exp(-y**2 / 2), 3.0, 21, 5, 1)
run('gauss', lambda y: np.exp(-y**2 / 2), 3.0, 21, 8, 1)
