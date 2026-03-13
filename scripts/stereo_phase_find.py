"""
Fast stereographic QSP phase-finding with analytical gradients.

Parallelizes random restarts across CPU cores.
Uses forward-accumulation for exact gradient of the decoded cost.

Usage:
    python scripts/stereo_phase_find.py
"""

import os

for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(var, "1")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


# ---- Core: QSP product with prefix/suffix for gradient ----


def _matmul_batch(A, B):
    """Batch 2x2 matrix multiply: (n,2,2) x (n,2,2) -> (n,2,2)."""
    return np.einsum("nij,njk->nik", A, B)


def _cost_and_grad(phis, a, f_target):
    """
    Compute cost = sum |Re(decoded(r)) - f_target|^2 and its gradient.
    a = r/sqrt(1+r^2) array, f_target = target values at r points.
    """
    d = len(phis) - 1
    n = len(a)
    s = np.sqrt(np.maximum(1.0 - a**2, 0.0))

    # Signal matrix (constant)
    Sa = np.zeros((n, 2, 2), dtype=complex)
    Sa[:, 0, 0] = a
    Sa[:, 0, 1] = 1j * s
    Sa[:, 1, 0] = 1j * s
    Sa[:, 1, 1] = a

    # Phase matrices and their derivatives
    ep = np.exp(1j * phis)  # shape (d+1,)
    em = np.exp(-1j * phis)

    # Build R[j] as (2,2) matrices (broadcast over n later)
    # R[j] = diag(e^{i phi_j}, e^{-i phi_j})
    # dR[j] = diag(i e^{i phi_j}, -i e^{-i phi_j})

    # Forward pass: prefix[j] = R0 * Sa * R1 * Sa * ... * R_{j-1} * Sa
    # prefix[0] = I
    I2 = np.zeros((n, 2, 2), dtype=complex)
    I2[:, 0, 0] = 1.0
    I2[:, 1, 1] = 1.0

    prefix = [None] * (d + 1)
    prefix[0] = I2.copy()

    for j in range(d):
        # prefix[j+1] = prefix[j] @ R[j] @ Sa
        # Apply R[j]: right-multiply prefix[j] by diag(ep[j], em[j])
        tmp = prefix[j].copy()
        tmp[:, :, 0] *= ep[j]
        tmp[:, :, 1] *= em[j]
        prefix[j + 1] = _matmul_batch(tmp, Sa)

    # Backward pass: suffix[j] = Sa * R[j+1] * Sa * R[j+2] * ... * Sa * R[d]
    # suffix[d] = I
    suffix = [None] * (d + 1)
    suffix[d] = I2.copy()

    for j in range(d - 1, -1, -1):
        # suffix[j] = Sa @ R[j+1] @ suffix[j+1]
        # Apply R[j+1] to left side of suffix[j+1]: diag * suffix
        tmp = suffix[j + 1].copy()
        tmp[:, 0, :] *= ep[j + 1][:, None] if isinstance(ep[j + 1], np.ndarray) else ep[j + 1]
        tmp[:, 1, :] *= em[j + 1][:, None] if isinstance(em[j + 1], np.ndarray) else em[j + 1]
        suffix[j] = _matmul_batch(Sa, tmp)

    # W = prefix[d] @ R[d]
    W = prefix[d].copy()
    W[:, :, 0] *= ep[d]
    W[:, :, 1] *= em[d]

    P_std = W[:, 0, 0]
    Q_std = W[:, 1, 0]

    # Decoded function: f = Re(P_std / (-i Q_std))
    f_dec = np.real(P_std / (-1j * Q_std))
    residual = f_dec - f_target
    cost = np.sum(residual**2)

    # Gradient
    grad = np.zeros(d + 1)
    for j in range(d + 1):
        # dW/dphi_j = prefix[j] @ dR[j] @ suffix[j]
        # dR[j] = diag(i*ep[j], -i*em[j])
        # prefix[j] @ dR[j]: multiply columns of prefix[j]
        tmp = prefix[j].copy()
        tmp[:, :, 0] *= 1j * ep[j]
        tmp[:, :, 1] *= -1j * em[j]
        dW = _matmul_batch(tmp, suffix[j])

        dP = dW[:, 0, 0]
        dQ = dW[:, 1, 0]

        # d(f_dec)/dphi_j = Re(i * (dP * Q_std - P_std * dQ) / Q_std^2)
        df = np.real(1j * (dP * Q_std - P_std * dQ) / Q_std**2)
        grad[j] = 2.0 * np.sum(residual * df)

    return cost, grad


# ---- Single trial worker ----


def _worker(args):
    """Single optimization trial. Top-level for pickling."""
    seed, d, a_samples, f_target_vals = args
    rng = np.random.RandomState(seed)
    phi0 = rng.uniform(-np.pi, np.pi, d + 1)

    res = minimize(
        _cost_and_grad,
        phi0,
        args=(a_samples, f_target_vals),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5000, "ftol": 1e-28},
    )
    return res.x, res.fun


# ---- Main API ----


def find_stereo_phases(f_target, d, r_samples=None, n_trials=50, n_workers=None, seed=42):
    """
    Find QSP phases for stereographic decoded target f(r).

    Parameters
    ----------
    f_target : callable
        Target f(r) for r > 0.
    d : int
        QSP degree.
    r_samples : array, optional
        Fitting points. Default: 25 points in [0.3, 6].
    n_trials : int
        Number of random restarts (parallelized).
    n_workers : int, optional
        CPU cores. Default: all available minus 2.
    seed : int
        Base random seed.

    Returns
    -------
    phis : array, shape (d+1,)
    cost : float
    """
    if r_samples is None:
        r_samples = np.linspace(0.3, 6, 25)

    a_samples = r_samples / np.sqrt(1.0 + r_samples**2)
    f_vals = np.array([f_target(r) for r in r_samples], dtype=float)

    if n_workers is None:
        cpu = os.cpu_count() or 1
        n_workers = max(1, cpu - 2) if cpu > 4 else max(1, cpu)

    seeds = [seed + i for i in range(n_trials)]
    trial_args = [(s, d, a_samples, f_vals) for s in seeds]

    best_cost = np.inf
    best_phis = None

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, a): i for i, a in enumerate(trial_args)}
        for future in as_completed(futures):
            phis, cost_val = future.result()
            if cost_val < best_cost:
                best_cost = cost_val
                best_phis = phis

    return best_phis, best_cost


# ---- Run convergence study ----

if __name__ == "__main__":
    from stereo_block_enc.numerical.qsp_phases import decoded_function

    r_plot = np.linspace(0.15, 8, 300)

    targets = [
        ("1/r", lambda r: 1.0 / r),
        ("1/(1+r^2)", lambda r: 1.0 / (1 + r**2)),
        ("exp(-r^2)", lambda r: np.exp(-(r**2))),
        ("1/sqrt(1+r^2)", lambda r: 1.0 / np.sqrt(1 + r**2)),
    ]

    degrees = [2, 4, 6, 8, 10, 14]

    for name, ft in targets:
        print(f"\n{name}:")
        for d in degrees:
            t0 = time.time()
            phis, cost = find_stereo_phases(ft, d, n_trials=50, seed=42)
            dt = time.time() - t0

            f_dec = decoded_function(phis, r_plot).real
            f_ex = ft(r_plot)
            mask = (r_plot >= 0.3) & (r_plot <= 6)
            max_err = np.max(np.abs(f_dec[mask] - f_ex[mask]))
            l2_err = np.sqrt(np.mean((f_dec[mask] - f_ex[mask]) ** 2))

            print(
                f"  d={d:2d}: cost={cost:.2e}, max_err={max_err:.2e}, "
                f"L2={l2_err:.2e}, time={dt:.1f}s"
            )

    print("\nDone.")
