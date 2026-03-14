"""
Numerical QSP phase-finding for stereographic encoding.

This module implements:
1. Vectorized QSP product evaluation (standard and stereographic)
2. Phase-finding via L-BFGS-B optimization (parallelized across trials)
3. Utility functions for the change of variable r <-> a = r/sqrt(1+r^2)

The key result (Proposition 7 in the manuscript) is that stereographic QSP
is unitarily equivalent to standard QSP via V = diag(1, i). Therefore
phase-finding reduces to the standard problem with a = r/sqrt(1+r^2).

Parallelism note:
  When running in parallel mode, each worker process should use single-threaded
  BLAS to avoid contention.  Set these BEFORE importing numpy:
      export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  The run_all.sh script does this automatically.
"""

import os
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Default parallelism: use all available cores, or override with env var
# ---------------------------------------------------------------------------


def _default_workers():
    """Number of parallel workers. Override with STEREO_QSP_WORKERS env var."""
    env = os.environ.get("STEREO_QSP_WORKERS")
    if env is not None:
        return int(env)
    cpu = os.cpu_count() or 1
    # Leave 1-2 cores free for the OS when many are available
    return max(1, cpu - 2) if cpu > 4 else max(1, cpu)


# ---------------------------------------------------------------------------
# Core QSP product evaluation
# ---------------------------------------------------------------------------


def qsp_product(phis: np.ndarray, a_arr: np.ndarray) -> np.ndarray:
    """
    Evaluate the standard QSP product for an array of signal values.

    W(a) = R(phi_0) * prod_{j=1}^{d} [S(a) * R(phi_j)]

    where S(a) = [[a, i*sqrt(1-a^2)], [i*sqrt(1-a^2), a]]
    and   R(phi) = diag(e^{i*phi}, e^{-i*phi}).

    Parameters
    ----------
    phis : array, shape (d+1,)
        Phase angles.
    a_arr : array, shape (n,)
        Signal values in [-1, 1].

    Returns
    -------
    W : array, shape (n, 2, 2), complex
        The 2x2 QSP product matrix at each signal value.
    """
    d = len(phis) - 1
    n = len(a_arr)
    a = np.asarray(a_arr, dtype=float)
    s = np.sqrt(np.maximum(1.0 - a**2, 0.0))

    # Initialize with R(phi_0)
    W = np.zeros((n, 2, 2), dtype=complex)
    W[:, 0, 0] = np.exp(1j * phis[0])
    W[:, 1, 1] = np.exp(-1j * phis[0])

    # Signal matrix (constant across iterations, only depends on a)
    Sa = np.zeros((n, 2, 2), dtype=complex)
    Sa[:, 0, 0] = a
    Sa[:, 0, 1] = 1j * s
    Sa[:, 1, 0] = 1j * s
    Sa[:, 1, 1] = a

    for j in range(d):
        W = np.einsum("nij,njk->nik", W, Sa)
        # Apply R(phi_{j+1}): right-multiply by diagonal
        ep = np.exp(1j * phis[j + 1])
        em = np.exp(-1j * phis[j + 1])
        W_new = W.copy()
        W_new[:, 0, 0] = W[:, 0, 0] * ep
        W_new[:, 0, 1] = W[:, 0, 1] * em
        W_new[:, 1, 0] = W[:, 1, 0] * ep
        W_new[:, 1, 1] = W[:, 1, 1] * em
        W = W_new

    return W


def stereo_qsp_product(phis: np.ndarray, r_arr: np.ndarray) -> np.ndarray:
    """
    Evaluate the stereographic QSP product.

    This is the standard QSP product with a = r/sqrt(1+r^2), related to
    the direct stereographic product by the basis change V = diag(1, i).

    Parameters
    ----------
    phis : array, shape (d+1,)
        Phase angles (same as for standard QSP).
    r_arr : array, shape (n,)
        Unbounded signal values r >= 0.

    Returns
    -------
    W : array, shape (n, 2, 2), complex
        The stereographic QSP product. W[0,0] = P_std(a), W[1,0] = -i*Q_std(a).
    """
    r = np.asarray(r_arr, dtype=float)
    a = r / np.sqrt(1.0 + r**2)

    W_std = qsp_product(phis, a)

    # Apply basis change: W_stereo = V^{-1} W_std V, V = diag(1, i)
    W_ste = W_std.copy()
    W_ste[:, 0, 1] *= 1j
    W_ste[:, 1, 0] *= -1j
    return W_ste


def decoded_function(phis: np.ndarray, r_arr: np.ndarray) -> np.ndarray:
    """
    Compute the stereographic decoded function f(r) = P(r~)/Q(r~).

    For the stereographic product, the decoded value is
    W_stereo[0,0] / W_stereo[1,0] = P_std(a) / (-i * Q_std(a)) = i * P_std/Q_std.

    For real-valued decoded functions (real polynomial QSP), this is real.

    Parameters
    ----------
    phis : array, shape (d+1,)
    r_arr : array, shape (n,)

    Returns
    -------
    f : array, shape (n,), complex
        The decoded function values.
    """
    W = stereo_qsp_product(phis, r_arr)
    return W[:, 0, 0] / W[:, 1, 0]


# ---------------------------------------------------------------------------
# Single-trial worker functions (top-level for pickling)
# ---------------------------------------------------------------------------


def _limit_blas_threads():
    """Restrict BLAS/OpenBLAS/MKL to 1 thread inside worker processes."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _run_single_trial_standard(args):
    """Run one L-BFGS-B trial for standard QSP. Picklable top-level function."""
    _limit_blas_threads()
    seed, d, a_samples, P_vals = args
    rng = np.random.RandomState(seed)
    phi0 = rng.uniform(-np.pi, np.pi, d + 1)

    def cost(phis):
        W = qsp_product(phis, a_samples)
        return np.sum(np.abs(W[:, 0, 0] - P_vals) ** 2)

    res = minimize(cost, phi0, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-15})
    return res.x, res.fun


def _run_single_trial_stereo(args):
    """Run one L-BFGS-B trial for stereographic QSP. Picklable top-level function."""
    _limit_blas_threads()
    seed, d, r_samples, f_vals, mask, real_only = args
    rng = np.random.RandomState(seed)
    phi0 = rng.uniform(-np.pi, np.pi, d + 1)

    def cost(phis):
        f_computed = decoded_function(phis, r_samples)
        if real_only:
            return np.sum((f_computed[mask].real - f_vals[mask].real) ** 2)
        return np.sum(np.abs(f_computed[mask] - f_vals[mask]) ** 2)

    res = minimize(cost, phi0, method="L-BFGS-B", options={"maxiter": 10000, "ftol": 1e-18})
    return res.x, res.fun


# ---------------------------------------------------------------------------
# Phase-finding (parallelized)
# ---------------------------------------------------------------------------


def find_phases_standard(
    P_target,
    d: int,
    a_samples: np.ndarray = None,
    n_trials: int = 50,
    verbose: bool = False,
    n_workers: int = None,
) -> tuple:
    """
    Find QSP phases for a target polynomial P(a) with |P(a)| <= 1.

    Uses optimization (L-BFGS-B with random restarts) to minimize
    ||<0|W|0> - P_target(a)||^2 over sample points.

    Trials are run in parallel across multiple CPU cores.

    Parameters
    ----------
    P_target : callable
        Target function P(a) for a in [-1, 1].
    d : int
        QSP degree (number of signal applications).
    a_samples : array, optional
        Sample points in [-1, 1]. Default: 50 Chebyshev nodes.
    n_trials : int
        Number of random restarts.
    verbose : bool
        Print progress.
    n_workers : int, optional
        Number of parallel workers. Default: auto-detect.

    Returns
    -------
    phis : array, shape (d+1,)
        Best phase angles found.
    cost : float
        Final cost (sum of squared errors).
    """
    if a_samples is None:
        k = np.arange(50)
        a_samples = np.cos((2 * k + 1) / (2 * 50) * np.pi)

    P_vals = np.array([P_target(a) for a in a_samples], dtype=complex)

    if n_workers is None:
        n_workers = _default_workers()

    # Generate reproducible seeds for each trial
    base_seed = np.random.randint(0, 2**31)
    seeds = [base_seed + i for i in range(n_trials)]

    # Build picklable argument tuples
    trial_args = [(s, d, a_samples, P_vals) for s in seeds]

    best_cost = np.inf
    best_phis = None

    if n_workers <= 1 or n_trials <= 2:
        # Sequential fallback
        for i, args in enumerate(trial_args):
            phis, cost_val = _run_single_trial_standard(args)
            if cost_val < best_cost:
                best_cost = cost_val
                best_phis = phis
                if verbose:
                    print(f"  trial {i}: cost = {cost_val:.2e}")
            if best_cost < 1e-25:
                break
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_single_trial_standard, a): idx
                for idx, a in enumerate(trial_args)
            }
            for future in as_completed(futures):
                phis, cost_val = future.result()
                idx = futures[future]
                if cost_val < best_cost:
                    best_cost = cost_val
                    best_phis = phis
                    if verbose:
                        print(f"  trial {idx}: cost = {cost_val:.2e}")
                # Note: can't early-stop already-submitted futures easily,
                # but we can skip processing if we found near-perfect solution.

    return best_phis, best_cost


def find_phases_stereo(
    f_target,
    d: int,
    r_samples: np.ndarray = None,
    n_trials: int = 50,
    verbose: bool = False,
    n_workers: int = None,
    real_only: bool = True,
) -> tuple:
    """
    Find QSP phases for a target decoded function f(r) on [0, infty).

    The decoded function is f(r) = W_stereo[0,0] / W_stereo[1,0].
    This works by optimizing the ratio directly.

    Trials are run in parallel across multiple CPU cores.

    Parameters
    ----------
    f_target : callable
        Target function f(r) for r > 0.
    d : int
        QSP degree.
    r_samples : array, optional
        Sample points r > 0. Default: 30 points in [0.1, 10].
    n_trials : int
        Number of random restarts.
    verbose : bool
        Print progress.
    n_workers : int, optional
        Number of parallel workers. Default: auto-detect.
    real_only : bool
        If True, only fit the real part of the decoded function.
        Default: True. This is correct for real-valued target functions
        since the imaginary part of the ratio is a gauge degree of freedom.

    Returns
    -------
    phis : array, shape (d+1,)
        Best phase angles found.
    cost : float
        Final cost.
    """
    if r_samples is None:
        r_samples = np.linspace(0.1, 10, 30)

    f_vals = np.array([f_target(r) for r in r_samples], dtype=complex)
    mask = np.isfinite(f_vals) & (np.abs(f_vals) < 1e6)

    if n_workers is None:
        n_workers = _default_workers()

    base_seed = np.random.randint(0, 2**31)
    seeds = [base_seed + i for i in range(n_trials)]

    trial_args = [(s, d, r_samples, f_vals, mask, real_only) for s in seeds]

    best_cost = np.inf
    best_phis = None

    if n_workers <= 1 or n_trials <= 2:
        for i, args in enumerate(trial_args):
            phis, cost_val = _run_single_trial_stereo(args)
            if cost_val < best_cost:
                best_cost = cost_val
                best_phis = phis
                if verbose:
                    print(f"  trial {i}: cost = {cost_val:.2e}")
            if best_cost < 1e-25:
                break
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_single_trial_stereo, a): idx
                for idx, a in enumerate(trial_args)
            }
            for future in as_completed(futures):
                phis, cost_val = future.result()
                idx = futures[future]
                if cost_val < best_cost:
                    best_cost = cost_val
                    best_phis = phis
                    if verbose:
                        print(f"  trial {idx}: cost = {cost_val:.2e}")

    return best_phis, best_cost


# ---------------------------------------------------------------------------
# Utility: change of variable
# ---------------------------------------------------------------------------


def r_to_a(r: np.ndarray) -> np.ndarray:
    """Map r in [0, infty) to a = r/sqrt(1+r^2) in [0, 1)."""
    return r / np.sqrt(1.0 + r**2)


def a_to_r(a: np.ndarray) -> np.ndarray:
    """Map a in [0, 1) to r = a/sqrt(1-a^2) in [0, infty)."""
    return a / np.sqrt(np.maximum(1.0 - a**2, 1e-30))


def cot_base_function(k: int, r: np.ndarray) -> np.ndarray:
    """
    Compute the base stereographic QSP function cot(k * arctan(1/r)).

    Parameters
    ----------
    k : int
        Degree.
    r : array
        Signal values r > 0.

    Returns
    -------
    f : array
        cot(k * arctan(1/r)).
    """
    return 1.0 / np.tan(k * np.arctan(1.0 / r))
