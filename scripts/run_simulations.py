#!/usr/bin/env python3
"""
Heavy numerical simulations for the stereographic QSP paper.

Saves results to data/ as .npz files. Run this once (may take hours),
then use scripts/generate_figures_from_data.py to produce figures quickly.

Simulations:
  1. Phase-finding for base cases cot(k*arctan(1/r)), k=2..12
  2. Phase-finding for 1/r inversion at degrees d=2,3,...,20
  3. Phase-finding for various target functions (step, Gaussian, Lorentzian, ...)
  4. Convergence: rational vs standard Chebyshev for multiple target functions
  5. Error analysis: Monte Carlo validation of error bounds
  6. Round-trip verification: generate from known phases, recover

Usage:
  python scripts/run_simulations.py              # run all
  python scripts/run_simulations.py --sim 1 2    # run specific sims
  python scripts/run_simulations.py --trials 200 # more random restarts
"""

import argparse
import json
import os
import sys
import time

# Force single-threaded BLAS BEFORE importing numpy.
# This is critical for parallel phase-finding: without it, each worker
# process spawns N BLAS threads, causing massive contention.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'BLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
from scipy.integrate import quad

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stereo_block_enc.numerical.qsp_phases import (
    qsp_product,
    stereo_qsp_product,
    decoded_function,
    find_phases_standard,
    find_phases_stereo,
    r_to_a,
    a_to_r,
    cot_base_function,
)

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def ensure_datadir():
    os.makedirs(DATADIR, exist_ok=True)


def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


# ============================================================================
# Helper: Chebyshev basis functions
# ============================================================================

def TB(k, r):
    """Rational Chebyshev of first kind: T_k(r/sqrt(1+r^2))."""
    from numpy.polynomial.chebyshev import chebval
    a = r / np.sqrt(1 + r**2)
    return chebval(a, np.eye(k + 1)[k])


def SB(k, r):
    """Rational Chebyshev of second kind: U_{k-1}(a)/sqrt(1+r^2)."""
    a = r / np.sqrt(1 + r**2)
    s = 1.0 / np.sqrt(1 + r**2)
    theta = np.arccos(np.clip(a, -1, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        Uk = np.where(
            np.abs(np.sin(theta)) > 1e-15,
            np.sin(k * theta) / np.sin(theta),
            k * np.cos(k * theta) / np.cos(theta),
        )
    return Uk * s


# ============================================================================
# Simulation 1: Phase-finding for base cases cot(k*arctan(1/r))
# ============================================================================

def sim1_base_case_phases(n_trials=100):
    """Find QSP phases for base cases cot(k*arctan(1/r)), k=2..12."""
    print(f'[{timestamp()}] SIM 1: Base case phase-finding (k=2..12, {n_trials} trials)')

    k_values = list(range(2, 13))
    r_sample = np.linspace(0.2, 10, 60)
    r_fine = np.linspace(0.1, 12, 500)

    results = {}
    for k in k_values:
        t0 = time.time()
        target = lambda r, kk=k: cot_base_function(kk, r)

        phis, cost = find_phases_stereo(
            target, d=k, r_samples=r_sample, n_trials=n_trials, verbose=False
        )
        elapsed = time.time() - t0

        # Evaluate on fine grid
        f_found = decoded_function(phis, r_fine)
        f_target = target(r_fine)
        max_err = np.max(np.abs(f_found.real - f_target)[np.isfinite(f_target)])

        results[f'k{k}_phis'] = phis
        results[f'k{k}_cost'] = cost
        results[f'k{k}_max_err'] = max_err
        results[f'k{k}_time'] = elapsed

        print(f'  k={k:2d}: cost={cost:.2e}, max_err={max_err:.2e}, time={elapsed:.1f}s')

    results['k_values'] = np.array(k_values)
    results['r_fine'] = r_fine
    results['r_sample'] = r_sample

    np.savez(os.path.join(DATADIR, 'sim1_base_cases.npz'), **results)
    print(f'[{timestamp()}] SIM 1 done.\n')


# ============================================================================
# Simulation 2: Eigenvalue inversion f(r)=1/r at various degrees
# ============================================================================

def sim2_inversion(n_trials=200):
    """Approximate f(r)=1/r via stereographic QSP at degrees d=2..20."""
    print(f'[{timestamp()}] SIM 2: Inversion f(r)=1/r (d=2..20, {n_trials} trials)')

    d_values = list(range(2, 21))
    r_sample = np.linspace(0.3, 15, 80)
    r_fine = np.linspace(0.1, 20, 600)
    target = lambda r: 1.0 / r

    results = {}
    for d in d_values:
        t0 = time.time()
        phis, cost = find_phases_stereo(
            target, d=d, r_samples=r_sample, n_trials=n_trials, verbose=False
        )
        elapsed = time.time() - t0

        # Evaluate
        f_found = decoded_function(phis, r_fine)
        f_target = target(r_fine)
        abs_err = np.abs(f_found.real - f_target)
        max_err = np.max(abs_err)
        # Weighted L2 error (more forgiving at large r)
        l2_err = np.sqrt(np.mean(abs_err**2))

        # Also get P(a) values to show internal polynomial
        a_fine = r_to_a(r_fine)
        W_std = qsp_product(phis, a_fine)
        P_vals = W_std[:, 0, 0]

        results[f'd{d}_phis'] = phis
        results[f'd{d}_cost'] = cost
        results[f'd{d}_max_err'] = max_err
        results[f'd{d}_l2_err'] = l2_err
        results[f'd{d}_f_found'] = f_found.real
        results[f'd{d}_P_vals'] = P_vals
        results[f'd{d}_abs_err'] = abs_err
        results[f'd{d}_time'] = elapsed

        print(f'  d={d:2d}: cost={cost:.2e}, max_err={max_err:.2e}, '
              f'L2={l2_err:.2e}, time={elapsed:.1f}s')

    results['d_values'] = np.array(d_values)
    results['r_fine'] = r_fine
    results['a_fine'] = r_to_a(r_fine)
    results['r_sample'] = r_sample

    np.savez(os.path.join(DATADIR, 'sim2_inversion.npz'), **results)
    print(f'[{timestamp()}] SIM 2 done.\n')


# ============================================================================
# Simulation 3: Phase-finding for various target functions
# ============================================================================

def sim3_various_targets(n_trials=150):
    """Phase-finding for diverse target functions at multiple degrees."""
    print(f'[{timestamp()}] SIM 3: Various target functions ({n_trials} trials)')

    r_sample = np.linspace(0.2, 10, 60)
    r_fine = np.linspace(0.05, 15, 500)

    # Target functions: name, callable, degrees to try
    targets = [
        ('step_at_2', lambda r: np.tanh(20 * (r_to_a(r) - 2/np.sqrt(5))),
         [3, 5, 7, 9, 11, 15, 21]),
        ('gaussian', lambda r: np.exp(-r**2 / 2),
         [3, 5, 7, 9, 11, 15]),
        ('lorentzian', lambda r: 1.0 / (1 + r**2),
         [2, 3, 4, 5, 7, 9]),
        ('sign_r', lambda r: np.tanh(10 * (r - 1)),
         [3, 5, 9, 13, 17, 21]),
        ('sqrt_inv', lambda r: 1.0 / np.sqrt(r + 0.1),
         [3, 5, 7, 9, 13]),
        ('oscillatory', lambda r: np.cos(3 * r) / (1 + r**2),
         [5, 9, 13, 17, 21]),
        ('linear_capped', lambda r: np.tanh(r / 3),
         [3, 5, 7, 9]),
        ('inv_sqrt_1pr2', lambda r: 1.0 / np.sqrt(1 + r**2),
         [2, 3, 5, 7, 9]),
    ]

    results = {}
    results['r_fine'] = r_fine
    results['r_sample'] = r_sample

    for name, f_target, d_list in targets:
        print(f'  Target: {name}')
        f_target_fine = np.array([f_target(r) for r in r_fine])
        results[f'{name}_target'] = f_target_fine
        results[f'{name}_d_list'] = np.array(d_list)

        for d in d_list:
            t0 = time.time()
            phis, cost = find_phases_stereo(
                f_target, d=d, r_samples=r_sample, n_trials=n_trials, verbose=False
            )
            elapsed = time.time() - t0

            f_found = decoded_function(phis, r_fine)
            abs_err = np.abs(f_found.real - f_target_fine)
            mask = np.isfinite(abs_err)
            max_err = np.max(abs_err[mask]) if np.any(mask) else np.inf

            results[f'{name}_d{d}_phis'] = phis
            results[f'{name}_d{d}_cost'] = cost
            results[f'{name}_d{d}_max_err'] = max_err
            results[f'{name}_d{d}_f_found'] = f_found.real
            results[f'{name}_d{d}_abs_err'] = abs_err
            results[f'{name}_d{d}_time'] = elapsed

            print(f'    d={d:2d}: cost={cost:.2e}, max_err={max_err:.2e}, time={elapsed:.1f}s')

    np.savez(os.path.join(DATADIR, 'sim3_various_targets.npz'), **results)
    print(f'[{timestamp()}] SIM 3 done.\n')


# ============================================================================
# Simulation 4: Convergence of rational vs standard Chebyshev
# ============================================================================

def sim4_convergence():
    """Chebyshev expansion convergence for multiple target functions."""
    print(f'[{timestamp()}] SIM 4: Chebyshev convergence analysis')

    K_max = 50  # max expansion terms

    def compute_rational_cheb_coeffs(f, K):
        coeffs = []
        for k in range(K):
            integrand = lambda r: f(r) * TB(k, np.array([r]))[0] / (1 + r**2)
            val, _ = quad(integrand, -200, 200, limit=300)
            if k == 0:
                val /= np.pi
            else:
                val *= 2.0 / np.pi
            coeffs.append(val)
        return np.array(coeffs)

    def compute_standard_cheb_coeffs(g, K):
        coeffs = []
        for k in range(K):
            integrand = lambda a: g(a) * np.cos(k * np.arccos(a)) / np.sqrt(1 - a**2)
            val, _ = quad(integrand, -1 + 1e-12, 1 - 1e-12, limit=300)
            if k == 0:
                val /= np.pi
            else:
                val *= 2.0 / np.pi
            coeffs.append(val)
        return np.array(coeffs)

    # Targets
    targets = [
        ('inv_1pr4', lambda r: 1.0 / (1 + r**4),
         lambda a: (1 - a**2)**2 / ((1 - a**2)**2 + a**4)
             if abs(a) < 1 - 1e-10 else 0.0),
        ('sech', lambda r: 1.0 / np.cosh(r),
         lambda a: 1.0 / np.cosh(a / np.sqrt(max(1 - a**2, 1e-30)))
             if abs(a) < 1 - 1e-6 else 0.0),
        ('lorentzian', lambda r: 1.0 / (1 + r**2),
         lambda a: 1 - a**2 if abs(a) < 1 else 0.0),
        ('gaussian', lambda r: np.exp(-r**2),
         lambda a: np.exp(-a**2 / max(1 - a**2, 1e-30))
             if abs(a) < 1 - 1e-6 else 0.0),
        ('inv_1pr6', lambda r: 1.0 / (1 + r**6),
         lambda a: (1 - a**2)**3 / ((1 - a**2)**3 + a**6)
             if abs(a) < 1 - 1e-10 else 0.0),
    ]

    r_test = np.linspace(-20, 20, 1000)
    a_test = r_test / np.sqrt(1 + r_test**2)

    results = {}
    results['K_max'] = K_max
    results['r_test'] = r_test
    results['a_test'] = a_test

    for name, f_r, g_a in targets:
        print(f'  Target: {name}')
        t0 = time.time()

        c_rat = compute_rational_cheb_coeffs(f_r, K_max)
        c_std = compute_standard_cheb_coeffs(g_a, K_max)

        # Compute truncation errors
        f_target_r = f_r(r_test)
        g_target_a = np.array([g_a(a) for a in a_test])

        err_rat = []
        err_std = []
        for Ktrunc in range(1, K_max + 1):
            # Rational Chebyshev partial sum
            f_approx = np.zeros_like(r_test)
            for k in range(Ktrunc):
                f_approx += c_rat[k] * TB(k, r_test)
            err_rat.append(np.max(np.abs(f_approx - f_target_r)))

            # Standard Chebyshev partial sum
            g_approx = np.zeros_like(a_test)
            for k in range(Ktrunc):
                g_approx += c_std[k] * np.cos(k * np.arccos(np.clip(a_test, -1, 1)))
            err_std.append(np.max(np.abs(g_approx - g_target_a)))

        elapsed = time.time() - t0

        results[f'{name}_c_rat'] = c_rat
        results[f'{name}_c_std'] = c_std
        results[f'{name}_err_rat'] = np.array(err_rat)
        results[f'{name}_err_std'] = np.array(err_std)

        print(f'    Done in {elapsed:.1f}s, final err_rat={err_rat[-1]:.2e}, '
              f'err_std={err_std[-1]:.2e}')

    np.savez(os.path.join(DATADIR, 'sim4_convergence.npz'), **results)
    print(f'[{timestamp()}] SIM 4 done.\n')


# ============================================================================
# Simulation 5: Error analysis — Monte Carlo validation
# ============================================================================

def sim5_error_analysis(n_mc=50000):
    """Monte Carlo validation of stereographic decoding error bounds."""
    print(f'[{timestamp()}] SIM 5: Error analysis Monte Carlo ({n_mc} samples)')

    r_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    delta_values = np.array([0.1, 0.01, 0.001])

    results = {}
    results['r_values'] = r_values
    results['delta_values'] = delta_values
    results['n_mc'] = n_mc

    # Analytical error bound: |dz| <= (1+r^2)(1+r)/2 * delta
    analytical_bound = lambda r, delta: (1 + r**2) * (1 + r) / 2 * delta

    for delta in delta_values:
        print(f'  delta = {delta}')
        empirical_max_err = []
        empirical_mean_err = []
        empirical_99th_err = []
        bound_vals = []

        for r in r_values:
            # True Bloch components
            X_true = 2 * r / (1 + r**2)
            Z_true = (r**2 - 1) / (1 + r**2)
            z_true = r

            # Perturb
            X_noisy = X_true + delta * np.random.randn(n_mc)
            Z_noisy = Z_true + delta * np.random.randn(n_mc)

            # Decode
            denom = 1 - Z_noisy
            # Avoid division by zero
            good = np.abs(denom) > 1e-15
            z_hat = np.full(n_mc, np.nan)
            z_hat[good] = X_noisy[good] / denom[good]

            errs = np.abs(z_hat[good] - z_true)
            empirical_max_err.append(np.max(errs))
            empirical_mean_err.append(np.mean(errs))
            empirical_99th_err.append(np.percentile(errs, 99))
            bound_vals.append(analytical_bound(r, delta))

            print(f'    r={r:5.1f}: bound={bound_vals[-1]:.3e}, '
                  f'max={empirical_max_err[-1]:.3e}, '
                  f'99th={empirical_99th_err[-1]:.3e}, '
                  f'mean={empirical_mean_err[-1]:.3e}')

        results[f'delta{delta}_max_err'] = np.array(empirical_max_err)
        results[f'delta{delta}_mean_err'] = np.array(empirical_mean_err)
        results[f'delta{delta}_99th_err'] = np.array(empirical_99th_err)
        results[f'delta{delta}_bound'] = np.array(bound_vals)

    # Also compute continuous curves for plotting
    r_cont = np.linspace(0.01, 60, 500)
    results['r_cont'] = r_cont
    results['amp_abs'] = (1 + r_cont**2) * (1 + r_cont) / 2
    results['amp_rel'] = (1 + r_cont**2) * (1 + r_cont) / (2 * r_cont)

    # Shot counts for various epsilon targets
    for eps in [0.1, 0.01, 0.001]:
        results[f'shots_eps{eps}'] = ((1 + r_cont**2) * (1 + r_cont) / 2)**2 / eps**2

    np.savez(os.path.join(DATADIR, 'sim5_error_analysis.npz'), **results)
    print(f'[{timestamp()}] SIM 5 done.\n')


# ============================================================================
# Simulation 6: Round-trip verification (phases -> function -> recover phases)
# ============================================================================

def sim6_roundtrip(n_trials=100):
    """Generate random QSP phases, compute decoded function, recover phases."""
    print(f'[{timestamp()}] SIM 6: Round-trip verification ({n_trials} trials)')

    d_values = [2, 3, 4, 5, 6, 7, 8]
    n_tests_per_d = 20
    r_sample = np.linspace(0.2, 10, 60)
    r_fine = np.linspace(0.1, 12, 400)

    results = {}
    results['d_values'] = np.array(d_values)
    results['r_fine'] = r_fine

    np.random.seed(2026)

    for d in d_values:
        recovery_costs = []
        max_errs = []
        original_phases_list = []
        recovered_phases_list = []

        for test_idx in range(n_tests_per_d):
            # Generate random phases
            original_phis = np.random.uniform(-np.pi, np.pi, d + 1)

            # Compute the decoded function
            f_original = decoded_function(original_phis, r_fine).real
            f_at_samples = decoded_function(original_phis, r_sample).real

            # Try to recover phases
            target_fn = lambda r, f=f_at_samples, rs=r_sample: np.interp(
                r, rs, f, left=f[0], right=f[-1]
            )
            recovered_phis, cost = find_phases_stereo(
                target_fn, d=d, r_samples=r_sample, n_trials=n_trials, verbose=False
            )

            # Compare
            f_recovered = decoded_function(recovered_phis, r_fine).real
            mask = np.isfinite(f_original) & np.isfinite(f_recovered) & (np.abs(f_original) < 1e6)
            if np.any(mask):
                max_err = np.max(np.abs(f_original[mask] - f_recovered[mask]))
            else:
                max_err = np.inf

            recovery_costs.append(cost)
            max_errs.append(max_err)
            original_phases_list.append(original_phis)
            recovered_phases_list.append(recovered_phis)

        results[f'd{d}_costs'] = np.array(recovery_costs)
        results[f'd{d}_max_errs'] = np.array(max_errs)
        results[f'd{d}_original_phis'] = np.array(original_phases_list)
        results[f'd{d}_recovered_phis'] = np.array(recovered_phases_list)

        print(f'  d={d}: median_cost={np.median(recovery_costs):.2e}, '
              f'median_max_err={np.median(max_errs):.2e}, '
              f'worst_max_err={np.max(max_errs):.2e}')

    np.savez(os.path.join(DATADIR, 'sim6_roundtrip.npz'), **results)
    print(f'[{timestamp()}] SIM 6 done.\n')


# ============================================================================
# Main
# ============================================================================

ALL_SIMS = {
    1: ('Base case phases (cot_k)', sim1_base_case_phases),
    2: ('Eigenvalue inversion 1/r', sim2_inversion),
    3: ('Various target functions', sim3_various_targets),
    4: ('Chebyshev convergence', sim4_convergence),
    5: ('Error analysis Monte Carlo', sim5_error_analysis),
    6: ('Round-trip verification', sim6_roundtrip),
}


def main():
    parser = argparse.ArgumentParser(description='Run numerical simulations for stereographic QSP')
    parser.add_argument('--sim', type=int, nargs='*', default=None,
                        help='Which simulations to run (1-6). Default: all.')
    parser.add_argument('--trials', type=int, default=None,
                        help='Override number of random restarts for phase-finding.')
    parser.add_argument('--mc', type=int, default=None,
                        help='Override number of Monte Carlo samples for error analysis.')
    args = parser.parse_args()

    ensure_datadir()

    sims_to_run = args.sim if args.sim else list(ALL_SIMS.keys())

    print('=' * 70)
    print('Stereographic QSP — Numerical Simulations')
    print('=' * 70)
    print(f'Start: {timestamp()}')
    print(f'Simulations: {sims_to_run}')
    if args.trials:
        print(f'Trials override: {args.trials}')
    print(f'Output directory: {os.path.abspath(DATADIR)}')
    print('=' * 70)
    print()

    t_total = time.time()

    for sim_id in sims_to_run:
        if sim_id not in ALL_SIMS:
            print(f'WARNING: Unknown simulation {sim_id}, skipping.')
            continue
        name, func = ALL_SIMS[sim_id]
        print(f'=== Simulation {sim_id}: {name} ===')

        # Pass overrides
        kwargs = {}
        if args.trials and sim_id in [1, 2, 3, 6]:
            kwargs['n_trials'] = args.trials
        if args.mc and sim_id == 5:
            kwargs['n_mc'] = args.mc

        try:
            func(**kwargs)
        except Exception as e:
            print(f'ERROR in simulation {sim_id}: {e}')
            import traceback
            traceback.print_exc()
            print()

    total_elapsed = time.time() - t_total
    print('=' * 70)
    print(f'All simulations complete. Total time: {total_elapsed:.1f}s '
          f'({total_elapsed/60:.1f} min)')
    print(f'Data saved to: {os.path.abspath(DATADIR)}')
    print('=' * 70)


if __name__ == '__main__':
    main()
