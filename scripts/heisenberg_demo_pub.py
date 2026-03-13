#!/usr/bin/env python3
"""
Publication-quality Heisenberg model end-to-end demo.

Pipeline:
  1. Build H = J(XX + YY + ZZ) + h(Z⊗I) for 2 qubits
  2. Diagonalize → eigenvalues, shift to positive
  3. For each degree d, find stereographic QSP phases for f(λ) = 1/λ
  4. Evaluate decoded function at eigenvalues and on continuous grid
  5. Also run standard QSP (with normalization) for comparison
  6. Save data to data/sim7_heisenberg_demo.npz

Usage:
  python scripts/heisenberg_demo_pub.py --trials 500 --degrees 4 6 8 10 14 18
"""

import argparse
import os
import sys
import time

# BLAS thread control BEFORE numpy
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'BLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stereo_block_enc.numerical.qsp_phases import (
    find_phases_stereo,
    find_phases_standard,
    decoded_function,
    qsp_product,
    r_to_a,
)

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def build_heisenberg(J, h):
    """Build H = J(XX + YY + ZZ) + h(Z⊗I) as 4x4 matrix."""
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return J * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)) + h * np.kron(Z, I2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--degrees', type=int, nargs='+', default=[4, 6, 8, 10, 14, 18])
    parser.add_argument('--J', type=float, default=1.0)
    parser.add_argument('--h', type=float, default=0.5)
    args = parser.parse_args()

    print('=' * 70)
    print('Heisenberg Model End-to-End Demo (Publication Quality)')
    print('=' * 70)
    print(f'  J = {args.J}, h = {args.h}')
    print(f'  Degrees: {args.degrees}')
    print(f'  Trials per degree: {args.trials}')
    print(f'  Start: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    # ── Step 1: Build & diagonalize ─────────────────────────────────────────
    H = build_heisenberg(args.J, args.h)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    print(f'\nEigenvalues: {eigenvalues}')

    # Shift to positive with a comfortable buffer
    shift = -min(eigenvalues) + 0.5
    eig_shifted = eigenvalues + shift
    print(f'Shift = {shift:.4f}')
    print(f'Shifted eigenvalues: {eig_shifted}')

    # ── Step 2: Define fitting range ────────────────────────────────────────
    r_max = max(eig_shifted) + 3.0
    r_sample = np.linspace(0.2, r_max, 80)  # more sample points for quality
    r_fine = np.linspace(0.15, r_max, 400)  # fine grid for plotting
    f_exact_fine = 1.0 / r_fine

    # Target
    def f_target(r_arr):
        return 1.0 / r_arr

    # ── Step 3: Stereographic QSP for each degree ──────────────────────────
    n_workers = max(1, (os.cpu_count() or 1) - 2)
    print(f'\nUsing {n_workers} parallel workers')

    results = {}
    results['J'] = args.J
    results['h'] = args.h
    results['eigenvalues'] = eigenvalues
    results['eigenvalues_shifted'] = eig_shifted
    results['shift'] = shift
    results['r_sample'] = r_sample
    results['r_fine'] = r_fine
    results['f_exact_fine'] = f_exact_fine
    results['degrees'] = np.array(args.degrees)

    for deg in args.degrees:
        t0 = time.time()
        print(f'\n--- Degree d={deg} ({args.trials} trials) ---')

        phis, cost = find_phases_stereo(
            f_target, d=deg,
            r_samples=r_sample,
            n_trials=args.trials,
            n_workers=n_workers,
            verbose=False,
        )
        elapsed = time.time() - t0
        print(f'  Phase-finding: cost={cost:.3e}, time={elapsed:.1f}s')

        # Evaluate on fine grid
        f_fine = decoded_function(phis, r_fine)

        # Evaluate at eigenvalues
        f_at_eig = decoded_function(phis, eig_shifted)
        f_exact_eig = 1.0 / eig_shifted
        errs_eig = np.abs(f_at_eig.real - f_exact_eig)

        # Internal polynomial |P(a)|
        a_fine = r_to_a(r_fine)
        W_std = qsp_product(phis, a_fine)
        P_vals = np.abs(W_std[:, 0, 0])

        for j in range(len(eig_shifted)):
            print(f'  λ\'={eig_shifted[j]:.3f}: '
                  f'f_stereo={f_at_eig[j].real:.6f}, '
                  f'f_exact={f_exact_eig[j]:.6f}, '
                  f'err={errs_eig[j]:.2e}')

        results[f'd{deg}_phis'] = phis
        results[f'd{deg}_cost'] = cost
        results[f'd{deg}_time'] = elapsed
        results[f'd{deg}_f_fine'] = f_fine.real
        results[f'd{deg}_P_vals'] = P_vals
        results[f'd{deg}_f_at_eig'] = f_at_eig.real
        results[f'd{deg}_errs_eig'] = errs_eig

    # ── Step 4: Standard QSP comparison (normalized) ───────────────────────
    # For comparison: standard QSP for f(λ) = 1/λ requires normalizing by α
    alpha = max(eig_shifted)  # normalization factor
    a_eig = eig_shifted / alpha  # normalized eigenvalues in [-1, 1]

    print(f'\n--- Standard QSP comparison (α = {alpha:.3f}) ---')

    # The standard target is P(a) = 1/(α·a), but bounded by 1
    # We approximate with a polynomial that's close to 1/(α·a) for a ∈ [a_min, 1]
    a_min = min(eig_shifted) / alpha
    print(f'  a_min = {a_min:.4f}, α = {alpha:.3f}')

    # For a fair comparison, use same degrees and evaluate the standard approach
    # Standard QSP target: bounded approx of 1/(α·a) for a ∈ [a_min, 1]
    def std_target_bounded(a_arr):
        """Bounded approximation of 1/(α·a), clamped to [-1, 1]."""
        result = np.zeros_like(a_arr)
        mask = np.abs(a_arr) > a_min / 2
        result[mask] = np.clip(1.0 / (alpha * a_arr[mask]), -1, 1)
        return result

    for deg in args.degrees:
        t0 = time.time()
        print(f'\n  Standard QSP d={deg}...')

        # Standard approach: find phases for bounded polynomial
        a_samples_std = np.linspace(a_min * 0.8, 1.0, 60)
        phis_std, cost_std = find_phases_standard(
            std_target_bounded, d=deg,
            a_samples=a_samples_std,
            n_trials=min(args.trials, 200),  # fewer trials for comparison
            n_workers=n_workers,
            verbose=False,
        )
        elapsed_std = time.time() - t0

        # Evaluate at eigenvalues (in normalized coordinates)
        W_at_eig = qsp_product(phis_std, a_eig)
        P_at_eig = W_at_eig[:, 0, 0].real

        # The standard output is P(a) ≈ 1/(α·a), so f(λ) = α · P(λ/α)
        # But P is bounded by 1, so the recovered value is just P(a)
        # and the actual function value is α·P(a) ... no, P(a) ≈ 1/(α·a)
        # so the inverse is λ⁻¹ ≈ α · P(λ/α)... actually standard QSVT
        # gives P(a) = subnormalized value. Let's just report the raw comparison.
        f_std_eig = P_at_eig  # This is the bounded polynomial evaluated at a
        f_std_target = 1.0 / (alpha * a_eig)  # Ideal value (may exceed 1)
        errs_std = np.abs(f_std_eig - np.clip(f_std_target, -1, 1))

        print(f'  cost={cost_std:.3e}, time={elapsed_std:.1f}s')
        for j in range(len(eig_shifted)):
            print(f'    λ\'={eig_shifted[j]:.3f} (a={a_eig[j]:.4f}): '
                  f'P(a)={f_std_eig[j]:.6f}, '
                  f'target={np.clip(f_std_target[j], -1, 1):.6f}, '
                  f'err={errs_std[j]:.2e}')

        results[f'std_d{deg}_phis'] = phis_std
        results[f'std_d{deg}_cost'] = cost_std
        results[f'std_d{deg}_time'] = elapsed_std
        results[f'std_d{deg}_P_at_eig'] = f_std_eig
        results[f'std_d{deg}_errs_eig'] = errs_std

    results['alpha'] = alpha
    results['a_eig'] = a_eig

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(DATADIR, exist_ok=True)
    outpath = os.path.join(DATADIR, 'sim7_heisenberg_demo.npz')
    np.savez(outpath, **results)
    print(f'\nData saved to {outpath}')
    print(f'Total time: {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
