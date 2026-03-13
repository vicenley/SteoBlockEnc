"""
Diagnostic script: validate QSP phase-finding pipeline.

Tests:
1. Standard QSP on Chebyshev polynomials T_k(a) — should recover phases = 0
2. Standard QSP on a smooth bounded target — verify |P(a)| <= 1
3. Stereographic QSP via the reduction: find standard phases for P(a_tilde),
   then decode to get f(r) = P/Q on the unbounded domain
4. Direct stereographic optimization for comparison

Usage:
    python scripts/diagnostic_qsp.py
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
from stereo_block_enc.numerical.qsp_phases import (
    qsp_product,
    stereo_qsp_product,
    decoded_function,
    find_phases_standard,
    find_phases_stereo,
    r_to_a,
    cot_base_function,
)


def test_1_chebyshev():
    """Test 1: Standard QSP recovers T_k(a) with near-zero phases."""
    print("=" * 60)
    print("TEST 1: Standard QSP for Chebyshev polynomial T_k(a)")
    print("=" * 60)

    for k in [2, 3, 5]:
        P_target = lambda a, k=k: np.cos(k * np.arccos(np.clip(a, -1, 1)))

        # Use more Chebyshev nodes for better conditioning
        n_pts = max(80, 2 * k + 20)
        idx = np.arange(n_pts)
        a_samples = np.cos((2 * idx + 1) / (2 * n_pts) * np.pi)

        phis, cost = find_phases_standard(
            P_target, d=k, a_samples=a_samples, n_trials=20, n_workers=1, verbose=False
        )
        print(f"  k={k}: cost = {cost:.2e}, phases = {np.round(phis, 4)}")

        # Verify: evaluate and compare
        W = qsp_product(phis, a_samples)
        P_found = W[:, 0, 0].real
        P_exact = np.array([P_target(a) for a in a_samples])
        max_err = np.max(np.abs(P_found - P_exact))
        print(f"         max |P_found - T_k| = {max_err:.2e}")
    print()


def test_2_smooth_bounded():
    """Test 2: Standard QSP for a smooth bounded target."""
    print("=" * 60)
    print("TEST 2: Standard QSP for smooth bounded targets")
    print("=" * 60)

    # Target: 0.9 * a^2 (even parity, bounded by 1)
    for d, P_target, name in [
        (4, lambda a: 0.9 * a**2, "0.9*a^2 (d=4)"),
        (6, lambda a: 0.5 * (3 * a**2 - 1) * 0.9, "0.9*P_2(a) (d=6)"),
        (8, lambda a: a * (1 - a**2) * 0.9, "0.9*a(1-a^2) (d=8)"),
    ]:
        n_pts = max(80, 2 * d + 20)
        idx = np.arange(n_pts)
        a_samples = np.cos((2 * idx + 1) / (2 * n_pts) * np.pi)

        phis, cost = find_phases_standard(
            P_target, d=d, a_samples=a_samples, n_trials=30, n_workers=1, verbose=False
        )
        W = qsp_product(phis, a_samples)
        P_found = W[:, 0, 0].real
        P_exact = np.array([P_target(a) for a in a_samples])
        max_err = np.max(np.abs(P_found - P_exact))
        bounded = np.all(np.abs(W[:, 0, 0]) <= 1.0 + 1e-10)
        print(f"  {name}: cost = {cost:.2e}, max_err = {max_err:.2e}, |P|<=1: {bounded}")
    print()


def test_3_stereo_base_cases():
    """Test 3: Stereographic base cases z_k(r) = cot(k*arctan(1/r))."""
    print("=" * 60)
    print("TEST 3: Stereographic base cases (all-zero phases)")
    print("=" * 60)

    r_test = np.linspace(0.5, 10, 50)

    for k in [2, 3, 5]:
        # All-zero phases should give the base case
        phis_zero = np.zeros(k + 1)
        f_decoded = decoded_function(phis_zero, r_test)
        f_exact = cot_base_function(k, r_test)

        # The decoded function may have an overall sign/phase — check
        max_err = np.max(np.abs(f_decoded.real - f_exact))
        print(f"  k={k}: max |f_decoded - cot(k*arctan(1/r))| = {max_err:.2e}")

        # If that's not close, try the imaginary part or negative
        if max_err > 0.1:
            max_err2 = np.max(np.abs(-f_decoded.real - f_exact))
            max_err3 = np.max(np.abs(f_decoded.imag - f_exact))
            print(f"         (trying -Re: {max_err2:.2e}, Im: {max_err3:.2e})")
    print()


def test_4_stereo_via_reduction():
    """Test 4: Stereographic QSP via the basis-change reduction.

    Strategy: for a target f(r), we need P(a_tilde) such that
    P(a_tilde)/Q(a_tilde) = f(r), with |P|^2 + |Q|^2 = 1.

    For a smooth target like f(r) = r/(1+r^2) = a_tilde * sqrt(1-a_tilde^2),
    we have P(a_tilde) = a_tilde * sqrt(1-a_tilde^2) * Q(a_tilde).
    But Q is determined by unitarity: Q = sqrt(1 - |P|^2).

    Actually, the simplest approach: find standard QSP phases for
    P(a) = T_k(a) at degree k, then the stereographic decoded function
    is z_k(r) = T_k(a_tilde) / Q_k(a_tilde) = cot(k*arctan(1/r)).
    This is the base case — we already tested it.

    For a general smooth bounded target P(a), find standard phases,
    then the decoded function is P(a_tilde)/Q(a_tilde) where Q comes
    from unitarity.
    """
    print("=" * 60)
    print("TEST 4: Stereographic QSP via reduction (standard phases)")
    print("=" * 60)

    # Target: P(a) = 0.5*a for a in [-1,1] (odd, degree 1 parity -> need d odd)
    # Decoded: f(r) = P(a_tilde) / Q(a_tilde)

    for d, P_target, name in [
        (3, lambda a: 0.5 * a, "P(a)=0.5*a, d=3"),
        (5, lambda a: 0.9 * np.cos(3 * np.arccos(np.clip(a, -1, 1))), "P(a)=0.9*T_3, d=5"),
    ]:
        n_pts = 80
        idx = np.arange(n_pts)
        a_samples = np.cos((2 * idx + 1) / (2 * n_pts) * np.pi)

        phis, cost = find_phases_standard(
            P_target, d=d, a_samples=a_samples, n_trials=30, n_workers=1, verbose=False
        )
        print(f"  {name}: standard cost = {cost:.2e}")

        # Now evaluate the stereographic decoded function
        r_test = np.linspace(0.2, 8, 40)
        a_test = r_to_a(r_test)

        # Standard QSP output
        W_std = qsp_product(phis, a_test)
        P_vals = W_std[:, 0, 0]
        Q_vals = W_std[:, 1, 0]  # this is i*sqrt(1-a^2)*... term

        # Stereographic decoded
        f_stereo = decoded_function(phis, r_test)

        # Check: P should match target
        P_exact = np.array([P_target(a) for a in a_test])
        P_err = np.max(np.abs(P_vals.real - P_exact))
        print(f"         max |P_std - P_target| = {P_err:.2e}")
        print(
            f"         f_stereo range: [{np.min(f_stereo.real):.3f}, {np.max(f_stereo.real):.3f}]"
        )
        print(f"         |Q| range: [{np.min(np.abs(Q_vals)):.4f}, {np.max(np.abs(Q_vals)):.4f}]")
    print()


def test_5_direct_stereo_optimization():
    """Test 5: Direct stereographic optimization for f(r) = 1/(1+r^2)."""
    print("=" * 60)
    print("TEST 5: Direct stereo optimization for smooth targets")
    print("=" * 60)

    for d, f_target, name in [
        (4, lambda r: 1.0 / (1 + r**2), "1/(1+r^2), d=4"),
        (8, lambda r: 1.0 / (1 + r**2), "1/(1+r^2), d=8"),
        (6, lambda r: np.exp(-(r**2)), "exp(-r^2), d=6"),
        (4, lambda r: 1.0 / r, "1/r, d=4"),
        (8, lambda r: 1.0 / r, "1/r, d=8"),
    ]:
        r_samples = np.linspace(0.2, 8, 50)
        phis, cost = find_phases_stereo(
            f_target, d=d, r_samples=r_samples, n_trials=30, n_workers=1, verbose=False
        )

        f_found = decoded_function(phis, r_samples).real
        f_exact = np.array([f_target(r) for r in r_samples])
        max_err = np.max(np.abs(f_found - f_exact))
        print(f"  {name}: cost = {cost:.2e}, max |f-f_target| = {max_err:.2e}")
    print()


if __name__ == "__main__":
    np.random.seed(42)
    test_1_chebyshev()
    test_2_smooth_bounded()
    test_3_stereo_base_cases()
    test_4_stereo_via_reduction()
    test_5_direct_stereo_optimization()
    print("All diagnostics complete.")
