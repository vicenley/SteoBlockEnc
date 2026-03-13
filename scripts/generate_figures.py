#!/usr/bin/env python3
"""
Generate publication-quality figures for the stereographic QSP paper.

Figures:
  1. Rational Chebyshev basis functions and cot_k(r) decoded functions
  2. Phase-finding demonstration: achieving target decoded functions
  3. Convergence of rational Chebyshev vs standard Chebyshev expansions

Usage:
  python scripts/generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numpy.polynomial.chebyshev import chebval

# Publication style
rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,  # Set True if LaTeX is available for nicer fonts
    'mathtext.fontset': 'cm',
})

FIGDIR = 'ms/figures'
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f']


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def TB(k, r):
    """Rational Chebyshev of first kind: T_k(r/sqrt(1+r^2))."""
    a = r / np.sqrt(1 + r**2)
    return chebval(a, np.eye(k + 1)[k])


def SB(k, r):
    """Rational Chebyshev of second kind: U_{k-1}(a) * sqrt(1-a^2)."""
    a = r / np.sqrt(1 + r**2)
    s = 1.0 / np.sqrt(1 + r**2)
    # U_{k-1}(a) via recurrence
    if k == 0:
        return np.ones_like(r)  # SB_0 = 1 by convention (not really used)
    elif k == 1:
        return s * 2 * a  # U_0(a) = 1, so SB_1 = s
    # Actually U_{k-1} via Chebyshev U
    # sin(k * arccos(a)) / sin(arccos(a)) = sin(k*theta)/sin(theta) = U_{k-1}
    theta = np.arccos(np.clip(a, -1, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        Uk = np.where(np.abs(np.sin(theta)) > 1e-15,
                       np.sin(k * theta) / np.sin(theta),
                       k * np.cos(k * theta) / np.cos(theta))  # L'Hopital at theta=0
    return Uk * s


def cot_k(k, r):
    """cot(k * arctan(1/r))."""
    return 1.0 / np.tan(k * np.arctan(1.0 / r))


# ---------------------------------------------------------------------------
# Figure 1: Rational Chebyshev basis and decoded cot functions
# ---------------------------------------------------------------------------

def figure_1():
    """Rational Chebyshev basis and decoded cot_k functions."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    r = np.linspace(-8, 8, 500)
    r_pos = np.linspace(0.05, 8, 500)

    # Panel (a): TB_k(r) for k = 0, 1, 2, 3, 4
    ax = axes[0]
    for k in range(5):
        ax.plot(r, TB(k, r), color=COLORS[k], linewidth=0.9,
                label=f'$k={k}$')
    ax.set_xlabel('$r$')
    ax.set_ylabel('$\\mathrm{TB}_k(r)$')
    ax.set_title('(a) Rational Chebyshev $\\mathrm{TB}_k$')
    ax.set_ylim(-1.3, 1.3)
    ax.legend(ncol=2, loc='lower left', frameon=False, fontsize=7)
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)

    # Panel (b): SB_k(r) for k = 1, 2, 3, 4
    ax = axes[1]
    for k in range(1, 5):
        ax.plot(r, SB(k, r), color=COLORS[k], linewidth=0.9,
                label=f'$k={k}$')
    ax.set_xlabel('$r$')
    ax.set_ylabel('$\\mathrm{SB}_k(r)$')
    ax.set_title('(b) Rational Chebyshev $\\mathrm{SB}_k$')
    ax.set_ylim(-1.3, 1.3)
    ax.legend(ncol=2, loc='lower left', frameon=False, fontsize=7)
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)

    # Panel (c): cot_k(r) = TB_k / SB_k, the decoded functions
    ax = axes[2]
    for k in [2, 3, 4, 5]:
        y = cot_k(k, r_pos)
        # Clip for visibility near poles
        y_clip = np.where(np.abs(y) < 15, y, np.nan)
        ax.plot(r_pos, y_clip, color=COLORS[k - 2], linewidth=0.9,
                label=f'$k={k}$')
    ax.set_xlabel('$r$')
    ax.set_ylabel('$z_k(r)$')
    ax.set_title('(c) Decoded: $\\cot(k\\arctan(1/r))$')
    ax.set_ylim(-12, 12)
    ax.legend(loc='upper right', frameon=False, fontsize=7)
    ax.axhline(0, color='gray', linewidth=0.3)

    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/rational_chebyshev.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGDIR}/rational_chebyshev.png', bbox_inches='tight')
    print('Figure 1 saved: rational_chebyshev.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Phase-finding demonstration
# ---------------------------------------------------------------------------

def figure_2():
    """Phase-finding: achieving target decoded functions via QSP."""
    from stereo_block_enc.numerical.qsp_phases import (
        find_phases_stereo, decoded_function, qsp_product, r_to_a
    )

    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    r_fine = np.linspace(0.15, 8, 300)
    r_sample = np.linspace(0.2, 8, 40)

    # Panel (a): cot(C + d*arctan(1/r)) for various C, d=2
    ax = axes[0]
    for C, ls, lbl in [(0, '-', '$C=0$'), (0.5, '--', '$C=\\pi/6$'),
                        (-0.3, '-.', '$C=-0.3$')]:
        target = lambda r, c=C: 1.0 / np.tan(c + 2 * np.arctan(1.0 / r))
        y = target(r_fine)
        y_clip = np.where(np.abs(y) < 15, y, np.nan)
        ax.plot(r_fine, y_clip, ls, color=COLORS[0], linewidth=0.9, alpha=0.4)

        # Find phases
        phis, cost = find_phases_stereo(target, d=2, r_samples=r_sample, n_trials=40)
        f_found = decoded_function(phis, r_fine)
        f_clip = np.where(np.abs(f_found.real) < 15, f_found.real, np.nan)
        ax.plot(r_fine, f_clip, 'o', color=COLORS[1], markersize=1.5, alpha=0.6)

    ax.set_xlabel('$r$')
    ax.set_ylabel('$f(r)$')
    ax.set_title('(a) $d=2$: $\\cot(C + 2\\arctan(1/r))$')
    ax.set_ylim(-12, 12)
    ax.axhline(0, color='gray', linewidth=0.3)

    # Panel (b): Higher-degree base cases cot(k*arctan(1/r)) recovered by phase-finding
    ax = axes[1]
    for k, col in [(3, COLORS[0]), (5, COLORS[2]), (7, COLORS[3])]:
        target = lambda r, kk=k: 1.0 / np.tan(kk * np.arctan(1.0 / r))
        y = target(r_fine)
        y_clip = np.where(np.abs(y) < 15, y, np.nan)
        ax.plot(r_fine, y_clip, '-', color=col, linewidth=0.9, alpha=0.4,
                label=f'$k={k}$ target')

        phis, cost = find_phases_stereo(target, d=k, r_samples=r_sample, n_trials=60)
        f_found = decoded_function(phis, r_fine)
        f_clip = np.where(np.abs(f_found.real) < 15, f_found.real, np.nan)
        ax.plot(r_fine, f_clip, '.', color=col, markersize=1, alpha=0.7)
        print(f'  k={k}: cost={cost:.2e}')

    ax.set_xlabel('$r$')
    ax.set_ylabel('$z_k(r)$')
    ax.set_title('(b) Phase-finding: base cases')
    ax.set_ylim(-12, 12)
    ax.legend(loc='upper right', frameon=False, fontsize=7)
    ax.axhline(0, color='gray', linewidth=0.3)

    # Panel (c): Phase-finding for non-trivial targets via standard QSP
    # Target: degree-d Chebyshev expansion of sign(r-2) (step function at r=2)
    # We approximate via Chebyshev in the a-variable, then decode.
    ax = axes[2]

    # Target in a-domain: step at a_0 = 2/sqrt(5) ≈ 0.894
    a0 = 2.0 / np.sqrt(5.0)
    a_fine = r_to_a(r_fine)

    for d, col in [(5, COLORS[0]), (11, COLORS[1]), (21, COLORS[2])]:
        # Chebyshev approximation of sign(a - a0) on [-1, 1]
        # Use direct optimization for P(a) with |P| <= 1
        N_cheb = 200
        a_nodes = np.cos((2 * np.arange(N_cheb) + 1) / (2 * N_cheb) * np.pi)
        target_vals = np.tanh(20 * (a_nodes - a0))  # smooth step, |target| < 1

        phis, cost = find_phases_stereo(
            lambda r: np.tanh(20 * (r / np.sqrt(1 + r**2) - a0)),
            d=d, r_samples=r_sample, n_trials=80
        )
        f_found = decoded_function(phis, r_fine)
        f_clip = np.where(np.abs(f_found.real) < 15, f_found.real, np.nan)
        ax.plot(r_fine, f_clip, '-', color=col, linewidth=0.9,
                label=f'$d={d}$')
        print(f'  step approx d={d}: cost={cost:.2e}')

    # Plot the target
    target_fine = np.tanh(20 * (a_fine - a0))
    # Decode the target: it's bounded, so decode just plots the same (no poles)
    ax.plot(r_fine, target_fine, 'k--', linewidth=0.7, alpha=0.5, label='target')

    ax.set_xlabel('$r$')
    ax.set_ylabel('$f(r)$')
    ax.set_title('(c) Step-function approx.')
    ax.set_ylim(-2, 2)
    ax.legend(loc='lower right', frameon=False, fontsize=7)
    ax.axhline(0, color='gray', linewidth=0.3)

    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/phase_finding.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGDIR}/phase_finding.png', bbox_inches='tight')
    print('Figure 2 saved: phase_finding.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Convergence comparison
# ---------------------------------------------------------------------------

def figure_3():
    """Convergence of rational Chebyshev vs standard Chebyshev."""
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.6))

    # Target function: f(r) = 1/(1 + r^2) (Lorentzian, decays algebraically)
    # This is a classic test case from Boyd (1987).
    # In the a-variable: a = r/sqrt(1+r^2), so 1+r^2 = 1/(1-a^2),
    # and f = 1-a^2. That's just a degree-2 polynomial in a — trivial!
    # Let's use a harder target: f(r) = sech(r) = 2/(e^r + e^{-r})
    # This decays exponentially, so both bases should converge well,
    # but the rational Chebyshev should need fewer terms for algebraically
    # decaying functions.
    #
    # Better target: f(r) = 1/(1 + r^4) — algebraic decay, not a rational
    # function of r/sqrt(1+r^2).

    # Panel (a): Expansion coefficients
    ax = axes[0]

    def compute_rational_cheb_coeffs(f, K):
        """Compute rational Chebyshev expansion coefficients of f(r)."""
        # Use numerical integration: c_k = (2/pi) * int f(r) TB_k(r) / (1+r^2) dr
        # (c_0 has factor 1/pi instead)
        from scipy.integrate import quad
        coeffs = []
        for k in range(K):
            integrand = lambda r: f(r) * TB(k, r) / (1 + r**2)
            val, _ = quad(integrand, -100, 100, limit=200)
            if k == 0:
                val /= np.pi
            else:
                val *= 2.0 / np.pi
            coeffs.append(val)
        return np.array(coeffs)

    def compute_standard_cheb_coeffs(g, K):
        """Compute standard Chebyshev expansion coefficients of g(a) on [-1,1]."""
        from scipy.integrate import quad
        coeffs = []
        for k in range(K):
            integrand = lambda a: g(a) * np.cos(k * np.arccos(a)) / np.sqrt(1 - a**2)
            val, _ = quad(integrand, -1 + 1e-10, 1 - 1e-10, limit=200)
            if k == 0:
                val /= np.pi
            else:
                val *= 2.0 / np.pi
            coeffs.append(val)
        return np.array(coeffs)

    K = 30

    # Target 1: f(r) = 1/(1 + r^4) — algebraic decay O(1/r^4)
    f1 = lambda r: 1.0 / (1.0 + r**4)
    # In a-variable: g(a) = f(a/sqrt(1-a^2)) = (1-a^2)^2 / (1-a^2)^2 + a^4)
    #                     = (1-a^2)^2 / (1 - 2a^2 + 2a^4)  [messy but finite on [-1,1]]
    g1 = lambda a: (1 - a**2)**2 / ((1 - a**2)**2 + a**4) if abs(a) < 1 - 1e-10 else 0.0

    c_rat1 = compute_rational_cheb_coeffs(f1, K)
    c_std1 = compute_standard_cheb_coeffs(g1, K)

    ax.semilogy(range(K), np.abs(c_rat1) + 1e-18, 'o-', color=COLORS[0],
                markersize=3, linewidth=0.8, label='Rational Cheb.')
    ax.semilogy(range(K), np.abs(c_std1) + 1e-18, 's--', color=COLORS[1],
                markersize=3, linewidth=0.8, label='Standard Cheb.')
    ax.set_xlabel('$k$')
    ax.set_ylabel('$|c_k|$')
    ax.set_title('(a) $f(r) = 1/(1+r^4)$')
    ax.legend(frameon=False)
    ax.set_ylim(1e-16, 2)

    # Panel (b): Truncation error vs degree
    ax = axes[1]

    # Compute partial-sum approximation error on a fine grid
    r_test = np.linspace(-10, 10, 500)
    a_test = r_test / np.sqrt(1 + r_test**2)

    max_K = 25
    err_rat = []
    err_std = []

    for Ktrunc in range(1, max_K + 1):
        # Rational Chebyshev partial sum
        f_approx_rat = np.zeros_like(r_test)
        for k in range(Ktrunc):
            f_approx_rat += c_rat1[k] * TB(k, r_test)
        err_r = np.max(np.abs(f_approx_rat - f1(r_test)))
        err_rat.append(err_r)

        # Standard Chebyshev partial sum in a-variable, evaluated at corresponding a
        f_approx_std = np.zeros_like(a_test)
        for k in range(Ktrunc):
            f_approx_std += c_std1[k] * np.cos(k * np.arccos(np.clip(a_test, -1, 1)))
        f_target_a = np.array([g1(a) for a in a_test])
        err_s = np.max(np.abs(f_approx_std - f_target_a))
        err_std.append(err_s)

    ax.semilogy(range(1, max_K + 1), err_rat, 'o-', color=COLORS[0],
                markersize=3, linewidth=0.8, label='Rational Cheb.')
    ax.semilogy(range(1, max_K + 1), err_std, 's--', color=COLORS[1],
                markersize=3, linewidth=0.8, label='Standard Cheb.')
    ax.set_xlabel('Truncation degree $K$')
    ax.set_ylabel('$\\|f - f_K\\|_\\infty$')
    ax.set_title('(b) Approximation error')
    ax.legend(frameon=False)
    ax.set_ylim(1e-16, 2)

    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/convergence.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGDIR}/convergence.png', bbox_inches='tight')
    print('Figure 3 saved: convergence.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Generating figures...')
    print()
    print('--- Figure 1: Rational Chebyshev basis ---')
    figure_1()
    print()
    print('--- Figure 2: Phase-finding demonstration ---')
    figure_2()
    print()
    print('--- Figure 3: Convergence comparison ---')
    figure_3()
    print()
    print('All figures generated.')
