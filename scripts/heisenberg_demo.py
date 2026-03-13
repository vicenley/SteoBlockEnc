#!/usr/bin/env python3
"""
End-to-end numerical demonstration of stereographic QSP
applied to the 2-qubit Heisenberg model.

Pipeline:
  1. Build H = J(XX + YY + ZZ) + h(Z⊗I)
  2. Diagonalize → eigenvalues λ_j
  3. Shift to make positive: H' = H + shift·I
  4. For each eigenvalue, build stereographic encoding
  5. Apply QSP with phases for f(λ) = 1/λ (inversion)
  6. Decode and compare with exact
  7. Generate figure

Output: data/sim7_heisenberg_demo.npz and ms/figures/heisenberg_demo.pdf
"""

import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stereo_block_enc.numerical.qsp_phases import (
    find_phases_stereo, decoded_function, r_to_a
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito palette
OI = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00', 'purple': '#CC79A7', 'cyan': '#56B4E9',
    'yellow': '#F0E442', 'black': '#000000',
}

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
})


def build_heisenberg(J, h):
    """Build H = J(XX + YY + ZZ) + h(Z⊗I) as 4x4 matrix."""
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return J * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)) + h * np.kron(Z, I2)


def main():
    print("=" * 60)
    print("Heisenberg Model End-to-End Demo")
    print("=" * 60)

    J, h = 1.0, 0.5
    H = build_heisenberg(J, h)

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    print(f"Eigenvalues: {eigenvalues}")

    # Shift to make positive (avoid zero eigenvalue)
    shift = -min(eigenvalues) + 0.5
    eig_shifted = eigenvalues + shift
    print(f"Shift = {shift:.4f}, Shifted eigenvalues: {eig_shifted}")

    # Target: f(r) = 1/r
    def f_target(r_arr):
        return 1.0 / r_arr

    # r samples for fitting — avoid zero, cover the eigenvalue range
    r_samples = np.linspace(0.3, max(eig_shifted) + 2.0, 60)

    degrees = [4, 8, 14]
    results = {}

    for deg in degrees:
        n_trials = 200
        print(f"\nFinding phases for d={deg} (n_trials={n_trials})...")
        phases, cost = find_phases_stereo(
            f_target, deg,
            r_samples=r_samples,
            n_trials=n_trials,
            n_workers=24,
            verbose=False,
        )
        print(f"  Best cost: {cost:.2e}")

        # Evaluate at eigenvalues
        f_vals = decoded_function(phases, eig_shifted)
        f_exact = 1.0 / eig_shifted
        errs = np.abs(f_vals.real - f_exact)
        for j in range(len(eig_shifted)):
            print(f"  λ'={eig_shifted[j]:.3f}: f_stereo={f_vals[j].real:.6f}, "
                  f"f_exact={f_exact[j]:.6f}, err={errs[j]:.2e}")

        results[deg] = {
            'phases': phases, 'cost': cost,
            'f_stereo': f_vals.real, 'f_exact': f_exact, 'errs': errs,
        }

    # Continuous evaluation for plotting
    r_cont = np.linspace(0.3, max(eig_shifted) + 2.0, 200)
    f_exact_cont = 1.0 / r_cont
    f_cont = {}
    for deg in degrees:
        f_cont[deg] = decoded_function(results[deg]['phases'], r_cont).real

    # Save data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_dict = {
        'J': J, 'h': h,
        'eigenvalues': eigenvalues, 'eigenvalues_shifted': eig_shifted,
        'shift': shift, 'r_cont': r_cont, 'f_exact_cont': f_exact_cont,
        'degrees': np.array(degrees),
    }
    for deg in degrees:
        save_dict[f'd{deg}_phases'] = results[deg]['phases']
        save_dict[f'd{deg}_cost'] = results[deg]['cost']
        save_dict[f'd{deg}_f_stereo'] = results[deg]['f_stereo']
        save_dict[f'd{deg}_errs'] = results[deg]['errs']
        save_dict[f'd{deg}_f_cont'] = f_cont[deg]
    np.savez(os.path.join(data_dir, 'sim7_heisenberg_demo.npz'), **save_dict)
    print(f"\nData saved.")

    # ============================================================
    # Generate figure
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    colors_eig = [OI['blue'], OI['orange'], OI['green'], OI['red']]

    # Panel (a): Spectrum
    ax = axes[0]
    for j in range(4):
        ax.plot([0, 1], [eigenvalues[j], eig_shifted[j]],
                'o-', color=colors_eig[j], ms=5, lw=1.2,
                label=f'$\\lambda_{j+1}={eigenvalues[j]:.1f}$')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Original', 'Shifted'], fontsize=8)
    ax.set_ylabel('Eigenvalue')
    ax.legend(fontsize=6.5, loc='upper left')
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
            fontweight='bold', fontsize=10, va='top')

    # Panel (b): Decoded function
    ax = axes[1]
    ax.plot(r_cont, f_exact_cont, 'k--', lw=1.5, label='$1/r$ (exact)')
    deg_colors = {4: OI['blue'], 8: OI['orange'], 14: OI['green']}
    for deg in degrees:
        ax.plot(r_cont, f_cont[deg], color=deg_colors[deg],
                lw=1.2, label=f'$d={deg}$')
    for j, lam in enumerate(eig_shifted):
        ax.axvline(lam, color=colors_eig[j], ls=':', lw=0.6, alpha=0.4)
    ax.set_xlabel('$r$')
    ax.set_ylabel('$f(r)$')
    ax.set_ylim(-0.5, 5)
    ax.legend(fontsize=6.5)
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes,
            fontweight='bold', fontsize=10, va='top')

    # Panel (c): Error at eigenvalues vs degree
    ax = axes[2]
    for j, lam in enumerate(eig_shifted):
        errs = [results[deg]['errs'][j] for deg in degrees]
        ax.semilogy(degrees, errs, 'o-', color=colors_eig[j],
                     ms=5, lw=1.2, label=f"$\\lambda'={lam:.1f}$")
    ax.set_xlabel('QSP degree $d$')
    ax.set_ylabel('Absolute error')
    ax.legend(fontsize=6.5)
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes,
            fontweight='bold', fontsize=10, va='top')

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '..', 'ms', 'figures',
                            'heisenberg_demo.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.close()
    print("Done!")


if __name__ == '__main__':
    main()
