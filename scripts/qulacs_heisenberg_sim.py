#!/usr/bin/env python3
"""
Quantum circuit simulation of stereographic QSP applied to the 2-qubit
Heisenberg model, using Qulacs (C++ backend).

This script builds the actual quantum circuit:
  1. Diagonalize H into the eigenbasis
  2. Apply the full QSP sequence in the diagonal basis
  3. Undiagonalize
  4. Decode the ancilla via Pauli expectations

Compares the quantum circuit output against the classical 2x2 matrix
calculation to verify exact agreement (statevector simulation).

Qubit layout:
  q0 = ancilla (stereographic encoding)
  q1, q2 = system (Heisenberg model)

Convention notes:
  - Qulacs RX/RY/RZ use the convention R(θ) = e^{+iθ/2 σ}
  - We use DenseMatrix for the signal operator to avoid sign issues
  - The standard QSP signal operator W(a) = [[a, i√(1-a²)], [i√(1-a²), a]]
  - Gate ordering: gates are added and applied left-to-right (first added = first applied)
  - Matrix product: M = G_n ⋯ G_1, so circuit order is G_1, G_2, ..., G_n
"""

import numpy as np
import sys
import os

# Thread control (before any numpy-heavy imports)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

from qulacs import QuantumState, QuantumCircuit, Observable
from qulacs.gate import DenseMatrix

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.stereo_block_enc.numerical.qsp_phases import (
    find_phases_stereo, decoded_function, qsp_product
)


def build_heisenberg_hamiltonian(J, h):
    """Build H = J(XX + YY + ZZ) + h(ZI) on 2 qubits."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    H = J * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)) + h * np.kron(Z, I)
    return H


def diagonalize(H):
    """Return eigenvalues and eigenvector matrix V such that H = V D V†."""
    eigenvalues, V = np.linalg.eigh(H)
    return eigenvalues, V


def make_phase_gate(phi):
    """Phase gate e^{iφZ} as a 2x2 matrix."""
    return np.diag([np.exp(1j * phi), np.exp(-1j * phi)])


def make_signal_gate(a):
    """Standard QSP signal unitary W(a) = [[a, i√(1-a²)], [i√(1-a²), a]]."""
    s = np.sqrt(max(1.0 - a**2, 0.0))
    return np.array([[a, 1j * s], [1j * s, a]], dtype=complex)


def standard_to_qulacs_perm(n_sys_qubits):
    """
    Permutation matrix converting standard qubit ordering to Qulacs ordering.

    Standard (numpy): index = b_{n-1} * 2^{n-1} + ... + b_0 (big-endian)
    Qulacs DenseMatrix([q0,...,q_{n-1}]): index = q0 + 2*q1 + ... (little-endian)

    For 2 qubits: swaps indices 1 (|01⟩) and 2 (|10⟩).
    """
    N = 2 ** n_sys_qubits
    P = np.zeros((N, N))
    for i in range(N):
        # Convert standard index to bit string, reverse it for Qulacs
        bits = [(i >> b) & 1 for b in range(n_sys_qubits - 1, -1, -1)]
        # Qulacs index: reverse the bit order
        j = sum(bits[b] * (2 ** b) for b in range(n_sys_qubits))
        P[j, i] = 1.0
    return P


def build_qsp_circuit(n_qubits, V_diag, eigenvalues_shifted, phases):
    """
    Build the full QSP circuit on n_qubits = 3 (1 ancilla + 2 system).

    Qubit layout: q0=ancilla, q1,q2=system.

    The QSP product is:
      M = e^{iφ₀Z} W(a) e^{iφ₁Z} W(a) ⋯ W(a) e^{iφ_dZ}

    The signal operator is uniformly controlled: for eigenvalue λ_k,
    W(ã_k) with ã_k = λ_k/√(1+λ_k²).

    IMPORTANT: Qulacs uses little-endian qubit ordering for DenseMatrix,
    so the eigenvector matrix V must be permuted to match.
    """
    circuit = QuantumCircuit(n_qubits)
    d = len(phases) - 1  # number of signal operator applications

    # Permute V_diag from standard (numpy) to Qulacs ordering
    P = standard_to_qulacs_perm(2)  # 2 system qubits
    V_q = P @ V_diag  # rows permuted for Qulacs gate convention

    # Compressed eigenvalues
    a_vals = eigenvalues_shifted / np.sqrt(1.0 + eigenvalues_shifted**2)

    # Build the uniformly-controlled signal operator (8x8 matrix)
    # For each eigenstate |k⟩ of the system, apply W(a_k) on the ancilla
    def build_uc_signal():
        mat = np.zeros((8, 8), dtype=complex)
        for k in range(4):
            W_k = make_signal_gate(a_vals[k])
            # Qubit ordering: index = q0 + 2*q1 + 4*q2
            # System state |k⟩: q1 = k % 2, q2 = k // 2
            q1_val = k % 2
            q2_val = k // 2
            base_0 = 0 + 2 * q1_val + 4 * q2_val  # ancilla=0
            base_1 = 1 + 2 * q1_val + 4 * q2_val  # ancilla=1
            mat[base_0, base_0] = W_k[0, 0]
            mat[base_0, base_1] = W_k[0, 1]
            mat[base_1, base_0] = W_k[1, 0]
            mat[base_1, base_1] = W_k[1, 1]
        return mat

    uc_signal = build_uc_signal()

    # Build the phase gate on ancilla (extended to 8x8 by identity on system)
    def build_phase_8x8(phi):
        mat = np.zeros((8, 8), dtype=complex)
        pg = make_phase_gate(phi)
        for k in range(4):
            q1_val = k % 2
            q2_val = k // 2
            base_0 = 0 + 2 * q1_val + 4 * q2_val
            base_1 = 1 + 2 * q1_val + 4 * q2_val
            mat[base_0, base_0] = pg[0, 0]
            mat[base_1, base_1] = pg[1, 1]
        return mat

    # === Build circuit ===
    # The QSP matrix product is: M = R(φ₀) W R(φ₁) W ⋯ W R(φ_d)
    # The state after the circuit is M|0⟩.
    #
    # In the multi-qubit setting with diagonalization:
    #   Full = V ⊗ I_a · [R₀ W R₁ W ⋯ R_d] · V† ⊗ I_a
    # where V† maps eigenstates to computational basis (V†|ψ_j⟩ = |j⟩),
    # the QSP operates in the computational/diagonal basis, and V maps back.
    #
    # Applied to |0⟩_a ⊗ |ψ_j⟩_s:
    #   state = V · M_diag · V† · |ψ_j⟩ = V · M_diag · |j⟩
    #   = V · [P(ã_j)|0⟩ + Q(ã_j)|1⟩] ⊗ |j⟩
    #   = [P(ã_j)|0⟩ + Q(ã_j)|1⟩] ⊗ |ψ_j⟩
    #
    # Circuit order (first gate applied first to state):
    #   V†, R(φ_d), W, R(φ_{d-1}), ..., R(φ₁), W, R(φ₀), V

    # Step 1: V_q† on system (maps eigenstate → comp basis, applied first)
    # V_q is permuted for Qulacs ordering
    circuit.add_gate(DenseMatrix([1, 2], V_q.conj().T))

    # Step 2: QSP in diagonal basis (reversed order for circuit)
    # Last phase rotation applied second
    circuit.add_gate(DenseMatrix([0, 1, 2], build_phase_8x8(phases[d])))

    for j in range(d - 1, -1, -1):
        # Signal operator
        circuit.add_gate(DenseMatrix([0, 1, 2], uc_signal))
        # Phase rotation
        circuit.add_gate(DenseMatrix([0, 1, 2], build_phase_8x8(phases[j])))

    # Step 3: V_q on system (maps comp basis → eigenstate, applied last)
    circuit.add_gate(DenseMatrix([1, 2], V_q))

    return circuit


def decode_ancilla_statevector(state_vec, n_qubits, V_diag, eigenstate_idx):
    """
    Decode the stereographic value from the ancilla qubit for a specific
    eigenstate of the system, using the statevector directly.

    After the full QSP circuit, the state is:
      |out⟩ = (P(ã)|0⟩_a + Q(ã)|1⟩_a) ⊗ |ψ_j⟩_s

    Since the system is in the eigenstate |ψ_j⟩ (a superposition of comp
    basis states), we project by taking the inner product:
      α = Σ_k V_diag[k,j]* · state_vec[idx(a=0, sys=k)]
      β = Σ_k V_diag[k,j]* · state_vec[idx(a=1, sys=k)]

    The decoded value is α / (-iβ).
    """
    j = eigenstate_idx
    P_perm = standard_to_qulacs_perm(2)
    eigvec_q = P_perm @ V_diag[:, j]  # Qulacs-ordered eigenvector

    alpha = 0.0 + 0.0j  # ancilla = |0⟩ component
    beta = 0.0 + 0.0j   # ancilla = |1⟩ component

    for k in range(4):
        q1_val = k % 2
        q2_val = k // 2
        idx_0 = 0 + 2 * q1_val + 4 * q2_val  # ancilla=0, system=|k⟩
        idx_1 = 1 + 2 * q1_val + 4 * q2_val  # ancilla=1, system=|k⟩

        alpha += eigvec_q[k].conj() * state_vec[idx_0]
        beta += eigvec_q[k].conj() * state_vec[idx_1]

    # Stereographic decoded value: P(ã) / (-i Q_std(ã))
    if abs(beta) < 1e-15:
        return np.inf
    return alpha / (-1j * beta)


def decode_ancilla_pauli(state, n_qubits):
    """
    Decode using Pauli expectations on the ancilla (q0).
    This gives the stereographic decoded value for the FULL state.
    Only valid when system is in a single eigenstate.
    """
    obs_x = Observable(n_qubits)
    obs_x.add_operator(1.0, 'X 0')
    obs_y = Observable(n_qubits)
    obs_y.add_operator(1.0, 'Y 0')
    obs_z = Observable(n_qubits)
    obs_z.add_operator(1.0, 'Z 0')

    ex = obs_x.get_expectation_value(state)
    ey = obs_y.get_expectation_value(state)
    ez = obs_z.get_expectation_value(state)

    denom = 1.0 - ez
    if abs(denom) < 1e-15:
        return np.inf
    return (ex + 1j * ey) / denom


def run_simulation(J=1.0, h=0.5, degrees=None, n_trials=200, verbose=True):
    """
    Full end-to-end quantum simulation.

    1. Build Heisenberg Hamiltonian
    2. Diagonalize and shift eigenvalues
    3. For each QSP degree:
       a. Find phases classically for target f(r) = 1/r
       b. Build quantum circuit
       c. For each eigenstate |j⟩, initialize and run circuit
       d. Decode ancilla and compare to classical prediction
    """
    if degrees is None:
        degrees = [4, 6, 8, 10, 14, 18]

    # Build Hamiltonian
    H = build_heisenberg_hamiltonian(J, h)
    eigenvalues, V_diag = diagonalize(H)

    if verbose:
        print(f"Heisenberg model: J={J}, h={h}")
        print(f"Eigenvalues: {eigenvalues}")

    # Shift to positive
    shift = -min(eigenvalues) + 0.5
    eig_shifted = eigenvalues + shift

    if verbose:
        print(f"Shift = {shift:.4f}")
        print(f"Shifted eigenvalues: {eig_shifted}")
        print()

    n_qubits = 3  # 1 ancilla + 2 system
    results = {
        'eigenvalues': eigenvalues,
        'eigenvalues_shifted': eig_shifted,
        'shift': shift,
        'V_diag': V_diag,
        'degrees': degrees,
        'J': J, 'h': h,
    }

    # r samples for phase finding
    r_max = max(eig_shifted) + 3.0
    r_samples = np.linspace(0.3, r_max, 60)

    for d in degrees:
        if verbose:
            print(f"=== Degree d={d} ===")

        # Find QSP phases classically
        phases, cost = find_phases_stereo(
            f_target=lambda r: 1.0 / r,
            d=d,
            r_samples=r_samples,
            n_trials=n_trials,
        )

        if verbose:
            print(f"  Phase-finding cost: {cost:.2e}")

        # Build quantum circuit
        circuit = build_qsp_circuit(n_qubits, V_diag, eig_shifted, phases)

        if verbose:
            print(f"  Circuit: {circuit.calculate_depth()} depth, "
                  f"{circuit.get_gate_count()} gates")

        # Run for each eigenstate
        quantum_decoded = []
        classical_decoded = []
        pauli_decoded = []
        exact_values = []

        # Permutation for initial state
        P = standard_to_qulacs_perm(2)

        for j in range(4):
            # Initialize: |0⟩_ancilla ⊗ |ψ_j⟩_system
            # |ψ_j⟩ = V_diag[:,j] in standard ordering
            # Must permute to Qulacs ordering for the 3-qubit state vector
            state = QuantumState(n_qubits)
            init_vec = np.zeros(8, dtype=complex)
            eigvec = V_diag[:, j]
            # Permute eigenvector to Qulacs ordering
            eigvec_q = P @ eigvec
            for k in range(4):
                # k is Qulacs system index: q1 = k%2, q2 = k//2
                # Full index with ancilla=0: 0 + 2*(k%2) + 4*(k//2)
                full_idx = 2 * (k % 2) + 4 * (k // 2)
                init_vec[full_idx] = eigvec_q[k]
            state.load(init_vec)

            # Apply QSP circuit
            circuit.update_quantum_state(state)

            # Decode: statevector method
            sv = state.get_vector()
            z_sv = decode_ancilla_statevector(sv, n_qubits, V_diag, j)

            # Decode: Pauli expectation method
            z_pauli = decode_ancilla_pauli(state, n_qubits)

            # Classical comparison
            z_classical = decoded_function(phases, np.array([eig_shifted[j]]))[0]
            z_exact = 1.0 / eig_shifted[j]

            quantum_decoded.append(z_sv)
            classical_decoded.append(z_classical)
            pauli_decoded.append(z_pauli)
            exact_values.append(z_exact)

            if verbose:
                err_qc = abs(z_sv - z_classical)
                err_pauli = abs(z_pauli - z_classical)
                print(f"  λ'={eig_shifted[j]:.4f}: "
                      f"circuit={z_sv.real:.10f}, "
                      f"classical={z_classical.real:.10f}, "
                      f"|Δ|={err_qc:.2e}, "
                      f"pauli_err={err_pauli:.2e}")

        results[f'd{d}'] = {
            'phases': phases,
            'cost': cost,
            'quantum_decoded': np.array(quantum_decoded),
            'classical_decoded': np.array(classical_decoded),
            'pauli_decoded': np.array(pauli_decoded),
            'exact_values': np.array(exact_values),
            'circuit_depth': circuit.calculate_depth(),
            'gate_count': circuit.get_gate_count(),
        }

        if verbose:
            print()

    return results


def generate_figure(results, outpath='ms/figures/quantum_circuit_verification.pdf'):
    """Generate publication figure comparing quantum circuit vs classical."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eig_shifted = results['eigenvalues_shifted']
    degrees = results['degrees']
    J, h = results['J'], results['h']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Panel (a): Quantum vs classical decoded values ---
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(degrees)))
    for i, d in enumerate(degrees):
        data = results[f'd{d}']
        q_vals = np.array([z.real for z in data['quantum_decoded']])
        c_vals = np.array([z.real for z in data['classical_decoded']])
        ax.scatter(c_vals, q_vals, c=[colors[i]], s=40, zorder=3,
                   label=f'd={d}', edgecolors='k', linewidths=0.5)

    # Perfect agreement line
    all_vals = []
    for d in degrees:
        data = results[f'd{d}']
        all_vals.extend([z.real for z in data['classical_decoded']])
        all_vals.extend([z.real for z in data['quantum_decoded']])
    vmin, vmax = min(all_vals), max(all_vals)
    margin = 0.1 * (vmax - vmin + 0.01)
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
            'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Classical 2x2 matrix result')
    ax.set_ylabel('Quantum circuit (Qulacs)')
    ax.set_title('(a) Quantum vs classical')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_aspect('equal')

    # --- Panel (b): Circuit–matrix discrepancy ---
    ax = axes[1]
    for j in range(4):
        errors = []
        for d in degrees:
            data = results[f'd{d}']
            err = abs(data['quantum_decoded'][j] - data['classical_decoded'][j])
            errors.append(max(err, 1e-16))
        ax.semilogy(degrees, errors, 'o-', label=f"$\\lambda'$={eig_shifted[j]:.2f}",
                    markersize=5)

    ax.set_xlabel('QSP degree $d$')
    ax.set_ylabel('$|f_{\\mathrm{circuit}} - f_{\\mathrm{matrix}}|$')
    ax.set_title('(b) Circuit vs matrix agreement')
    ax.legend(fontsize=7)
    ax.axhline(y=1e-13, color='gray', linestyle=':', alpha=0.5)
    ax.text(degrees[-1], 2e-13, 'machine prec.', fontsize=7, color='gray',
            ha='right')

    # --- Panel (c): Eigenvalue inversion accuracy ---
    ax = axes[2]
    best_d = degrees[-1]
    data = results[f'd{best_d}']
    y_q = np.array([z.real for z in data['quantum_decoded']])
    y_exact = 1.0 / eig_shifted

    x_pos = np.arange(4)
    ax.bar(x_pos - 0.15, y_exact, 0.3, label='Exact $1/\\lambda\'$',
           color='steelblue', alpha=0.8)
    ax.bar(x_pos + 0.15, y_q, 0.3, label=f'Circuit ($d={best_d}$)',
           color='coral', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{l:.2f}" for l in eig_shifted], fontsize=8)
    ax.set_xlabel("Shifted eigenvalue $\\lambda'$")
    ax.set_ylabel('Decoded value $f(\\lambda\')$')
    ax.set_title(f'(c) Eigenvalue inversion ($d={best_d}$)')
    ax.legend(fontsize=8)

    fig.suptitle(f'Qulacs statevector verification: '
                 f'$H = J(XX{{+}}YY{{+}}ZZ) + h(Z \\otimes I)$, '
                 f'$J={J}$, $h={h}$',
                 fontsize=11, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {outpath}")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Qulacs quantum circuit simulation of stereographic QSP')
    parser.add_argument('--J', type=float, default=1.0, help='Exchange coupling')
    parser.add_argument('--h_field', type=float, default=0.5,
                        help='Field strength')
    parser.add_argument('--degrees', nargs='+', type=int,
                        default=[4, 6, 8, 10, 14, 18],
                        help='QSP degrees to test')
    parser.add_argument('--trials', type=int, default=200,
                        help='Phase-finding trials')
    parser.add_argument('--no-figure', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--outpath', type=str,
                        default='ms/figures/quantum_circuit_verification.pdf',
                        help='Output figure path')
    args = parser.parse_args()

    results = run_simulation(
        J=args.J, h=args.h_field, degrees=args.degrees,
        n_trials=args.trials, verbose=True,
    )

    # Summary statistics
    print("=" * 60)
    print("SUMMARY: Quantum Circuit vs Classical Matrix Agreement")
    print("=" * 60)
    max_err = 0
    for d in args.degrees:
        data = results[f'd{d}']
        errs = [abs(data['quantum_decoded'][j] - data['classical_decoded'][j])
                for j in range(4)]
        me = max(errs)
        max_err = max(max_err, me)
        print(f"  d={d:2d}: max|circuit-matrix| = {me:.2e}  "
              f"(depth={data['circuit_depth']}, gates={data['gate_count']})")

    print(f"\nOverall max discrepancy: {max_err:.2e}")
    if max_err < 1e-10:
        print("PASS: Quantum circuit exactly reproduces classical calculation")
    else:
        print("WARNING: Discrepancy exceeds 1e-10")

    if not args.no_figure:
        generate_figure(results, outpath=args.outpath)
