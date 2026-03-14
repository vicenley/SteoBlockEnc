#!/usr/bin/env python3
"""
Simulation driver for Paper 2: Stereographic Block Encoding for
Multi-Qubit Eigenvalue Transformations.

Simulations:
  1. Heisenberg model verification — parameter sweep over (J, h)
  2. Gate count benchmarks — diagonal, Pauli-Z, Heisenberg
  3. Eigenvalue transformation accuracy — full QSVT for 1/λ, convergence vs degree
  4. Noise analysis — sampling noise and depolarizing noise

Usage:
  python scripts/paper2/run_simulations.py --sim 1 2 3 4
  python scripts/paper2/run_simulations.py --sim 1 --trials 100
"""

# === BLAS thread control (MUST be before numpy import) ===
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

import numpy as np
import sys
import time
import argparse
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from stereo_block_enc.numerical.qsp_phases import (
    find_phases_stereo,
    decoded_function,
    qsp_product,
    r_to_a,
)

# Import Qulacs
from qulacs import QuantumState, QuantumCircuit, Observable
from qulacs.gate import DenseMatrix

DATADIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


# ================================================================
#  Shared utility functions (adapted from qulacs_heisenberg_sim.py)
# ================================================================


def build_heisenberg_hamiltonian(J, h):
    """Build H = J(XX + YY + ZZ) + h(ZI) on 2 qubits."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    H = J * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)) + h * np.kron(Z, I2)
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
    """Permutation matrix converting standard (big-endian) to Qulacs (little-endian) ordering."""
    N = 2**n_sys_qubits
    P = np.zeros((N, N))
    for i in range(N):
        bits = [(i >> b) & 1 for b in range(n_sys_qubits - 1, -1, -1)]
        j = sum(bits[b] * (2**b) for b in range(n_sys_qubits))
        P[j, i] = 1.0
    return P


def build_qsp_circuit(n_qubits, V_diag, eigenvalues_shifted, phases):
    """
    Build the full QSP circuit on n_qubits (1 ancilla + n_sys system).

    Qubit layout: q0=ancilla, q1,...,q_{n_sys}=system.
    """
    circuit = QuantumCircuit(n_qubits)
    n_sys = n_qubits - 1
    d = len(phases) - 1
    N_sys = 2**n_sys
    N_full = 2**n_qubits

    # Permute V_diag from standard to Qulacs ordering
    P = standard_to_qulacs_perm(n_sys)
    V_q = P @ V_diag

    # Compressed eigenvalues
    a_vals = eigenvalues_shifted / np.sqrt(1.0 + eigenvalues_shifted**2)

    # Build the uniformly-controlled signal operator
    def build_uc_signal():
        mat = np.zeros((N_full, N_full), dtype=complex)
        for k in range(N_sys):
            W_k = make_signal_gate(a_vals[k])
            # Compute system qubit values for state k (Qulacs little-endian)
            sys_bits = [(k >> b) & 1 for b in range(n_sys)]
            # Full index with ancilla value a_val:
            # index = a_val + 2*sys_bits[0] + 4*sys_bits[1] + ...
            offset = sum(sys_bits[b] * (2 ** (b + 1)) for b in range(n_sys))
            base_0 = 0 + offset  # ancilla = 0
            base_1 = 1 + offset  # ancilla = 1
            mat[base_0, base_0] = W_k[0, 0]
            mat[base_0, base_1] = W_k[0, 1]
            mat[base_1, base_0] = W_k[1, 0]
            mat[base_1, base_1] = W_k[1, 1]
        return mat

    uc_signal = build_uc_signal()

    # Phase gate on ancilla (identity on system)
    def build_phase_full(phi):
        mat = np.zeros((N_full, N_full), dtype=complex)
        pg = make_phase_gate(phi)
        for k in range(N_sys):
            sys_bits = [(k >> b) & 1 for b in range(n_sys)]
            offset = sum(sys_bits[b] * (2 ** (b + 1)) for b in range(n_sys))
            base_0 = 0 + offset
            base_1 = 1 + offset
            mat[base_0, base_0] = pg[0, 0]
            mat[base_1, base_1] = pg[1, 1]
        return mat

    # System qubit indices
    sys_qubits = list(range(1, n_qubits))
    all_qubits = list(range(n_qubits))

    # Circuit construction: V†, R(φ_d), [W, R(φ_j)]_{j=d-1..0}, V
    circuit.add_gate(DenseMatrix(sys_qubits, V_q.conj().T))
    circuit.add_gate(DenseMatrix(all_qubits, build_phase_full(phases[d])))
    for j in range(d - 1, -1, -1):
        circuit.add_gate(DenseMatrix(all_qubits, uc_signal))
        circuit.add_gate(DenseMatrix(all_qubits, build_phase_full(phases[j])))
    circuit.add_gate(DenseMatrix(sys_qubits, V_q))

    return circuit


def decode_ancilla_statevector(state_vec, n_qubits, V_diag, eigenstate_idx):
    """Decode the stereographic value from statevector for a specific eigenstate."""
    n_sys = n_qubits - 1
    N_sys = 2**n_sys
    j = eigenstate_idx
    P_perm = standard_to_qulacs_perm(n_sys)
    eigvec_q = P_perm @ V_diag[:, j]

    alpha = 0.0 + 0.0j
    beta = 0.0 + 0.0j

    for k in range(N_sys):
        sys_bits = [(k >> b) & 1 for b in range(n_sys)]
        offset = sum(sys_bits[b] * (2 ** (b + 1)) for b in range(n_sys))
        idx_0 = 0 + offset
        idx_1 = 1 + offset
        alpha += eigvec_q[k].conj() * state_vec[idx_0]
        beta += eigvec_q[k].conj() * state_vec[idx_1]

    if abs(beta) < 1e-15:
        return np.inf
    return alpha / (-1j * beta)


def decode_ancilla_pauli(state, n_qubits):
    """Decode using Pauli expectations on the ancilla (q0).

    The stereographic decoded function f = alpha/(-i*beta) maps to Pauli
    expectations as Re(f) = -<Y>/(1-<Z>), Im(f) = <X>/(1-<Z>).
    This follows from the V = diag(1, i) basis change in the stereographic
    encoding.
    """
    obs_x = Observable(n_qubits)
    obs_x.add_operator(1.0, "X 0")
    obs_y = Observable(n_qubits)
    obs_y.add_operator(1.0, "Y 0")
    obs_z = Observable(n_qubits)
    obs_z.add_operator(1.0, "Z 0")

    ex = obs_x.get_expectation_value(state)
    ey = obs_y.get_expectation_value(state)
    ez = obs_z.get_expectation_value(state)

    denom = 1.0 - ez
    if abs(denom) < 1e-15:
        return np.inf
    # Stereographic decoding: Re(f) = -Y/(1-Z), Im(f) = X/(1-Z)
    return (-ey + 1j * ex) / denom


def initialize_eigenstate(n_qubits, V_diag, eigenstate_idx):
    """Create |0⟩_a ⊗ |ψ_j⟩_s in Qulacs ordering."""
    n_sys = n_qubits - 1
    N_sys = 2**n_sys
    N_full = 2**n_qubits
    P = standard_to_qulacs_perm(n_sys)
    eigvec_q = P @ V_diag[:, eigenstate_idx]

    state = QuantumState(n_qubits)
    init_vec = np.zeros(N_full, dtype=complex)
    for k in range(N_sys):
        sys_bits = [(k >> b) & 1 for b in range(n_sys)]
        offset = sum(sys_bits[b] * (2 ** (b + 1)) for b in range(n_sys))
        init_vec[0 + offset] = eigvec_q[k]  # ancilla = 0
    state.load(init_vec)
    return state


# ================================================================
#  Simulation 1: Heisenberg model verification — parameter sweep
# ================================================================


def sim1_heisenberg_verification(n_trials=100, verbose=True):
    """
    Verify stereographic block encoding on the 2-qubit Heisenberg model
    over a grid of (J, h) values. For each parameter set, build the
    circuit, run for all eigenstates, and compare decoded vs exact.
    """
    print("=" * 70)
    print("SIM 1: Heisenberg Model Verification — Parameter Sweep")
    print("=" * 70)

    J_values = [0.5, 1.0, 2.0]
    h_values = [0.0, 0.25, 0.5, 1.0, 2.0]
    degree = 14  # high enough for good 1/r approximation

    all_results = []
    n_qubits = 3

    for J in J_values:
        for h in h_values:
            if verbose:
                print(f"\n  J={J}, h={h}:")

            H = build_heisenberg_hamiltonian(J, h)
            eigenvalues, V_diag = diagonalize(H)

            # Shift to positive
            shift = -min(eigenvalues) + 0.5
            eig_shifted = eigenvalues + shift

            # Find QSP phases — use well-spaced samples covering eigenvalue range
            r_max = max(eig_shifted) + 2.0
            r_samples = np.linspace(0.2, r_max, 80)
            phases, cost = find_phases_stereo(
                f_target=lambda r: 1.0 / r,
                d=degree,
                r_samples=r_samples,
                n_trials=n_trials,
            )

            # Build circuit
            circuit = build_qsp_circuit(n_qubits, V_diag, eig_shifted, phases)

            # Run for each eigenstate
            errors_circuit_vs_classical = []
            errors_circuit_vs_exact = []
            decoded_values = []
            exact_values = []

            for j in range(4):
                state = initialize_eigenstate(n_qubits, V_diag, j)
                circuit.update_quantum_state(state)
                sv = state.get_vector()

                z_circuit = decode_ancilla_statevector(sv, n_qubits, V_diag, j)
                z_classical = decoded_function(phases, np.array([eig_shifted[j]]))[0]
                z_exact = 1.0 / eig_shifted[j]

                errors_circuit_vs_classical.append(abs(z_circuit - z_classical))
                errors_circuit_vs_exact.append(abs(z_circuit.real - z_exact))
                decoded_values.append(z_circuit.real)
                exact_values.append(z_exact)

            max_err_cc = max(errors_circuit_vs_classical)
            max_err_ce = max(errors_circuit_vs_exact)

            if verbose:
                print(f"    eigenvalues (shifted): {eig_shifted}")
                print(f"    max|circuit-classical| = {max_err_cc:.2e}")
                print(f"    max|circuit-exact| = {max_err_ce:.2e}")
                print(f"    phase cost = {cost:.2e}")

            all_results.append(
                {
                    "J": J,
                    "h": h,
                    "eigenvalues": eigenvalues,
                    "eigenvalues_shifted": eig_shifted,
                    "shift": shift,
                    "degree": degree,
                    "phase_cost": cost,
                    "decoded_values": np.array(decoded_values),
                    "exact_values": np.array(exact_values),
                    "errors_circuit_vs_classical": np.array(errors_circuit_vs_classical),
                    "errors_circuit_vs_exact": np.array(errors_circuit_vs_exact),
                }
            )

    # Save results
    os.makedirs(DATADIR, exist_ok=True)
    save_data = {}
    for i, r in enumerate(all_results):
        for key, val in r.items():
            if isinstance(val, np.ndarray):
                save_data[f"{i}_{key}"] = val
            else:
                save_data[f"{i}_{key}"] = np.array([val])
    save_data["n_results"] = np.array([len(all_results)])
    outpath = os.path.join(DATADIR, "paper2_sim1_heisenberg_verification.npz")
    np.savez(outpath, **save_data)
    print(f"\nSaved to {outpath}")

    return all_results


# ================================================================
#  Simulation 2: Gate count benchmarks
# ================================================================


def sim2_gate_counts(verbose=True):
    """
    Compare gate counts for stereographic block encoding across
    three Hamiltonian classes and compare with standard approaches.
    """
    print("\n" + "=" * 70)
    print("SIM 2: Gate Count Benchmarks")
    print("=" * 70)

    results = {}

    # --- Diagonal Hamiltonians ---
    print("\n  Class 1: Diagonal Hamiltonians")
    for n in [2, 3, 4, 5]:
        N = 2**n
        # Random positive diagonal Hamiltonian
        np.random.seed(42 + n)
        eigenvalues = np.sort(np.abs(np.random.randn(N))) + 0.5

        # Stereographic block encoding:
        # - Uniformly controlled R_y on ancilla: 2^n CNOTs + 2^n single-qubit rotations
        # - No diagonalization circuit needed (already diagonal)
        stereo_cnots = N  # UCR_y decomposition
        stereo_single = N
        stereo_ancilla = 1

        # Standard block encoding (LCU for diagonal):
        # - Prepare oracle: O(N) gates, Reflect oracle: O(N) gates
        # - Number of ancilla qubits: ceil(log2(N)) for state preparation
        # - Normalization factor alpha = sum(|eigenvalues|)
        alpha = np.sum(np.abs(eigenvalues))
        standard_ancilla = int(np.ceil(np.log2(N)))
        # LCU: prepare + select + unprepare ~ O(N) total gates
        standard_cnots = 2 * N  # conservative estimate
        standard_single = 2 * N

        if verbose:
            print(
                f"    n={n}: stereo ({stereo_cnots} CNOT, {stereo_single} 1Q, "
                f"{stereo_ancilla} anc) vs standard ({standard_cnots} CNOT, "
                f"{standard_single} 1Q, {standard_ancilla} anc, α={alpha:.1f})"
            )

        results[f"diag_n{n}"] = {
            "n_qubits": n,
            "N_dim": N,
            "stereo_cnots": stereo_cnots,
            "stereo_single": stereo_single,
            "stereo_ancilla": stereo_ancilla,
            "standard_cnots": standard_cnots,
            "standard_single": standard_single,
            "standard_ancilla": standard_ancilla,
            "alpha": alpha,
        }

    # --- Pauli-Z sums ---
    print("\n  Class 2: Pauli-Z Sum Hamiltonians")
    for m in [2, 4, 8, 16]:
        # H = sum_{k=1}^m c_k Z_{i_k}
        # Stereographic: compute eigenvalue via reversible adder + UCR_y
        # Adder: O(m) gates, UCR_y conditioned on adder output
        stereo_cnots = 3 * m  # adder + encoding + uncompute
        stereo_single = 2 * m
        stereo_ancilla = 1  # plus O(log m) for adder scratch

        # Standard block encoding (LCU):
        # Each Pauli-Z is its own unitary, so LCU with m terms
        # Select: O(m) multi-controlled gates
        # Prepare: O(m) rotations
        alpha_typical = m * 1.0  # ||H|| ~ m for unit coefficients
        standard_cnots = 4 * m
        standard_single = 4 * m
        standard_ancilla = int(np.ceil(np.log2(m)))

        if verbose:
            print(
                f"    m={m}: stereo ({stereo_cnots} CNOT, {stereo_single} 1Q, "
                f"{stereo_ancilla}+O(log m) anc) vs standard ({standard_cnots} CNOT, "
                f"{standard_single} 1Q, {standard_ancilla} anc, α~{alpha_typical:.0f})"
            )

        results[f"pauli_z_m{m}"] = {
            "m_terms": m,
            "stereo_cnots": stereo_cnots,
            "stereo_single": stereo_single,
            "stereo_ancilla": stereo_ancilla,
            "standard_cnots": standard_cnots,
            "standard_single": standard_single,
            "standard_ancilla": standard_ancilla,
            "alpha_typical": alpha_typical,
        }

    # --- Heisenberg model ---
    print("\n  Class 3: Heisenberg Model (2 qubits)")
    for J, h in [(1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (2.0, 0.5)]:
        H = build_heisenberg_hamiltonian(J, h)
        eigenvalues, V_diag = diagonalize(H)
        shift = -min(eigenvalues) + 0.5
        eig_shifted = eigenvalues + shift
        alpha = np.max(np.abs(eig_shifted))

        # Build actual circuit and count gates
        n_qubits = 3
        d_test = 10
        phases = np.random.randn(d_test + 1) * 0.1  # dummy phases for gate counting
        circuit = build_qsp_circuit(n_qubits, V_diag, eig_shifted, phases)

        # Stereographic:
        # - Givens rotation: O(1) gates
        # - UCR_y on 2 system qubits: 4 CNOTs + 4 single-qubit
        # - Total diag circuit: O(1) + 4 + 4 = O(1)
        # Per QSP step: 1 phase gate + 1 signal operator = O(1)
        # Total: O(d) + O(1) for V, V†
        stereo_depth = circuit.calculate_depth()
        stereo_gates = circuit.get_gate_count()

        # Standard block encoding:
        # LCU of 4 Pauli terms (XX + YY + ZZ + ZI)
        # Select: 4 controlled-Pauli operations ~ 16 CNOTs
        # Prepare: O(4) rotations
        # Normalization: alpha = 3J + h
        alpha_lcu = 3 * J + h
        # Per QSVT step: 2 reflections + 1 query = O(16) gates
        standard_depth_per_step = 20  # conservative estimate
        standard_total_depth = d_test * standard_depth_per_step
        standard_ancilla = 2  # log2(4 terms)

        if verbose:
            print(
                f"    J={J}, h={h}: stereo (depth={stereo_depth}, gates={stereo_gates}, "
                f"1 anc, α=N/A) vs standard (~depth={standard_total_depth}, "
                f"{standard_ancilla} anc, α={alpha_lcu:.1f})"
            )

        results[f"heisenberg_J{J}_h{h}"] = {
            "J": J,
            "h": h,
            "eigenvalues": eigenvalues,
            "stereo_depth": stereo_depth,
            "stereo_gates": stereo_gates,
            "stereo_ancilla": 1,
            "standard_depth_estimate": standard_total_depth,
            "standard_ancilla": standard_ancilla,
            "alpha_lcu": alpha_lcu,
            "d_test": d_test,
        }

    # Save results
    os.makedirs(DATADIR, exist_ok=True)
    save_data = {}
    for key, val in results.items():
        for subkey, subval in val.items():
            if isinstance(subval, np.ndarray):
                save_data[f"{key}_{subkey}"] = subval
            else:
                save_data[f"{key}_{subkey}"] = np.array([subval])
    outpath = os.path.join(DATADIR, "paper2_sim2_gate_counts.npz")
    np.savez(outpath, **save_data)
    print(f"\nSaved to {outpath}")

    return results


# ================================================================
#  Simulation 3: Eigenvalue transformation accuracy
# ================================================================


def sim3_eigenvalue_transform(n_trials=100, verbose=True):
    """
    Apply full stereographic QSVT to compute f(λ) = 1/λ on the
    Heisenberg model and measure convergence as QSP degree increases.
    """
    print("\n" + "=" * 70)
    print("SIM 3: Eigenvalue Transformation Accuracy — Convergence vs Degree")
    print("=" * 70)

    J, h = 1.0, 0.5
    H = build_heisenberg_hamiltonian(J, h)
    eigenvalues, V_diag = diagonalize(H)
    shift = -min(eigenvalues) + 0.5
    eig_shifted = eigenvalues + shift
    n_qubits = 3

    degrees = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28]
    r_max = max(eig_shifted) + 2.0
    r_samples = np.linspace(0.2, r_max, 80)

    all_errors = []  # shape: (len(degrees), 4) — max error per eigenstate
    all_decoded = []  # shape: (len(degrees), 4)
    all_exact = 1.0 / eig_shifted
    all_costs = []
    all_depths = []

    for d in degrees:
        if verbose:
            print(f"\n  Degree d={d}:")

        phases, cost = find_phases_stereo(
            f_target=lambda r: 1.0 / r,
            d=d,
            r_samples=r_samples,
            n_trials=n_trials,
        )
        all_costs.append(cost)

        circuit = build_qsp_circuit(n_qubits, V_diag, eig_shifted, phases)
        all_depths.append(circuit.calculate_depth())

        decoded_vals = []
        errors = []
        for j in range(4):
            state = initialize_eigenstate(n_qubits, V_diag, j)
            circuit.update_quantum_state(state)
            sv = state.get_vector()
            z_circuit = decode_ancilla_statevector(sv, n_qubits, V_diag, j)
            z_exact = all_exact[j]
            decoded_vals.append(z_circuit.real)
            errors.append(abs(z_circuit.real - z_exact))

        all_decoded.append(decoded_vals)
        all_errors.append(errors)

        if verbose:
            print(f"    cost={cost:.2e}, depth={all_depths[-1]}")
            for j in range(4):
                print(
                    f"    λ'={eig_shifted[j]:.3f}: "
                    f"decoded={decoded_vals[j]:.8f}, "
                    f"exact={all_exact[j]:.8f}, "
                    f"err={errors[j]:.2e}"
                )

    # Save results
    os.makedirs(DATADIR, exist_ok=True)
    outpath = os.path.join(DATADIR, "paper2_sim3_convergence.npz")
    np.savez(
        outpath,
        J=J,
        h=h,
        eigenvalues=eigenvalues,
        eigenvalues_shifted=eig_shifted,
        shift=shift,
        degrees=np.array(degrees),
        errors=np.array(all_errors),
        decoded=np.array(all_decoded),
        exact=all_exact,
        costs=np.array(all_costs),
        depths=np.array(all_depths),
    )
    print(f"\nSaved to {outpath}")

    return {
        "degrees": degrees,
        "errors": np.array(all_errors),
        "decoded": np.array(all_decoded),
        "exact": all_exact,
    }


# ================================================================
#  Simulation 4: Noise analysis
# ================================================================


def sim4_noise_analysis(n_trials=50, verbose=True):
    """
    Analyze sensitivity to:
      (a) Finite measurement statistics (sampling noise on decoded value)
      (b) Depolarizing gate noise (density matrix simulation)

    Uses eigenstate-resolved decoding throughout: the decoded value is
    extracted by projecting the output state onto the known eigenstate,
    yielding ancilla amplitudes alpha, beta with f = alpha/(-i*beta).
    """
    print("\n" + "=" * 70)
    print("SIM 4: Noise Analysis")
    print("=" * 70)

    J, h = 1.0, 0.5
    H = build_heisenberg_hamiltonian(J, h)
    eigenvalues, V_diag = diagonalize(H)
    shift = -min(eigenvalues) + 0.5
    eig_shifted = eigenvalues + shift
    n_qubits = 3
    degree = 14

    # Find phases
    r_max = max(eig_shifted) + 2.0
    r_samples = np.linspace(0.2, r_max, 80)
    phases, cost = find_phases_stereo(
        f_target=lambda r: 1.0 / r,
        d=degree,
        r_samples=r_samples,
        n_trials=n_trials,
    )

    print(f"  Phase-finding cost: {cost:.2e}")

    # Build noiseless circuit
    circuit = build_qsp_circuit(n_qubits, V_diag, eig_shifted, phases)

    # ---- Part A: Sampling noise ----
    # Model: the statevector decoding gives exact alpha_j, beta_j.
    # Finite measurement (N shots) introduces Gaussian noise on the
    # decoded real value with std ~ (1 + |f|^2) / sqrt(N), derived from
    # the stereographic error amplification factor.
    print("\n  Part A: Sampling noise (stereographic decoding)")
    shot_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
    n_repetitions = 200  # statistical repetitions for stable statistics

    sampling_results = {}
    for j_eig in [0, 1, 2, 3]:
        r_val = eig_shifted[j_eig]
        exact_val = 1.0 / r_val

        # Get exact decoded value from statevector
        state = initialize_eigenstate(n_qubits, V_diag, j_eig)
        circuit.update_quantum_state(state)
        sv = state.get_vector()
        z_exact_sv = decode_ancilla_statevector(sv, n_qubits, V_diag, j_eig)

        if verbose:
            print(
                f"\n    Eigenstate j={j_eig}, λ'={r_val:.3f}, "
                f"exact=1/λ'={exact_val:.8f}, "
                f"SV decoded={z_exact_sv.real:.8f}"
            )

        # Extract ancilla amplitudes for this eigenstate
        P_perm = standard_to_qulacs_perm(2)
        eigvec_q = P_perm @ V_diag[:, j_eig]
        alpha = 0.0 + 0.0j
        beta = 0.0 + 0.0j
        for k in range(4):
            sys_bits = [(k >> b) & 1 for b in range(2)]
            offset = sum(sys_bits[b] * (2 ** (b + 1)) for b in range(2))
            alpha += eigvec_q[k].conj() * sv[0 + offset]
            beta += eigvec_q[k].conj() * sv[1 + offset]

        # Eigenstate-projected ancilla Bloch vector
        x_j = 2 * np.real(alpha * np.conj(beta))
        y_j = 2 * np.imag(alpha * np.conj(beta))
        z_j = abs(alpha) ** 2 - abs(beta) ** 2
        # Verify: -y_j/(1-z_j) should equal the decoded value
        f_from_bloch = -y_j / (1 - z_j)

        if verbose:
            print(f"    Bloch: X={x_j:.6f}, Y={y_j:.6f}, Z={z_j:.6f}, -Y/(1-Z)={f_from_bloch:.8f}")

        errors_by_shots = []
        for N_shots in shot_counts:
            errors_rep = []
            for _ in range(n_repetitions):
                # Simulate finite measurement of the eigenstate-projected
                # Pauli expectations. Each has variance (1 - <P>^2) / N.
                x_noisy = x_j + np.random.normal(0, np.sqrt(max(1 - x_j**2, 0) / N_shots))
                y_noisy = y_j + np.random.normal(0, np.sqrt(max(1 - y_j**2, 0) / N_shots))
                z_noisy = z_j + np.random.normal(0, np.sqrt(max(1 - z_j**2, 0) / N_shots))

                denom = 1.0 - z_noisy
                if abs(denom) < 1e-15:
                    f_noisy = np.inf
                else:
                    f_noisy = -y_noisy / denom

                errors_rep.append(abs(f_noisy - exact_val))

            mean_err = np.mean(errors_rep)
            std_err = np.std(errors_rep)
            errors_by_shots.append((mean_err, std_err))

            if verbose:
                print(f"      N={N_shots:>7d}: mean_err={mean_err:.4e} ± {std_err:.4e}")

        sampling_results[f"eig{j_eig}"] = {
            "r_val": r_val,
            "exact_val": exact_val,
            "errors_by_shots": np.array(errors_by_shots),
            "x_proj": x_j,
            "y_proj": y_j,
            "z_proj": z_j,
        }

    # ---- Part B: Depolarizing noise ----
    print("\n  Part B: Depolarizing gate noise")
    noise_rates = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    from qulacs.gate import DepolarizingNoise
    from qulacs import DensityMatrix as QulacsDensityMatrix

    depol_results = {}
    for j_eig in [0, 2]:  # Test smallest and largest eigenvalues
        r_val = eig_shifted[j_eig]
        exact_val = 1.0 / r_val
        errors_by_noise = []

        P_perm = standard_to_qulacs_perm(2)
        eigvec_q = P_perm @ V_diag[:, j_eig]

        for p in noise_rates:
            if p == 0.0:
                # Noiseless reference via statevector
                state = initialize_eigenstate(n_qubits, V_diag, j_eig)
                circuit.update_quantum_state(state)
                sv = state.get_vector()
                z_dec = decode_ancilla_statevector(sv, n_qubits, V_diag, j_eig)
                errors_by_noise.append(abs(z_dec.real - exact_val))
            else:
                # Build noisy circuit
                noisy_circuit = QuantumCircuit(n_qubits)
                d_qsp = len(phases) - 1
                V_q = P_perm @ V_diag
                a_vals = eig_shifted / np.sqrt(1.0 + eig_shifted**2)

                def add_gate_with_noise(circ, gate, qubits, noise_rate):
                    circ.add_gate(gate)
                    for q in qubits:
                        circ.add_gate(DepolarizingNoise(q, noise_rate))

                # V†
                gate_vdag = DenseMatrix([1, 2], V_q.conj().T)
                add_gate_with_noise(noisy_circuit, gate_vdag, [1, 2], p)

                # Build signal and phase matrices
                N_full = 8
                N_sys = 4

                def build_uc_sig():
                    mat = np.zeros((N_full, N_full), dtype=complex)
                    for k in range(N_sys):
                        W_k = make_signal_gate(a_vals[k])
                        q1_val = k % 2
                        q2_val = k // 2
                        b0 = 0 + 2 * q1_val + 4 * q2_val
                        b1 = 1 + 2 * q1_val + 4 * q2_val
                        mat[b0, b0] = W_k[0, 0]
                        mat[b0, b1] = W_k[0, 1]
                        mat[b1, b0] = W_k[1, 0]
                        mat[b1, b1] = W_k[1, 1]
                    return mat

                def build_ph(phi):
                    mat = np.zeros((N_full, N_full), dtype=complex)
                    pg = make_phase_gate(phi)
                    for k in range(N_sys):
                        q1_val = k % 2
                        q2_val = k // 2
                        b0 = 0 + 2 * q1_val + 4 * q2_val
                        b1 = 1 + 2 * q1_val + 4 * q2_val
                        mat[b0, b0] = pg[0, 0]
                        mat[b1, b1] = pg[1, 1]
                    return mat

                uc_sig = build_uc_sig()

                # Phase d
                gate_phase = DenseMatrix([0, 1, 2], build_ph(phases[d_qsp]))
                add_gate_with_noise(noisy_circuit, gate_phase, [0, 1, 2], p)

                for jj in range(d_qsp - 1, -1, -1):
                    gate_sig = DenseMatrix([0, 1, 2], uc_sig)
                    add_gate_with_noise(noisy_circuit, gate_sig, [0, 1, 2], p)
                    gate_phase = DenseMatrix([0, 1, 2], build_ph(phases[jj]))
                    add_gate_with_noise(noisy_circuit, gate_phase, [0, 1, 2], p)

                # V
                gate_v = DenseMatrix([1, 2], V_q)
                add_gate_with_noise(noisy_circuit, gate_v, [1, 2], p)

                # Run noisy simulation with density matrix
                dm = QulacsDensityMatrix(n_qubits)
                init_vec = np.zeros(8, dtype=complex)
                for k in range(4):
                    q1v = k % 2
                    q2v = k // 2
                    idx = 2 * q1v + 4 * q2v
                    init_vec[idx] = eigvec_q[k]
                dm.load(init_vec)
                noisy_circuit.update_quantum_state(dm)

                # Decode from density matrix via eigenstate projection
                # Get the 8x8 density matrix, project onto eigenstate j
                rho = dm.get_matrix()
                alpha_dm = 0.0 + 0.0j
                beta_dm = 0.0 + 0.0j
                # Compute Tr_sys[|psi_j><psi_j| rho] projected ancilla state
                # alpha = sum_k eigvec_q[k]* <k,0|rho|alpha,beta>...
                # Simpler: compute the ancilla reduced density matrix
                # conditioned on system eigenstate j
                rho_anc = np.zeros((2, 2), dtype=complex)
                for k1 in range(4):
                    for k2 in range(4):
                        sys_bits_1 = [(k1 >> b) & 1 for b in range(2)]
                        sys_bits_2 = [(k2 >> b) & 1 for b in range(2)]
                        off_1 = sum(sys_bits_1[b] * (2 ** (b + 1)) for b in range(2))
                        off_2 = sum(sys_bits_2[b] * (2 ** (b + 1)) for b in range(2))
                        coeff = eigvec_q[k1].conj() * eigvec_q[k2]
                        for a1 in range(2):
                            for a2 in range(2):
                                rho_anc[a1, a2] += coeff * rho[a1 + off_1, a2 + off_2]

                # Normalize
                tr = np.real(rho_anc[0, 0] + rho_anc[1, 1])
                if tr > 1e-15:
                    rho_anc /= tr

                # Extract Bloch components
                x_dm = 2 * np.real(rho_anc[0, 1])
                y_dm = 2 * np.imag(rho_anc[0, 1])
                z_dm = np.real(rho_anc[0, 0] - rho_anc[1, 1])

                denom = 1.0 - z_dm
                if abs(denom) < 1e-15:
                    f_dm = np.inf
                else:
                    f_dm = -y_dm / denom
                errors_by_noise.append(abs(f_dm - exact_val))

            if verbose:
                print(f"    eig{j_eig}, p={p:.1e}: error={errors_by_noise[-1]:.4e}")

        depol_results[f"eig{j_eig}"] = {
            "r_val": r_val,
            "exact_val": exact_val,
            "errors_by_noise": np.array(errors_by_noise),
        }

    # Save results
    os.makedirs(DATADIR, exist_ok=True)
    save_data = {
        "J": np.array([J]),
        "h": np.array([h]),
        "degree": np.array([degree]),
        "shot_counts": np.array(shot_counts),
        "noise_rates": np.array(noise_rates),
        "eigenvalues_shifted": eig_shifted,
    }
    for key, val in sampling_results.items():
        save_data[f"sampling_{key}_errors"] = val["errors_by_shots"]
        save_data[f"sampling_{key}_r_val"] = np.array([val["r_val"]])
    for key, val in depol_results.items():
        save_data[f"depol_{key}_errors"] = val["errors_by_noise"]
        save_data[f"depol_{key}_r_val"] = np.array([val["r_val"]])

    outpath = os.path.join(DATADIR, "paper2_sim4_noise.npz")
    np.savez(outpath, **save_data)
    print(f"\nSaved to {outpath}")

    return {"sampling": sampling_results, "depol": depol_results}


# ================================================================
#  Main
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper 2 simulation driver")
    parser.add_argument(
        "--sim", nargs="+", type=int, default=[1, 2, 3, 4], help="Which simulations to run (1-4)"
    )
    parser.add_argument(
        "--trials", type=int, default=100, help="Phase-finding trials per parameter set"
    )
    args = parser.parse_args()

    t0 = time.time()

    for sim_num in args.sim:
        try:
            if sim_num == 1:
                sim1_heisenberg_verification(n_trials=args.trials)
            elif sim_num == 2:
                sim2_gate_counts()
            elif sim_num == 3:
                sim3_eigenvalue_transform(n_trials=args.trials)
            elif sim_num == 4:
                sim4_noise_analysis(n_trials=args.trials)
            else:
                print(f"Unknown simulation: {sim_num}")
        except Exception:
            print(f"\nERROR in simulation {sim_num}:")
            traceback.print_exc()
            print("Continuing with next simulation...\n")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
