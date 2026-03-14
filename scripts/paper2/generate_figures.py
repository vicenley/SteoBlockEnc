#!/usr/bin/env python3
"""
Generate publication figures for Paper 2 from saved simulation data.

Figures:
  1. Heisenberg verification: decoded vs exact across (J, h) parameter sweep
  2. Gate count comparison table (printed, not plotted)
  3. Convergence of eigenvalue transformation vs QSP degree
  4. Noise analysis: (a) sampling noise, (b) depolarizing noise

Usage:
  python scripts/paper2/generate_figures.py
"""

import os
import sys
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# Publication style
COLWIDTH = 3.4  # inches (PRA single column)
TEXTWIDTH = 7.0  # inches (PRA full width)

# Color palette (Okabe-Ito inspired, matching Paper 1)
BLUE = "#1A5276"
RED = "#C0392B"
TEAL = "#117A65"
DARKGRAY = "#444444"
MEDGRAY = "#777777"
ORANGE = "#D35400"
PURPLE = "#7D3C98"

COLORS = [BLUE, RED, TEAL, ORANGE, PURPLE, DARKGRAY]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    }
)

DATADIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "..", "ms2", "figures")


def load_data(filename):
    """Load .npz data file."""
    path = os.path.join(DATADIR, filename)
    if not os.path.exists(path):
        print(f"Data file not found: {path}")
        print("Run the simulations first: python scripts/paper2/run_simulations.py")
        return None
    return np.load(path, allow_pickle=True)


# ================================================================
#  Figure 1: Heisenberg Verification
# ================================================================


def figure_heisenberg_verification():
    """
    Figure: Decoded eigenvalue inversion vs exact for various (J, h).
    Two panels: (a) scatter plot of decoded vs exact, (b) error heatmap.
    """
    data = load_data("paper2_sim1_heisenberg_verification.npz")
    if data is None:
        return

    n_results = int(data["n_results"][0])

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.45 * TEXTWIDTH))

    # Panel (a): Scatter plot — all decoded vs exact values
    ax = axes[0]
    all_decoded = []
    all_exact = []
    all_labels = []

    for i in range(n_results):
        J = float(data[f"{i}_J"][0])
        h = float(data[f"{i}_h"][0])
        decoded = data[f"{i}_decoded_values"]
        exact = data[f"{i}_exact_values"]
        all_decoded.extend(decoded)
        all_exact.extend(exact)

    all_decoded = np.array(all_decoded)
    all_exact = np.array(all_exact)

    ax.scatter(all_exact, all_decoded, s=12, c=BLUE, alpha=0.7, edgecolors="none", zorder=3)

    # Perfect agreement line
    vmin = min(all_exact.min(), all_decoded.min())
    vmax = max(all_exact.max(), all_decoded.max())
    margin = 0.05 * (vmax - vmin)
    ax.plot(
        [vmin - margin, vmax + margin],
        [vmin - margin, vmax + margin],
        "k--",
        alpha=0.4,
        linewidth=0.8,
    )
    ax.set_xlabel(r"Exact $1/\lambda^\prime$")
    ax.set_ylabel(r"Decoded $f(\lambda^\prime)$")
    ax.set_title(r"(a) Circuit vs exact ($d=14$)")
    ax.set_aspect("equal")

    # Panel (b): Max error for each (J, h) pair
    ax = axes[1]
    J_values = sorted(set(float(data[f"{i}_J"][0]) for i in range(n_results)))
    h_values = sorted(set(float(data[f"{i}_h"][0]) for i in range(n_results)))

    error_grid = np.zeros((len(J_values), len(h_values)))
    for i in range(n_results):
        J = float(data[f"{i}_J"][0])
        h = float(data[f"{i}_h"][0])
        errs = data[f"{i}_errors_circuit_vs_exact"]
        ji = J_values.index(J)
        hi = h_values.index(h)
        error_grid[ji, hi] = np.max(errs)

    im = ax.imshow(
        np.log10(error_grid + 1e-16),
        aspect="auto",
        cmap="RdYlGn_r",
        origin="lower",
        vmin=-12,
        vmax=-1,
    )
    ax.set_xticks(range(len(h_values)))
    ax.set_xticklabels([f"{hv:.2g}" for hv in h_values])
    ax.set_yticks(range(len(J_values)))
    ax.set_yticklabels([f"{Jv:.1g}" for Jv in J_values])
    ax.set_xlabel(r"Field $h$")
    ax.set_ylabel(r"Coupling $J$")
    ax.set_title(r"(b) $\log_{10}$ max error")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$\log_{10}|\mathrm{error}|$", fontsize=8)

    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    outpath = os.path.join(FIGDIR, "heisenberg_verification.pdf")
    plt.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()


# ================================================================
#  Figure 2: Convergence vs Degree
# ================================================================


def figure_convergence():
    """
    Figure: Eigenvalue transformation error vs QSP degree for each eigenstate.
    """
    data = load_data("paper2_sim3_convergence.npz")
    if data is None:
        return

    degrees = data["degrees"]
    errors = data["errors"]  # shape: (n_degrees, 4)
    eig_shifted = data["eigenvalues_shifted"]
    depths = data["depths"]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.42 * TEXTWIDTH))

    # Panel (a): Error vs degree for each eigenstate
    ax = axes[0]
    for j in range(4):
        errs = errors[:, j]
        # Replace zeros with small value for log plot
        errs = np.maximum(errs, 1e-16)
        ax.semilogy(
            degrees,
            errs,
            "o-",
            color=COLORS[j],
            markersize=4,
            linewidth=1.2,
            label=rf"$\lambda'={eig_shifted[j]:.2f}$",
        )

    ax.set_xlabel(r"QSP degree $d$")
    ax.set_ylabel(r"$|f_{\mathrm{decoded}} - 1/\lambda^\prime|$")
    ax.set_title(r"(a) Convergence vs degree")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=1e-15)
    ax.grid(True, alpha=0.2)

    # Panel (b): Max error vs circuit depth
    ax = axes[1]
    max_errors = np.max(errors, axis=1)
    max_errors = np.maximum(max_errors, 1e-16)
    ax.semilogy(depths, max_errors, "s-", color=BLUE, markersize=5, linewidth=1.2)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel(r"Max $|f - 1/\lambda^\prime|$")
    ax.set_title(r"(b) Max error vs depth")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    outpath = os.path.join(FIGDIR, "convergence_vs_degree.pdf")
    plt.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()


# ================================================================
#  Figure 3: Noise Analysis
# ================================================================


def figure_noise():
    """
    Figure: (a) Sampling noise — error vs shot count
            (b) Depolarizing noise — error vs noise rate
    """
    data = load_data("paper2_sim4_noise.npz")
    if data is None:
        return

    shot_counts = data["shot_counts"]
    noise_rates = data["noise_rates"]
    eig_shifted = data["eigenvalues_shifted"]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.42 * TEXTWIDTH))

    # Panel (a): Sampling noise
    ax = axes[0]
    for idx, j_eig in enumerate([0, 1, 2, 3]):
        key = f"sampling_eig{j_eig}_errors"
        if key not in data:
            continue
        errors = data[key]  # shape: (n_shots, 2) — (mean, std)
        mean_err = errors[:, 0]
        std_err = errors[:, 1]
        r_val = float(data[f"sampling_eig{j_eig}_r_val"][0])
        ax.loglog(
            shot_counts,
            mean_err,
            "o-",
            color=COLORS[idx],
            markersize=4,
            linewidth=1.2,
            label=rf"$\lambda'={r_val:.2f}$",
        )
        ax.fill_between(
            shot_counts, mean_err - std_err, mean_err + std_err, alpha=0.15, color=COLORS[idx]
        )

    # Reference line: 1/sqrt(N)
    N_ref = np.array(shot_counts, dtype=float)
    scale = mean_err[0] * np.sqrt(shot_counts[0])
    ax.loglog(
        N_ref,
        scale / np.sqrt(N_ref),
        "--",
        color=MEDGRAY,
        linewidth=0.8,
        label=r"$\propto 1/\sqrt{N}$",
    )

    ax.set_xlabel(r"Number of shots $N$")
    ax.set_ylabel(r"Mean decoded error")
    ax.set_title(r"(a) Sampling noise")
    ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
    ax.grid(True, alpha=0.2, which="both")

    # Panel (b): Depolarizing noise
    ax = axes[1]
    for idx, j_eig in enumerate([0, 2]):
        key = f"depol_eig{j_eig}_errors"
        if key not in data:
            continue
        errors = data[key]
        r_val = float(data[f"depol_eig{j_eig}_r_val"][0])
        # Plot only nonzero noise rates (skip p=0 noiseless reference)
        nr = noise_rates[1:]  # skip p=0
        er = errors[1:]
        noiseless = errors[0]
        ax.loglog(
            nr,
            er,
            "s-",
            color=COLORS[idx],
            markersize=5,
            linewidth=1.2,
            label=rf"$\lambda'={r_val:.2f}$",
        )
        ax.axhline(y=noiseless, color=COLORS[idx], linestyle=":", alpha=0.4)

    ax.set_xlabel(r"Depolarizing rate $p$")
    ax.set_ylabel(r"Decoded error")
    ax.set_title(r"(b) Gate noise")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    outpath = os.path.join(FIGDIR, "noise_analysis.pdf")
    plt.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()


# ================================================================
#  Gate count table (LaTeX)
# ================================================================


def print_gate_count_table():
    """Print gate count comparison as LaTeX table for inclusion in the manuscript."""
    data = load_data("paper2_sim2_gate_counts.npz")
    if data is None:
        return

    print("\n% LaTeX table: Gate count comparison")
    print("% Paste into ms2/sections/07_numerics.tex")
    print(r"\begin{table*}")
    print(r"\caption{Gate count comparison for stereographic vs.\ standard block encoding.")
    print(r"CNOT and single-qubit (1Q) gate counts are shown alongside ancilla qubit")
    print(r"requirements and normalization factor $\alpha$.}")
    print(r"\label{tab:gate-counts}")
    print(r"\begin{ruledtabular}")
    print(r"\begin{tabular}{llcccccccc}")
    print(r" & & \multicolumn{4}{c}{Stereographic} & \multicolumn{4}{c}{Standard} \\")
    print(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    print(
        r"Class & Params & CNOT & 1Q & Anc.\ & $\alpha$ "
        r"& CNOT & 1Q & Anc.\ & $\alpha$ \\"
    )
    print(r"\hline")

    # Diagonal
    for n in [2, 3, 4, 5]:
        key = f"diag_n{n}"
        sc = int(data[f"{key}_stereo_cnots"][0])
        ss = int(data[f"{key}_stereo_single"][0])
        sa = int(data[f"{key}_stereo_ancilla"][0])
        tc = int(data[f"{key}_standard_cnots"][0])
        ts = int(data[f"{key}_standard_single"][0])
        ta = int(data[f"{key}_standard_ancilla"][0])
        alpha = float(data[f"{key}_alpha"][0])
        print(
            rf"Diagonal & $n={n}$ & {sc} & {ss} & {sa} & --- "
            rf"& {tc} & {ts} & {ta} & {alpha:.1f} \\"
        )

    # Pauli-Z
    for m in [2, 4, 8, 16]:
        key = f"pauli_z_m{m}"
        sc = int(data[f"{key}_stereo_cnots"][0])
        ss = int(data[f"{key}_stereo_single"][0])
        sa = int(data[f"{key}_stereo_ancilla"][0])
        tc = int(data[f"{key}_standard_cnots"][0])
        ts = int(data[f"{key}_standard_single"][0])
        ta = int(data[f"{key}_standard_ancilla"][0])
        alpha = float(data[f"{key}_alpha_typical"][0])
        print(
            rf"$Z$-sum & $m={m}$ & {sc} & {ss} & {sa} & --- "
            rf"& {tc} & {ts} & {ta} & {alpha:.0f} \\"
        )

    # Heisenberg
    for J, h in [(1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (2.0, 0.5)]:
        key = f"heisenberg_J{J}_h{h}"
        sd = int(data[f"{key}_stereo_depth"][0])
        sg = int(data[f"{key}_stereo_gates"][0])
        sa = int(data[f"{key}_stereo_ancilla"][0])
        td = int(data[f"{key}_standard_depth_estimate"][0])
        ta = int(data[f"{key}_standard_ancilla"][0])
        alpha = float(data[f"{key}_alpha_lcu"][0])
        print(
            rf"Heisenberg & $J\!={J:.0f},h\!={h:.1f}$ & \multicolumn{{2}}{{c}}"
            rf"{{depth {sd}}} & {sa} & --- "
            rf"& \multicolumn{{2}}{{c}}{{$\sim${td}}} & {ta} & {alpha:.1f} \\"
        )

    print(r"\end{tabular}")
    print(r"\end{ruledtabular}")
    print(r"\end{table*}")


# ================================================================
#  Figure 4: Total Cost Comparison (Sim 5)
# ================================================================


def figure_total_cost():
    """
    Figure: Comparison of stereographic vs standard QSVT for diagonal
    Hamiltonians.  Three panels:
      (a) Circuit depth (total gates) vs condition number κ
      (b) Total cost (gates × shots) vs κ
      (c) Coherence-limited feasibility: maximum accessible κ vs depth budget
    """
    data = load_data("paper2_sim5_total_cost.npz")
    if data is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 0.38 * TEXTWIDTH))

    # --- Panel (a): Circuit depth vs κ (Part A data) ---
    ax = axes[0]
    kappa_a = data["partA_kappa"]
    gates_std_a = data["partA_total_gates_std"]
    gates_ste_a = data["partA_total_gates_ste"]

    ax.loglog(
        kappa_a,
        gates_std_a,
        "s-",
        color=RED,
        markersize=5,
        linewidth=1.2,
        label="Standard QSVT",
    )
    ax.loglog(
        kappa_a,
        gates_ste_a,
        "o-",
        color=BLUE,
        markersize=5,
        linewidth=1.2,
        label="Stereographic",
    )

    ax.set_xlabel(r"Condition number $\kappa$")
    ax.set_ylabel(r"Total gates")
    ax.set_title(r"(a) Circuit depth")
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.2, which="both")

    # Annotate the depth ratio at largest κ
    max_idx = np.argmax(kappa_a)
    ratio = gates_std_a[max_idx] / gates_ste_a[max_idx]
    ax.annotate(
        rf"${ratio:.0f}\times$",
        xy=(kappa_a[max_idx], np.sqrt(gates_std_a[max_idx] * gates_ste_a[max_idx])),
        fontsize=7,
        ha="right",
        color=DARKGRAY,
    )

    # --- Panel (b): Total cost vs κ ---
    ax = axes[1]
    cost_std_a = data["partA_total_cost_std"]
    cost_ste_a = data["partA_total_cost_ste"]

    ax.loglog(
        kappa_a,
        cost_std_a,
        "s-",
        color=RED,
        markersize=5,
        linewidth=1.2,
        label="Standard QSVT",
    )
    ax.loglog(
        kappa_a,
        cost_ste_a,
        "o-",
        color=BLUE,
        markersize=5,
        linewidth=1.2,
        label="Stereographic",
    )

    ax.set_xlabel(r"Condition number $\kappa$")
    ax.set_ylabel(r"Total cost (gates $\times$ shots)")
    ax.set_title(r"(b) Total cost")
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.2, which="both")

    # --- Panel (c): Feasibility diagram ---
    ax = axes[2]
    budgets = data["partD_depth_budget"]
    kappa_max_std = data["partD_kappa_max_std"]

    # Stereographic is feasible for any κ when budget ≥ ~66 gates
    stereo_min_budget = float(data["partD_stereo_total_gates"][0])

    ax.semilogx(
        budgets,
        kappa_max_std,
        "s-",
        color=RED,
        markersize=5,
        linewidth=1.2,
        label="Standard QSVT",
    )

    # Stereographic: any κ is feasible above threshold
    ax.axhline(y=100, color=BLUE, linestyle="-", linewidth=1.2, alpha=0.6)
    ax.fill_between(
        budgets,
        [100 if b >= stereo_min_budget else 0 for b in budgets],
        alpha=0.12,
        color=BLUE,
    )
    ax.annotate(
        r"Stereo: any $\kappa$",
        xy=(200, 85),
        fontsize=7,
        color=BLUE,
        ha="left",
    )

    ax.set_xlabel("Gate budget")
    ax.set_ylabel(r"Max accessible $\kappa$")
    ax.set_title(r"(c) Coherence-limited feasibility")
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0, top=110)

    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    outpath = os.path.join(FIGDIR, "total_cost_comparison.pdf")
    plt.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()


# ================================================================
#  Main
# ================================================================

if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)

    print("Generating Paper 2 figures from saved data...\n")

    figure_heisenberg_verification()
    figure_convergence()
    figure_noise()
    figure_total_cost()
    print_gate_count_table()

    print("\nDone. Figures saved to", FIGDIR)
