"""
Generate publication figures for stereographic QSP.

Figure 4: Decoded functions at increasing degree for selected targets.
Figure 5: Convergence plot (max error vs degree) for multiple targets.

Usage:
    python scripts/generate_new_figures.py
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stereo_block_enc.numerical.qsp_phases import decoded_function, qsp_product, r_to_a
from stereo_phase_find import find_stereo_phases
import time

# ---- Publication style ----
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "text.usetex": False,
    }
)

# Okabe-Ito colorblind-safe palette
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
OI_RED = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_CYAN = "#56B4E9"
OI_YELLOW = "#F0E442"

TEXTWIDTH = 7.0  # inches (two-column)
COLWIDTH = 3.375

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "ms", "figures")
DATADIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)


# ---- Phase 1: Generate data ----


def generate_data():
    """Run phase-finding for all targets and degrees, save to .npz."""
    r_fit = np.linspace(0.3, 6, 25)
    r_plot = np.linspace(0.15, 10, 500)

    targets = {
        "1/r": lambda r: 1.0 / r,
        "1/(1+r^2)": lambda r: 1.0 / (1 + r**2),
        "exp(-r^2)": lambda r: np.exp(-(r**2)),
        "1/sqrt(1+r^2)": lambda r: 1.0 / np.sqrt(1 + r**2),
    }

    degrees = [2, 4, 6, 8, 10, 14, 18]

    results = {}

    for name, ft in targets.items():
        print(f"\n  {name}:")
        results[name] = {}
        for d in degrees:
            t0 = time.time()
            phis, cost = find_stereo_phases(ft, d, r_samples=r_fit, n_trials=80, seed=42)
            dt = time.time() - t0

            f_dec = decoded_function(phis, r_plot).real
            f_exact = ft(r_plot)

            # Internal polynomial
            a_plot = r_to_a(r_plot)
            W = qsp_product(phis, a_plot)
            P_vals = np.abs(W[:, 0, 0])

            mask = (r_plot >= 0.3) & (r_plot <= 6)
            max_err = np.max(np.abs(f_dec[mask] - f_exact[mask]))
            l2_err = np.sqrt(np.mean((f_dec[mask] - f_exact[mask]) ** 2))

            results[name][d] = {
                "phis": phis,
                "cost": cost,
                "f_dec": f_dec,
                "P_vals": P_vals,
                "max_err": max_err,
                "l2_err": l2_err,
            }
            print(
                f"    d={d:2d}: cost={cost:.2e}, max_err={max_err:.2e}, "
                f"L2={l2_err:.2e}, time={dt:.1f}s"
            )

    # Save
    save_dict = {"r_plot": r_plot, "r_fit": r_fit, "degrees": np.array(degrees)}
    for name in targets:
        for d in degrees:
            r = results[name][d]
            prefix = f"{name.replace('/', '_').replace('(', '').replace(')', '')}_d{d}"
            save_dict[f"{prefix}_phis"] = r["phis"]
            save_dict[f"{prefix}_cost"] = r["cost"]
            save_dict[f"{prefix}_f_dec"] = r["f_dec"]
            save_dict[f"{prefix}_P_vals"] = r["P_vals"]
            save_dict[f"{prefix}_max_err"] = r["max_err"]
            save_dict[f"{prefix}_l2_err"] = r["l2_err"]

    outpath = os.path.join(DATADIR, "stereo_qsp_results.npz")
    np.savez(outpath, **save_dict)
    print(f"\n  Data saved to {outpath}")

    return results, r_plot, degrees


# ---- Phase 2: Generate Figure 4 ----


def figure_inversion(results, r_plot, degrees):
    """
    Figure 4: Eigenvalue inversion 1/r.

    (a) Internal bounded polynomial |P(a~)| vs a~
    (b) Decoded f(r) vs exact 1/r at several degrees
    (c) Pointwise error |f(r) - 1/r|
    """
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4))

    name = "1/r"
    f_exact = 1.0 / r_plot
    a_plot = r_to_a(r_plot)

    show_degrees = [2, 4, 8, 14]
    colors = [OI_BLUE, OI_ORANGE, OI_GREEN, OI_RED]

    # Panel (a): |P(a~)|
    ax = axes[0]
    ax.axhline(1.0, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.fill_between([0, 1], 0, 1, color=OI_CYAN, alpha=0.08)
    for d, c in zip(show_degrees, colors):
        r = results[name][d]
        ax.plot(a_plot, r["P_vals"], color=c, lw=1.0, label=f"$d={d}$")
    # Ideal |Q| = sqrt(1-a^2) for reference (the complementary)
    ax.plot(
        a_plot,
        np.sqrt(1.0 - a_plot**2),
        color="gray",
        ls="--",
        lw=0.8,
        label=r"$\sqrt{1{-}\tilde{r}^2}$",
    )
    ax.set_xlabel(r"$\tilde{r} = r/\sqrt{1+r^2}$")
    ax.set_ylabel(r"$|P(\tilde{r})|$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontweight="bold", va="top", fontsize=10)

    # Panel (b): decoded f(r) vs 1/r
    ax = axes[1]
    ax.plot(r_plot, f_exact, "k--", lw=1.2, label="$1/r$")
    for d, c in zip(show_degrees, colors):
        r = results[name][d]
        ax.plot(r_plot, r["f_dec"], color=c, lw=1.0, label=f"$d={d}$")
    ax.set_xlabel("$r$")
    ax.set_ylabel("$f(r)$")
    ax.set_xlim(0.15, 8)
    ax.set_ylim(-0.5, 5)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontweight="bold", va="top", fontsize=10)

    # Panel (c): pointwise error
    ax = axes[2]
    for d, c in zip(show_degrees, colors):
        r = results[name][d]
        err = np.abs(r["f_dec"] - f_exact)
        ax.semilogy(r_plot, err, color=c, lw=1.0, label=f"$d={d}$")
    ax.set_xlabel("$r$")
    ax.set_ylabel("$|f(r) - 1/r|$")
    ax.set_xlim(0.15, 8)
    ax.set_ylim(1e-10, 1e1)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, fontweight="bold", va="top", fontsize=10)

    fig.tight_layout(w_pad=1.5)
    outpath = os.path.join(FIGDIR, "inversion_example.pdf")
    fig.savefig(outpath)
    print(f"  Saved {outpath}")
    plt.close(fig)


# ---- Phase 3: Generate Figure 5 ----


def figure_convergence(results, r_plot, degrees):
    """
    Figure 5: Convergence of stereographic QSP.

    (a) Decoded functions for 1/(1+r^2) at increasing degree
    (b) Max error vs degree for all four targets
    """
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))

    # Panel (a): 1/(1+r^2) decoded functions
    ax = axes[0]
    name = "1/(1+r^2)"
    f_exact = 1.0 / (1 + r_plot**2)
    ax.plot(r_plot, f_exact, "k--", lw=1.2, label=r"$1/(1{+}r^2)$")

    show_degrees = [4, 8, 14]
    colors_a = [OI_BLUE, OI_ORANGE, OI_GREEN]
    for d, c in zip(show_degrees, colors_a):
        r = results[name][d]
        ax.plot(r_plot, r["f_dec"], color=c, lw=1.0, label=f"$d={d}$")

    ax.set_xlabel("$r$")
    ax.set_ylabel("$f(r)$")
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.15, 1.1)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontweight="bold", va="top", fontsize=10)

    # Panel (b): convergence — max error vs degree
    ax = axes[1]

    target_info = [
        ("1/r", "$1/r$", OI_BLUE, "o"),
        ("1/(1+r^2)", "$1/(1{+}r^2)$", OI_ORANGE, "s"),
        ("1/sqrt(1+r^2)", "$1/\\sqrt{1{+}r^2}$", OI_GREEN, "^"),
        ("exp(-r^2)", "$e^{-r^2}$", OI_RED, "D"),
    ]

    for tname, label, color, marker in target_info:
        errs = []
        ds = []
        for d in degrees:
            if d in results[tname]:
                errs.append(results[tname][d]["max_err"])
                ds.append(d)
        ax.semilogy(ds, errs, color=color, marker=marker, markersize=4, lw=1.2, label=label)

    ax.set_xlabel("QSP degree $d$")
    ax.set_ylabel("Max error (on $[0.3, 6]$)")
    ax.set_xlim(1, 19)
    ax.set_ylim(1e-8, 1e1)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontweight="bold", va="top", fontsize=10)

    fig.tight_layout(w_pad=1.5)
    outpath = os.path.join(FIGDIR, "convergence.pdf")
    fig.savefig(outpath)
    print(f"  Saved {outpath}")
    plt.close(fig)


# ---- Main ----

if __name__ == "__main__":
    print("Phase 1: Generating data...")
    results, r_plot, degrees = generate_data()

    print("\nPhase 2: Figure 4 (inversion)...")
    figure_inversion(results, r_plot, degrees)

    print("\nPhase 3: Figure 5 (convergence)...")
    figure_convergence(results, r_plot, degrees)

    print("\nAll done.")
