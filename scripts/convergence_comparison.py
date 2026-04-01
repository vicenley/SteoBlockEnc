#!/usr/bin/env python3
"""
Convergence comparison: rational Chebyshev (stereographic) vs
standard Chebyshev (truncated domain) approximations.

For each target function, computes:
  Top row:    Fourier coefficients of the stereographic pullback
  Bottom row: L∞ approximation error vs polynomial degree

Produces:
  data/convergence_comparison.npz
  ms/figures/convergence_comparison.pdf

Usage:
  python scripts/convergence_comparison.py [--plot] [--dmax 60]
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG  = ROOT / "ms" / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# ── target functions ──────────────────────────────────────────

def lorentzian(y):
    return 1.0 / (1.0 + y**2)

def sech(y):
    return 1.0 / np.cosh(np.clip(y, -500, 500))

def gaussian(y):
    return np.exp(-y**2 / 2.0)

def shifted_lorentzian(y):
    return 1.0 / ((y - 2.0)**2 + 1.0)

#                    (label,              func,                L,  is_even)
TARGETS = [
    (r"$1/(1+y^2)$",           lorentzian,         1.0,  True),
    (r"$\mathrm{sech}(y)$",    sech,               1.0,  True),
    (r"$e^{-y^2/2}$",          gaussian,           1.0,  True),
    (r"$1/((y{-}2)^2{+}1)$",  shifted_lorentzian, 2.0,  False),
]

# ── rational Chebyshev coefficients & errors ──────────────────

def _pullback_on_grid(f, L, N):
    """Evaluate g(θ) = f(L cot θ) on N midpoints in (0,π)."""
    theta = np.pi * (2 * np.arange(N) + 1) / (2 * N)
    return theta, f(L / np.tan(theta))


def cosine_coeffs(g_vals, N):
    """
    Cosine-series coefficients on (0,π) midpoint grid.
    {cos(kθ)} ARE mutually orthogonal under this quadrature.
    Returns a_k with convention g ≈ a_0 + Σ_{k≥1} a_k cos(kθ).
    """
    theta = np.pi * (2 * np.arange(N) + 1) / (2 * N)
    K = N // 2
    a = np.zeros(K)
    for k in range(K):
        a[k] = (2.0 / N) * np.sum(g_vals * np.cos(k * theta))
    a[0] /= 2.0
    return a


def rational_cheb_errors(f, L, d_max, is_even, N_grid=8192, N_eval=4096):
    """
    L∞ error of degree-d rational Chebyshev approximation, d=0..d_max.

    For even f: cosine-only expansion (exact under midpoint quadrature).
    For general f: least-squares in {1,cos,sin,...,cos(d),sin(d)}.
    """
    theta_g, g_grid = _pullback_on_grid(f, L, N_grid)
    theta_e, g_eval = _pullback_on_grid(f, L, N_eval)

    if is_even:
        a_k = cosine_coeffs(g_grid, N_grid)
        coeffs_mag = np.abs(a_k[:d_max + 1])

        errors = np.zeros(d_max + 1)
        for d in range(d_max + 1):
            g_approx = np.zeros(N_eval)
            for k in range(d + 1):
                g_approx += a_k[k] * np.cos(k * theta_e)
            errors[d] = np.max(np.abs(g_eval - g_approx))
    else:
        # least-squares: build full basis on eval grid
        coeffs_mag = np.zeros(d_max + 1)
        errors = np.zeros(d_max + 1)

        for d in range(d_max + 1):
            ncols = 2 * d + 1
            A = np.ones((N_eval, ncols))
            for k in range(1, d + 1):
                A[:, 2*k - 1] = np.cos(k * theta_e)
                A[:, 2*k]     = np.sin(k * theta_e)
            c, *_ = np.linalg.lstsq(A, g_eval, rcond=None)
            g_approx = A @ c
            errors[d] = np.max(np.abs(g_eval - g_approx))

            # magnitude of k-th harmonic
            if d > 0:
                coeffs_mag[d] = np.sqrt(c[2*d - 1]**2 + c[2*d]**2)
            else:
                coeffs_mag[0] = np.abs(c[0])

    return errors, coeffs_mag


# ── standard Chebyshev on [-L, L] ────────────────────────────

def standard_cheb_errors(f, L, d_max, N=8192, N_eval=4096):
    """
    L∞ error of degree-d Chebyshev approximation of f on [-L, L].
    Uses DCT (Chebyshev nodes are orthogonal under this quadrature).
    """
    # Chebyshev coefficients via type-II nodes
    j = np.arange(N)
    nodes = np.cos(np.pi * (2*j + 1) / (2*N))      # in [-1,1]
    fvals = f(L * nodes)

    c = np.zeros(d_max + 1)
    for k in range(d_max + 1):
        c[k] = (2.0 / N) * np.sum(fvals * np.cos(
            k * np.arccos(nodes)))
    c[0] /= 2.0

    # evaluate on fine grid
    x_fine = np.cos(np.pi * (2*np.arange(N_eval) + 1) / (2*N_eval))
    f_fine = f(L * x_fine)

    errors = np.zeros(d_max + 1)
    for d in range(d_max + 1):
        f_approx = np.zeros(N_eval)
        for k in range(d + 1):
            f_approx += c[k] * np.cos(k * np.arccos(x_fine))
        errors[d] = np.max(np.abs(f_fine - f_approx))

    return errors, np.abs(c)


# ── driver ────────────────────────────────────────────────────

def compute_all(d_max=60):
    results = {}
    L_trunc = [10.0, 50.0, 200.0]

    for label, f, L_map, is_even in TARGETS:
        tag = label.split("(")[0].strip("$ \\").split("/")[0].replace("{","").replace("}","")
        if "sech" in label: tag = "sech"
        elif "y^2" in label and "1+y" in label: tag = "lorentzian"
        elif "e^" in label: tag = "gaussian"
        else: tag = "shifted"
        print(f"  {tag}")

        rc_err, rc_coeff = rational_cheb_errors(
            f, L_map, d_max, is_even)
        results[f"{tag}_rc_errors"] = rc_err
        results[f"{tag}_rc_coeffs"] = rc_coeff

        for Ld in L_trunc:
            sc_err, sc_c = standard_cheb_errors(f, Ld, d_max)
            results[f"{tag}_sc_L{int(Ld)}_errors"] = sc_err
            results[f"{tag}_sc_L{int(Ld)}_coeffs"] = sc_c

    results["degrees"] = np.arange(d_max + 1)
    results["L_values"] = np.array(L_trunc)
    return results


# ── figure ────────────────────────────────────────────────────

def make_figure(results):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.size": 9, "axes.labelsize": 10,
        "legend.fontsize": 7.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "serif",
    })

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6),
                             gridspec_kw={"wspace": 0.35})
    fig.subplots_adjust(bottom=0.22)

    tags   = ["lorentzian", "sech", "gaussian", "shifted"]
    titles = [r"$1/(1+y^2)$", r"$\mathrm{sech}(y)$",
              r"$e^{-y^2/2}$", r"$1/((y{-}2)^2{+}1)$"]
    deg = results["degrees"]

    for col, (tg, title) in enumerate(zip(tags, titles)):
        ax = axes[col]
        rc = results[f"{tg}_rc_errors"]
        m_rc = rc > 1e-16
        ax.semilogy(deg[m_rc], rc[m_rc], "s-", ms=2.5, lw=1.2,
                     color="C0",
                     label=r"Rational Cheb.\ $(\mathbb{R})$",
                     zorder=4)

        styles = ["^--", "v--", "d--"]
        for i, Ld in enumerate([10, 50, 200]):
            sc = results[f"{tg}_sc_L{int(Ld)}_errors"]
            m_sc = sc > 1e-16
            ax.semilogy(deg[m_sc], sc[m_sc], styles[i], ms=2,
                         lw=0.7, color=f"C{i+1}",
                         label=rf"Std.\ Cheb.\ $[-{int(Ld)},{int(Ld)}]$")

        ax.set_xlabel(r"Degree $d$")
        if col == 0:
            ax.set_ylabel(r"$L^\infty$ error")
        ax.set_title(rf"({chr(97+col)})\ \ {title}", fontsize=9)
        ax.set_ylim(1e-16, 10)
        ax.set_xlim(-1, deg[-1]+1)
        ax.grid(True, alpha=0.25, lw=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=4, frameon=True, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))

    out = FIG / "convergence_comparison.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"  Figure → {out}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dmax", type=int, default=60)
    args = parser.parse_args()

    npz = DATA / "convergence_comparison.npz"

    if args.plot and npz.exists():
        print("Loading saved data …")
        results = dict(np.load(npz, allow_pickle=True))
    else:
        print("Computing …")
        results = compute_all(d_max=args.dmax)
        np.savez(npz, **results)
        print(f"  Data → {npz}")

    print("Plotting …")
    make_figure(results)


if __name__ == "__main__":
    main()
