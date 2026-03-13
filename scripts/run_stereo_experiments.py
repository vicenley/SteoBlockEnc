#!/usr/bin/env python
"""
Run stereographic QSP experiments.

Two-stage pipeline:
  1.  python scripts/run_stereo_experiments.py run   -- compute phases (slow)
  2.  python scripts/run_stereo_experiments.py plot   -- generate figures (fast)

The 'run' stage saves all results to data/stereo_qsp_results.npz.
The 'plot' stage reads that file and writes ms/figures/*.pdf.

Tunable parameters (for 'run'):
  --trials N       Number of random restarts per (target, degree) pair.
                   More trials = better chance of finding the global optimum.
                   Default: 200.  Paper run: 500.
  --maxiter N      Max L-BFGS-B iterations per trial.  Default: 8000.
  --n-fit N        Number of fitting points in [r_min, r_max].  Default: 30.
  --r-min R        Left edge of fitting interval.   Default: 0.3.
  --r-max R        Right edge of fitting interval.  Default: 6.0.
  --degrees D,...  Comma-separated list of QSP degrees.
                   Default: 2,4,6,8,10,14,18.
  --targets T,...  Comma-separated list of target names (see TARGETS dict).
                   Default: all four.
  --workers N      Number of CPU cores.  Default: auto (ncpu - 2).
  --seed N         Base random seed.  Default: 42.

Examples:
  # Quick test (~2 min on 28 cores):
  python scripts/run_stereo_experiments.py run --trials 50 --degrees 2,4,8

  # Full publication run (~30 min on 28 cores):
  python scripts/run_stereo_experiments.py run --trials 500 --maxiter 10000

  # Generate figures from saved data:
  python scripts/run_stereo_experiments.py plot
"""

import os

# CRITICAL: set BLAS threads BEFORE importing numpy (see AGENTS.md)
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
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
BASEDIR = os.path.join(os.path.dirname(__file__), "..")
DATADIR = os.path.join(BASEDIR, "data")
FIGDIR = os.path.join(BASEDIR, "ms", "figures")
DATA_FILE = os.path.join(DATADIR, "stereo_qsp_results.npz")

os.makedirs(DATADIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


# =====================================================================
# Target functions
# =====================================================================

TARGETS = {
    "1/r": lambda r: 1.0 / r,
    "1/(1+r^2)": lambda r: 1.0 / (1 + r**2),
    "exp(-r^2)": lambda r: np.exp(-(r**2)),
    "1/sqrt(1+r^2)": lambda r: 1.0 / np.sqrt(1 + r**2),
}


# =====================================================================
# Core: QSP product with analytical gradient
# =====================================================================


def _cost_and_grad(phis, a, f_target):
    """
    Cost = sum |Re(decoded(r)) - f_target|^2 and its gradient w.r.t. phis.

    Uses forward-accumulation (prefix/suffix products) for exact gradient.

    Parameters
    ----------
    phis : array, shape (d+1,)
        QSP phase angles.
    a : array, shape (n,)
        Compressed signal values a = r/sqrt(1+r^2).
    f_target : array, shape (n,)
        Target decoded-function values.
    """
    d = len(phis) - 1
    n = len(a)
    s = np.sqrt(np.maximum(1.0 - a**2, 0.0))

    # Signal matrix S(a) — constant across optimization
    Sa = np.zeros((n, 2, 2), dtype=complex)
    Sa[:, 0, 0] = a
    Sa[:, 0, 1] = 1j * s
    Sa[:, 1, 0] = 1j * s
    Sa[:, 1, 1] = a

    ep = np.exp(1j * phis)  # shape (d+1,)
    em = np.exp(-1j * phis)

    # Identity for batch initialization
    I2 = np.zeros((n, 2, 2), dtype=complex)
    I2[:, 0, 0] = 1.0
    I2[:, 1, 1] = 1.0

    # Forward: prefix[j] = R(phi_0) S R(phi_1) S ... R(phi_{j-1}) S
    # prefix[0] = I
    prefix = [None] * (d + 1)
    prefix[0] = I2.copy()
    for j in range(d):
        tmp = prefix[j].copy()
        tmp[:, :, 0] *= ep[j]
        tmp[:, :, 1] *= em[j]
        prefix[j + 1] = np.einsum("nij,njk->nik", tmp, Sa)

    # Backward: suffix[j] = S R(phi_{j+1}) S R(phi_{j+2}) ... S R(phi_d)
    # suffix[d] = I
    suffix = [None] * (d + 1)
    suffix[d] = I2.copy()
    for j in range(d - 1, -1, -1):
        tmp = suffix[j + 1].copy()
        tmp[:, 0, :] *= ep[j + 1]
        tmp[:, 1, :] *= em[j + 1]
        suffix[j] = np.einsum("nij,njk->nik", Sa, tmp)

    # Full product: W = prefix[d] @ R(phi_d)
    W = prefix[d].copy()
    W[:, :, 0] *= ep[d]
    W[:, :, 1] *= em[d]

    P_std = W[:, 0, 0]
    Q_std = W[:, 1, 0]

    # Decoded: f = Re( P / (-i Q) )
    f_dec = np.real(P_std / (-1j * Q_std))
    residual = f_dec - f_target
    cost = np.sum(residual**2)

    # Gradient via chain rule
    grad = np.zeros(d + 1)
    for j in range(d + 1):
        tmp = prefix[j].copy()
        tmp[:, :, 0] *= 1j * ep[j]
        tmp[:, :, 1] *= -1j * em[j]
        dW = np.einsum("nij,njk->nik", tmp, suffix[j])
        dP = dW[:, 0, 0]
        dQ = dW[:, 1, 0]
        df = np.real(1j * (dP * Q_std - P_std * dQ) / Q_std**2)
        grad[j] = 2.0 * np.sum(residual * df)

    return cost, grad


# =====================================================================
# Worker for parallel optimization
# =====================================================================


def _worker(args):
    """Single L-BFGS-B trial. Top-level function for pickling."""
    seed, d, a_samples, f_target_vals, maxiter, warm_phis = args
    rng = np.random.RandomState(seed)
    if warm_phis is not None:
        # Perturbation around warm-start guess
        phi0 = warm_phis + rng.uniform(-0.3, 0.3, d + 1)
    else:
        phi0 = rng.uniform(-np.pi, np.pi, d + 1)
    res = minimize(
        _cost_and_grad,
        phi0,
        args=(a_samples, f_target_vals),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-30},
    )
    return res.x, res.fun


def find_stereo_phases(
    f_target,
    d,
    r_samples,
    n_trials=200,
    n_workers=None,
    maxiter=8000,
    seed=42,
    warm_phis=None,
):
    """
    Find QSP phases for a stereographic decoded target f(r).

    Parameters
    ----------
    f_target : callable
        Target function f(r) for r > 0.
    d : int
        QSP degree (number of signal operator applications).
    r_samples : array
        Fitting points r > 0.
    n_trials : int
        Number of independent random restarts (parallelized).
    n_workers : int or None
        CPU cores.  None = auto-detect.
    maxiter : int
        Max L-BFGS-B iterations per trial.
    seed : int
        Base random seed (trial i uses seed + i).
    warm_phis : array or None, optional
        Phase angles from a lower-degree solution, zero-padded to length d+1.
        If provided, half the trials are seeded with perturbations around this.

    Returns
    -------
    phis : array, shape (d+1,)
        Best phase angles found.
    cost : float
        Final cost value.
    """
    a_samples = r_samples / np.sqrt(1.0 + r_samples**2)
    f_vals = np.array([f_target(r) for r in r_samples], dtype=float)

    if n_workers is None:
        cpu = os.cpu_count() or 1
        n_workers = max(1, cpu - 2) if cpu > 4 else max(1, cpu)

    seeds = [seed + i for i in range(n_trials)]
    if warm_phis is not None:
        # Half random, half warm-started
        n_warm = n_trials // 2
        trial_args = [(s, d, a_samples, f_vals, maxiter, warm_phis) for s in seeds[:n_warm]] + [
            (s, d, a_samples, f_vals, maxiter, None) for s in seeds[n_warm:]
        ]
    else:
        trial_args = [(s, d, a_samples, f_vals, maxiter, None) for s in seeds]

    best_cost = np.inf
    best_phis = None

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, a): i for i, a in enumerate(trial_args)}
        for future in as_completed(futures):
            phis, cost_val = future.result()
            if cost_val < best_cost:
                best_cost = cost_val
                best_phis = phis

    return best_phis, best_cost


# =====================================================================
# Stage 1: run experiments
# =====================================================================


def cmd_run(args):
    """Run phase-finding experiments and save results."""
    from stereo_block_enc.numerical.qsp_phases import decoded_function, qsp_product, r_to_a

    degrees = [int(x) for x in args.degrees.split(",")]
    target_names = (
        [x.strip() for x in args.targets.split(",")] if args.targets else list(TARGETS.keys())
    )

    r_fit = np.linspace(args.r_min, args.r_max, args.n_fit)
    r_plot = np.linspace(0.1, max(10, args.r_max + 2), 500)

    print(f"Configuration:")
    print(f"  Targets:  {target_names}")
    print(f"  Degrees:  {degrees}")
    print(f"  Trials:   {args.trials}")
    print(f"  Maxiter:  {args.maxiter}")
    print(f"  Fit pts:  {args.n_fit} in [{args.r_min}, {args.r_max}]")
    print(f"  Workers:  {args.workers or 'auto'}")
    print(f"  Seed:     {args.seed}")
    print()

    # Load existing results if incremental
    save_dict = {}
    if os.path.exists(DATA_FILE) and not args.overwrite:
        existing = dict(np.load(DATA_FILE, allow_pickle=True))
        save_dict.update(existing)
        print(f"  Loaded existing data from {DATA_FILE}")
        print(f"  (use --overwrite to start fresh)\n")

    save_dict["r_plot"] = r_plot
    save_dict["r_fit"] = r_fit
    save_dict["degrees"] = np.array(degrees)

    total_jobs = len(target_names) * len(degrees)
    done = 0
    t_total = time.time()

    for name in target_names:
        if name not in TARGETS:
            print(f'  WARNING: unknown target "{name}", skipping')
            continue
        ft = TARGETS[name]
        print(f"  {name}:")
        prev_best_phis = None  # warm-start chain

        for d in degrees:
            done += 1
            t0 = time.time()

            # Build warm-start: zero-pad previous-degree solution to length d+1
            warm = None
            if prev_best_phis is not None and len(prev_best_phis) < d + 1:
                warm = np.zeros(d + 1)
                warm[: len(prev_best_phis)] = prev_best_phis

            phis, cost = find_stereo_phases(
                ft,
                d,
                r_fit,
                n_trials=args.trials,
                n_workers=args.workers,
                maxiter=args.maxiter,
                seed=args.seed,
                warm_phis=warm,
            )

            # Evaluate on dense grid
            f_dec = decoded_function(phis, r_plot).real
            f_exact = ft(r_plot)

            a_plot = r_to_a(r_plot)
            W = qsp_product(phis, a_plot)
            P_vals = np.abs(W[:, 0, 0])

            mask = (r_plot >= args.r_min) & (r_plot <= args.r_max)
            max_err = np.max(np.abs(f_dec[mask] - f_exact[mask]))
            l2_err = np.sqrt(np.mean((f_dec[mask] - f_exact[mask]) ** 2))

            dt = time.time() - t0
            print(
                f"    d={d:2d}: cost={cost:.2e}, max_err={max_err:.2e}, "
                f"L2={l2_err:.2e}  ({dt:.1f}s)  [{done}/{total_jobs}]"
            )

            # Store in save_dict — keep existing result if it has lower max_err
            prefix = name.replace("/", "_").replace("(", "").replace(")", "").replace("^", "")
            existing_err_key = f"{prefix}_d{d}_max_err"
            existing_err = (
                float(save_dict[existing_err_key]) if existing_err_key in save_dict else np.inf
            )
            if max_err < existing_err:
                save_dict[f"{prefix}_d{d}_phis"] = phis
                save_dict[f"{prefix}_d{d}_cost"] = np.float64(cost)
                save_dict[f"{prefix}_d{d}_f_dec"] = f_dec
                save_dict[f"{prefix}_d{d}_P_vals"] = P_vals
                save_dict[f"{prefix}_d{d}_max_err"] = np.float64(max_err)
                save_dict[f"{prefix}_d{d}_l2_err"] = np.float64(l2_err)
                prev_best_phis = phis
            else:
                print(f"           (kept existing: max_err={existing_err:.2e})")
                # Use existing best phases for warm-starting next degree
                phis_key = f"{prefix}_d{d}_phis"
                if phis_key in save_dict:
                    prev_best_phis = save_dict[phis_key]

    # Save
    np.savez(DATA_FILE, **save_dict)
    elapsed = time.time() - t_total
    print(f"\n  Results saved to {DATA_FILE}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


# =====================================================================
# Stage 2: generate figures from saved data
# =====================================================================


def cmd_plot(args):
    """Generate publication figures from saved data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stereo_block_enc.numerical.qsp_phases import r_to_a

    if not os.path.exists(DATA_FILE):
        print(f'ERROR: {DATA_FILE} not found. Run "run" first.')
        sys.exit(1)

    data = dict(np.load(DATA_FILE, allow_pickle=True))
    r_plot = data["r_plot"]
    degrees = list(data["degrees"])

    # Publication style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )

    # Okabe-Ito palette
    C = {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "red": "#D55E00",
        "purple": "#CC79A7",
        "cyan": "#56B4E9",
    }
    TEXTWIDTH = 7.0

    def _get(name, d, field):
        """Retrieve a field from data dict, trying both key conventions."""
        # Current convention: strip ^ from name
        prefix = name.replace("/", "_").replace("(", "").replace(")", "").replace("^", "")
        key = f"{prefix}_d{d}_{field}"
        if key in data:
            v = data[key]
            return float(v) if v.ndim == 0 else v
        # Legacy convention: keep ^ in name
        prefix_legacy = name.replace("/", "_").replace("(", "").replace(")", "")
        key_legacy = f"{prefix_legacy}_d{d}_{field}"
        if key_legacy in data:
            v = data[key_legacy]
            return float(v) if v.ndim == 0 else v
        return None

    # ==================================================================
    # Figure 4: Gallery — all targets, 2 rows x 4 columns
    #   Top row:    decoded f(r) vs exact (black dashed)
    #   Bottom row: pointwise error (log scale)
    # ==================================================================

    targets_gallery = [
        ("1/r", "$1/r$", lambda r: 1.0 / r),
        ("1/(1+r^2)", r"$1/(1{+}r^2)$", lambda r: 1.0 / (1 + r**2)),
        ("1/sqrt(1+r^2)", r"$1/\sqrt{1{+}r^2}$", lambda r: 1.0 / np.sqrt(1 + r**2)),
        ("exp(-r^2)", r"$e^{-r^2}$", lambda r: np.exp(-(r**2))),
    ]

    # Unified color per degree — same palette across all panels
    _deg_palette = [
        C["blue"],
        C["orange"],
        C["green"],
        C["red"],
        C["purple"],
        C["cyan"],
        "#999999",
    ]
    all_degrees_sorted = sorted(set(degrees))
    degree_color_map = {
        d: _deg_palette[i % len(_deg_palette)] for i, d in enumerate(all_degrees_sorted)
    }

    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, 3.6))
    col_labels = ["(a)", "(b)", "(c)", "(d)"]
    row2_labels = ["(e)", "(f)", "(g)", "(h)"]

    for col, (tname, tlabel, tfunc) in enumerate(targets_gallery):
        f_exact = tfunc(r_plot)

        # --- Top row: decoded functions ---
        ax = axes[0, col]
        ax.plot(r_plot, f_exact, "k--", lw=1.2, zorder=10)
        for d in all_degrees_sorted:
            f_dec = _get(tname, d, "f_dec")
            if f_dec is not None:
                ax.plot(
                    r_plot,
                    f_dec,
                    color=degree_color_map[d],
                    lw=0.9,
                    label=f"$d={d}$" if col == 0 else None,
                )

        ax.set_xlim(0.1, 8)
        if tname == "1/r":
            ax.set_ylim(-0.5, 5)
        else:
            ax.set_ylim(-0.15, 1.15)
        ax.set_title(tlabel, fontsize=9)
        if col == 0:
            ax.set_ylabel("$f(r)$")
        # Suppress x-axis tick labels on top row
        ax.tick_params(labelbottom=False)
        # Single legend in panel (a) only
        if col == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=6, handlelength=1.2)
        ax.text(
            0.03,
            0.95,
            col_labels[col],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            fontsize=9,
        )

        # --- Bottom row: pointwise error ---
        ax = axes[1, col]
        for d in all_degrees_sorted:
            f_dec = _get(tname, d, "f_dec")
            if f_dec is not None:
                err = np.abs(f_dec - f_exact)
                err = np.where(err > 0, err, 1e-16)  # avoid log(0)
                ax.semilogy(
                    r_plot,
                    err,
                    color=degree_color_map[d],
                    lw=0.9,
                )

        ax.set_xlim(0.1, 8)
        ax.set_ylim(1e-10, 1e1)
        ax.set_xlabel("$r$")
        if col == 0:
            ax.set_ylabel("Pointwise error")
        ax.text(
            0.03,
            0.95,
            row2_labels[col],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            fontsize=9,
        )

    # ==================================================================
    # Inset: Convergence — max error vs degree (in panel (e))
    # ==================================================================
    ax_host = axes[1, 0]  # bottom-left panel (e)
    ax_inset = ax_host.inset_axes([0.32, 0.38, 0.65, 0.58])

    target_info = [
        ("1/r", "$1/r$", C["blue"], "o"),
        ("1/(1+r^2)", "$1/(1{+}r^2)$", C["orange"], "s"),
        ("1/sqrt(1+r^2)", "$1/\\sqrt{1{+}r^2}$", C["green"], "^"),
        ("exp(-r^2)", "$e^{-r^2}$", C["red"], "D"),
    ]
    for tname, label, color, marker in target_info:
        ds, errs = [], []
        for d in degrees:
            e = _get(tname, d, "max_err")
            if e is not None:
                ds.append(d)
                errs.append(e)
        if ds:
            ax_inset.semilogy(
                ds, errs, color=color, marker=marker, markersize=3, lw=0.9, label=label
            )

    ax_inset.set_xlabel("degree $d$", fontsize=5.5, labelpad=1)
    ax_inset.set_ylabel("max error", fontsize=5.5, labelpad=1)
    dmax = max(degrees) if degrees else 18
    ax_inset.set_xlim(1, dmax + 1)
    ax_inset.set_ylim(1e-8, 1e1)
    ax_inset.tick_params(labelsize=5, pad=1)
    ax_inset.legend(
        loc="lower left",
        frameon=False,
        fontsize=4.5,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    ax_inset.patch.set_facecolor("white")
    ax_inset.patch.set_alpha(0.92)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.5)

    fig.tight_layout(w_pad=0.8, h_pad=0.6)
    path_gallery = os.path.join(FIGDIR, "target_gallery.pdf")
    fig.savefig(path_gallery)
    plt.close(fig)
    print(f"  Saved {path_gallery}")

    print("\n  Figures generated.")


# =====================================================================
# CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stereographic QSP experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    p_run = sub.add_parser("run", help="Run phase-finding experiments")
    p_run.add_argument(
        "--trials", type=int, default=200, help="Random restarts per (target, degree). Default: 200"
    )
    p_run.add_argument(
        "--maxiter", type=int, default=8000, help="Max L-BFGS-B iterations per trial. Default: 8000"
    )
    p_run.add_argument("--n-fit", type=int, default=30, help="Fitting points. Default: 30")
    p_run.add_argument(
        "--r-min", type=float, default=0.3, help="Left edge of fitting interval. Default: 0.3"
    )
    p_run.add_argument(
        "--r-max", type=float, default=6.0, help="Right edge of fitting interval. Default: 6.0"
    )
    p_run.add_argument(
        "--degrees",
        type=str,
        default="2,4,6,8,14,18",
        help="Comma-separated degrees. Default: 2,4,6,8,14,18",
    )
    p_run.add_argument(
        "--targets", type=str, default=None, help="Comma-separated targets. Default: all four"
    )
    p_run.add_argument("--workers", type=int, default=None, help="CPU cores. Default: auto")
    p_run.add_argument("--seed", type=int, default=42, help="Base random seed. Default: 42")
    p_run.add_argument(
        "--overwrite", action="store_true", help="Discard existing results and start fresh"
    )

    # --- plot ---
    p_plot = sub.add_parser("plot", help="Generate figures from saved data")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
