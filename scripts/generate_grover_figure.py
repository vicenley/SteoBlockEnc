#!/usr/bin/env python3
"""
Generate the Grover/QSP geometric figure (panel a) for Section II.

Produces ms/figures/grover_bloch.pdf — a single-column-width figure showing
the geometric interpretation of Grover's algorithm in the 2D plane spanned
by {|ψ_*⟩, |ψ_*⊥⟩}, bridging to the QSP framework.

Convention (following Benenti et al., Fig 3.24):
  - Vertical axis   = |ψ_*⟩  (target state)
  - Horizontal axis  = |ψ_*⊥⟩ (complement of target)
  - |ψ₀⟩ = |S⟩ (equal superposition) at angle θ from horizontal
  - Oracle O reflects about |ψ_*⊥⟩ axis
  - Grover operator G rotates by 2θ toward |ψ_*⟩
  - After k iterations: state at angle (2k+1)θ, target amplitude = sin((2k+1)θ)

QSP bridge: a = cos θ is the signal; amplitudes are Chebyshev polynomials.

Style matches the other publication figures (serif fonts, Okabe-Ito palette).
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Arc

# ── Publication style ────────────────────────────────────────────────────────
rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}",
        "mathtext.fontset": "cm",
        "lines.linewidth": 1.0,
    }
)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "ms", "figures")

# Okabe-Ito palette
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "black": "#000000",
    "gray": "#999999",
    "lightgray": "#CCCCCC",
    "darkgray": "#555555",
}

COLWIDTH = 3.4  # REVTeX single-column width in inches


def _arrow(ax, x, y, color, lw=0.9, zorder=5):
    """Draw an arrow from origin to (x, y)."""
    ax.annotate(
        "",
        xy=(x, y),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=1.5,
            mutation_scale=8,
        ),
        zorder=zorder,
    )


def draw_grover_qsp(ax):
    """
    Draw Grover's geometric interpretation bridged to QSP.

    2D plane: horizontal = |ψ_*⊥⟩, vertical = |ψ_*⟩.
    |ψ₀⟩ at angle θ from horizontal.  Oracle reflects about horizontal.
    Grover operator G rotates by 2θ toward the target.
    """
    R = 1.0

    # ── Unit circle ──────────────────────────────────────────────────────
    theta_circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        R * np.cos(theta_circ),
        R * np.sin(theta_circ),
        color=C["gray"],
        linewidth=0.8,
        zorder=1,
    )

    # ── Axes ─────────────────────────────────────────────────────────────
    # Light gray cross-hairs
    ax.plot([0, 0], [-1.12, 1.12], color=C["lightgray"], lw=0.4, zorder=0)
    ax.plot([-1.12, 1.12], [0, 0], color=C["lightgray"], lw=0.4, zorder=0)

    # ── Parameters ───────────────────────────────────────────────────────
    # θ = angle between |ψ₀⟩ and horizontal axis
    # Choose θ ≈ 20° for N ≈ 8 (sin θ ≈ 1/√N)
    theta = np.radians(20)

    # ── Key states ───────────────────────────────────────────────────────
    # All angles measured from the positive horizontal axis (|ψ_*⊥⟩)

    # |ψ₀⟩ = |S⟩ — initial superposition at angle θ
    psi0_angle = theta
    psi0 = (R * np.cos(psi0_angle), R * np.sin(psi0_angle))

    # O|ψ₀⟩ — oracle reflection about horizontal axis (flip sign of |ψ_*⟩)
    Opsi_angle = -theta
    Opsi = (R * np.cos(Opsi_angle), R * np.sin(Opsi_angle))

    # G|ψ₀⟩ — after 1 Grover iteration, at angle 3θ
    Gpsi_angle = 3 * theta
    Gpsi = (R * np.cos(Gpsi_angle), R * np.sin(Gpsi_angle))

    # G²|ψ₀⟩ — after 2 iterations, at angle 5θ
    G2psi_angle = 5 * theta
    G2psi = (R * np.cos(G2psi_angle), R * np.sin(G2psi_angle))

    # ── Draw O|ψ₀⟩ first (dashed, behind everything) ─────────────────────
    ax.plot(
        [0, Opsi[0]],
        [0, Opsi[1]],
        color=C["darkgray"],
        lw=0.7,
        ls="--",
        zorder=2,
    )
    ax.plot(Opsi[0], Opsi[1], "o", color=C["darkgray"], markersize=2.5, zorder=3)
    ax.annotate(
        r"$O\ket{\psi_0}$",
        Opsi,
        textcoords="offset points",
        xytext=(6, -6),
        fontsize=7,
        color=C["darkgray"],
        ha="left",
        va="top",
        zorder=7,
    )

    # Dotted reflection line connecting |ψ₀⟩ to O|ψ₀⟩
    ax.plot(
        [psi0[0], Opsi[0]],
        [psi0[1], Opsi[1]],
        color=C["lightgray"],
        lw=0.5,
        ls=":",
        zorder=1,
    )

    # ── State vectors ────────────────────────────────────────────────────
    # |ψ₀⟩ = |S⟩
    _arrow(ax, *psi0, C["blue"], lw=1.2)
    ax.plot(*psi0, "o", color=C["blue"], markersize=3.5, zorder=6)

    # G|ψ₀⟩
    _arrow(ax, *Gpsi, C["green"], lw=1.0)
    ax.plot(*Gpsi, "o", color=C["green"], markersize=3.0, zorder=6)

    # G²|ψ₀⟩
    _arrow(ax, *G2psi, C["orange"], lw=0.9)
    ax.plot(*G2psi, "o", color=C["orange"], markersize=3.0, zorder=6)

    # ── State labels ─────────────────────────────────────────────────────
    ax.annotate(
        r"$\ket{\psi_0}\!=\!\ket{S}$",
        psi0,
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=7.5,
        color=C["blue"],
        ha="left",
        va="bottom",
        zorder=7,
    )
    ax.annotate(
        r"$G\ket{\psi_0}$",
        Gpsi,
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=7.5,
        color=C["green"],
        ha="left",
        va="bottom",
        zorder=7,
    )
    ax.annotate(
        r"$G^2\ket{\psi_0}$",
        G2psi,
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=7.5,
        color=C["orange"],
        ha="left",
        va="bottom",
        zorder=7,
    )

    # ── Angle θ: from horizontal to |ψ₀⟩ ─────────────────────────────────
    arc_r = 0.50
    arc_theta = Arc(
        (0, 0),
        2 * arc_r,
        2 * arc_r,
        angle=0,
        theta1=0,
        theta2=np.degrees(theta),
        color=C["black"],
        linewidth=0.7,
        zorder=3,
    )
    ax.add_patch(arc_theta)
    # θ label
    mid_th = theta / 2
    lr = arc_r + 0.10
    ax.text(
        lr * np.cos(mid_th),
        lr * np.sin(mid_th),
        r"$\theta$",
        fontsize=8.5,
        ha="center",
        va="center",
        zorder=7,
    )

    # ── 2θ rotation arc: from |ψ₀⟩ to G|ψ₀⟩ ─────────────────────────────
    arc_2th_r = 0.70
    arc_2th = Arc(
        (0, 0),
        2 * arc_2th_r,
        2 * arc_2th_r,
        angle=0,
        theta1=np.degrees(theta),
        theta2=np.degrees(3 * theta),
        color=C["red"],
        linewidth=0.8,
        zorder=3,
    )
    ax.add_patch(arc_2th)
    # 2θ label
    mid_2th = 2 * theta
    lr2 = arc_2th_r + 0.11
    ax.text(
        lr2 * np.cos(mid_2th),
        lr2 * np.sin(mid_2th),
        r"$2\theta$",
        fontsize=7.5,
        ha="center",
        va="center",
        color=C["red"],
        zorder=7,
    )

    # ── 2θ rotation arc: from G|ψ₀⟩ to G²|ψ₀⟩ ───────────────────────────
    arc_2th2_r = 0.55
    arc_2th2 = Arc(
        (0, 0),
        2 * arc_2th2_r,
        2 * arc_2th2_r,
        angle=0,
        theta1=np.degrees(3 * theta),
        theta2=np.degrees(5 * theta),
        color=C["gray"],
        linewidth=0.5,
        linestyle=":",
        zorder=2,
    )
    ax.add_patch(arc_2th2)

    # ── Projection: sin(3θ) for G|ψ₀⟩ onto vertical axis ────────────────
    # This shows the target-state amplitude after 1 iteration
    Gpsi_proj_y = Gpsi[1]  # = sin(3θ)
    ax.plot(
        [0, Gpsi[0]],
        [Gpsi_proj_y, Gpsi_proj_y],
        color=C["green"],
        linewidth=0.5,
        linestyle="--",
        zorder=2,
    )
    ax.plot(
        [-0.03, 0.03],
        [Gpsi_proj_y, Gpsi_proj_y],
        color=C["green"],
        linewidth=0.8,
        zorder=3,
    )
    ax.text(
        -0.07,
        Gpsi_proj_y,
        r"$\sin 3\theta$",
        fontsize=6.5,
        ha="right",
        va="center",
        color=C["green"],
        zorder=7,
    )

    # ── Axis labels ──────────────────────────────────────────────────────
    # |ψ_*⟩ at top of vertical axis
    ax.text(
        0.06,
        R + 0.07,
        r"$\ket{\psi_*}$",
        fontsize=8.5,
        ha="left",
        va="bottom",
        color=C["black"],
        zorder=7,
    )

    # |ψ_*⊥⟩ at right end of horizontal axis
    ax.text(
        R + 0.06,
        -0.07,
        r"$\ket{\psi_*^{\!\perp}}$",
        fontsize=8,
        ha="left",
        va="top",
        color=C["black"],
        zorder=7,
    )

    # ── QSP bridge annotation ────────────────────────────────────────────
    # a = cos θ, sin θ = ⟨ψ_*|ψ₀⟩
    ax.text(
        -0.55,
        -1.15,
        r"$a = \cos\theta$",
        fontsize=7.5,
        ha="center",
        va="center",
        color=C["black"],
        zorder=7,
    )

    # ── Clean up ─────────────────────────────────────────────────────────
    ax.set_xlim(-1.15, 1.55)
    ax.set_ylim(-1.22, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    figsize = COLWIDTH * 0.75
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    draw_grover_qsp(ax)

    os.makedirs(FIGDIR, exist_ok=True)
    outpath = os.path.join(FIGDIR, "grover_bloch.pdf")
    fig.savefig(outpath, format="pdf")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
