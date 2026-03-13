#!/usr/bin/env python3
"""
Generate the Grover/QSP Bloch-sphere figure (panel a) for Section II.

Produces ms/figures/grover_bloch.pdf — a single-column-width figure showing
the 2D great-circle cross-section of the Bloch sphere.  The signal unitary
W(a) rotates by θ = arccos(a) in the xz-plane; successive applications walk
the state along the circle, and the |0⟩-amplitude is cos(kθ) = T_k(a).

Convention:
  - Vertical axis  = |0⟩ (top)  / |1⟩ (bottom)
  - Angle θ from the |0⟩ axis defines the signal a = cos θ
  - Each W(a) application rotates clockwise by θ

Style matches the other publication figures (serif fonts, Okabe-Ito palette).
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Arc

# ── Publication style (matches generate_figures_from_data.py) ────────────────
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
}

COLWIDTH = 3.4  # REVTeX single-column width in inches


def draw_grover_bloch(ax):
    """
    Draw the 2D great-circle cross-section with QSP signal rotations.

    |0⟩ at top, |1⟩ at bottom.  The signal unitary W(a) rotates by
    θ = arccos(a) clockwise; after k applications the state is at angle
    kθ from |0⟩ and its |0⟩-amplitude is cos(kθ) = T_k(a).
    """
    R = 1.0

    # ── Unit circle ──────────────────────────────────────────────────────
    theta_circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        R * np.cos(theta_circ),
        R * np.sin(theta_circ),
        color=C["gray"],
        linewidth=0.6,
        zorder=1,
    )

    # Thin gray cross-hairs
    ax.plot([0, 0], [-1.12, 1.12], color=C["gray"], lw=0.3, zorder=0)
    ax.plot([-1.12, 1.12], [0, 0], color=C["gray"], lw=0.3, zorder=0)

    # ── Parameters ───────────────────────────────────────────────────────
    theta_deg = 22
    theta = np.radians(theta_deg)

    # State after k applications: math polar angle = 90° - kθ
    n_states = 4
    colors = [C["blue"], C["green"], C["orange"], C["red"]]
    labels = [
        r"$|0\rangle$",
        r"$W|0\rangle$",
        r"$W^{\!2}|0\rangle$",
        r"$W^{\!3}|0\rangle$",
    ]

    states = []
    for k in range(n_states):
        polar = np.pi / 2 - k * theta
        x = R * np.cos(polar)
        y = R * np.sin(polar)
        states.append((x, y, polar))

    # ── State vectors (arrows from origin) ───────────────────────────────
    for k, (x, y, _) in enumerate(states):
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=colors[k],
                lw=1.0 if k == 0 else 0.7,
                shrinkA=0,
                shrinkB=1.5,
                mutation_scale=7,
            ),
            zorder=5,
        )
        ax.plot(x, y, "o", color=colors[k], markersize=2.5, zorder=6)

    # ── State labels ─────────────────────────────────────────────────────
    label_offsets = [
        (-4, 7),  # |0⟩: above-left
        (5, 5),  # W|0⟩
        (5, 2),  # W²|0⟩
        (5, -3),  # W³|0⟩
    ]
    label_ha = ["right", "left", "left", "left"]
    for k, (x, y, _) in enumerate(states):
        ax.annotate(
            labels[k],
            (x, y),
            textcoords="offset points",
            xytext=label_offsets[k],
            fontsize=7,
            color=colors[k],
            ha=label_ha[k],
            va="center",
            zorder=7,
        )

    # ── Single θ arc (from |0⟩ to W|0⟩ only) ────────────────────────────
    # A clean arc near the circle, not overlapping the state arrows
    arc_r = 0.38
    arc = Arc(
        (0, 0),
        2 * arc_r,
        2 * arc_r,
        angle=0,
        theta1=np.degrees(states[1][2]),  # W|0⟩ angle
        theta2=90,  # |0⟩ angle
        color=C["black"],
        linewidth=0.6,
        zorder=3,
    )
    ax.add_patch(arc)

    # θ label — placed just outside the arc midpoint
    mid_angle = np.pi / 2 - 0.5 * theta
    lr = arc_r + 0.11
    ax.text(
        lr * np.cos(mid_angle),
        lr * np.sin(mid_angle),
        r"$\theta$",
        fontsize=7.5,
        ha="center",
        va="center",
        zorder=7,
    )

    # ── Projection: show cos(2θ) = T_2(a) for the k=2 state ─────────────
    k_proj = 2
    x_p, y_p, _ = states[k_proj]
    # Dashed horizontal line from state to vertical axis
    ax.plot(
        [0, x_p],
        [y_p, y_p],
        color=colors[k_proj],
        linewidth=0.5,
        linestyle="--",
        zorder=2,
    )
    # Small tick on vertical axis
    ax.plot(
        [-0.03, 0.03],
        [y_p, y_p],
        color=colors[k_proj],
        linewidth=0.7,
        zorder=3,
    )
    # Label
    ax.text(
        -0.07,
        y_p,
        r"$T_2(a)$",
        fontsize=6.5,
        ha="right",
        va="center",
        color=colors[k_proj],
        zorder=7,
    )

    # ── Pole label: |1⟩ at bottom ────────────────────────────────────────
    ax.text(
        0,
        -R - 0.08,
        r"$|1\rangle$",
        fontsize=8,
        ha="center",
        va="top",
        color=C["black"],
        zorder=7,
    )

    # ── Signal annotation ────────────────────────────────────────────────
    ax.text(
        -0.72,
        -1.18,
        r"$a = \cos\theta$",
        fontsize=7.5,
        ha="center",
        va="center",
        color=C["black"],
        zorder=7,
    )

    # ── Clean up ─────────────────────────────────────────────────────────
    ax.set_xlim(-1.15, 1.50)
    ax.set_ylim(-1.35, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    fig, ax = plt.subplots(1, 1, figsize=(COLWIDTH * 0.55, COLWIDTH * 0.55))
    draw_grover_bloch(ax)

    outpath = os.path.join(FIGDIR, "grover_bloch.pdf")
    fig.savefig(outpath, format="pdf")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
