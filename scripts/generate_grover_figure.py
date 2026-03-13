#!/usr/bin/env python3
"""
Generate the Grover/QSP Bloch-sphere figure (panel a) for Section II.

Produces ms/figures/grover_bloch.pdf — a single-column-width figure showing
the 2D great-circle cross-section of the Bloch sphere.  The signal unitary
W(a) rotates by θ = arccos(a) in the xz-plane; successive applications walk
the state along the circle, and the |0⟩-amplitude is cos(kθ) = T_k(a).

Convention:
  - Vertical axis  = |0⟩ (top)  / |1⟩ (bottom)  [= cos component]
  - Horizontal axis = sin component
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
from matplotlib.patches import Arc, FancyArrowPatch

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
    "lightgray": "#CCCCCC",
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
        linewidth=0.8,
        zorder=1,
    )

    # Thin gray cross-hairs (axes through origin)
    ax.plot([0, 0], [-1.15, 1.15], color=C["lightgray"], lw=0.4, zorder=0)
    ax.plot([-1.15, 1.15], [0, 0], color=C["lightgray"], lw=0.4, zorder=0)

    # ── Parameters ───────────────────────────────────────────────────────
    theta_deg = 25  # slightly larger angle for clearer visualization
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
                lw=1.2 if k == 0 else 0.9,
                shrinkA=0,
                shrinkB=1.5,
                mutation_scale=8,
            ),
            zorder=5,
        )
        ax.plot(x, y, "o", color=colors[k], markersize=3.0, zorder=6)

    # ── State labels ─────────────────────────────────────────────────────
    label_offsets = [
        (-5, 8),  # |0⟩: above-left
        (5, 6),  # W|0⟩: above-right
        (6, 2),  # W²|0⟩: right
        (6, -2),  # W³|0⟩: right-below
    ]
    label_ha = ["right", "left", "left", "left"]
    for k, (x, y, _) in enumerate(states):
        ax.annotate(
            labels[k],
            (x, y),
            textcoords="offset points",
            xytext=label_offsets[k],
            fontsize=8,
            color=colors[k],
            ha=label_ha[k],
            va="center",
            zorder=7,
        )

    # ── θ arc (from |0⟩ to W|0⟩) ─────────────────────────────────────────
    arc_r = 0.48
    arc = Arc(
        (0, 0),
        2 * arc_r,
        2 * arc_r,
        angle=0,
        theta1=np.degrees(states[1][2]),  # W|0⟩ angle
        theta2=90,  # |0⟩ angle
        color=C["black"],
        linewidth=0.7,
        zorder=3,
    )
    ax.add_patch(arc)

    # θ label — placed just outside the arc midpoint
    mid_angle = np.pi / 2 - 0.5 * theta
    lr = arc_r + 0.12
    ax.text(
        lr * np.cos(mid_angle),
        lr * np.sin(mid_angle),
        r"$\theta$",
        fontsize=8.5,
        ha="center",
        va="center",
        zorder=7,
    )

    # ── Small rotation arrows between successive states ──────────────────
    # Show θ arcs between states 1→2 and 2→3 to reinforce "rotate by θ"
    for k_start in [1, 2]:
        arc_rk = 0.30 + k_start * 0.07
        a_start = np.degrees(states[k_start + 1][2])
        a_end = np.degrees(states[k_start][2])
        small_arc = Arc(
            (0, 0),
            2 * arc_rk,
            2 * arc_rk,
            angle=0,
            theta1=a_start,
            theta2=a_end,
            color=C["gray"],
            linewidth=0.4,
            linestyle=":",
            zorder=2,
        )
        ax.add_patch(small_arc)

    # ── Projection: show cos(2θ) = T_2(a) for the k=2 state ─────────────
    k_proj = 2
    x_p, y_p, _ = states[k_proj]

    # Dashed horizontal line from state to vertical axis
    ax.plot(
        [0, x_p],
        [y_p, y_p],
        color=C["orange"],
        linewidth=0.6,
        linestyle="--",
        zorder=2,
    )
    # Small tick on vertical axis
    ax.plot(
        [-0.03, 0.03],
        [y_p, y_p],
        color=C["orange"],
        linewidth=0.8,
        zorder=3,
    )
    # T_2(a) label on vertical axis
    ax.text(
        -0.08,
        y_p,
        r"$T_2(a)$",
        fontsize=7,
        ha="right",
        va="center",
        color=C["orange"],
        zorder=7,
    )

    # Dashed vertical line from state down to horizontal axis
    ax.plot(
        [x_p, x_p],
        [0, y_p],
        color=C["orange"],
        linewidth=0.4,
        linestyle=":",
        zorder=2,
    )

    # ── Axis labels ──────────────────────────────────────────────────────
    # Vertical axis: |0⟩ at top, |1⟩ at bottom  (= cos component)
    # Note: |0⟩ is already labeled as a state; add standalone pole labels
    # only for |1⟩ at bottom
    ax.text(
        0.06,
        -R - 0.07,
        r"$|1\rangle$",
        fontsize=8,
        ha="left",
        va="top",
        color=C["black"],
        zorder=7,
    )

    # a = cos θ annotation — placed cleanly to the left of the vertical axis
    ax.text(
        -0.55,
        -1.18,
        r"$a = \cos\theta$",
        fontsize=7.5,
        ha="center",
        va="center",
        color=C["black"],
        zorder=7,
    )

    # ── Clean up ─────────────────────────────────────────────────────────
    ax.set_xlim(-1.20, 1.55)
    ax.set_ylim(-1.30, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    # Larger figure: 0.75 × COLWIDTH for better readability in two-column format
    figsize = COLWIDTH * 0.75
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    draw_grover_bloch(ax)

    outpath = os.path.join(FIGDIR, "grover_bloch.pdf")
    fig.savefig(outpath, format="pdf")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
