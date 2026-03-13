#!/usr/bin/env python3
"""
Generate Figure 2 (schematic.pdf) — Stereographic QSP framework overview.

Three-panel schematic showing:
  (a) Encoding:   r ∈ [0,∞) → stereographic compression → r̃ ∈ [0,1) → Bloch sphere
  (b) QSP Circuit: alternating phase and signal gates, unitarity constraint
  (c) Decoding:   Pauli measurement → f(r) = P/Q unbounded rational function

Output: ms/figures/schematic.pdf (full textwidth, ~3.0 inches tall)

Color palette: professional blue / vermilion / teal triad, all dark enough
for crisp print reproduction.  Secondary text uses dark gray (#444444),
never light gray.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Ellipse,
    Rectangle,
)

# ── Publication style ────────────────────────────────────────────────────────
rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
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
        "savefig.pad_inches": 0.04,
        "text.usetex": True,
        "text.latex.preamble": (r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}"),
        "mathtext.fontset": "cm",
        "lines.linewidth": 1.0,
    }
)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "ms", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────────
# Three semantic roles:
#   BOUNDED  = blue   (encoding output, unitarity text, |P| curve)
#   UNBOUNDED = vermilion/red (unbounded domain, P/Q curve, decoding output)
#   CIRCUIT  = teal   (SP arrow, connecting arrows)
# All colours chosen for ≥ 4.5:1 contrast ratio against white.

BLUE = "#1A5276"  # dark teal-blue — bounded quantities
RED = "#C0392B"  # vermilion-red  — unbounded quantities
TEAL = "#117A65"  # deep teal      — circuit / signal processing
BLACK = "#1C1C1C"
DARKGRAY = "#444444"  # secondary text — always readable
MEDGRAY = "#777777"  # tertiary (axes ticks, equator, dashed lines)

# Gate fills — muted pastels that read well in print
PHASE_FILL = "#D4E6F1"  # pale blue
SIGNAL_FILL = "#FADBD8"  # pale rose
UNITARY_FILL = "#D5F5E3"  # pale green
MEAS_FILL = "#EAECEE"  # pale gray

# Panel backgrounds — very subtle tints
BG_ENC = "#EBF0F5"  # faint blue
BG_CIRC = "#F5F5F5"  # near-white
BG_DEC = "#FDF2E9"  # faint warm

# Box accents
INSIGHT_FILL = "#FDEDEC"
INSIGHT_EDGE = RED

TEXTWIDTH = 7.0  # inches
FIG_H = 3.4  # inches — extra vertical space to avoid crowding


# ── Helpers ──────────────────────────────────────────────────────────────────


def _gate_box(ax, x, y, label, facecolor="white", w=0.32, h=0.26, fontsize=6.5):
    """Draw a sharp-cornered gate box centred at (x, y), matching quantikz style."""
    rect = Rectangle(
        (x - w / 2, y - h / 2),
        w,
        h,
        facecolor="white",
        edgecolor=BLACK,
        linewidth=0.4,
        zorder=5,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, fontsize=fontsize, ha="center", va="center", zorder=6)


# ── Main drawing ─────────────────────────────────────────────────────────────


def draw_all(ax):
    """Draw the entire three-panel schematic in inch coordinates."""
    W = TEXTWIDTH
    H = FIG_H

    # ── Panel geometry ────────────────────────────────────────────────────
    gap = 0.30  # horizontal gap between panels
    enc_L, enc_R = 0.05, 1.82
    circ_L = enc_R + gap
    circ_R = 5.08
    dec_L = circ_R + gap
    dec_R = W - 0.05

    panel_top = H - 0.50  # leave room for overall flow arrow above
    panel_bot = 0.05

    # ── Background panels ─────────────────────────────────────────────────
    for xL, xR, bg in [(enc_L, enc_R, BG_ENC), (circ_L, circ_R, BG_CIRC), (dec_L, dec_R, BG_DEC)]:
        ax.add_patch(
            FancyBboxPatch(
                (xL, panel_bot),
                xR - xL,
                panel_top - panel_bot,
                boxstyle="round,pad=0.06",
                facecolor=bg,
                edgecolor="#C0C0C0",
                linewidth=0.4,
                zorder=0,
            )
        )

    # vertical centre of the connecting-arrow corridor
    # Sits between the bounded interval (y ≈ 1.84) and the unitarity brace
    y_connect = 1.75

    # ==================================================================
    #  ENCODING (left)
    # ==================================================================
    enc_cx = (enc_L + enc_R) / 2
    line_L = enc_L + 0.15
    line_R = enc_R - 0.15
    span = line_R - line_L

    # title
    ax.text(
        enc_cx,
        panel_top - 0.07,
        r"\textsf{Encoding}",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
        color=BLACK,
    )

    # ── unbounded number line ─────────────────────────────────────────
    y1 = panel_top - 0.42
    ax.annotate(
        "",
        xy=(line_R, y1),
        xytext=(line_L, y1),
        arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.7),
    )
    for i, lab in enumerate(["0", "1", "2", "3"]):
        xp = line_L + i * span / 4
        ax.plot([xp, xp], [y1 - 0.04, y1 + 0.04], color=BLACK, lw=0.5)
        ax.text(xp, y1 - 0.07, f"${lab}$", fontsize=6, ha="center", va="top", color=BLACK)
    ax.text(line_R + 0.06, y1, r"$r$", fontsize=7.5, ha="left", va="center", color=BLACK)
    ax.text(
        enc_cx, y1 + 0.09, r"$r \in [0,\infty)$", fontsize=7, ha="center", va="bottom", color=RED
    )
    # dotted extension towards ∞
    ax.plot([line_L + 0.78 * span, line_R - 0.02], [y1, y1], color=RED, lw=1.0, ls=":")

    # ── stereographic compression arrow ───────────────────────────────
    y_atop = y1 - 0.14
    y_abot = y1 - 0.52
    ax.annotate(
        "",
        xy=(enc_cx, y_abot),
        xytext=(enc_cx, y_atop),
        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.0, mutation_scale=8),
    )
    # formula to the RIGHT but still inside the panel
    ax.text(
        enc_cx + 0.08,
        (y_atop + y_abot) / 2,
        r"$\tilde{r} = \frac{r}{\sqrt{1{+}r^2}}$",
        fontsize=6.5,
        ha="left",
        va="center",
        color=TEAL,
    )
    ax.text(
        enc_cx - 0.12, (y_atop + y_abot) / 2, r"SP", fontsize=6, ha="right", va="center", color=TEAL
    )

    # ── bounded interval [0,1) ────────────────────────────────────────
    y2 = y_abot - 0.12
    ax.annotate(
        "",
        xy=(line_R, y2),
        xytext=(line_L, y2),
        arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.7),
    )
    ax.text(line_R + 0.06, y2, r"$\tilde{r}$", fontsize=7.5, ha="left", va="center", color=BLACK)
    # closed dot at 0
    ax.plot(line_L, y2, "o", color=BLUE, ms=3.5, zorder=5)
    ax.text(line_L, y2 - 0.07, r"$0$", fontsize=6, ha="center", va="top", color=BLACK)
    # open circle at 1
    open_x = line_L + 0.78 * span
    ax.add_patch(Ellipse((open_x, y2), 0.06, 0.06, fill=False, edgecolor=BLUE, lw=0.7, zorder=5))
    ax.text(open_x, y2 - 0.07, r"$1$", fontsize=6, ha="center", va="top", color=BLACK)
    ax.text(
        enc_cx,
        y2 - 0.14,
        r"$\tilde{r} \in [0,1)$ bounded",
        fontsize=6.5,
        ha="center",
        va="top",
        color=BLUE,
    )

    # ── Bloch sphere ──────────────────────────────────────────────────
    yB = 0.95  # centred in available space between bounded label and formula
    rB = 0.30  # slightly smaller to fit in panel
    ax.add_patch(
        Ellipse((enc_cx, yB), 2 * rB, 2 * rB, fill=False, edgecolor=BLACK, lw=0.7, zorder=3)
    )
    # equator
    eq_h = rB * 0.28
    th_back = np.linspace(np.pi, 2 * np.pi, 60)
    ax.plot(
        enc_cx + rB * np.cos(th_back),
        yB + eq_h * np.sin(th_back),
        color=MEDGRAY,
        lw=0.5,
        ls="--",
        zorder=2,
    )
    th_front = np.linspace(0, np.pi, 60)
    ax.plot(
        enc_cx + rB * np.cos(th_front),
        yB + eq_h * np.sin(th_front),
        color=MEDGRAY,
        lw=0.5,
        zorder=4,
    )
    # poles
    ax.plot(enc_cx, yB + rB, "o", color=BLACK, ms=2.5, zorder=5)
    ax.text(
        enc_cx + 0.05,
        yB + rB + 0.02,
        r"$|0\rangle$",
        fontsize=6,
        ha="left",
        va="bottom",
        color=BLACK,
    )
    ax.plot(enc_cx, yB - rB, "o", color=BLACK, ms=2.5, zorder=5)
    ax.text(
        enc_cx + 0.08,
        yB - rB + 0.01,
        r"$|1\rangle$",
        fontsize=5.5,
        ha="left",
        va="top",
        color=BLACK,
    )
    # state vector
    ang = np.radians(50)
    xr = enc_cx + rB * 0.75 * np.sin(ang)
    yr = yB + rB * 0.75 * np.cos(ang)
    ax.annotate(
        "",
        xy=(xr, yr),
        xytext=(enc_cx, yB),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8, mutation_scale=6),
    )
    ax.text(xr + 0.04, yr + 0.02, r"$|r\rangle$", fontsize=6.5, color=BLUE, ha="left", va="bottom")

    # encoding state formula — well below Bloch sphere
    ax.text(
        enc_cx,
        0.15,
        r"$|r\rangle = \frac{r|0\rangle + |1\rangle}{\sqrt{r^2+1}}$",
        fontsize=6,
        ha="center",
        va="top",
        color=BLACK,
    )

    # ==================================================================
    #  CONNECTING ARROW  Encoding → QSP
    # ==================================================================
    ax.annotate(
        "",
        xy=(circ_L + 0.05, y_connect),
        xytext=(enc_R + 0.02, y_connect),
        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.0, mutation_scale=9),
    )
    ax.text(
        (enc_R + circ_L) / 2,
        y_connect + 0.07,
        r"$|r\rangle$",
        fontsize=7,
        ha="center",
        va="bottom",
        color=TEAL,
    )

    # ==================================================================
    #  QSP CIRCUIT (centre)
    # ==================================================================
    circ_cx = (circ_L + circ_R) / 2

    ax.text(
        circ_cx,
        panel_top - 0.07,
        r"\textsf{QSP Circuit}",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
        color=BLACK,
    )

    # helper text — use DARKGRAY, not light gray
    ax.text(
        circ_cx,
        panel_top - 0.30,
        r"degree $d$\quad ($d{+}1$ phases, $d$ signal operators)",
        fontsize=5.5,
        ha="center",
        va="bottom",
        color=DARKGRAY,
    )
    ax.text(
        circ_cx,
        panel_top - 0.45,
        r"$S_z = V^{-1}\,W(\tilde{r})\,V$,\quad $V = \mathrm{diag}(1,i)$",
        fontsize=5.5,
        ha="center",
        va="bottom",
        color=DARKGRAY,
    )

    # ── circuit wire + gates ─────────────────────────────────────────
    y_w = y_connect
    wL = circ_L + 0.30  # extra left margin for |0⟩ label
    wR = circ_R - 0.20
    ax.text(wL - 0.03, y_w, r"$|0\rangle$", fontsize=5.5, ha="right", va="center", color=BLACK)

    gates = [
        ("phase", r"$\phi_d$"),
        ("signal", r"$S_z$"),
        ("phase", r"$\phi_{d\text{-}1}$"),
        ("dots", ""),
        ("signal", r"$S_z$"),
        ("phase", r"$\phi_1$"),
        ("signal", r"$S_z$"),
        ("phase", r"$\phi_0$"),
    ]
    n_g = len(gates)
    gw = 0.32  # gate box width
    dots_half = 0.12  # half-width of visual space for "..." element
    spacing = (wR - wL - 0.10) / (n_g - 1)
    gate_xs = [wL + 0.05 + i * spacing for i in range(n_g)]

    # Draw wire segments between gates (wire passes through each gate)
    wire_lw = 0.7
    # Segment from wire start to first gate's left edge
    ax.plot([wL, gate_xs[0] - gw / 2], [y_w, y_w], color=BLACK, lw=wire_lw, zorder=1)
    for i in range(n_g - 1):
        # right edge of gate i to left edge of gate i+1
        gt_i = gates[i][0]
        gt_next = gates[i + 1][0]
        x_right = gate_xs[i] + (dots_half if gt_i == "dots" else gw / 2)
        x_left = gate_xs[i + 1] - (dots_half if gt_next == "dots" else gw / 2)
        ax.plot([x_right, x_left], [y_w, y_w], color=BLACK, lw=wire_lw, zorder=1)
    # Segment from last gate's right edge to wire end
    ax.plot([gate_xs[-1] + gw / 2, wR], [y_w, y_w], color=BLACK, lw=wire_lw, zorder=1)

    # Draw gate boxes (on top of wire segments)
    for i, (gt, lab) in enumerate(gates):
        gx = gate_xs[i]
        if gt == "dots":
            ax.text(
                gx, y_w, r"$\cdots$", fontsize=9, ha="center", va="center", zorder=5, color=BLACK
            )
        else:
            fc = PHASE_FILL if gt == "phase" else SIGNAL_FILL
            fs = 5.5 if len(lab) > 6 else 6
            _gate_box(ax, gx, y_w, lab, fc, w=gw, h=0.26, fontsize=fs)

    # output state label — above the wire, centred over the last 3 gates
    ax.text(
        wR - 0.65,
        y_w + 0.22,
        r"$P|0\rangle{+}Q|1\rangle$",
        fontsize=5.5,
        ha="center",
        va="bottom",
        color=BLACK,
    )

    # ── unitarity brace ───────────────────────────────────────────────
    yB2 = y_w - 0.24
    bL = wL + 0.03
    bR = wR - 0.03
    bM = (bL + bR) / 2
    tk = 0.04
    for seg in [
        ([bL, bR], [yB2, yB2]),
        ([bL, bL], [yB2, yB2 + tk]),
        ([bR, bR], [yB2, yB2 + tk]),
        ([bM, bM], [yB2, yB2 - tk]),
    ]:
        ax.plot(seg[0], seg[1], color=BLUE, lw=0.5)
    ax.text(
        bM,
        yB2 - 0.08,
        r"$|P(\tilde{r})|^2 + |Q(\tilde{r})|^2 = 1$"
        r"$\;\;\Rightarrow\;\;|P|, |Q| \leq 1$ bounded",
        fontsize=5.5,
        ha="center",
        va="top",
        color=BLUE,
    )

    # ── block-encoding inset ──────────────────────────────────────────
    y_be = 0.80
    ax.text(
        circ_cx,
        y_be,
        r"Block-encoding context:",
        fontsize=5.5,
        ha="center",
        va="top",
        color=DARKGRAY,
    )

    y_a = y_be - 0.22  # ancilla wire
    y_s = y_a - 0.27  # system wire
    bL2 = circ_L + 0.55
    bR2 = circ_R - 0.55
    ax.text(bL2 - 0.06, y_a, r"$|0\rangle_a$", fontsize=5, ha="right", va="center", color=BLACK)
    ax.text(bL2 - 0.06, y_s, r"$|\psi\rangle_s$", fontsize=5, ha="right", va="center", color=BLACK)

    # Gate positions and sizes
    v_cx = bL2 + 0.25
    v_w = 0.25
    v_h = 0.22
    qsp_cx = (bL2 + bR2) / 2
    qsp_w = 0.8
    qsp_h = 0.20
    vdag_cx = bR2 - 0.25
    vdag_w = 0.28
    vdag_h = 0.22
    m_cx = bR2 - 0.05
    m_w = 0.18
    m_h = 0.18
    be_lw = 0.5

    # -- Ancilla wire segments (y_a): start → QSP left, QSP right → M left, M right → end
    ax.plot([bL2, qsp_cx - qsp_w / 2], [y_a, y_a], color=BLACK, lw=be_lw, zorder=1)
    ax.plot([qsp_cx + qsp_w / 2, m_cx - m_w / 2], [y_a, y_a], color=BLACK, lw=be_lw, zorder=1)
    ax.plot([m_cx + m_w / 2, bR2], [y_a, y_a], color=BLACK, lw=be_lw, zorder=1)

    # -- System wire segments (y_s): start → V left, V right → control dot,
    #    control dot → V† left, V† right → end
    ax.plot([bL2, v_cx - v_w / 2], [y_s, y_s], color=BLACK, lw=be_lw, zorder=1)
    ax.plot([v_cx + v_w / 2, qsp_cx], [y_s, y_s], color=BLACK, lw=be_lw, zorder=1)
    ax.plot([qsp_cx, vdag_cx - vdag_w / 2], [y_s, y_s], color=BLACK, lw=be_lw, zorder=1)
    ax.plot([vdag_cx + vdag_w / 2, bR2], [y_s, y_s], color=BLACK, lw=be_lw, zorder=1)

    # -- Gates --
    _gate_box(ax, v_cx, y_s, r"$V$", UNITARY_FILL, w=v_w, h=v_h, fontsize=5)
    ax.add_patch(
        Rectangle(
            (qsp_cx - qsp_w / 2, y_a - qsp_h / 2),
            qsp_w,
            qsp_h,
            facecolor="white",
            edgecolor=BLACK,
            linewidth=0.4,
            zorder=5,
        )
    )
    ax.text(
        qsp_cx,
        y_a,
        r"$\textsf{QSP}(\bm{\phi})$",
        fontsize=5.5,
        ha="center",
        va="center",
        zorder=6,
        color=BLACK,
    )
    # control line (from bottom of QSP box to top of control dot on system wire)
    ax.plot([qsp_cx, qsp_cx], [y_a - qsp_h / 2, y_s + 0.02], color=BLACK, lw=be_lw, zorder=3)
    ax.plot(qsp_cx, y_s, "o", color=BLACK, ms=2, zorder=5)
    _gate_box(ax, vdag_cx, y_s, r"$V^\dagger$", UNITARY_FILL, w=vdag_w, h=vdag_h, fontsize=5)
    # Measurement gate: quantikz-style meter symbol
    ax.add_patch(
        Rectangle(
            (m_cx - m_w / 2, y_a - m_h / 2),
            m_w,
            m_h,
            facecolor="white",
            edgecolor=BLACK,
            linewidth=0.4,
            zorder=5,
        )
    )
    # Meter arc (bottom-half semicircle)
    arc_r = m_w * 0.30
    arc_y = y_a - m_h * 0.10
    th_arc = np.linspace(0, np.pi, 40)
    ax.plot(
        m_cx + arc_r * np.cos(th_arc),
        arc_y + arc_r * np.sin(th_arc),
        color=BLACK,
        lw=0.4,
        zorder=6,
    )
    # Meter needle (diagonal line from arc centre to upper-right)
    ax.plot(
        [m_cx, m_cx + arc_r * 0.75 * np.cos(np.pi / 4)],
        [arc_y, arc_y + arc_r * 0.85 * np.sin(np.pi / 4) + arc_r * 0.2],
        color=BLACK,
        lw=0.4,
        zorder=6,
    )

    # ==================================================================
    #  CONNECTING ARROW  QSP → Decoding
    # ==================================================================
    ax.annotate(
        "",
        xy=(dec_L + 0.05, y_connect),
        xytext=(circ_R + 0.02, y_connect),
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0, mutation_scale=9),
    )
    mid_x = (circ_R + dec_L) / 2
    ax.text(
        mid_x,
        y_connect + 0.10,
        r"$\langle\sigma_Z\rangle$",
        fontsize=5.5,
        ha="center",
        va="bottom",
        color=RED,
    )
    ax.text(
        mid_x,
        y_connect - 0.10,
        r"$\langle\sigma_X\rangle$",
        fontsize=5.5,
        ha="center",
        va="top",
        color=RED,
    )

    # ==================================================================
    #  DECODING (right)
    # ==================================================================
    dec_cx = (dec_L + dec_R) / 2

    ax.text(
        dec_cx,
        panel_top - 0.07,
        r"\textsf{Decoding}",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
        color=BLACK,
    )
    ax.text(
        dec_cx,
        panel_top - 0.32,
        r"Pauli measurement",
        fontsize=7,
        ha="center",
        va="center",
        color=DARKGRAY,
    )
    ax.text(
        dec_cx,
        panel_top - 0.58,
        r"$f(r) = \frac{P(\tilde{r})}{Q(\tilde{r})}$",
        fontsize=9,
        ha="center",
        va="center",
        color=BLACK,
    )

    # ── key-insight box ───────────────────────────────────────────────
    bw = dec_R - dec_L - 0.20
    bh = 0.48
    bx = dec_cx - bw / 2
    by = panel_top - 1.22
    ax.add_patch(
        FancyBboxPatch(
            (bx, by),
            bw,
            bh,
            boxstyle="round,pad=0.04",
            facecolor=INSIGHT_FILL,
            edgecolor=INSIGHT_EDGE,
            linewidth=0.7,
            zorder=3,
        )
    )
    ax.text(
        dec_cx,
        by + bh * 0.76,
        r"$|P|, |Q| \leq 1$ \textit{bounded}",
        fontsize=6.5,
        ha="center",
        va="center",
        zorder=5,
        color=BLACK,
    )
    ax.text(
        dec_cx,
        by + bh * 0.44,
        r"but $P/Q$ can \textbf{diverge}",
        fontsize=6.5,
        ha="center",
        va="center",
        zorder=5,
        color=BLACK,
    )
    ax.text(
        dec_cx,
        by + bh * 0.16,
        r"where $Q(\tilde{r}) = 0$",
        fontsize=6,
        ha="center",
        va="center",
        zorder=5,
        color=DARKGRAY,
    )

    # ── inset plot ────────────────────────────────────────────────────
    inset_L = (dec_L + 0.10) / W  # figure-fraction coords
    inset_B = 0.06
    inset_W = (dec_R - dec_L - 0.20) / W
    inset_H = 0.34
    inset = ax.get_figure().add_axes([inset_L, inset_B, inset_W, inset_H])

    rv = np.linspace(0.1, 8.0, 300)
    P_bnd = np.sin(1.8 * rv) * np.exp(-0.12 * rv)
    r_pole = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        PQ = 1.5 * rv / (rv - r_pole)
    PQ = np.clip(PQ, -6, 6)

    # bounded band
    inset.axhspan(-1, 1, color=BLUE, alpha=0.06, zorder=0)
    inset.axhline(1, color=MEDGRAY, lw=0.4, ls="--", zorder=1)
    inset.axhline(-1, color=MEDGRAY, lw=0.4, ls="--", zorder=1)

    # curves
    inset.plot(rv, P_bnd, color=BLUE, lw=1.1, zorder=3)
    mL = rv < r_pole - 0.08
    mR = rv > r_pole + 0.08
    inset.plot(rv[mL], PQ[mL], color=RED, lw=1.1, zorder=3)
    inset.plot(rv[mR], PQ[mR], color=RED, lw=1.1, zorder=3)

    # pole
    inset.axvline(r_pole, color="#990000", lw=0.5, ls=":", zorder=2)
    inset.text(r_pole + 0.15, 4.8, r"pole", fontsize=4.5, color="#990000", va="top")
    # curve labels — inside the plotting area, well clear of edges
    inset.text(5.5, 2.0, r"$|P(\tilde{r})|$", fontsize=5, color=BLUE, ha="right")
    inset.text(5.5, -4.2, r"$P/Q$ \textbf{unbounded}", fontsize=4.5, color=RED, ha="right")
    inset.set_ylim(-6, 6)
    inset.set_xlim(0, 8.5)
    inset.set_xlabel(r"$r$", fontsize=6, labelpad=1, color=BLACK)
    inset.set_yticks([-1, 0, 1])
    inset.set_yticklabels([r"$-1$", r"$0$", r"$1$"], fontsize=5, color=BLACK)
    inset.tick_params(axis="x", labelsize=5, pad=1, colors=BLACK)
    inset.tick_params(axis="y", labelsize=5, pad=1, colors=BLACK)
    for spine in inset.spines.values():
        spine.set_color(MEDGRAY)
        spine.set_linewidth(0.4)

    # ==================================================================
    #  OVERALL FLOW ARROW
    # ==================================================================
    y_fl = H - 0.12
    ax.add_patch(
        FancyArrowPatch(
            (enc_L + 0.15, y_fl),
            (dec_R - 0.15, y_fl),
            arrowstyle="-|>",
            color=DARKGRAY,
            lw=1.4,
            mutation_scale=11,
            connectionstyle="arc3,rad=-0.06",
            zorder=10,
        )
    )
    ax.text(
        W / 2,
        y_fl + 0.08,
        r"$r \in [0,\infty) \;\longrightarrow\; "
        r"f(r) \in (-\infty, +\infty)$:"
        r" unbounded input $\to$ unbounded output",
        fontsize=6,
        ha="center",
        va="bottom",
        color=DARKGRAY,
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    fig, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH, FIG_H))
    ax.set_xlim(0, TEXTWIDTH)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_all(ax)

    outpath = os.path.join(FIGDIR, "schematic.pdf")
    fig.savefig(outpath, format="pdf")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
