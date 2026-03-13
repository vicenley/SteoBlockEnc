# AGENTS.md — Coding Agent Instructions for SteoBlockEnc

Research repository for the paper *"Lifting the Boundedness Constraint in
Quantum Signal Processing via Stereographic Encoding"*. Contains a Python
library (`stereo_block_enc`), numerical simulation scripts, and a LaTeX
manuscript (REVTeX 4.2 / Physical Review A).

## Build & Install

```bash
uv sync               # install core deps into .venv/
uv sync --extra dev   # include pytest, black, ruff, jupyter
source .venv/bin/activate
```

Package uses src-layout: importable as `from stereo_block_enc.numerical import ...`.

## Test Commands

```bash
pytest                          # run all tests (testpaths = tests/)
pytest tests/test_foo.py        # single file
pytest tests/test_foo.py::test_bar  # single test
pytest -x                       # stop on first failure
pytest -k "phase"               # keyword filter
pytest --no-cov                 # skip coverage (faster)
```

Tests directory is currently empty — if you write tests, place them in `tests/`
following the pattern `test_*.py` with functions named `test_*`.

## Lint & Format

```bash
ruff check .                    # lint (E, W, F, I, B, C4, UP rules)
ruff check --fix .              # auto-fix
black --check .                 # formatting check
black .                         # auto-format
mypy src/                       # type checking (lenient config)
```

Line length: **100** characters (black + ruff). Target: Python 3.9+.

## LaTeX Manuscript

```bash
cd ms/
make manual    # pdflatex -> bibtex -> pdflatex -> pdflatex
make quick     # single pdflatex pass (no bib update)
make check     # grep for TODOs, empty citations, undefined refs
make clean     # remove aux files
```

Format: REVTeX 4.2, PRA style, two-column reprint, `floatfix` option.
Title/author/abstract go INSIDE `\begin{document}`. Circuit diagrams use
`quantikz`. All figures use `figure*` (full-width) with `\textwidth` sizing.

## Simulation Pipeline

```bash
# Full publication run (500 trials, ~2 hours on 28 cores):
bash run_publication.sh

# Specific simulations:
python scripts/run_simulations.py --sim 1 2 --trials 50

# Regenerate figures from saved data (fast):
python scripts/generate_figures_from_data.py
python scripts/generate_heisenberg_figure.py

# Quantum circuit verification (Qulacs):
python scripts/qulacs_heisenberg_sim.py
```

Data files live in `data/*.npz`. Figures output to `ms/figures/*.pdf`.

## Critical: BLAS Thread Control

**MUST** set these environment variables BEFORE importing numpy in any
parallel script. Without this, each of N workers spawns 28 BLAS threads,
causing massive contention on the 28-core machine.

```python
import os
for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'BLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')
import numpy as np  # AFTER setting env vars
```

## SageMath

Installed in conda environment `sage` (v10.7). Access via:
```bash
conda run -n sage sage -c "..."
```
Note: Sage's `random.seed()` only accepts `None, int, float, str, bytes,
bytearray` — not Sage integers.

## Code Style

### Imports
- Standard library → third-party → local. Ruff enforces isort ordering.
- Scripts use `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))`
  before importing from the package.
- `__init__.py` files use star-imports (`from .module import *`); F401 is suppressed.

### Naming
- **Classes**: `PascalCase` — `StereographicEncoding`, `QSPStereographic`
- **Functions/methods**: `snake_case` — `find_phases_stereo`, `encoding_state`
- **Constants**: `SCREAMING_SNAKE_CASE` — `DATADIR`, `FIGDIR`, `TEXTWIDTH`
- **Math variables**: short names allowed — `r`, `z`, `a`, `k`, `d`, `P`, `Q`, `W`

### Docstrings
NumPy-style with Parameters/Returns sections:
```python
def encoding_state(self, z=None):
    """
    Return the symbolic quantum state |z>.

    Parameters
    ----------
    z : sympy expression, optional
        Complex number to encode. If None, uses symbolic z.

    Returns
    -------
    state : sympy Matrix
        2x1 column vector representing |z>
    """
```

### Type Hints
- Optional — not enforced strictly (`disallow_untyped_defs = false` in mypy).
- Used in numerical code: `np.ndarray`, `int`, `float`, `tuple`, `str`.
- Symbolic modules omit type hints entirely; rely on docstrings.

### Error Handling
- `ValueError` for invalid math inputs (e.g., singular Mobius matrices).
- `try/except` with traceback printing in simulation scripts (non-fatal).
- `np.errstate(divide='ignore', invalid='ignore')` for expected singularities.
- Division-by-zero guards: `if abs(x) < 1e-15: return np.inf`.
- Shell scripts use `set -euo pipefail`.

### Figures
- `matplotlib.use('Agg')` for headless rendering in scripts.
- Okabe-Ito colorblind-safe palette.
- Serif fonts, 9pt, 300 DPI for publication quality.
- Two-tier: heavy sim → `.npz` data, then fast figure generation from data.

## Qulacs Conventions

Qulacs (v0.6.13) has non-standard conventions — watch for these:
- **Rotation sign**: `RX/RY/RZ(θ) = exp(+iθ/2 σ)` (opposite from textbook).
  Use `DenseMatrix(indices, M)` with explicit 2×2 matrices to avoid sign bugs.
- **Gate ordering**: first gate added = first applied to state. For the QSP
  product M = R₀ W R₁ W ··· Rₐ, add gates in REVERSE order.
- **Qubit ordering**: `DenseMatrix([q0, q1], M)` uses little-endian:
  index = q0 + 2*q1. This swaps indices 1↔2 vs numpy's big-endian convention.

## Project Layout

```
src/stereo_block_enc/
├── __init__.py
├── symbolic/          # SymPy: encoding, Mobius, QSP algebra
│   ├── stereographic.py
│   ├── mobius.py
│   └── qsp.py
└── numerical/         # NumPy/SciPy: phase-finding, simulations
    └── qsp_phases.py

scripts/               # Standalone simulation & figure scripts
tests/                 # pytest tests (empty — write tests here)
ms/                    # LaTeX manuscript (REVTeX 4.2)
data/                  # Simulation output (.npz)
notebooks/             # Jupyter notebooks for exploration
```

## Inkscape & Figure Workflow

Inkscape 1.4.3 is installed at `/snap/bin/inkscape`. TexText extension is
installed in `~/.config/inkscape/extensions/textext/`.

**inkscape-mcp** (grumpydevorg/inkscape-mcps) is installed and configured in
`.mcp.json` (project scope) and `~/.claude.json` (user scope). Status:
connected but tools may not load into Claude Code sessions due to a known
bug. **Workaround**: use Inkscape CLI directly via bash:

```bash
# List available actions:
/snap/bin/inkscape --action-list

# Run actions on an SVG:
/snap/bin/inkscape --actions="action1;action2" input.svg -o output.svg

# Export SVG to PDF:
/snap/bin/inkscape input.svg --export-type=pdf --export-filename=output.pdf
```

**Figure 1** is currently inline TikZ in `ms/sections/qsp_01_introduction.tex`
with known overlapping/crowding issues. Plan: redo as SVG using Inkscape +
TexText, export to PDF, include via `\includegraphics`. Old Inkscape source:
`ms/QSPExtended/figs/Steofig.svg`.

## Writing Style (Manuscript)

- **No lists in the introduction.** Narrative prose only — all
  enumerate/itemize environments were removed and content woven into
  flowing paragraphs.
- **Sections II–VI** may use lists where technically appropriate.

## LSP False Positives

Ignore "module not resolved" errors for `sympy`, `numpy`, `qulacs`,
`scipy`, `matplotlib`. These are installed in `.venv/` and work at runtime.
