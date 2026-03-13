#!/usr/bin/env bash
#
# run_publication.sh — Run all simulations at publication quality.
#
# This script produces the final data and figures for submission.
# It re-runs every simulation with high trial counts, adds the
# Heisenberg end-to-end demo (sim 7), regenerates all figures,
# and recompiles the manuscript.
#
# Expected runtime: 2–4 hours on a 28-core machine.
#
# Usage:
#   ./run_publication.sh            # Full publication run
#   ./run_publication.sh --quick    # Quick sanity check (~10 min)
#   ./run_publication.sh --sim 7    # Only the Heisenberg demo
#
# Output:
#   data/sim1_base_cases.npz      — Phase-finding k=2..12
#   data/sim2_inversion.npz       — 1/r inversion d=2..20
#   data/sim3_various_targets.npz — 8 targets × multiple degrees
#   data/sim4_convergence.npz     — Chebyshev convergence
#   data/sim5_error_analysis.npz  — MC error validation
#   data/sim6_roundtrip.npz       — Round-trip verification
#   data/sim7_heisenberg_demo.npz — Heisenberg end-to-end demo
#   ms/figures/*.pdf               — All publication figures
#   ms/main.pdf                    — Compiled manuscript
#   logs/pub_TIMESTAMP.log         — Full log
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Configuration ──────────────────────────────────────────────────────────

PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# Publication defaults (heavy)
TRIALS=500
MC_SAMPLES=500000
HEISENBERG_TRIALS=500
HEISENBERG_DEGREES="4 6 8 10 14 18"
QUICK=false
CUSTOM_SIMS=""

# ─── Parse arguments ────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            TRIALS=50
            MC_SAMPLES=50000
            HEISENBERG_TRIALS=100
            HEISENBERG_DEGREES="4 8 14"
            QUICK=true
            echo ">>> Quick mode"
            shift ;;
        --sim)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CUSTOM_SIMS="${CUSTOM_SIMS} $1"
                shift
            done ;;
        --trials)
            shift; TRIALS="$1"; HEISENBERG_TRIALS="$1"; shift ;;
        -h|--help)
            head -30 "$0" | tail -26; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Environment ────────────────────────────────────────────────────────────
# CRITICAL: single-threaded BLAS per worker to avoid contention.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p data logs ms/figures

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/pub_${TIMESTAMP}.log"

# ─── Validate ───────────────────────────────────────────────────────────────

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Run 'uv sync' first."
    exit 1
fi

cat <<BANNER
======================================================================
  Stereographic QSP — Publication Simulation Pipeline
======================================================================
  Start:             $(date)
  Python:            $PYTHON
  Trials (sims 1-6): $TRIALS
  Heisenberg trials: $HEISENBERG_TRIALS
  Heisenberg degrees:$HEISENBERG_DEGREES
  MC samples:        $MC_SAMPLES
  Log:               $LOGFILE
======================================================================
BANNER

{
    echo "Publication run — $(date)"
    echo "Trials=$TRIALS, MC=$MC_SAMPLES, Heisenberg=$HEISENBERG_TRIALS"
    echo "======================================================================"
} > "$LOGFILE"

# Helper: run with tee and time tracking
run_step() {
    local label="$1"; shift
    echo ""
    echo ">>> $label"
    echo "  Command: $*"
    local t0=$(date +%s)
    "$@" 2>&1 | tee -a "$LOGFILE"
    local rc=${PIPESTATUS[0]}
    local t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    echo "  [$label] finished in ${elapsed}s (exit $rc)"
    echo ""
    return $rc
}

T_START=$(date +%s)

# ─── Step 1: Core simulations (sims 1–6) ───────────────────────────────────

should_run() {
    # Return 0 (true) if we should run sim $1
    local sim=$1
    if [[ -z "$CUSTOM_SIMS" ]]; then
        return 0  # run all
    fi
    for s in $CUSTOM_SIMS; do
        [[ "$s" == "$sim" ]] && return 0
    done
    return 1
}

if should_run "1-6" || should_run 1 || should_run 2 || should_run 3 || \
   should_run 4 || should_run 5 || should_run 6; then

    SIM_CMD="$PYTHON scripts/run_simulations.py --trials $TRIALS --mc $MC_SAMPLES"

    # If specific sims requested, filter to 1-6 only
    if [[ -n "$CUSTOM_SIMS" ]]; then
        CORE_SIMS=""
        for s in $CUSTOM_SIMS; do
            if [[ "$s" -ge 1 && "$s" -le 6 ]]; then
                CORE_SIMS="$CORE_SIMS $s"
            fi
        done
        if [[ -n "$CORE_SIMS" ]]; then
            SIM_CMD="$SIM_CMD --sim $CORE_SIMS"
            run_step "Simulations 1-6 (selected:$CORE_SIMS)" $SIM_CMD || true
        fi
    else
        run_step "Simulations 1-6 (all)" $SIM_CMD || true
    fi
fi

# ─── Step 2: Heisenberg end-to-end demo (sim 7) ────────────────────────────

if should_run 7 || [[ -z "$CUSTOM_SIMS" ]]; then
    run_step "Simulation 7: Heisenberg demo" \
        $PYTHON scripts/heisenberg_demo_pub.py \
        --trials "$HEISENBERG_TRIALS" \
        --degrees $HEISENBERG_DEGREES \
    || true
fi

# ─── Step 3: Generate all figures ───────────────────────────────────────────

run_step "Figure generation (sims 1-6)" \
    $PYTHON scripts/generate_figures_from_data.py || true

# The Heisenberg figure is generated by the demo script itself,
# but let's also regenerate from saved data for consistency.
if [[ -f "data/sim7_heisenberg_demo.npz" ]]; then
    run_step "Figure generation (Heisenberg)" \
        $PYTHON scripts/generate_heisenberg_figure.py || true
fi

# ─── Step 4: Compile manuscript ─────────────────────────────────────────────

echo ">>> Compiling manuscript..."
cd ms
if command -v pdflatex &> /dev/null; then
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
    bibtex main > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
    PAGES=$(pdflatex -interaction=nonstopmode main.tex 2>&1 \
        | grep "Output written" | grep -oP '\(\K[0-9]+' || echo "?")
    echo "  Manuscript: main.pdf ($PAGES pages)"
else
    echo "  pdflatex not found, skipping."
fi
cd "$SCRIPT_DIR"

# ─── Summary ────────────────────────────────────────────────────────────────

T_END=$(date +%s)
TOTAL_SEC=$(( T_END - T_START ))
TOTAL_MIN=$(( TOTAL_SEC / 60 ))

echo ""
cat <<SUMMARY
======================================================================
  Publication pipeline complete!
  End:       $(date)
  Duration:  ${TOTAL_SEC}s (${TOTAL_MIN} min)
  Log:       $LOGFILE
----------------------------------------------------------------------
  Data files:
$(ls -lhS data/*.npz 2>/dev/null | awk '{print "    " $NF " (" $5 ")"}')
  Figures:
$(ls -lhS ms/figures/*.pdf 2>/dev/null | awk '{print "    " $NF " (" $5 ")"}')
  Manuscript:
    ms/main.pdf ($(du -h ms/main.pdf 2>/dev/null | cut -f1 || echo "?"))
======================================================================
SUMMARY
