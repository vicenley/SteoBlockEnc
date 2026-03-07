#!/bin/bash
# Simple compilation script for the manuscript

set -e  # Exit on error

echo "======================================"
echo "  Compiling Stereographic Encoding"
echo "======================================"
echo ""

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "❌ ERROR: pdflatex not found!"
    echo ""
    echo "Please install LaTeX:"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-publishers"
    echo "  Fedora:        sudo dnf install texlive-scheme-medium"
    echo "  Arch:          sudo pacman -S texlive-core texlive-publishers"
    exit 1
fi

# Check if REVTeX is installed
if ! kpsewhich revtex4-2.cls &> /dev/null; then
    echo "❌ ERROR: REVTeX 4.2 not found!"
    echo ""
    echo "Please install REVTeX:"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-publishers"
    exit 1
fi

echo "✓ pdflatex found: $(which pdflatex)"
echo "✓ REVTeX found"
echo ""

# Compilation
echo "📝 Compiling main.tex (pass 1/3)..."
if ! pdflatex -interaction=batchmode main.tex > /dev/null 2>&1; then
    # Check if PDF was still generated (warnings vs errors)
    if [ ! -f main.pdf ]; then
        echo "❌ Compilation failed! Check main.log for errors"
        tail -50 main.log
        exit 1
    fi
fi

# Check if bibliography exists and has entries
if [ -f references.bib ] && grep -q "@" references.bib; then
    echo "📚 Processing bibliography..."
    bibtex main > /dev/null 2>&1 || true
    
    echo "📝 Compiling main.tex (pass 2/3)..."
    pdflatex -interaction=batchmode main.tex > /dev/null 2>&1
fi

echo "📝 Compiling main.tex (pass 3/3)..."
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1

echo ""
echo "======================================"
echo "✅ SUCCESS!"
echo "======================================"
echo ""
echo "Output: main.pdf"
ls -lh main.pdf
echo ""

# Count warnings
WARNINGS=$(grep -c "Warning" main.log || true)
if [ "$WARNINGS" -gt 0 ]; then
    echo "⚠️  $WARNINGS warnings found (see main.log)"
else
    echo "✓ No warnings"
fi

# Check for undefined references
if grep -q "undefined" main.log; then
    echo "⚠️  Undefined references found (add entries to references.bib)"
fi

echo ""
echo "To view: make view"
echo "To clean: make clean"
