# Makefile for REVTeX manuscript compilation

# Main file
MAIN = main
BIB = references

# Default target
all: $(MAIN).pdf

# Compile with latexmk (recommended)
$(MAIN).pdf: $(MAIN).tex $(BIB).bib sections/*.tex
	latexmk -pdf $(MAIN).tex

# Alternative: manual compilation
manual:
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

# Clean auxiliary files
clean:
	latexmk -C
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.synctex.gz
	rm -f sections/*.aux

# Clean everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# Quick recompile (no bibliography)
quick:
	pdflatex $(MAIN).tex

# View PDF (adjust viewer as needed)
view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || start $(MAIN).pdf

# Continuous compilation (watches for changes)
watch:
	latexmk -pdf -pvc $(MAIN).tex

# Check for common LaTeX errors
check:
	@echo "Checking for common issues..."
	@grep -n "TODO" sections/*.tex || echo "No TODOs found"
	@grep -n "\\cite{}" sections/*.tex || echo "No empty citations"
	@grep -n "???" $(MAIN).log 2>/dev/null || echo "No undefined references"

.PHONY: all manual clean cleanall quick view watch check
