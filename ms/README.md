# Manuscript: Stereographic Encoding

## Structure

- `main.tex` - Main manuscript file
- `sections/` - Individual sections
  - `01_introduction.tex` - Introduction and motivation
  - `02_preliminaries.tex` - Mathematical background
  - `03_encoding.tex` - Stereographic encoding definition and properties
  - `04_mobius.tex` - Möbius transformations from quantum gates
  - `05_qsp.tex` - Quantum signal processing
  - `06_dynamics.tex` - Hamiltonian dynamics and Berry phase
  - `07_applications.tex` - Applications
  - `08_discussion.tex` - Discussion and future work
  - `appendix_*.tex` - Appendices
- `figures/` - Figures and diagrams
- `references.bib` - Bibliography

## Building

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk:
```bash
latexmk -pdf main.tex
```

## Equation Verification

All equations are verified symbolically using SymPy in:
`../notebooks/symbolic/02_equation_verification.ipynb`

Run this notebook to:
1. Verify all formulas from the theory
2. Generate LaTeX code for equations
3. Check consistency between sections

## Status

- [x] Outline created
- [x] Section 1: Introduction (complete)
- [x] Section 2: Preliminaries (complete)
- [x] Section 3: Encoding (complete)
- [ ] Section 4: Möbius transformations (TODO)
- [ ] Section 5: QSP (TODO)
- [ ] Section 6: Dynamics (TODO)
- [ ] Section 7: Applications (TODO)
- [ ] Section 8: Discussion (TODO)
- [ ] Appendices (TODO)
- [ ] Figures (TODO)
- [ ] Bibliography (TODO)

## Next Steps

1. Run `notebooks/symbolic/02_equation_verification.ipynb` to verify equations
2. Complete Section 4 with Möbius transformation formulas (Eqs. 24-30, 36-37)
3. Complete Section 5 with QSP analysis (Eqs. 41-44, 57-59)
4. Add figures for geometric visualization
5. Complete applications section
