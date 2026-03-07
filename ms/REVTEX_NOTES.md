# REVTeX Conversion Notes

## What Changed

The manuscript has been converted from standard LaTeX `article` class to **REVTeX 4.2** format, which is the standard for Physical Review journals (APS).

## Key Differences

### Document Class
**Before:**
```latex
\documentclass[11pt,a4paper]{article}
```

**After:**
```latex
\documentclass[
    reprint,           % Two-column format
    amsmath,amssymb,   % AMS math support
    aps,               % American Physical Society
    pra,               % Physical Review A (quantum computing)
    showpacs,          % Show PACS numbers
    superscriptaddress,% Superscript author affiliations
    floatfix           % Improved float placement
]{revtex4-2}
```

### Author Information

**⚠️ CRITICAL: Title must be INSIDE `\begin{document}`!**

**Before (standard LaTeX):**
```latex
\title{My Title}
\author{Tyler Kharazi \and Vicente Leyton-Ortega}
\begin{document}
\maketitle
```

**After (REVTeX):**
```latex
\begin{document}
\title{My Title}
\author{Tyler Kharazi}
\affiliation{Institution TBD}

\author{Vicente Leyton-Ortega}
\affiliation{Institution TBD}

\begin{abstract}
...
\end{abstract}

\pacs{...}
\maketitle
```

**Key points:**
- `\title`, `\author`, `\affiliation` go **INSIDE** `\begin{document}`
- They come **BEFORE** `\begin{abstract}`
- `\maketitle` comes **AFTER** `\pacs` and abstract
- Otherwise the title won't appear in the PDF!

### Abstract and Metadata
**Before:**
```latex
\maketitle
\begin{abstract}
...
\end{abstract}
```

**After:**
```latex
\begin{abstract}
...
\end{abstract}

\pacs{03.67.Lx, 03.67.Ac, 02.30.Fn}

\maketitle
```

### Bibliography
**Before:**
```latex
\bibliographystyle{plain}
\bibliography{references}
```

**After:**
```latex
\bibliography{references}
```
(REVTeX handles bibliography style automatically)

### Removed Features
- **Table of contents**: Not typically used in journal articles
- **Custom theorem environments**: REVTeX provides its own
- **Explicit theorem styling**: Built into REVTeX

### Added Features
- **PACS numbers**: Physics classification codes
  - `03.67.Lx` - Quantum computation architectures
  - `03.67.Ac` - Quantum algorithms and protocols
  - `02.30.Fn` - Complex variables and analytic spaces
- **Acknowledgments section**: Standard for journal submissions
- **Superscript affiliations**: Professional author formatting

## Packages

### Removed (built into REVTeX):
- `inputenc`, `fontenc` (handled automatically)
- `amsthm` (REVTeX provides theorem environments)
- `physics` (bra-ket notation included)
- `mathtools` (amsmath extension included)
- `cleveref` (REVTeX has its own cross-referencing)
- `float` (replaced by `floatfix` option)

### Kept:
- `graphicx` - For figures
- `hyperref` - For hyperlinks
- `algorithm`, `algorithmic` - For algorithms
- `tikz` - For diagrams
- `bm` - Bold math symbols
- `dcolumn` - Decimal-aligned columns (REVTeX recommended)

## Building

### Quick Build:
```bash
make
```

### Watch Mode (auto-recompile on save):
```bash
make watch
```

### Clean:
```bash
make clean
```

## Journal Target

**Physical Review A (PRA)**
- Focus: Atomic, molecular, and optical physics, quantum information
- Perfect fit for quantum algorithms and quantum computing theory
- Two-column format in reprint mode
- Single-column for submission mode (change `reprint` to `preprint`)

## Alternative Journals

If targeting a different journal, modify the document class options:

### PRX Quantum (more applied):
```latex
\documentclass[prxquantum,...]{revtex4-2}
```

### Quantum (open access):
```latex
% Use Quantum template instead (different format)
```

### Physical Review Letters (short):
```latex
\documentclass[prl,...]{revtex4-2}
```

## Submission Modes

### For Submission (single column):
```latex
\documentclass[preprint,...]{revtex4-2}
```

### For Journal (two column):
```latex
\documentclass[reprint,...]{revtex4-2}
```

### For ArXiv (single column, larger):
```latex
\documentclass[preprint,12pt,...]{revtex4-2}
```

## Notes

1. REVTeX automatically handles figure and table placement
2. Equations are numbered automatically per section
3. References are formatted according to APS style
4. Cross-references work with standard `\ref{}`
5. Bra-ket notation: Use `\ket{}`, `\bra{}`, `\braket{}{}`
