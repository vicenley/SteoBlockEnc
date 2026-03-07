# How to Compile the Manuscript

## ⚠️ IMPORTANT: Use `pdflatex`, NOT `pdftex`

### The Difference

- **`pdftex`** - Low-level TeX engine (doesn't understand LaTeX)
- **`pdflatex`** - LaTeX compiler that uses pdftex engine (correct one!)

## ✅ Correct Compilation

### Method 1: Using Make (Easiest)

```bash
cd ms/
make
```

This runs all necessary commands automatically.

### Method 2: Manual Compilation

```bash
cd ms/

# First pass
pdflatex main.tex

# Process bibliography (when you add citations)
bibtex main

# Second pass (resolve references)
pdflatex main.tex

# Third pass (finalize)
pdflatex main.tex
```

### Method 3: Using latexmk (Recommended for development)

```bash
cd ms/

# Single command - does everything
latexmk -pdf main.tex

# Watch mode - auto-recompiles on file changes
latexmk -pdf -pvc main.tex
```

## 🔍 Quick Build (No Bibliography)

If you just want to see the output quickly:

```bash
cd ms/
pdflatex main.tex
```

This is enough for a quick preview. You'll see warnings about missing citations, but the PDF will be generated.

## 🛠️ Available Make Commands

```bash
make          # Compile the manuscript
make clean    # Remove auxiliary files
make cleanall # Remove everything including PDF
make quick    # Quick compile (no bibliography)
make watch    # Auto-recompile on changes
make view     # Open the PDF
```

## ❌ Common Errors

### Error: "pdftex: command not found"

**Problem:** You're using `pdftex` instead of `pdflatex`

**Solution:** Use `pdflatex main.tex`

### Error: "! Undefined control sequence"

**Problem:** Missing package or incompatible LaTeX installation

**Solution:** 
```bash
# Install REVTeX and dependencies
sudo apt-get install texlive-publishers texlive-latex-extra
```

### Error: "! LaTeX Error: File 'revtex4-2.cls' not found"

**Problem:** REVTeX not installed

**Solution:**
```bash
sudo apt-get install texlive-publishers
```

### Error: "! LaTeX Error: File 'braket.sty' not found"

**Problem:** Missing braket package

**Solution:**
```bash
sudo apt-get install texlive-latex-extra
```

## 📦 Required Packages

Make sure you have these installed:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-base texlive-latex-base texlive-latex-extra texlive-publishers

# Fedora
sudo dnf install texlive-scheme-medium texlive-revtex

# Arch Linux
sudo pacman -S texlive-core texlive-publishers
```

## 🧪 Test Your Installation

Create and compile a minimal test file:

```bash
cd ms/
cat > test.tex << 'TESTEOF'
\documentclass[reprint,amsmath,aps,pra]{revtex4-2}
\usepackage{braket}
\title{Test}
\author{Test Author}
\begin{document}
\maketitle
Test: $\ket{0}$
\end{document}
TESTEOF

pdflatex test.tex
```

If this works, your installation is correct!

## 📊 What Gets Generated

After successful compilation:

```
ms/
├── main.pdf          ← The compiled manuscript
├── main.aux          ← Auxiliary file
├── main.log          ← Compilation log (check for errors/warnings)
├── main.out          ← Hyperref output
├── main.bbl          ← Bibliography (after bibtex)
└── main.blg          ← Bibliography log
```

## 🎯 Summary

**DO THIS:**
```bash
cd ms/
pdflatex main.tex
```

**NOT THIS:**
```bash
pdftex main.tex  # ❌ WRONG - won't work with LaTeX!
```

**Or even better:**
```bash
cd ms/
make
```
