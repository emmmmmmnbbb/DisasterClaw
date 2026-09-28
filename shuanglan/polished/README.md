# DisasterClaw CJA manuscript

This is the single maintained LaTeX manuscript project. The entry point, section sources, tables, figures, appendices, and bibliography are all stored in this directory; compilation does not depend on another manuscript directory.

Build from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Or build from the repository root:

```sh
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error shuanglan/polished/main.tex
```

The PDF is written to `main.pdf`. MESSI experiment records and reproducibility materials remain under `../../experiments/messi_case/`; they are not required to compile the manuscript.
