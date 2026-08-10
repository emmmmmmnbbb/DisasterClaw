# DisasterClaw paper

CVPR-style anonymous draft:

```bash
cd /home/lc/disasterclaw
python scripts/benchmarks/export_paper_assets.py
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

The source of every generated table and plot is recorded in
`generated/provenance.json`. Historical results are explicitly labeled in the
text and must be replaced only by completed strict-split runs.

Key editorial contracts:

- `CLAIMS.md`: allowed and prohibited scientific claims.
- `EXPERIMENT_PROTOCOL.md`: pre-registered strict evaluation.
- `sections/`: main English paper.
- `appendix/`: protocol and implementation details.
- `ref/arXiv-2604.07765v2.tar.gz`: structural reference only; its method and
  prose are not copied.

Clean generated LaTeX files with:

```bash
latexmk -C -cd paper/main.tex
```
