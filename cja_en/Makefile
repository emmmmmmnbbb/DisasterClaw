.PHONY: all audit assets section6-figures pdf clean

all: pdf

audit: section6-figures

assets: section6-figures

section6-figures:
	python3 scripts/build_section6_figures.py

pdf: section6-figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -c main.tex
