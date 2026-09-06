.PHONY: all audit assets pdf clean

all: pdf

audit:
	python3 scripts/audit_downloaded_results.py

assets: audit
	python3 scripts/build_assets.py

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -c main.tex
