#!/usr/bin/env python3
"""Technical manuscript checks; does not certify scientific submission readiness."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    issues = []
    files = [HERE / 'main.tex', *sorted((HERE / 'sections').glob('*.tex')),
             *sorted((HERE / 'appendix').glob('*.tex')), *sorted((HERE / 'figures').glob('*.tex')),
             *sorted((HERE / 'tables').glob('*.tex'))]
    tex = '\n'.join(p.read_text() for p in files)
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    duplicates = [k for k, n in Counter(labels).items() if n > 1]
    refs = set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', tex))
    bib = (HERE / 'references.bib').read_text()
    bib_keys = re.findall(r'@\w+\{([^,]+)', bib)
    citations = {k.strip() for group in re.findall(r'\\cite\w*(?:\[[^\]]*\])?\{([^}]+)\}', tex) for k in group.split(',')}
    missing_cites = sorted(citations - set(bib_keys))
    missing_refs = sorted(refs - set(labels))
    if duplicates or missing_cites or missing_refs:
        issues.append('Duplicate labels or unresolved source references')
    missing_inputs = []
    for name in re.findall(r'\\input\{([^}]+)\}', tex):
        p = HERE / name
        if not p.suffix:
            p = p.with_suffix('.tex')
        if not p.is_file():
            missing_inputs.append(name)
    if missing_inputs:
        issues.append('Missing input assets')
    if re.search('[\u4e00-\u9fff]', tex):
        issues.append('Chinese characters in English TeX source')
    log = (HERE / 'main.log').read_text(errors='replace')
    latex_warnings = [line for line in log.splitlines() if 'Warning:' in line or 'Overfull' in line or 'Underfull' in line]
    if re.search(r'^!|undefined|Overfull|multiply defined', log, re.M):
        issues.append('Unresolved LaTeX diagnostic')
    bib_warnings = [line for line in (HERE / 'main.blg').read_text().splitlines() if line.startswith('Warning--')]
    info = subprocess.check_output(['pdfinfo', str(HERE / 'main.pdf')], text=True)
    page_count = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
    text = subprocess.check_output(['pdftotext', '-layout', str(HERE / 'main.pdf'), '-'], text=True)
    if '??' in text or 'Appendix Appendix' in text:
        issues.append('Unresolved or duplicated reference in PDF text')
    pdf_pages = [p for p in text.split('\f') if p.strip()]
    if len(pdf_pages) != page_count:
        issues.append('Unexpected empty PDF page')
    src = json.loads((HERE / 'review/source_manifest.json').read_text())
    changed = [r['path'] for r in src['files'] if not (ROOT / r['path']).is_file() or sha(ROOT / r['path']) != r['sha256']]
    if changed:
        issues.append('Original source fingerprint changed; inspect user edits before proceeding')
    audit = json.loads((HERE / 'review/downloaded_results_audit.json').read_text())
    changed_runs = [p for p, r in audit['sources'].items() if not (ROOT / p).is_file() or sha(ROOT / p) != r['sha256']]
    if changed_runs or audit['offline_numeric_mismatches'] or audit['vqa_duplicates']:
        issues.append('Raw input or arithmetic audit mismatch')
    analysis = json.loads((HERE / 'review/manuscript_analysis.json').read_text())
    if analysis['input_audit_sha256'] != sha(HERE / 'review/downloaded_results_audit.json'):
        issues.append('Analysis is not bound to the current audit')
    visual_path = HERE / 'review/visual_qa.json'
    visual = json.loads(visual_path.read_text()) if visual_path.exists() else {}
    visual_current = visual.get('pdf_sha256') == sha(HERE / 'main.pdf') and visual.get('pages_reviewed') == list(range(1, page_count + 1))
    result = {'scope': 'technical consistency only; submission readiness remains incomplete',
              'errors': issues, 'pdf_pages': page_count, 'pdf_sha256': sha(HERE / 'main.pdf'),
              'latex_warnings': latex_warnings, 'bibtex_warnings': bib_warnings,
              'citations': len(citations), 'bib_entries': len(bib_keys),
              'unused_bib_entries': sorted(set(bib_keys) - citations), 'missing_citations': missing_cites,
              'missing_references': missing_refs, 'duplicate_labels': duplicates,
              'missing_inputs': missing_inputs, 'source_files_checked': len(src['files']),
              'changed_original_sources': changed, 'run_files_checked': len(audit['sources']),
              'changed_run_inputs': changed_runs, 'visual_qa_matches_current_pdf': visual_current}
    (HERE / 'review/technical_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
