"""Validate data lineage, numerical tables, PDF structure and render all pages."""
from pathlib import Path
import csv
import hashlib
import json
import re
import pymupdf as fitz
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'report'

def main():
    hashes = json.loads((HERE/'sources.json').read_text(encoding='utf-8'))
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    print(f'PASS: {len(hashes)} frozen input SHA256 checks')
    for csv_name, tex_name, group in [('method_summary.csv','method_rows.tex','semantic'),
                                       ('scale_summary.csv','scale_rows.tex',None)]:
        with (ROOT/'results/raw'/csv_name).open(encoding='utf-8-sig', newline='') as f:
            rs = list(csv.DictReader(f))
        if group:
            rs = [r for r in rs if r['group'] == group]
        lines = (HERE/tex_name).read_text(encoding='utf-8').splitlines()
        assert len(lines) == len(rs)+1
        for r,line in zip(rs,lines):
            cells = [c.strip() for c in line.removesuffix('\\\\').split('&')]
            expected = ([f"{float(r['recall'])*100:.2f}", f"{float(r['mean_candidate_ratio'])*100:.4f}",
                         f"{float(r['mean_time_ms']):.4f}", f"{float(r['mean_comparisons']):.2f}", f"{float(r['index_mib']):.3f}"]
                        if group else [f"{float(r['recall'])*100:.2f}",f"{float(r['mean_candidates']):.2f}",
                         f"{float(r['mean_false_candidates']):.2f}",f"{float(r['mean_time_ms']):.3f}",f"{int(r['index_bytes'])/2**20:.3f}"])
            assert cells[1:] == expected, (csv_name, cells, expected)
    print('PASS: all 30 numerical result-table cells match raw CSV rounding')
    tex = (HERE/'experiment_report.tex').read_text(encoding='utf-8')
    reflection = tex.split(r'\textbf{最后思考。}',1)[1].split(r'\end{document}')[0].strip()
    assert len(reflection) <= 300
    print(f'PASS: final reflection {len(reflection)} characters <= 300')
    log = (ROOT/'tmp/pdfs/experiment_report.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'Overfull|Missing character|Undefined control sequence|LaTeX Error', log)
    print('PASS: no overfull boxes, missing glyphs or LaTeX errors')
    path = ROOT/'output/pdf/experiment_report.pdf'
    reader = PdfReader(path)
    assert len(reader.pages) == 4
    document = fitz.open(path)
    assert len(document) == 4
    assert all(abs(p.rect.width - 595.28) < 1 and abs(p.rect.height - 841.89) < 1 for p in document)
    pages = [p.get_text() for p in document]
    expected_sections = [['问题、数据与表示','两层索引'], ['核心算法','统一评测'],
                         ['三方法实测','Exact 子组'], ['规模关系','最后思考']]
    for i,(page,required) in enumerate(zip(document,expected_sections),1):
        text = pages[i-1]
        for token in required:
            assert token in text, (i,token)
        assert '\ufffd' not in text
        page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(ROOT/f'tmp/pdfs/verified-page-{i}.png')
    for token in ['96.77','153.07','139.005','15,307','没有实测']:
        assert token in '\n'.join(pages), token
    (ROOT/'tmp/pdfs/verified_text.txt').write_text('\n\f\n'.join(pages),encoding='utf-8')
    print('PASS: 4 A4 pages, required sections and numerical landmarks extracted; 4 PNGs rendered')
    print('VISUAL REVIEW REQUIRED: inspect tmp/pdfs/verified-page-1.png through -4.png')

if __name__ == '__main__':
    main()
