"""Build presentation artifacts from frozen experiment CSVs; never run retrieval."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'results/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'report'
TMP = ROOT / 'tmp/pdfs'
OUT = ROOT / 'output/pdf'

def rows(name):
    with (ROOT / 'results/raw' / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / 'figures').mkdir(exist_ok=True)
    grid = rows('validation_grid.csv')
    methods = rows('method_summary.csv')
    scale = rows('scale_summary.csv')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.5, 2.55), layout='constrained')
    for k, marker, style in [(8, 'o', '-'), (12, 's', '--'), (16, '^', ':')]:
        rs = [r for r in grid if int(r['k']) == k]
        ax.plot([float(r['mean_candidate_ratio'])*100 for r in rs],
                [float(r['recall']) for r in rs], marker=marker, linestyle=style,
                color='black', label=f'k={k}; L=4,8,16,32')
        for r in rs:
            ax.annotate(r['L'], (float(r['mean_candidate_ratio'])*100, float(r['recall'])),
                        xytext=(5, -11 if k == 12 else 5), textcoords='offset points', fontsize=7)
    selected = next(r for r in grid if r['selected'].lower() == 'true')
    ax.plot(float(selected['mean_candidate_ratio'])*100, float(selected['recall']),
            marker='*', color='black', markersize=14)
    ax.axvline(5, color='0.45', linestyle=':', linewidth=1)
    ax.axhline(.9, color='0.45', linestyle=':', linewidth=1)
    ax.set(xscale='log', xlabel='Mean candidate ratio (%) [log scale]', ylabel='Conditional recall', ylim=(.19, 1.07))
    ax.legend(loc='lower right', fontsize=7)
    fig.savefig(HERE / 'figures/recall_candidate.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.05), layout='constrained')
    ns = [int(r['N']) for r in scale]
    axes[0].plot(ns, [float(r['mean_candidates']) for r in scale], 'ko-', label='Candidates')
    axes[0].plot(ns, [float(r['mean_false_candidates']) for r in scale], 'k--^', label='False candidates')
    axes[0].set(xlabel='Measured corpus size N', ylabel='Mean count / query')
    axes[0].legend(fontsize=7)
    axes[1].plot(ns, [float(r['mean_time_ms']) for r in scale], 'ko-')
    axes[1].set(xlabel='Measured corpus size N', ylabel='Mean query time (ms)')
    for ax in axes:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    fig.savefig(HERE / 'figures/scale.pdf')
    plt.close(fig)
    def table(name, lines):
        (HERE / name).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    table('method_rows.tex', [
        f"{ {'Brute-force':'Brute-force', 'Exact Hash Only':'Exact-only', 'Hash + LSH':'Hash + LSH'}[r['method']]} & "
        f"{float(r['recall'])*100:.2f} & {float(r['mean_candidate_ratio'])*100:.4f} & "
        f"{float(r['mean_time_ms']):.4f} & {float(r['mean_comparisons']):.2f} & {float(r['index_mib']):.3f} \\\\"
        for r in methods if r['group'] == 'semantic'] + [r'\bottomrule'])
    table('scale_rows.tex', [
        f"{int(r['N']):,} & {float(r['recall'])*100:.2f} & {float(r['mean_candidates']):.2f} & "
        f"{float(r['mean_false_candidates']):.2f} & {float(r['mean_time_ms']):.3f} & {int(r['index_bytes'])/2**20:.3f} \\\\"
        for r in scale] + [r'\bottomrule'])
    sources = ['results/raw/'+n for n in ['validation_grid.csv', 'method_summary.csv', 'scale_summary.csv',
               'scale_predictions.csv', 'threshold_analysis.json', 'environment.json', 'phase5_manifest.json', 'scale_manifest.json']]
    sources += ['configs/selected_experiment.yaml', 'data/processed/manifest.json']
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    (HERE/'sources.json').write_text(json.dumps(hashes, indent=2)+'\n', encoding='utf-8')
    for _ in range(2):
        result = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                                 f'-output-directory={TMP}', 'experiment_report.tex'],
                                cwd=HERE, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (TMP/'build-console.log').write_text(result.stdout+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('XeLaTeX failed; see tmp/pdfs/build-console.log')
    target = OUT/'experiment_report.pdf'
    target.write_bytes((TMP/'experiment_report.pdf').read_bytes())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    print(f'Built {target}; {len(hashes)} frozen inputs unchanged.')

if __name__ == '__main__':
    main()
