#!/usr/bin/env python3
"""Stratify unchanged paper predictions by oracle first-block completion.

Run after report_joint_allocation.py. Selection uses oracle measurements only;
errors are computed after summing predictions and outcomes within each subset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ('solved_geometric', 'neither_regularized', 'both_regularized')
PANELS = {1: ('gpt54', 'gpt55', 'opus'), 2: ('gpt54', 'gpt55', 'opus')}
NAMES = {'gpt54': 'GPT-5.4', 'gpt55': 'GPT-5.5', 'opus': 'Opus'}
GROUP_LABELS = {'complete': 'Saturated', 'incomplete': 'Unsaturated'}


def summarize_subset(rows):
    if not rows:
        raise ValueError('Empty execution subset')
    weights = np.array([r['weight'] for r in rows])
    observed = weights @ np.array([r['observed'] for r in rows])
    metrics = {}
    for method in METHODS:
        predicted = weights @ np.array([r['predictions'][method] for r in rows])
        metrics[method] = dict(predicted=predicted.tolist(),
                              rmse=float(np.sqrt(np.mean((predicted-observed)**2))))
    return dict(problems=len(rows), trials=int(weights.sum()), observed=observed.tolist(),
                members=[dict(dataset=r['dataset'], problem=r['problem']) for r in rows],
                metrics=metrics)


def build(data):
    result = {}
    for n, models in PANELS.items():
        for model in models:
            key = f'{model}-n{n}'
            rows = data['reports'][key]['rows']
            if len(rows) != 57 or any(r['weight'] != 3 for r in rows):
                raise ValueError(f'{key}: requires all 57 problems and three trials each')
            if len({(r['dataset'], r['problem']) for r in rows}) != 57:
                raise ValueError(f'{key}: duplicate problems')
            for r in rows:
                e = np.asarray(r['epsilon'])
                if (len(e) != 8 or not np.isfinite(e).all() or
                        np.any(e < 0) or np.any(e > 1) or np.any(np.diff(e) < 0) or
                        r['oracle_n'] != 3):
                    raise ValueError(f'{key}: invalid oracle curve')
                if len(r['observed']) != 8//n:
                    raise ValueError(f'{key}: wrong target horizon')
            complete = [r for r in rows if r['epsilon'][0] == 1]
            incomplete = [r for r in rows if r['epsilon'][0] < 1]
            # The complete subset reduces exactly to plug-in geometric for DE.
            for r in complete:
                q = r['solved']/r['parallel_n']
                expected = 1-(1-q)**(n*np.arange(1, 8//n+1))
                if not np.allclose(r['predictions']['neither_regularized'], expected,
                                   atol=1e-10, rtol=0):
                    raise ValueError(f'{key}: DE does not reduce to geometric')
            result[key] = {name: summarize_subset(subset) for name, subset in
                           [('complete', complete), ('incomplete', incomplete)]}
            combined = np.array(result[key]['complete']['observed']) + result[key]['incomplete']['observed']
            if not np.allclose(combined, summarize_subset(rows)['observed']):
                raise ValueError('Subsets do not reconstruct the full observed curve')
    delayed = [r for r in data['reports']['gpt54-n2']['rows']
               if r['epsilon'][3] > r['epsilon'][0]]
    gains = summarize_subset(delayed)
    plugin = sum(r['weight']*(1-(1-r['solved']/r['parallel_n'])**(2*np.arange(1, 5)))
                 for r in delayed)
    gains['metrics']['plug_in_geometric'] = dict(predicted=plugin.tolist(),
        rmse=float(np.sqrt(np.mean((plugin-np.array(gains['observed']))**2))))
    return dict(panels=result, gpt54_n2_early_gains=gains,
                audit_sha256=data.get('audit_sha256', {}),
                selection='Saturated: all three oracle trajectories solve by block 1; unsaturated: otherwise.',
                fitting='Reuse full-57 intervention fits; no subset refitting.',
                metric='RMSE of aggregate cumulative solved-trial counts, not individual errors.')


def render(report, opus_only=False):
    lines = [r'\begin{table}[t]', r'\centering\small\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{llrrrr}',
             (r'\toprule $N$' if opus_only else r'\toprule Model') + r' & Execution & Trials & SG & DE & R-DE \\']
    for n, models in PANELS.items():
        if opus_only:
            models = ('opus',)
            lines.append(r'\midrule')
        else:
            lines += [r'\midrule', r'\multicolumn{6}{l}{\textit{$N='+str(n)+r'$}} \\', r'\midrule']
        for group in ('complete', 'incomplete'):
            if group == 'incomplete' and not opus_only:
                lines.append(r'\addlinespace')
            for model in models:
                item = report['panels'][f'{model}-n{n}'][group]
                values = [item['metrics'][m]['rmse'] for m in METHODS]
                cells = [r'\textbf{'+f'{v:.2f}'+'}' if v == min(values) else f'{v:.2f}' for v in values]
                first = str(n) if opus_only else NAMES[model]
                lines.append(' & '.join([first, GROUP_LABELS[group], str(item['trials']), *cells])+r' \\')
    title = 'Opus prediction error by first-block execution saturation.' if opus_only else 'Prediction error by first-block execution saturation for the GPT models and Opus.'
    label = 'tab:execution-regimes' if opus_only else 'tab:execution-regimes-full'
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{'+title+r' Saturated means all three oracle trajectories solve by block 1 ($\widehat\varepsilon_n^{\rm oracle}(1)=1$); unsaturated means at least one does not. Entries are aggregate solved-trial-count RMSE over eight checkpoints for $N=1$ and four for $N=2$, using unchanged full-set fits. Each problem contributes three trials. Bold marks row minima.}',
              r'\label{'+label+'}', r'\end{table}']
    return '\n'.join(lines)+'\n'


def render_early_gains(report):
    item = report['gpt54_n2_early_gains']
    lines = [r'\begin{table}[htbp]', r'\centering\small', r'\begin{tabular}{lrrrr}',
             r'\toprule Checkpoint & Observed & SG & DE & R-DE \\', r'\midrule']
    for i, observed in enumerate(item['observed']):
        values = [item['metrics'][m]['predicted'][i] for m in METHODS]
        lines.append(' & '.join([f'${i+1}\\times$', f'{observed:.0f}', *[f'{v:.2f}' for v in values]])+r' \\')
    lines += [r'\midrule', 'RMSE & --- & '+' & '.join(f"{item['metrics'][m]['rmse']:.2f}" for m in METHODS)+r' \\',
              r'\bottomrule', r'\end{tabular}',
              r'\caption{GPT-5.4 at $N=2$ on the '+str(item['problems'])+' problems ('+str(item['trials'])+r' allocation trials) where empirical oracle completion increases between blocks 1 and 4. Checkpoints give compute per trajectory. Values are cumulative solved-trial counts; predictions use the unchanged full-set fit.}',
              r'\label{tab:early-execution-gains}', r'\end{table}']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'paper/img/allocation_report.json')
    parser.add_argument('--output', type=Path, default=ROOT/'local_data/execution_regimes_report.json')
    parser.add_argument('--tex-dir', type=Path, default=ROOT/'paper/img')
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    if not data.get('audit_sha256'):
        raise ValueError('Source report has no audit fingerprints')
    for path, expected in data['audit_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Stale source report: {path}; rerun report_joint_allocation.py')
    report = build(data)
    report['source_report_sha256'] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    args.tex_dir.mkdir(parents=True, exist_ok=True)
    (args.tex_dir/'execution_regimes_table.tex').write_text(render(report, opus_only=True))
    (args.tex_dir/'execution_regimes_full_table.tex').write_text(render(report))
    (args.tex_dir/'early_execution_gains_table.tex').write_text(render_early_gains(report))
    print(f'Wrote {args.output} and execution subset tables to {args.tex_dir}')


if __name__ == '__main__':
    main()
