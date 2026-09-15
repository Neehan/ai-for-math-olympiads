#!/usr/bin/env python3
"""Audited AOBench standalone sketch controls; one count per seed, no majority vote."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

MODELS = {
    'muse-spark-1.2-contributor': 'Muse Spark~1.2',
    'litellm-gpt-5.4': 'GPT-5.4',
    'litellm-gpt-5.5': 'GPT-5.5',
    'claude-opus-4-8': 'Claude Opus~4.8',
}


def load_arm(base, arm, problems, hashes):
    records = {}
    for pid in sorted(problems):
        for seed in (1, 2, 3):
            directory = base / arm / pid / f'seed_{seed}'
            path = directory / 'audit.json'
            record = json.loads(path.read_text())
            if (record['problem_id'], record['seed'], record['arm']) != (pid, seed, arm):
                raise ValueError(f'Audit identity mismatch: {path}')
            score = record['audit_score']
            if type(score) is not int or not 0 <= score <= 7:
                raise ValueError(f'Invalid score: {path}')
            proof = (directory / 'solution.md').read_bytes()
            if hashlib.sha256(proof).hexdigest() != record.get('solution_sha256'):
                if proof.strip() or score != 0 or record.get('solution_sha256'):
                    raise ValueError(f'Stale proof audit: {path}')
            meta = json.loads((directory / 'meta.json').read_text())
            if meta['mode'] != 'single' or meta['budget_output_tokens'] != 200000:
                raise ValueError(f'Not a standalone 1x run: {directory}')
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            records[pid, seed] = record
    # Individual files are authoritative while a batch is being compiled.
    compiled = base / arm / 'audit.jsonl'
    if compiled.exists():
        seen = set()
        for line in compiled.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = record['problem_id'], record['seed']
            if key in records:
                if key in seen or records[key] != record:
                    raise ValueError(f'Duplicate/stale compiled audit: {compiled}: {key}')
                seen.add(key)
    return records


def collect(root):
    hashes, content, alternate = {}, [], []
    common_alt = None
    for model, label in MODELS.items():
        base = root / 'results' / model
        problems = {p.name for p in (base / 'baseline').iterdir()
                    if p.is_dir() and (p / 'seed_1/meta.json').exists()}
        if len(problems) != 35 or any(p.startswith('PB-Advanced-') for p in problems):
            raise ValueError(f'Expected 35 AOBench problems: {model}')
        arms = {arm: load_arm(base, arm, problems, hashes)
                for arm in ('baseline', 'placebo-hint', 'hint')}
        content.append(dict(model=label, trajectories=105,
                            counts={arm: sum(r['audit_score'] >= 5 for r in rows.values())
                                    for arm, rows in arms.items()},
                            judges={arm: dict(Counter(r['audit_model'] for r in rows.values()))
                                    for arm, rows in arms.items()}))
        if model not in ('muse-spark-1.2-contributor', 'litellm-gpt-5.5'):
            continue
        alt_ids = {p.name for p in (base / 'alt-hint').iterdir() if p.is_dir()} & problems
        if len(alt_ids) != 24 or (common_alt is not None and alt_ids != common_alt):
            raise ValueError('Alternate-sketch models must share the same 24 AOBench problems')
        common_alt = alt_ids
        alt = load_arm(base, 'alt-hint', alt_ids, hashes)
        paired = [dict(problem=p, seed=s, original=arms['hint'][p, s]['audit_score'],
                       alternate=alt[p, s]['audit_score']) for p, s in sorted(alt)]
        alternate.append(dict(model=label, trajectories=len(paired),
                              original=sum(r['original'] >= 5 for r in paired),
                              alternate=sum(r['alternate'] >= 5 for r in paired),
                              rows=paired,
                              judges=dict(Counter(r['audit_model'] for r in alt.values()))))
    return dict(protocol='AOBench only; standalone 200k-output-token runs; seeds 1,2,3; score >=5/7; individual proof hashes verified.',
                content=content, alternate=alternate, audit_sha256=hashes)


def table(headers, rows, caption, label):
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\begin{tabular}{l' + 'r' * (len(headers)-1) + '}',
             r'\toprule ' + ' & '.join(headers) + r' \\', r'\midrule']
    lines.extend(' & '.join(map(str, row)) + r' \\' for row in rows)
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
              r'\label{' + label + '}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def render(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'alternate_sketch_table.tex').write_text(table(
        ['Model', 'Trajectories', 'Original sketch', 'Alternate sketch'],
        [[r['model'], r['trajectories'], r['original'], r['alternate']] for r in report['alternate']],
        r'Original versus alternate oracle sketches on the same 24 AOBench problems and three seeds. The last two columns count solved standalone $1\times$ trajectories (score $\geq5/7$).',
        'tab:alternate-sketch'))
    (output / 'sketch_content_table.tex').write_text(table(
        ['Model', 'Trajectories', 'No sketch', 'Shuffled sketch', 'Oracle sketch'],
        [[r['model'], r['trajectories']] + [r['counts'][a] for a in ('baseline', 'placebo-hint', 'hint')]
         for r in report['content']],
        r'Sketch-content control on all 35 AOBench problems and three seeds. The last three columns count solved standalone $1\times$ trajectories (score $\geq5/7$); seeds are counted individually.',
        'tab:placebo'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tex-dir', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    render(result, args.tex_dir)
    for group in ('alternate', 'content'):
        for row in result[group]:
            print(group, {k: v for k, v in row.items() if k != 'rows'})
