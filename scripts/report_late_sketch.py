"""Reproduce the paper's AOBench recorded-3x-failure late-sketch comparison.

Select the control arm's recorded 3x audit score <5, not cumulative failure.
This recorded token checkpoint need not coincide with the native fork.
Reports terminal branch proofs (not cumulative solved status). No LLM calls.
"""
import argparse
import hashlib
import json
from pathlib import Path


COHORTS = (
    ('results', 'AOBench', 'muse-spark-1.2-contributor', 'Muse Spark~1.2', 35),
    ('results', 'AOBench', 'litellm-gpt-5.4', 'GPT-5.4', 35),
    ('results', 'AOBench', 'litellm-gpt-5.5', 'GPT-5.5', 35),
    ('results', 'AOBench', 'claude-opus-4-8', 'Claude Opus~4.8', 35),
)


def passed(score):
    if type(score) is not int or not 0 <= score <= 7:
        raise ValueError(f'Invalid audit score: {score!r}')
    return score >= 5


def selected_at_3x(record):
    return not passed(record['budget_cuts']['3x']['audit_score'])


def verify_cut(directory, record, block):
    audit = record['budget_cuts'][f'{block}x']
    passed(audit['audit_score'])
    path = directory / f'solution_{block}x.md'
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit.get('solution_sha256'):
            raise ValueError(f'Stale checkpoint audit: {path}')
    elif audit.get('solution_sha256') or audit['audit_score'] != 0:
        raise ValueError(f'Missing checkpoint proof: {path}')
    return audit


def load_audits(base, arm, fingerprints):
    path = base / arm / 'audit.jsonl'
    fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r['seed'] not in (1, 2, 3):
            continue
        key = r['problem_id'], r['seed']
        if key in rows:
            raise ValueError(f'Duplicate audit: {path}, {key}')
        directory = path.parent / key[0] / f'seed_{key[1]}'
        if json.loads((directory / 'audit.json').read_text()) != r:
            raise ValueError(f'Stale compiled audit: {directory}')
        proof = (directory / 'solution.md').read_bytes()
        if not proof.strip() and r['audit_score'] == 0 and not r.get('solution_sha256'):
            pass  # Explicitly recorded absence of a gradeable final proof.
        elif hashlib.sha256(proof).hexdigest() != r.get('solution_sha256'):
            raise ValueError(f'Stale final-proof audit: {directory}')
        passed(r['audit_score'])
        rows[key] = r
    return rows


def collect(root):
    report = {'protocol': 'Seeds 1--3; select late-baseline-sequential recorded 3x audit score <5; not cumulative failure and no token-consumption filter; terminal branch outcomes; start sketch uses hint-sequential 1x; pass >=5/7.',
              'cohorts': [], 'audit_sha256': {}}
    for dataset, label, model, display, expected in COHORTS:
        base = root / dataset / model
        control = load_audits(base, 'late-baseline-sequential', report['audit_sha256'])
        late = load_audits(base, 'late-hint-sequential', report['audit_sha256'])
        early = load_audits(base, 'hint-sequential', report['audit_sha256'])
        problems = {p for p, _ in late}
        required = {(p, s) for p in problems for s in (1, 2, 3)}
        if len(problems) != expected or any(set(r) != required for r in (control, late, early)):
            raise ValueError(f'Incomplete matched cohort: {base}')
        selected = []
        for problem, seed in sorted(required):
            suffix = Path(problem) / f'seed_{seed}'
            cm = json.loads((base / 'late-baseline-sequential' / suffix / 'meta.json').read_text())
            hm = json.loads((base / 'late-hint-sequential' / suffix / 'meta.json').read_text())
            if not hm.get('source_session_id') or cm.get('native_prefix_session_id') != hm['source_session_id']:
                raise ValueError(f'Unmatched native prefix: {base / suffix}')
            if cm['prefix_output_tokens_spent'] != hm['prefix_output_tokens_spent']:
                raise ValueError(f'Unmatched prefix consumption: {base / suffix}')
            verify_cut(base / 'late-baseline-sequential' / suffix, control[problem, seed], 3)
            if not selected_at_3x(control[problem, seed]):
                continue
            er = verify_cut(base / 'hint-sequential' / suffix, early[problem, seed], 1)
            selected.append(dict(problem=problem, seed=seed,
                                 early=int(passed(er['audit_score'])),
                                 control=int(passed(control[problem, seed]['audit_score'])),
                                 late=int(passed(late[problem, seed]['audit_score']))))
        report['cohorts'].append(dict(dataset=label, model=display, total=len(required),
                                     count=len(selected), problems=len({r['problem'] for r in selected}),
                                     **{k: sum(r[k] for r in selected) for k in ('early', 'control', 'late')},
                                     selected=selected))
    return report


def render(report):
    lines = [r'\begin{table}[htbp]', r'\centering\small', r'\begin{tabular}{lrrrrr}',
             r'\toprule Model & Trajectories & No sketch & Start sketch & Late sketch & $\Delta$ \\', r'\midrule']
    for r in report['cohorts']:
        if r['dataset'] != 'AOBench':
            raise ValueError('The timing table is restricted to AOBench')
        n = r['count']
        cells = [str(r[k]) if n else '--' for k in ('control', 'early', 'late')]
        delta = r['late'] - r['early']
        cells.append(f'${delta:+d}$' if delta else '$0$')
        lines.append(f"{r['model']} & {n} & " + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{AOBench sketch-timing comparison. Condition columns count solved trajectories (score $\geq5/7$); $\Delta$ is late sketch minus start sketch. No sketch and late sketch use final continuation proofs at a nominal total budget of $4\times$ ($3\times$ prefix plus $1\times$ continuation); start sketch uses the oracle $1\times$ checkpoint for the same problem and seed. Selection uses the recorded no-sketch $3\times$ score below 5, not cumulative failure.}',
              r'\label{tab:late-oracle}', r'\end{table}']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tex', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    args.tex.write_text(render(result))
    for row in result['cohorts']:
        print({k: v for k, v in row.items() if k != 'selected'})
