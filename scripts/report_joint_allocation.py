#!/usr/bin/env python3
"""Recompute the paper's DE/R-DE curves and comparisons from correctness audits.

Fresh and oracle seeds define the intervention fit, independently of target
coverage. N=2 reuses the full N=1 intervention fit. Missing required audits
are errors, never failures; Opus N=2 is an explicitly partial replication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.allocation_estimators import Joint, fit_de, posterior_n2, check_posterior_n1, check_posterior_n2
from scripts.report_allocation_model import _read_jsonl, _proof_curve

MODELS = {
    'muse': 'muse-spark-1.2-contributor',
    'gpt54': 'litellm-gpt-5.4',
    'gpt55': 'litellm-gpt-5.5',
    'opus': 'claude-opus-4-8',
}
METHODS = {
    'solved_geometric': 'Plain-geometric',
    'oracle_slope_linear': 'Linear',
    'oracle_gain_transfer': 'OGT',
    'neither_regularized': 'DE',
    'both_regularized': 'R-DE',
}
DATASETS = {'results': 35, 'results-imobench': 22}
SEEDS = (1, 2, 3)
PROFILES = [f'{m}-n{n}' for n in (1, 2) for m in MODELS]


def passing(score, threshold):
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 7:
        raise ValueError(f'Missing or invalid correctness-audit score: {score!r}')
    return score >= threshold


def indexed_audits(path, seeds, fingerprints, *, optional=False):
    if optional and not path.exists():
        return {}
    records = _read_jsonl(path)
    fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    index = {}
    for record in records:
        seed = record.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f'Invalid seed in {path}: {seed!r}')
        if seed not in seeds:
            continue
        problem = record.get('problem_id')
        if not isinstance(problem, str) or not problem:
            raise ValueError(f'Missing problem ID in {path}')
        key = problem, seed
        if key in index:
            raise ValueError(f'Duplicate audit in {path}: {key}')
        index[key] = record
    return index


def proof_curve(record, final_block, threshold):
    # The legacy cumulative helper treats ungraded cuts as false; this current
    # pipeline must reject them before calling it.
    passing(record.get('audit_score'), threshold)
    cuts = record.get('budget_cuts', {})
    if not isinstance(cuts, dict):
        raise ValueError('budget_cuts must be an object')
    for k in range(1, final_block):
        cut = cuts.get(f'{k}x')
        if not isinstance(cut, dict):
            raise ValueError(f'Missing proof audit for {k}x: {record.get("problem_id")}')
        passing(cut.get('audit_score'), threshold)
    values = _proof_curve(record, final_block=final_block, threshold=threshold)
    return [int(values[k]) for k in range(1, final_block+1)]


def collect_interventions(root, model, threshold, fingerprints, datasets=DATASETS):
    rows = []
    parallel_seeds = (1, 2) if model == 'gpt55' else SEEDS
    for dataset, expected in datasets.items():
        base = root/dataset/MODELS[model]
        fresh = indexed_audits(base/'baseline-parallel/audit.jsonl', parallel_seeds, fingerprints)
        oracle = indexed_audits(base/'hint-sequential/audit.jsonl', SEEDS, fingerprints)
        problems = sorted({p for p, _ in fresh} | {p for p, _ in oracle})
        if len(problems) != expected:
            raise ValueError(f'{model}/{dataset}: expected {expected} intervention problems, found {len(problems)}')
        for problem in problems:
            solved = 0
            for seed in parallel_seeds:
                if (problem, seed) not in fresh:
                    raise ValueError(f'Missing Parallel-8 audit: {model}/{dataset}/{problem}/seed_{seed}')
                bank = fresh[problem, seed].get('runs')
                if not isinstance(bank, list) or len(bank) != 8:
                    raise ValueError(f'Incomplete Parallel-8 bank: {model}/{dataset}/{problem}/seed_{seed}')
                if any(not isinstance(r, dict) or type(r.get('run')) is not int for r in bank) or {r['run'] for r in bank} != set(range(1, 9)):
                    raise ValueError(f'Duplicate/missing run IDs: {model}/{dataset}/{problem}/seed_{seed}')
                solved += sum(passing(r.get('audit_score'), threshold) for r in bank)
            curves = []
            for seed in SEEDS:
                if (problem, seed) not in oracle:
                    raise ValueError(f'Missing oracle audit: {model}/{dataset}/{problem}/seed_{seed}')
                curves.append(proof_curve(oracle[problem, seed], 8, threshold))
            rows.append(dict(dataset=dataset, problem=problem, parallel_n=8*len(parallel_seeds),
                             solved=solved, oracle_n=3, epsilon=np.mean(curves, axis=0).tolist()))
    return rows


def geometric(solved, total, attempts):
    if not 1 <= attempts <= total:
        raise ValueError(f'Need {attempts} distinct fresh observations; have {total}')
    return 1-math.comb(total-solved, attempts)/math.comb(total, attempts)


def fit_interventions(rows):
    # This interface receives no target outcomes, weights, or acquisition audits.
    joint = Joint(rows)
    optimum, starts = joint.fit()
    rde1, alpha = joint.predict(optimum.x)
    rde2 = posterior_n2(joint, optimum.x)
    predictions = []
    de_fits = []
    for i, row in enumerate(rows):
        q = row['solved']/row['parallel_n']
        epsilon = np.array(row['epsilon'])
        slope = max(0., float(np.dot(np.arange(1, 8), epsilon[1:]-epsilon[0])/140))
        de = fit_de(row)
        de_fits.append(de)
        single = dict(oracle_slope_linear=np.clip(q+slope*np.arange(8), 0, 1),
                      oracle_gain_transfer=q+(1-q)*(epsilon-epsilon[0]),
                      neither_regularized=np.array(de['prediction_unidentified_alpha0']))
        allocations = {}
        for n, horizon in [(1, 8), (2, 4)]:
            p = {name: (1-(1-values[:horizon])**n).tolist() for name, values in single.items()}
            p['solved_geometric'] = [geometric(row['solved'], row['parallel_n'], n*k) for k in range(1, horizon+1)]
            p['both_regularized'] = (rde1[i] if n == 1 else rde2[i]).tolist()
            allocations[n] = p
        predictions.append(allocations)
    _, _, _, _, _, _, d = joint.components(optimum.x)
    a, b = np.exp(optimum.x[:2])
    diagnostics = dict(starts=starts, log_hyperparameters=optimum.x.tolist(),
                       optimum_nll=float(optimum.fun),
                       hyper=dict(mu_alpha=float(a/(a+b)), tau_alpha=float(a+b),
                                  tau_execution=float(d.sum()), execution_mean=(d/d.sum()).tolist()),
                       posterior_alpha=alpha.tolist(), de_fits=de_fits)
    return predictions, diagnostics


def collect_targets(root, model, n, rows, predictions, threshold, fingerprints):
    result = []
    partial = model == 'opus' and n == 2
    for dataset in DATASETS:
        if partial and dataset != 'results-imobench':
            continue
        base = root/dataset/MODELS[model]
        first = indexed_audits(base/'baseline-sequential/audit.jsonl', SEEDS, fingerprints)
        second = indexed_audits(base/'late-baseline-sequential/audit.jsonl', SEEDS, fingerprints, optional=partial) if n == 2 else None
        for row, p in zip(rows, predictions):
            if row['dataset'] != dataset:
                continue
            curves = []
            for seed in SEEDS:
                ident = row['problem'], seed
                if ident not in first or (n == 2 and ident not in second):
                    if partial:
                        continue
                    raise ValueError(f'Missing N={n} target audit: {model}/{dataset}/{ident}')
                c = np.array(proof_curve(first[ident], 8, threshold))
                if n == 2:
                    c = np.maximum(c[:4], proof_curve(second[ident], 4, threshold))
                curves.append(c)
            if curves:
                result.append(dict(row, weight=len(curves), observed=np.mean(curves, axis=0).tolist(), predictions=p[n]))
    if not result:
        raise ValueError(f'No audited targets for {model}-n{n}')
    return result


def summarize(rows):
    weights = np.array([r['weight'] for r in rows])
    trials = int(weights.sum())
    observed = weights@np.array([r['observed'] for r in rows])
    metrics = {}
    for method in METHODS:
        predicted = weights@np.array([r['predictions'][method] for r in rows])
        error = predicted-observed
        mae, rmse = float(np.abs(error).mean()), float(np.sqrt(np.mean(error**2)))
        metrics[method] = dict(observed=observed.tolist(), predicted=predicted.tolist(),
                               mae=mae, rmse=rmse, rate_mae_pp=100*mae/trials, rate_rmse_pp=100*rmse/trials)
    return dict(problems=len(rows), trials=trials, metrics=metrics, rows=rows)


def build_reports(root, profiles, threshold=5):
    if not 0 <= threshold <= 7:
        raise ValueError('passing-score must be between 0 and 7')
    fingerprints, fits, reports = {}, {}, {}
    for model in MODELS:
        selected = [n for n in (1, 2) if f'{model}-n{n}' in profiles]
        if not selected:
            continue
        rows = collect_interventions(root, model, threshold, fingerprints)
        print(f'Fitting {model} from {len(rows)} intervention problems (six starts)...', file=sys.stderr, flush=True)
        predictions, fits[model] = fit_interventions(rows)
        # Only now read the unaided outcomes. N=2 never changes the fitted model.
        for n in selected:
            targets = collect_targets(root, model, n, rows, predictions, threshold, fingerprints)
            reports[f'{model}-n{n}'] = summarize(targets)
    pooled = {}
    for n in (1, 2):
        keys = [f'{m}-n{n}' for m in MODELS if not (n == 2 and m == 'opus')]
        if all(k in reports for k in keys):
            pooled[f'n{n}'] = {method: float(np.sqrt(np.mean([reports[k]['metrics'][method]['rate_rmse_pp']**2 for k in keys]))) for method in METHODS}
    return dict(estimator='DE/R-DE', passing_score=threshold, reports=reports, fits=fits,
                pooled_rmse_pp=pooled,
                audit_sha256={str(Path(path).resolve().relative_to(root.resolve())): sha for path, sha in fingerprints.items()},
                notes='Cumulative score-based solves; intervention-only fitting; N=2 posterior second moments; aggregate-curve RMSE. Opus N=2 is partial IMO-ProofBench only and excluded from pooled comparisons.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', action='append', choices=PROFILES)
    parser.add_argument('--json', action='store_true', help='Print complete machine-readable report to stdout.')
    parser.add_argument('--output-dir', type=Path, help='Write JSON and regenerated prediction plots/tables to this directory.')
    parser.add_argument('--results-base', type=Path, default=ROOT)
    parser.add_argument('--passing-score', type=int, default=5)
    args = parser.parse_args(argv)
    check_posterior_n1()
    check_posterior_n2()
    data = build_reports(args.results_base, set(args.profile or PROFILES), args.passing_score)
    if args.output_dir:
        from scripts.render_allocation_report import render
        render(data, args.output_dir)
    if args.json:
        print(json.dumps(data, indent=2, allow_nan=False))
    else:
        for key in PROFILES:
            if key not in data['reports']:
                continue
            report = data['reports'][key]
            print(f'\n{key}: {report["problems"]} problems, {report["trials"]} trials')
            for method, label in METHODS.items():
                m = report['metrics'][method]
                print(f'  {label}: predicted {[round(v, 3) for v in m["predicted"]]}; RMSE {m["rate_rmse_pp"]:.2f} pp')
            print('  observed:', [round(v) for v in report['metrics']['both_regularized']['observed']])
        print('\nPooled RMSE (pp):', data['pooled_rmse_pp'])


if __name__ == '__main__':
    main()
