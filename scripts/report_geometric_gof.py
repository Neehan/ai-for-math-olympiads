#!/usr/bin/env python3
"""Exploratory N=1 geometric goodness-of-fit; never reads oracle outcomes.

Null: per-problem q is shared by independent fresh Bernoulli trials and
geometric first-completion times, censored after block 8. The statistic is
aggregate solved-count RMSE across all eight checkpoints, with the paper's
finite-bank unbiased prediction (NOT the plug-in geometric curve).

Primary calibration: parametric bootstrap at the joint null MLE. This is an
approximate composite-null test, not a finite-sample exact p-value. Both the
fresh bank and target trajectories are regenerated; the prediction is
re-estimated from each simulated fresh bank. The null MLE is used only to
generate replicates, not to change the paper's predictions.

Sensitivity check: conditional Monte Carlo with no estimated q. Condition
per problem on S=x+d and E=sum(min(T_j,8)), where d is the number of solved
trajectories. For a labelled triple T in {1,...,8,9}^3 (9 means censored),
the joint likelihood is C(24,x)*q^S*(1-q)^(24+E-S). Conditional on S,E,
all q terms cancel. Enumerate the 729 triples; x=S-d is then determined.
This test conditions on informative statistics and can have different power
from the plug-in bootstrap. It is not a replacement chosen by its outcome.

All inference is conditional on independent runs, the specified shared-q
null, the fixed 57 problems, and the audited labels. No internal discovery
mechanism or superiority of DE/R-DE is tested here.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.report_joint_allocation import (
    DATASETS, MODELS, SEEDS, geometric, indexed_audits, passing,
)
from scripts.report_gpt55_allocations import load

HORIZON = 8
BANK = 24
CHECKPOINTS = np.arange(1, HORIZON + 1)
LOOKUP = np.array([[geometric(x, BANK, k) for k in CHECKPOINTS]
                   for x in range(BANK + 1)])
STATES = np.array(list(itertools.product(range(1, HORIZON + 2), repeat=3)))
STATE_EVENTS = (STATES <= HORIZON).sum(axis=1)
STATE_EXPOSURE = np.minimum(STATES, HORIZON).sum(axis=1)
STATE_CURVES = (STATES[:, :, None] <= CHECKPOINTS).sum(axis=1)


def collect(root, model):
    rows, fingerprints = [], {}
    for dataset, expected in DATASETS.items():
        base = root / dataset / MODELS[model]
        fresh = indexed_audits(base / 'baseline-parallel/audit.jsonl', SEEDS, fingerprints)
        # Verify individual sequential audits, compiled records, and proof hashes.
        sequential = load(base, 'baseline-sequential', HORIZON, SEEDS)
        seq_path = base / 'baseline-sequential/audit.jsonl'
        fingerprints[str(seq_path)] = hashlib.sha256(seq_path.read_bytes()).hexdigest()
        problems = sorted({p for p, _ in fresh})
        required = {(p, s) for p in problems for s in SEEDS}
        if len(problems) != expected or set(fresh) != required or set(sequential) != required:
            raise ValueError(f'Incomplete or mismatched cohort: {model}/{dataset}')
        for problem in problems:
            x = 0
            for seed in SEEDS:
                record = fresh[problem, seed]
                path = base / 'baseline-parallel' / problem / f'seed_{seed}' / 'audit.json'
                if json.loads(path.read_text()) != record:
                    raise ValueError(f'Stale compiled bank audit: {path}')
                bank = record.get('runs', [])
                if len(bank) != 8 or {r['run'] for r in bank} != set(range(1, 9)):
                    raise ValueError(f'Incomplete bank: {path}')
                x += sum(passing(r['audit_score'], 5) for r in bank)
            times = [next((k for k, v in enumerate(sequential[problem, s], 1) if v), 9)
                     for s in SEEDS]
            rows.append(dict(dataset=dataset, problem=problem, fresh_successes=x,
                             first_completion=times))
    return rows, fingerprints


def null_mle(x, times):
    """Binomial fresh observations plus right-censored geometric likelihood."""
    return (x + (times <= HORIZON).sum(axis=-1)) / (
        BANK + np.minimum(times, HORIZON).sum(axis=-1))


def residual(x, times):
    # Leading axes may index bootstrap replicates; last axes are problem, seed.
    observed = (times[..., None] <= CHECKPOINTS).sum(axis=(-3, -2))
    predicted = times.shape[-1] * LOOKUP[x].sum(axis=-2)
    return observed - predicted


def rmse(r):
    return np.sqrt(np.mean(r * r, axis=-1))


def simulate(q, count, rng):
    x = rng.binomial(BANK, q, size=(count, len(q)))
    t = rng.geometric(np.where(q > 0, q, 1), size=(count, 3, len(q))).transpose(0, 2, 1)
    t[:, q == 0, :] = HORIZON + 1
    return x, np.minimum(t, HORIZON + 1)


def conditional_support(x, times):
    successes = x + (times <= HORIZON).sum()
    exposure = np.minimum(times, HORIZON).sum()
    xs = successes - STATE_EVENTS
    valid = (STATE_EXPOSURE == exposure) & (xs >= 0) & (xs <= BANK)
    weights = np.array([math.comb(BANK, int(v)) for v in xs[valid]], dtype=float)
    weights /= weights.sum()
    return xs[valid], STATES[valid], weights


def tail_summary(values, observed):
    exceedances = int(np.count_nonzero(values >= observed - 1e-12))
    p = (exceedances + 1) / (len(values) + 1)
    return dict(null_rmse_quantiles_025_50_975=np.quantile(values, [.025, .5, .975]).tolist(),
                exceedances=exceedances, p_value=p,
                monte_carlo_standard_error=float(np.sqrt(p * (1 - p) / len(values))))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def analyze(rows, replicates, rng):
    x = np.array([r['fresh_successes'] for r in rows])
    t = np.array([r['first_completion'] for r in rows])
    q = null_mle(x, t)
    actual = float(rmse(residual(x, t)))
    values = []
    for start in range(0, replicates, 2000):
        bx, bt = simulate(q, min(2000, replicates - start), rng)
        values.append(rmse(residual(bx, bt)))
    conditional_residual = np.zeros((replicates, HORIZON))
    support_sizes = []
    for xi, ti in zip(x, t):
        xs, ts, weights = conditional_support(xi, ti)
        support_sizes.append(len(xs))
        r = (ts[:, :, None] <= CHECKPOINTS).sum(axis=1) - 3 * LOOKUP[xs]
        conditional_residual += r[rng.choice(len(xs), size=replicates, p=weights)]
    return dict(problems=len(rows), fresh_attempts=int(len(rows) * BANK),
                unaided_trajectories=int(t.size), observed_rmse=actual,
                observed_counts=(t[:, :, None] <= CHECKPOINTS).sum(axis=(0, 1)).tolist(),
                predicted_counts=(3 * LOOKUP[x].sum(axis=0)).tolist(),
                null_q=q.tolist(), null_q_zero=int(sum(q == 0)), null_q_one=int(sum(q == 1)),
                parametric_bootstrap=tail_summary(np.concatenate(values), actual),
                conditional_check=tail_summary(rmse(conditional_residual), actual),
                conditional_support_sizes=support_sizes, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--results-base', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error('--replicates must be positive')
    reports, fingerprints = {}, {}
    streams = np.random.SeedSequence(args.seed).spawn(len(MODELS))
    for model, stream in zip(MODELS, streams):
        rows, hashes = collect(args.results_base, model)
        fingerprints.update(hashes)
        reports[model] = analyze(rows, args.replicates, np.random.default_rng(stream))
    for method in ('parametric_bootstrap', 'conditional_check'):
        adjusted = holm([r[method]['p_value'] for r in reports.values()])
        for r, p in zip(reports.values(), adjusted):
            r[method]['holm_p_four_models'] = float(p)
    print(json.dumps(dict(replicates=args.replicates, seed=args.seed,
                         null='Independent shared-q fresh Bernoulli and censored geometric completion, problem-specific q',
                         statistic='Aggregate solved-count RMSE over blocks 1 through 8',
                         notes='Exploratory; parametric bootstrap is plug-in approximate. Conditional check eliminates q by conditioning on sufficient statistics. Holm correction covers four LLMs, not prior exploratory analyses. No oracle data used.',
                         reports=reports, audit_sha256=fingerprints), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
