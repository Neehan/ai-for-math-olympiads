"""Opus aggregate shape check: fit geometric curves to the target itself.

These are descriptive in-sample fits, NOT predictions or goodness-of-fit
p-values. In particular, aggregate fitting of problem-specific rates does
not identify which rate belongs to which problem. No paper files are edited.
The fresh-bank comparator retains the paper's finite-bank unbiased estimator.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from scripts.report_geometric_gof import CHECKPOINTS, LOOKUP, ROOT, collect


def geometric_counts(q, trials_per_problem=3):
    q = np.asarray(q, dtype=float)
    return trials_per_problem * (1 - (1 - q[:, None]) ** CHECKPOINTS).sum(axis=0)


def two_rate_counts(parameters, trials):
    weight, fast, slow = parameters
    return trials * (weight * (1 - (1 - fast) ** CHECKPOINTS)
                     + (1 - weight) * (1 - (1 - slow) ** CHECKPOINTS))


def summarize(predicted, observed, **extra):
    residual = predicted - observed
    return dict(rmse=float(np.sqrt(np.mean(residual**2))),
                mae=float(np.mean(np.abs(residual))),
                predicted_counts=predicted.tolist(), **extra)


def analyze(rows, starts=8, seed=35):
    times = np.array([r['first_completion'] for r in rows])
    observed = (times[:, :, None] <= CHECKPOINTS).sum(axis=(0, 1))
    trials = times.size
    single = minimize_scalar(
        lambda q: np.mean((trials * (1 - (1 - q)**CHECKPOINTS) - observed)**2),
        bounds=(0, 1), method='bounded', options={'xatol': 1e-12})
    if not single.success:
        raise RuntimeError(single.message)
    rng = np.random.default_rng(seed)
    mixture_fits = [least_squares(
        lambda z: two_rate_counts(z, trials) - observed,
        z, bounds=(0, 1), max_nfev=2000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12)
        for z in [[.7, .7, .02], *rng.uniform(.001, .999, (starts-1, 3))]]
    mixture = min(mixture_fits, key=lambda fit: fit.cost)
    rate_fits = [least_squares(
        lambda q: geometric_counts(q) - observed, rng.uniform(0, 1, len(rows)),
        jac=lambda q: (3 * CHECKPOINTS * (1-q[:, None])**(CHECKPOINTS-1)).T,
        bounds=(0, 1), max_nfev=2000, gtol=1e-9, ftol=1e-10, xtol=1e-10)
        for _ in range(starts)]
    rates = min(rate_fits, key=lambda fit: fit.cost)
    if not mixture.success or not rates.success:
        raise RuntimeError('Best aggregate fit did not converge')
    # Distinct objective: maximize each problem's censored time likelihood.
    q_mle = (times <= 8).sum(axis=1) / np.minimum(times, 8).sum(axis=1)
    fresh = np.array([r['fresh_successes'] for r in rows])
    return dict(problems=len(rows), trajectories=trials,
                observed_counts=observed.tolist(),
                fresh_calibrated=summarize(3*LOOKUP[fresh].sum(axis=0), observed),
                fitted_single_rate=summarize(trials*(1-(1-single.x)**CHECKPOINTS),
                                             observed, q=float(single.x)),
                fitted_two_rate_mixture=summarize(two_rate_counts(mixture.x, trials),
                                                  observed, parameters=mixture.x.tolist()),
                fitted_problem_rates=summarize(geometric_counts(rates.x), observed,
                                               sorted_q=np.sort(rates.x).tolist(),
                                               start_costs=[float(f.cost) for f in rate_fits]),
                problemwise_time_mle=summarize(geometric_counts(q_mle), observed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'local_data/free_geometric_opus.json')
    parser.add_argument('--starts', type=int, default=8)
    args = parser.parse_args()
    if args.starts < 1:
        parser.error('--starts must be positive')
    rows, hashes = collect(ROOT, 'opus')
    reports = {name: analyze([r for r in rows if dataset is None or r['dataset'] == dataset], args.starts)
               for name, dataset in [('All', None), ('AOBench', 'results'), ('IMO-ProofBench', 'results-imobench')]}
    output = dict(model='opus', starts=args.starts, seed=35,
                  metric='RMSE of cumulative solved counts over blocks 1 through 8',
                  interpretation='In-sample shape check only; no p-values, no oracle data, no held-out prediction claim.',
                  reports=reports, audit_sha256=hashes)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    for name, report in reports.items():
        print(name, {key: round(value['rmse'], 4) for key, value in report.items()
                     if isinstance(value, dict) and 'rmse' in value})
    print(args.output)


if __name__ == '__main__':
    main()
