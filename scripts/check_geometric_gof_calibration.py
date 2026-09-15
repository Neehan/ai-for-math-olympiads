#!/usr/bin/env python3
"""Validate geometric GOF false-positive rates by nested null simulation.

Outer experiments are generated with a KNOWN vector of per-problem q values.
Each outer experiment is analyzed just like the real one: estimate the joint
null q, regenerate a fresh bank and three censored trajectories from that
estimate, and recompute aggregate-curve RMSE from each simulated fresh bank.
The proportion of resulting p-values below a threshold measures the actual
false-positive rate at that generating parameter vector.

This is a finite-sample diagnostic at specified parameter settings, not a
proof of uniform validity for all q. No oracle observations, conditional
alternative test, regularized fit, or new LLM calls are used.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.report_geometric_gof import collect, null_mle, residual, rmse, simulate
from scripts.report_joint_allocation import MODELS


def wilson(successes, total):
    """95% Wilson interval for Monte Carlo rejection frequency."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., float(center-half)), min(1., float(center+half))]


def bootstrap_p(x, t, inner, rng):
    if inner < 1:
        raise ValueError('Need positive inner simulation count')
    qhat = null_mle(x, t)
    actual = float(rmse(residual(x, t)))
    exceed = 0
    for start in range(0, inner, 2000):
        bx, bt = simulate(qhat, min(2000, inner-start), rng)
        exceed += int(np.count_nonzero(rmse(residual(bx, bt)) >= actual - 1e-12))
    return (exceed+1)/(inner+1)


def rate_summary(pvalues, threshold):
    # Use <= for rejection, as in a conventional level-alpha test.
    count = int(np.count_nonzero(pvalues <= threshold))
    return dict(threshold=float(threshold), rejections=count,
                rate=float(count/len(pvalues)), monte_carlo_95_interval=wilson(count, len(pvalues)))


def calibration(q, outer, inner, seed, label, known_reference=100000):
    if outer < 1 or inner < 1 or known_reference < 1:
        raise ValueError('All simulation counts must be positive')
    children = np.random.SeedSequence(seed).spawn(outer+2)
    outer_rng = np.random.default_rng(children[0])
    reference_rng = np.random.default_rng(children[1])
    ox, ot = simulate(q, outer, outer_rng)
    actual = rmse(residual(ox, ot))
    # Control: independent reference distribution using TRUE q, not estimated q.
    reference = []
    for start in range(0, known_reference, 2000):
        bx, bt = simulate(q, min(2000, known_reference-start), reference_rng)
        reference.append(rmse(residual(bx, bt)))
    reference = np.sort(np.concatenate(reference))
    known_p = (1+len(reference)-np.searchsorted(reference, actual-1e-12))/(len(reference)+1)
    estimated_p = np.empty(outer)
    for i in range(outer):
        estimated_p[i] = bootstrap_p(ox[i], ot[i], inner, np.random.default_rng(children[i+2]))
        if (i+1) % 250 == 0:
            print(f'{label}: {i+1}/{outer} outer experiments completed', file=sys.stderr, flush=True)
    thresholds = [.01, .014269857301426986, .05, .1]
    return dict(label=label, generating_q=np.asarray(q).tolist(), outer=outer, inner=inner, seed=seed,
                known_q_reference_size=known_reference,
                fitted_q_procedure=[rate_summary(estimated_p, a) for a in thresholds],
                known_q_control=[rate_summary(known_p, a) for a in thresholds],
                fitted_q_p_quantiles=np.quantile(estimated_p, [.01, .025, .05, .1, .5, .9]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int, default=2000)
    parser.add_argument('--inner', type=int, default=1999)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--model', choices=list(MODELS), action='append')
    parser.add_argument('--opus-boundary-sensitivity', action='store_true',
                        help='Also generate Opus data at q clipped to [0.01,0.99]; inference is unchanged.')
    args = parser.parse_args()
    if min(args.outer, args.inner) < 1:
        parser.error('Simulation counts must be positive')
    chosen = args.model or list(MODELS)
    reports, hashes, base_q = {}, {}, {}
    for model in chosen:
        rows, fingerprints = collect(ROOT, model)
        hashes.update(fingerprints)
        x = np.array([r['fresh_successes'] for r in rows])
        t = np.array([r['first_completion'] for r in rows])
        q = null_mle(x, t)
        base_q[model] = q
        reports[model] = calibration(q, args.outer, args.inner,
                                    args.seed+list(MODELS).index(model), model)
    if args.opus_boundary_sensitivity:
        if 'opus' not in base_q:
            parser.error('Boundary sensitivity requires Opus among selected models')
        reports['opus_boundary_sensitivity'] = calibration(
            np.clip(base_q['opus'], .01, .99), args.outer, args.inner,
            args.seed+4, 'opus_boundary_sensitivity')
    print(json.dumps(dict(reports=reports, audit_sha256=hashes,
                          notes='Known geometric null throughout. Rates are false-positive frequencies; intervals describe Monte Carlo uncertainty, not experimental effect sizes. Fitted q is re-estimated in every outer experiment. Boundary sensitivity alters only the generating truth, not the estimator. This validates specified configurations, not all possible nuisance parameters.'),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
