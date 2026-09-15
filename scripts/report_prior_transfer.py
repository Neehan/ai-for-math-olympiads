#!/usr/bin/env python3
"""Transfer AOBench-fitted R-DE priors to IMO-ProofBench interventions.

No target unaided outcomes enter fitting. Strict transfer preserves the current
zero-support convention. A separate full-support sensitivity fits all nine
Dirichlet coordinates using the existing log-parameter bounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.allocation_estimators import Joint, posterior_n2, posterior_n4
from scripts.report_joint_allocation import (
    MODELS, collect_interventions, collect_targets, fit_interventions, summarize,
)
from scripts.report_sketch_controls import table


def render(report):
    names={'muse':'Muse Spark~1.2','gpt54':'GPT-5.4','gpt55':'GPT-5.5','opus':'Claude Opus~4.8'}
    rows=[]
    for model,name in names.items():
        r=report['reports'][model]['allocations']['1']
        if r['problems']!=22 or r['trials']!=66:
            raise ValueError('Prior-transfer evaluation requires 22 IMO-ProofBench problems and 66 trajectories')
        metrics=r['methods']
        rows.append([name,r['trials'],f"{metrics['positive_support_joint']['rmse']:.2f}",
                     f"{metrics['positive_support_transfer']['rmse']:.2f}"])
    return table(['Model','Trajectories','57-problem priors','AOBench-only priors'],rows,
                 r'Prior parameter sensitivity: $N=1$ RMSE in solved-trajectory counts on the same 22 IMO-ProofBench problems. Both fits retain positive prior support for all nine oracle outcomes; only the problems used to learn shared priors change. Each target problem supplies its own intervention measurements, not unaided outcomes, for estimation.',
                 'tab:prior-sensitivity')


def fit_full_support(rows):
    model = Joint(rows)
    model.active = np.arange(9)
    # Positive initialization only: no pseudo-observations enter the likelihood.
    g = (model.c.sum(axis=0) + 1 / 9) / (model.t.sum() + 1)
    starts = []
    for ta, te in [(0.5, 0.5), (2, 2), (10, 0.5), (0.5, 10), (10, 10), (50, 50)]:
        initial = np.log(np.r_[ta * .6, ta * .4, te * g])
        opt = minimize(model.objective, initial, method='L-BFGS-B',
                       bounds=[(-12, 12)] * 11,
                       options=dict(maxiter=1500, ftol=1e-12, gtol=1e-6, maxls=40))
        starts.append(opt)
    best = min(starts, key=lambda result: result.fun)
    if not best.success or not np.isfinite(best.fun):
        raise RuntimeError(str(best.message))
    return model, best, [dict(nll=float(s.fun), success=bool(s.success),
                              message=str(s.message)) for s in starts]


def frozen_curves(training_model, theta, target_rows):
    """Condition on target interventions without refitting any hyperparameter."""
    d = training_model.components(theta)[-1]
    target = Joint(target_rows)
    unsupported = np.flatnonzero((d == 0) & (target.c.sum(axis=0) > 0))
    if len(unsupported):
        return None, [int(k + 1) if k < 8 else '>8' for k in unsupported]
    target.active = training_model.active.copy()
    curves = {1: target.predict(theta)[0], 2: posterior_n2(target, theta),
              4: posterior_n4(target, theta)}
    # An identical fixed prior must give identical answers in batch or per row.
    for i, row in enumerate(target_rows):
        singleton = Joint([row]) if len(np.flatnonzero(target.c[i])) > 1 and target.c[i, 0] > 0 else None
        if singleton is not None:
            singleton.active = target.active.copy()
            np.testing.assert_allclose(singleton.predict(theta)[0][0], curves[1][i], atol=1e-10)
    return curves, []


def fit_details(model, optimum, starts):
    d = model.components(optimum.x)[-1]
    a, b = np.exp(optimum.x[:2])
    return dict(a=float(a), b=float(b), dirichlet=d.tolist(),
                active=model.active.tolist(), nll=float(optimum.fun), starts=starts,
                log_parameters=optimum.x.tolist())


def build_report():
    fingerprints, reports = {}, {}
    for name in MODELS:
        print(f'{name}: collecting intervention observations', file=sys.stderr, flush=True)
        rows = collect_interventions(ROOT, name, 5, fingerprints)
        train = [r for r in rows if r['dataset'] == 'results']
        target = [r for r in rows if r['dataset'] == 'results-imobench']
        assert len(train) == 35 and len(target) == 22
        assert not ({r['problem'] for r in train} & {r['problem'] for r in target})

        print(f'{name}: original joint fit', file=sys.stderr, flush=True)
        original_predictions, original_fit = fit_interventions(rows, prior_dataset=None)
        variants, fits = {}, dict(original_joint={k: v for k, v in original_fit.items()
                                                 if k not in ('de_fits', 'posterior_alpha')})
        for full_support in (False, True):
            label = 'positive_support_transfer' if full_support else 'strict_transfer'
            print(f'{name}: {label} (35 source problems, six starts)', file=sys.stderr, flush=True)
            if full_support:
                model, opt, starts = fit_full_support(train)
            else:
                model = Joint(train)
                opt, starts = model.fit()
            fits[label] = fit_details(model, opt, starts)
            curves, unsupported = frozen_curves(model, opt.x, target)
            variants[label] = dict(curves=curves, unsupported=unsupported)
        # Same support rule on all 57 distinguishes transfer from the support change.
        print(f'{name}: positive-support joint control', file=sys.stderr, flush=True)
        model, opt, starts = fit_full_support(rows)
        fits['positive_support_joint'] = fit_details(model, opt, starts)
        curves, unsupported = frozen_curves(model, opt.x, target)
        variants['positive_support_joint'] = dict(curves=curves, unsupported=unsupported)

        # Fit and freeze every variant before reading unaided outcomes.
        allocations = {}
        target_index = {r['problem']: i for i, r in enumerate(target)}
        for n in ([1, 2, 4] if name in ('muse', 'gpt55') else [1, 2]):
            evaluated = collect_targets(ROOT, name, n, rows, original_predictions, 5, fingerprints)
            evaluated = [r for r in evaluated if r['dataset'] == 'results-imobench']
            assert len(evaluated) == 22 and sum(r['weight'] for r in evaluated) == 66
            baseline = summarize(evaluated)
            result = dict(problems=22, trials=66, methods=baseline['metrics'])
            for label, variant in variants.items():
                if variant['curves'] is None:
                    result['methods'][label] = dict(undefined=True, unsupported=variant['unsupported'])
                    continue
                changed = []
                for row in evaluated:
                    pred = dict(row['predictions'])
                    pred['both_regularized'] = variant['curves'][n][target_index[row['problem']]].tolist()
                    changed.append(dict(row, predictions=pred))
                result['methods'][label] = summarize(changed)['metrics']['both_regularized']
            allocations[str(n)] = result
        reports[name] = dict(fits=fits, allocations=allocations)
        for n, report in allocations.items():
            print(name, n, {k: round(v['rmse'], 4) if 'rmse' in v else 'undefined'
                            for k, v in report['methods'].items()}, file=sys.stderr, flush=True)
    return dict(protocol='Shared priors fitted on AOBench only; target interventions condition the posterior; unaided IMO outcomes only evaluate predictions.',
                support_sensitivity='All nine Dirichlet parameters fitted with log bounds [-12,12], six starts; positive initialization is not extra observations.',
                reports=reports,
                audit_sha256={str(Path(p).relative_to(ROOT)): sha for p, sha in fingerprints.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tex', type=Path)
    args = parser.parse_args()
    report = build_report()
    data = json.dumps(report, indent=2, allow_nan=False)
    if args.tex:
        args.tex.write_text(render(report))
    if args.output:
        args.output.write_text(data + '\n')
        print(f'Saved {args.output}')
    else:
        print(data)
