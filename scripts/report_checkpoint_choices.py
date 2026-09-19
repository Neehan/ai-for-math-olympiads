"""Retrospective one-to-four-block continue/restart comparisons, without refitting.

Continue: baseline-sequential, conditional on cumulative failure through k.
Restart: average cumulative success of all three separate late-baseline-sequential
trajectories through h blocks (the 4x endpoint includes its no-sketch fork).
Fit: saved Parallel-8/oracle R-DE only. At each k, choose one aggregate action.
The same restart observation is reused across checkpoints; these are not
independent experiments or a rollout of an adaptive policy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import betaln

from scripts.allocation_estimators import Joint
from scripts.report_joint_allocation import MODELS, DATASETS, indexed_audits, proof_curve


def de_choices(curves):
    curves = np.asarray(curves, dtype=float)
    survival = 1 - curves[:, :-1]
    valid = survival > 1e-12
    continuation = np.divide(np.diff(curves, axis=1), survival,
                             out=np.full_like(survival, np.nan), where=valid)
    restart = np.broadcast_to(curves[:, :1], continuation.shape).copy()
    restart[~valid] = np.nan
    return continuation, restart


def rde_choices(joint, theta, horizon=1):
    """Integrate shared parameter uncertainty conditional on the current failure.

    Continue = (E[s(k+h)] - E[s(k)]) / (1 - E[s(k)]).
    Restart = (E[s(h)] - E[s(h)*s(k)]) / (1 - E[s(k)]).
    Integrate the product, not a product of posterior means.
    """
    curves, _ = joint.predict(theta)
    _, weights, ap, bp, ep, rp, d = joint.components(theta)
    e2 = ep * (ep + 1) / ((ep + rp) * (ep + rp + 1))
    e_cross = ep * rp / ((ep + rp) * (ep + rp + 1))
    rest2 = rp * (rp + 1) / ((ep + rp) * (ep + rp + 1))
    tail_den = (d[1:].sum() + joint.c[:, 1:].sum(axis=1))[:, None]
    if horizon not in (1, 2, 3, 4):
        raise ValueError('Additional budget must be 1, 2, 3, or 4')
    tail_mass = {ell: (d[1:ell].sum() + joint.c[:, 1:ell].sum(axis=1))[:, None]
                 for ell in range(1, 9)}
    cross = np.zeros((len(joint.rows), 8-horizon))
    for k in range(1, 9-horizon):
        for j in range(1, k + 1):
            ell = k - j + 1
            for i in range(1, horizon+1):
                t = horizon-i+1
                tail_prod = (tail_mass[ell]*tail_mass[t] + tail_mass[min(ell,t)]) / (tail_den*(tail_den+1))
                eps_prod = e2 + e_cross*(tail_mass[ell]+tail_mass[t])/tail_den + rest2*tail_prod
                alpha_moment = np.exp(betaln(ap + 2, bp + j + i - 2) - betaln(ap, bp))
                cross[:, k - 1] += (weights * alpha_moment * eps_prod).sum(axis=1)
    survival = 1-curves[:, :8-horizon]
    continuation = (curves[:, horizon:] - curves[:, :8-horizon])/survival
    restart = (curves[:, horizon-1: horizon] - cross) / survival
    for values in (continuation, restart):
        if not np.all(np.isfinite(values)) or np.any(values < -1e-9) or np.any(values > 1 + 1e-9):
            raise ValueError('Invalid posterior conditional probabilities')
    return continuation, np.clip(restart, 0, 1)


def action(delta):
    if not np.isfinite(delta):
        return 'undefined'
    return 'continue' if delta > 1e-9 else 'restart' if delta < -1e-9 else 'tie'


def evaluate(curves, restart_outcomes, predictions, horizon=1, restart_estimator='separate-three'):
    """Arrays are problem x seed x block, problem x seed, and problem x block.

    Cohort recommendations maximize predicted total next-block success over
    the actual at-risk cohort, never using either next-block outcome.
    Observed-best gaps are descriptive estimates, not population regret.
    """
    curves = np.asarray(curves)
    restart_outcomes = np.asarray(restart_outcomes)
    if curves.shape[1:] != (3, 8) or restart_outcomes.shape != curves.shape[:2]:
        raise ValueError('Expected three seeds and eight checkpoints per problem')
    if np.any(np.diff(curves.astype(int), axis=2) < 0):
        raise ValueError('Success curves must be cumulative')
    if restart_estimator == 'separate-three':
        restart_rates = np.broadcast_to(restart_outcomes.mean(axis=1)[:, None], curves.shape[:2])
    elif restart_estimator == 'leave-one-out-five':
        original = curves[:, :, horizon-1]
        restart_rates = (restart_outcomes.sum(axis=1)[:, None]
                         + original.sum(axis=1)[:, None] - original) / 5
    else:
        raise ValueError('Unknown restart estimator')
    result = []
    for k in range(8-horizon):
        risk = 1 - curves[:, :, k]
        weights = risk.sum(axis=1)
        c = (risk * curves[:, :, k + horizon]).sum(axis=1)
        r = (risk * restart_rates).sum(axis=1)
        row = dict(k=k + 1, horizon=horizon, at_risk=int(weights.sum()),
                   continue_solved=int(c.sum()), restart_solved=float(r.sum()),
                   observed_advantage=float(c.sum() - r.sum()), frameworks={})
        for name, (pc, pr) in predictions.items():
            valid = np.isfinite(pc[:, k]) & np.isfinite(pr[:, k])
            undefined = int(weights[~valid].sum())
            delta = float(weights[valid] @ (pc[valid, k] - pr[valid, k])) if not undefined else float('nan')
            choice = action(delta)
            achieved = {'continue': float(c.sum()), 'restart': float(r.sum()),
                        'tie': float((c.sum() + r.sum()) / 2), 'undefined': None}[choice]
            row['frameworks'][name] = dict(predicted_advantage=None if undefined else delta,
                                          choice=choice, achieved=achieved,
                                          empirical_gap=None if achieved is None else float(max(c.sum(), r.sum()) - achieved),
                                          undefined_at_risk=undefined)
            # Also retain per-problem decisions, not conflated with cohort-wide choices.
            diff = pc[:, k] - pr[:, k]
            choose_c = diff > 1e-9
            choose_r = diff < -1e-9
            ties = valid & ~choose_c & ~choose_r
            row['frameworks'][name]['per_problem'] = dict(
                continue_trials=int(weights[choose_c].sum()),
                restart_trials=int(weights[choose_r].sum()), tie_trials=int(weights[ties].sum()),
                achieved=None if undefined else float(c[choose_c].sum() + r[choose_r].sum() + .5*(c[ties]+r[ties]).sum()))
        result.append(row)
    return result


def verify_record(directory, record):
    if json.loads((directory / 'audit.json').read_text()) != record:
        raise ValueError(f'Stale compiled audit: {directory}')
    for label, audit in [('final', record), *record['budget_cuts'].items()]:
        file = directory / ('solution.md' if label == 'final' else f'solution_{label}.md')
        expected = audit.get('solution_sha256')
        if expected and (not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != expected):
            raise ValueError(f'Stale proof audit: {file}')


def collect(root, report, restart_estimator='separate-three', selected_model=None):
    # Freeze intervention data/hyperparameters; stop if their source banks changed.
    for path, digest in report['audit_sha256'].items():
        if any(f'/{arm}/' in path for arm in ('baseline-parallel', 'hint-sequential')):
            if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Intervention fit is stale: {path}')
    output = dict(protocol=__doc__, restart_estimator=restart_estimator, models={}, audit_sha256={})
    for model, dirname in MODELS.items():
        if selected_model is not None and model != selected_model:
            continue
        rows = report['reports'][f'{model}-n1']['rows']
        joint = Joint(rows)
        theta = np.array(report['fits'][model]['log_hyperparameters'])
        rde_curve, _ = joint.predict(theta)
        np.testing.assert_allclose(rde_curve, [r['predictions']['both_regularized'] for r in rows], atol=1e-9)
        predicted = {h: dict(RDE=rde_choices(joint, theta, h)) for h in (1, 2, 3, 4)}
        indices = {}
        for dataset, count in DATASETS.items():
            base = root / dataset / dirname
            indices[dataset] = {arm: indexed_audits(base / arm / 'audit.jsonl', (1, 2, 3), output['audit_sha256'])
                                for arm in ('baseline-sequential', 'late-baseline-sequential')}
            required = {(r['problem'], s) for r in rows if r['dataset'] == dataset for s in (1, 2, 3)}
            if len(required) != count * 3 or any(set(idx) != required for idx in indices[dataset].values()):
                raise ValueError('Incomplete matched cohort')
        curves, restarts, trial_rows = [], [], []
        for row in rows:
            base = root / row['dataset'] / dirname
            current, fresh = [], []
            for seed in (1, 2, 3):
                key = row['problem'], seed
                a = indices[row['dataset']]['baseline-sequential'][key]
                b = indices[row['dataset']]['late-baseline-sequential'][key]
                adir = base / 'baseline-sequential' / key[0] / f'seed_{seed}'
                bdir = base / 'late-baseline-sequential' / key[0] / f'seed_{seed}'
                am = json.loads((adir / 'meta.json').read_text())
                bm = json.loads((bdir / 'meta.json').read_text())
                ids_a = set(am.get('provider_session_ids', a.get('provider_session_ids', {})).values())
                ids_b = set(bm.get('provider_session_ids', b.get('provider_session_ids', {})).values())
                if not ids_a or not ids_b or not ids_a.isdisjoint(ids_b):
                    raise ValueError(f'Non-independent or undocumented source sessions: {base / key[0]}')
                verify_record(adir, a)
                verify_record(bdir, b)
                c = proof_curve(a, 8, 5)
                r = proof_curve(b, 4, 5)
                current.append(c)
                fresh.append(r)
                trial_rows.append(dict(dataset=row['dataset'], problem=key[0], seed=seed, cumulative=c, restart=r))
            np.testing.assert_allclose(np.mean(current, axis=0), row['observed'], atol=1e-12)
            curves.append(current)
            restarts.append(fresh)
        restarts = np.asarray(restarts)
        checkpoints = [entry for h in (1, 2, 3, 4)
                       for entry in evaluate(curves, restarts[:, :, h-1], predicted[h], h, restart_estimator)]
        summaries = {}
        for h in (1, 2, 3, 4):
            subset = [r for r in checkpoints if r['horizon'] == h]
            summaries[h] = dict(checkpoints=len(subset),
                RDE=float(np.mean([r['frameworks']['RDE']['empirical_gap'] for r in subset])),
                always_continue=float(np.mean([max(0., -r['observed_advantage']) for r in subset])),
                always_restart=float(np.mean([max(0., r['observed_advantage']) for r in subset])))
        averages = {policy: float(np.mean([s[policy] for s in summaries.values()]))
                    for policy in ('RDE', 'always_continue', 'always_restart')}
        output['models'][model] = dict(checkpoints=checkpoints, summaries=summaries,
                                      equal_horizon_average=averages, trials=trial_rows)
    return output


def render_tables(report):
    """Render measured-regret tables without changing fits or averaging rules."""
    names = {'muse': 'Muse Spark~1.2', 'gpt54': 'GPT-5.4',
             'gpt55': 'GPT-5.5', 'opus': 'Claude Opus~4.8'}
    policies = ('always_continue', 'always_restart', 'RDE')
    if set(report['models']) != set(names):
        raise ValueError('Paper tables require all four models')

    def cells(values):
        best = min(values)
        return ' & '.join(r'\textbf{' + f'{v:.2f}' + '}'
                          if np.isclose(v, best, atol=1e-10, rtol=0) else f'{v:.2f}'
                          for v in values)

    sensitivity = report['restart_estimator'] == 'leave-one-out-five'
    if report['restart_estimator'] not in ('separate-three', 'leave-one-out-five'):
        raise ValueError('Unknown restart estimator')
    label = 'checkpoint-regret-loo' if sensitivity else 'checkpoint-regret'
    caption = ('Mean measured regret using five leave-one-out restart trajectories. Predictions and selected actions are unchanged from Table~\\ref{tab:checkpoint-regret}; only restart success estimates change.'
               if sensitivity else
               'Mean measured regret in additional solved trials lost; lower is better. Values average checkpoints within each additional budget, then the four budgets. The final row averages the four models. Bold marks row minima.')
    lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{lrrr}',
             r'\toprule Model & Always continue & Always restart & R-DE \\', r'\midrule']
    values = []
    for model, name in names.items():
        row = [report['models'][model]['equal_horizon_average'][p] for p in policies]
        values.append(row)
        lines.append(name + ' & ' + cells(row) + r' \\')
    lines += [r'\midrule', 'Average & ' + cells(np.mean(values, axis=0)) + r' \\',
              r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
              r'\label{tab:' + label + '}', r'\end{table}']
    output = {label.replace('-', '_') + '_table.tex': '\n'.join(lines) + '\n'}
    if not sensitivity:
        lines = [r'\begin{table}[htbp]', r'\centering\small', r'\begin{tabular}{llrrr}',
                 r'\toprule Model & Additional budget & Always continue & Always restart & R-DE \\', r'\midrule']
        for model, name in names.items():
            for h in (1, 2, 3, 4):
                summaries = report['models'][model]['summaries']
                row = summaries[h] if h in summaries else summaries[str(h)]
                lines.append((name if h == 1 else '') + ' & $' + str(h) + r'\times$ & '
                             + cells([row[p] for p in policies]) + r' \\')
            if model != 'opus':
                lines.append(r'\midrule')
        lines += [r'\bottomrule', r'\end{tabular}',
                  r'\caption{Measured regret by additional budget, averaged over the $8-h$ eligible checkpoints for budget $h$. Restart success uses three separate trajectories per problem.}',
                  r'\label{tab:checkpoint-regret-budgets}', r'\end{table}']
        output['checkpoint_regret_budgets_table.tex'] = '\n'.join(lines) + '\n'
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('local_data/checkpoint_choices.json'))
    parser.add_argument('--restart-estimator', choices=('separate-three', 'leave-one-out-five'), default='separate-three')
    parser.add_argument('--model', choices=tuple(MODELS))
    parser.add_argument('--tex-dir', type=Path, help='Write paper tables after verifying the analysis')
    args = parser.parse_args()
    data = json.loads((args.root / 'paper/img/allocation_report.json').read_text())
    out = collect(args.root, data, args.restart_estimator, args.model)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    if args.tex_dir is not None:
        for name, content in render_tables(out).items():
            (args.tex_dir / name).write_text(content)
    for model, values in out['models'].items():
        print(model)
        for r in values['checkpoints']:
            print(r)
