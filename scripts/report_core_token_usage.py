"""Sum recorded output usage for the paper's three core arms, without double counting.

Reads one parent metadata file per problem/seed/arm. Parallel parent counters
already include all eight attempts. Excludes extra allocations, controls,
grading, and any runs outside the frozen 57-problem paper cohort.
"""
import hashlib
import json
from pathlib import Path

from scripts.report_joint_allocation import MODELS


def collect(root):
    report = json.loads((root/'paper/img/allocation_report.json').read_text())
    totals, fingerprints = {}, {}
    for model, directory in MODELS.items():
        rows = report['reports'][f'{model}-n1']['rows']
        if len({(r['dataset'], r['problem']) for r in rows}) != 57 or len(rows) != 57:
            raise ValueError('Expected 57 unique problems')
        totals[model] = {}
        for arm in ('baseline-parallel', 'hint-sequential', 'baseline-sequential'):
            used, budget = 0, 0
            for row in rows:
                for seed in (1, 2, 3):
                    path = root/row['dataset']/directory/arm/row['problem']/f'seed_{seed}'/'meta.json'
                    raw = path.read_bytes()
                    meta = json.loads(raw)
                    spent = meta['output_tokens_spent']
                    if type(spent) is not int or spent < 0 or meta['budget_output_tokens'] != 1600000:
                        raise ValueError(f'Invalid usage/budget: {path}')
                    used += spent
                    budget += meta['budget_output_tokens']
                    fingerprints[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            totals[model][arm] = dict(parent_records=171, recorded_output_tokens=used,
                                      allocated_output_token_budget=budget)
    return dict(scope=__doc__, models=totals,
                recorded_output_tokens=sum(a['recorded_output_tokens'] for m in totals.values() for a in m.values()),
                allocated_output_token_budget=sum(a['allocated_output_token_budget'] for m in totals.values() for a in m.values()),
                metadata_sha256=fingerprints)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    output = collect(root)
    target = root/'local_data/core_token_usage.json'
    target.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'metadata_sha256'}, indent=2))
    print(target)
