#!/usr/bin/env python3
"""Paired stratified bootstrap under AOBench-only shared-prior fitting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.report_measurement_checks import fit_sample


def statistics(rows, observed, predictions, baselines, weights):
    output={}
    for dataset in ('combined','results','results-imobench'):
        ix=np.array([dataset=='combined' or r['dataset']==dataset for r in rows])
        obs=3*weights[ix]@observed[ix];pred=3*weights[ix]@predictions[ix]
        output[dataset]={key:float(np.mean((3*weights[ix]@values[ix]-obs)**2)-np.mean((pred-obs)**2))
                         for key,values in baselines.items()}
    return output


def run_one(task):
    name,indices,rows,observed,baselines=task
    unique,weights=np.unique(indices,return_counts=True)
    selected=[rows[i] for i in unique]
    source=np.array([r['dataset']=='results' for r in selected])
    predictions,_=fit_sample([r for r in selected if r['dataset']=='results'],weights[source],
                              full_support=True,prediction_rows=selected)
    return name,statistics(selected,observed[unique],predictions,{k:v[unique] for k,v in baselines.items()},weights)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=1000);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();data=json.loads(args.report.read_text());rng=np.random.default_rng(20260914)
    indices=np.c_[rng.integers(0,35,(args.replicates,35)),rng.integers(35,57,(args.replicates,22))]
    tasks=[];out={}
    for name in ('muse','gpt54','gpt55','opus'):
        rows=data['reports'][name+'-n1']['rows']
        assert [r['dataset'] for r in rows]==['results']*35+['results-imobench']*22
        obs=np.array([r['observed'] for r in rows])
        base={m:np.array([r['predictions'][m] for r in rows]) for m in ('solved_geometric','oracle_slope_linear')}
        pred=np.array([r['predictions']['both_regularized'] for r in rows])
        out[name]={'point':statistics(rows,obs,pred,base,np.ones(57)),'bootstrap':[]}
        tasks.extend((name,ix,rows,obs,base) for ix in indices)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count,(name,value) in enumerate(pool.map(run_one,tasks,chunksize=5),1):
            out[name]['bootstrap'].append(value)
            if count%200==0:print(count,'/',len(tasks),flush=True)
    for name,r in out.items():
        r['ci95']={ds:{m:np.quantile([b[ds][m] for b in r['bootstrap']],[.025,.975]).tolist()
                       for m in r['point'][ds]} for ds in r['point']}
        print(name,r['point'],r['ci95'],flush=True)
    average={ds:{m:dict(point=float(np.mean([r['point'][ds][m] for r in out.values()])),
                       ci95=np.quantile(np.mean([[b[ds][m] for b in r['bootstrap']] for r in out.values()],axis=0),[.025,.975]).tolist())
                    for m in ('solved_geometric','oracle_slope_linear')} for ds in ('combined','results','results-imobench')}
    args.output.write_text(json.dumps(dict(models=out,average=average,replicates=args.replicates,seed=20260914,
        protocol='Resample 35 AOBench and 22 IMO problems separately; preserve all seeds/checkpoints; six-start full-support prior fit on resampled AOBench only; condition on each resampled problem interventions.'),indent=2)+'\n')

if __name__=='__main__':main()
