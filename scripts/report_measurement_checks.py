#!/usr/bin/env python3
"""Empirical DE constraint checks and paired, refitted N=1 problem bootstrap."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, digamma

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.allocation_estimators import Joint, fit_de
from scripts.report_joint_allocation import MODELS, collect_interventions, collect_targets, geometric

STARTS = ((.5, .5), (2, 2), (10, .5), (.5, 10), (10, 10), (50, 50))


def objective_gradient(model, theta, weights):
    """Analytic gradient of the existing Joint marginal likelihood."""
    ll, w, ap, bp, ep, rp, d = model.components(theta)
    a, b = np.exp(theta[:2])
    scores = [a*np.sum(w*(digamma(ap)-digamma(ap+bp)-digamma(a)+digamma(a+b)), axis=1),
              b*np.sum(w*(digamma(bp)-digamma(ap+bp)-digamma(b)+digamma(a+b)), axis=1)]
    tau = d.sum()
    e0 = (d[0]+model.c[:, 0])[:, None]
    er = (tau-d[0]+model.t-model.c[:, 0])[:, None]
    for i in model.active:
        direct = digamma(tau)-digamma(tau+model.t)+digamma(d[i]+model.c[:, i])-digamma(d[i])
        if i == 0:
            mix = digamma(ep)-digamma(ep+rp)-digamma(e0)+digamma(e0+er)
        else:
            mix = digamma(rp)-digamma(ep+rp)-digamma(er)+digamma(e0+er)
        scores.append(d[i]*(direct+np.sum(w*mix, axis=1)))
    return -float(weights@ll), -np.array([weights@s for s in scores])


def fit_sample(rows, weights, *, full_support=False, prediction_rows=None):
    """Six intervention-only starts; exact saturated-execution limit if needed."""
    if not full_support and all(r['epsilon'][0] == 1 for r in rows):
        x = np.array([r['solved'] for r in rows]); m = np.array([r['parallel_n'] for r in rows])
        def objective(theta):
            a, b = np.exp(theta); ap, bp = a+x, b+m-x
            ll = betaln(ap,bp)-betaln(a,b)
            ga = a*(digamma(ap)-digamma(ap+bp)-digamma(a)+digamma(a+b))
            gb = b*(digamma(bp)-digamma(ap+bp)-digamma(b)+digamma(a+b))
            return -float(weights@ll), -np.array([weights@ga,weights@gb])
        fits = [minimize(objective, np.log([t*.6,t*.4]), jac=True, method='L-BFGS-B',
                         bounds=[(-12,12)]*2, options=dict(maxiter=1500,ftol=1e-12,gtol=1e-6,maxls=40))
                for t in (.5,2,10,50)]
        best = min(fits,key=lambda r:r.fun)
        if not best.success: raise RuntimeError(str(best.message))
        a,b=np.exp(best.x)
        curves=np.array([1-np.exp(betaln(a+x,b+m-x+k)-betaln(a+x,b+m-x)) for k in range(1,9)]).T
        return curves, dict(nll=float(best.fun), saturated_execution=True)
    model = Joint(rows, full_support=full_support)
    counts = (weights@model.c)[model.active]
    g = (counts+1/9)/(weights@model.t+1) if full_support else counts/(weights@model.t)
    fits=[]
    for ta,te in STARTS:
        initial=np.log(np.r_[ta*.6,ta*.4,te*g])
        opt=minimize(lambda theta:objective_gradient(model,theta,weights),initial,jac=True,
                     method='L-BFGS-B',bounds=[(-12,12)]*len(initial),
                     options=dict(maxiter=1500,ftol=1e-12,gtol=1e-6,maxls=40))
        if not opt.success:
            opt=minimize(lambda theta:objective_gradient(model,theta,weights),opt.x,jac=True,
                         method='L-BFGS-B',bounds=[(-12,12)]*len(initial),
                         options=dict(maxiter=3000,ftol=1e-12,gtol=1e-6,maxls=80))
        fits.append(opt)
    best=min(fits,key=lambda r:r.fun)
    if not best.success or not np.isfinite(best.fun): raise RuntimeError(str(best.message))
    target = model if prediction_rows is None else Joint(prediction_rows, full_support=full_support)
    return target.predict(best.x)[0],dict(nll=float(best.fun),saturated_execution=False,
                                        theta=best.x.tolist(),
                                        starts_nll=[float(r.fun) for r in fits])


def constraint_check(rows):
    details=[]
    for r in rows:
        x,m,t=r['solved'],r['parallel_n'],r['oracle_n']; z=round(t*r['epsilon'][0])
        fit=fit_de(r)
        details.append(dict(dataset=r['dataset'],problem=r['problem'],fresh_solved=x,fresh_trials=m,
                            oracle_solved_1x=z,oracle_trials=t,excess_pp=100*(x/m-z/t),
                            violation=x*t>z*m,alpha_one=fit['alpha']==1,
                            unidentified=fit['unidentified']))
    positive=[r['excess_pp'] for r in details if r['violation']]
    return dict(problems=len(rows),violations=len(positive),alpha_one=sum(r['alpha_one'] for r in details),
                mean_excess_among_violations_pp=float(np.mean(positive)) if positive else 0.,
                max_excess_pp=max(positive,default=0.),mean_positive_excess_all_pp=sum(positive)/len(rows),
                details=details)


def mse_delta(observed, geo, rde, weights):
    o=3*weights@observed; g=3*weights@geo; r=3*weights@rde
    return float(np.mean((g-o)**2)-np.mean((r-o)**2))


def replicate(task):
    name,index,indices,rows,observed,geo=task
    unique,weights=np.unique(indices,return_counts=True)
    predicted,diag=fit_sample([rows[i] for i in unique],weights)
    return name,index,mse_delta(observed[unique],geo[unique],predicted,weights),diag['saturated_execution']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20260914)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fingerprints={}; data={}; tasks=[]
    indices=np.random.default_rng(args.seed).integers(0,57,size=(args.replicates,57))
    identities=None
    for name in MODELS:
        rows=collect_interventions(ROOT,name,5,fingerprints)
        keys=[(r['dataset'],r['problem']) for r in rows]
        if identities is None: identities=keys
        assert keys==identities and len(rows)==57
        # Freeze point estimates before loading target observations.
        rde,diag=fit_sample(rows,np.ones(57))
        targets=collect_targets(ROOT,name,1,rows,[{1:{'rde':p.tolist()}} for p in rde],5,fingerprints)
        assert [(r['dataset'],r['problem']) for r in targets]==keys
        assert all(r['weight']==3 for r in targets)
        observed=np.array([r['observed'] for r in targets])
        geo=np.array([[geometric(r['solved'],r['parallel_n'],k) for k in range(1,9)] for r in rows])
        data[name]=dict(constraints=constraint_check(rows),delta=mse_delta(observed,geo,rde,np.ones(57)),
                        fit=diag,bootstrap=[],saturated_replicates=0)
        print(name,'constraints', {k:v for k,v in data[name]['constraints'].items() if k!='details'},
              'point delta',data[name]['delta'],flush=True)
        tasks.extend((name,b,idx,rows,observed,geo) for b,idx in enumerate(indices))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count,(name,index,delta,saturated) in enumerate(pool.map(replicate,tasks,chunksize=5),1):
            data[name]['bootstrap'].append(delta)
            data[name]['saturated_replicates']+=int(saturated)
            if count%100==0: print(f'{count}/{len(tasks)} refitted comparisons finished',flush=True)
    for name,r in data.items():
        r['ci95']=np.quantile(r['bootstrap'],[.025,.975]).tolist()
        r['positive_fraction']=float(np.mean(np.array(r['bootstrap'])>0))
        print(name,'delta',r['delta'],'CI',r['ci95'],flush=True)
    args.output.write_text(json.dumps(dict(models=data,replicates=args.replicates,seed=args.seed,
        protocol='N=1, 57 problems, 171 trials, paired problem bootstrap; shared intervention priors refitted with six starts; no separate within-problem resampling. Delta=aggregate-count MSE_geo-MSE_RDE.',
        audit_sha256={str(Path(p).relative_to(ROOT)):v for p,v in fingerprints.items()}),indent=2)+'\n')


if __name__=='__main__': main()
