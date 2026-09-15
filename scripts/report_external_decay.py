#!/usr/bin/env python3
"""Discovery-decay sensitivity on all 57 problems with the main fit frozen."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.allocation_estimators import Joint
from scripts.report_sketch_controls import table


def render(data):
    names={'muse':'Muse Spark~1.2','gpt54':'GPT-5.4','gpt55':'GPT-5.5','opus':'Claude Opus~4.8'}
    rows=[]
    for key,name in names.items():
        r=data[key]
        if r['problems']!=57 or r['trajectories']!=171 or r['dataset'] is not None:
            raise ValueError('Decay table requires all 57 problems and 171 trajectories')
        rows.append([name,r['trajectories'],f"{r['gamma']:.3f}",
                     f"{r['delta_log_likelihood']:.2f}",f"{r['max_shift_count']:.2f}"])
    return table(['Model','Trajectories',r'$\widehat\gamma$',r'$\Delta\log L$','Max. count change'],rows,
                 r'Geometric-discovery sensitivity on all 57 problems. $\Delta\log L$ is the improvement over $\gamma=1$; the final column is the largest absolute change in predicted solved trajectories across the eight checkpoints.',
                 'tab:memoryless-acquisition')


def evaluate(rows,theta,seed,draws=16384,*,full_support=False,dataset=None):
    rng=np.random.default_rng(seed);model=Joint(rows,full_support=full_support)
    selected=np.array([dataset is None or r['dataset']==dataset for r in rows])
    if not selected.any():raise ValueError('No selected sensitivity problems')
    _,w,ap,bp,ep,rp,d=model.components(np.asarray(theta))
    ep=np.broadcast_to(ep,w.shape)
    alphas=[];epsilons=[]
    for i in range(len(rows)):
        h=rng.choice(w.shape[1],size=draws,p=w[i]/w[i].sum())
        a=rng.beta(ap[i,h],bp[i,h]);e=rng.beta(ep[i,h],rp[i,h])
        rest=rng.gamma(d[1:]+model.c[i,1:],1,size=(draws,8))
        rest/=rest.sum(axis=1,keepdims=True)
        eps=np.c_[e,e[:,None]+(1-e[:,None])*np.cumsum(rest[:,:7],axis=1)]
        alphas.append(a);epsilons.append(eps)
    a=np.array(alphas)[selected];eps=np.array(epsilons)[selected]
    selected_rows=[r for r,keep in zip(rows,selected) if keep]
    weights=np.array([r['weight'] for r in selected_rows],dtype=int)
    counts=np.rint(weights[:,None]*np.array([r['observed'] for r in selected_rows])).astype(int)
    first=np.diff(np.c_[np.zeros(len(counts),int),counts,weights],axis=1)
    if np.any(first<0):raise ValueError('Observed curves must be cumulative')
    def curve(gamma):
        s=np.zeros_like(eps);p=a.copy()
        for k in range(8):
            s[:,:,k:]+=p[:,:,None]*eps[:,:,:8-k]
            p*=((1-a)*gamma)
        return s
    def likelihood(gamma):
        s=curve(gamma)
        probabilities=np.concatenate([s[:,:,:1],np.diff(s,axis=2),1-s[:,:,-1:]],axis=2)
        logs=np.sum(first[:,None,:]*np.log(np.clip(probabilities,1e-300,1)),axis=2)
        return float(np.sum(logsumexp(logs,axis=1)-np.log(draws)))
    base=likelihood(1.)
    opt=minimize_scalar(lambda g:-likelihood(g),bounds=(0.,1.),method='bounded',options={'xatol':1e-5})
    candidates=[(base,1.),(likelihood(0.),0.),(-opt.fun,float(opt.x))]
    best,gamma=max(candidates)
    baseline=weights@curve(1.).mean(axis=1)
    changed=weights@curve(gamma).mean(axis=1)
    shift=np.max(np.abs(changed-baseline))
    exact,_=model.predict(np.asarray(theta))
    mc_error=float(np.max(np.abs(weights@exact[selected]-baseline)))
    return dict(gamma=gamma,delta_log_likelihood=best-base,max_shift_count=float(shift),
                trajectories=int(weights.sum()),problems=len(selected_rows),dataset=dataset,
                mc_baseline_max_error_count=mc_error,draws=draws,seed=seed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tex',type=Path)
    args=p.parse_args();data=json.loads(args.report.read_text());out={}
    for i,name in enumerate(('muse','gpt54','gpt55','opus')):
        out[name]=evaluate(data['reports'][name+'-n1']['rows'],data['fits'][name]['log_hyperparameters'],20260914+i,
                           full_support=data['fits'][name].get('full_support',False))
        print(name,out[name],flush=True)
    out['_audit_sha256']=data['audit_sha256']
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    if args.tex:args.tex.write_text(render(out))
