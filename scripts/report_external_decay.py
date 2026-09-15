#!/usr/bin/env python3
"""Discovery-decay sensitivity with AOBench-fitted shared priors frozen."""
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


def evaluate(rows,theta,seed,draws=16384):
    rng=np.random.default_rng(seed);model=Joint(rows,full_support=True)
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
    a=np.array(alphas);eps=np.array(epsilons)
    counts=np.rint(3*np.array([r['observed'] for r in rows])).astype(int)
    first=np.diff(np.c_[np.zeros(len(rows),int),counts,np.full(len(rows),3)],axis=1)
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
    shift=100*np.max(np.abs(curve(gamma).mean(axis=(0,1))-curve(1.).mean(axis=(0,1))))
    return dict(gamma=gamma,delta_log_likelihood=best-base,max_shift_pp=float(shift),draws=draws,seed=seed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.report.read_text());out={}
    for i,name in enumerate(('muse','gpt54','gpt55','opus')):
        out[name]=evaluate(data['reports'][name+'-n1']['rows'],data['fits'][name]['log_hyperparameters'],20260914+i)
        print(name,out[name],flush=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
