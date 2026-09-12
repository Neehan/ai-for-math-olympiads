"""DE and R-DE estimators fitted only to fresh solves and oracle completion times.

R-DE predictions integrate the joint posterior. They are not frequentist
unbiased estimates; N=2 uses the second moment, not squared posterior means.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, logsumexp


def validate_row(row):
    for field in ("solved", "parallel_n", "oracle_n"):
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{field} must be an integer")
    if not 0 <= row["solved"] <= row["parallel_n"] or row["parallel_n"] < 1 or row["oracle_n"] < 1:
        raise ValueError("Invalid fresh/oracle counts")
    e = np.asarray(row["epsilon"], dtype=float)
    if e.shape != (8,) or not np.all(np.isfinite(e)) or np.any((e < 0) | (e > 1)) or np.any(np.diff(e) < 0):
        raise ValueError("epsilon must be an eight-block cumulative probability curve")
    counts = e * row["oracle_n"]
    if not np.allclose(counts, np.rint(counts), atol=1e-9, rtol=0):
        raise ValueError("Oracle probabilities must correspond to integer completion counts")


def fit_de(row):
    """Constrained joint MLE; alpha=min(1,q/epsilon1) when epsilon1>0.

    At q>epsilon1 the joint fit also adjusts execution. At x=z=0,
    acquisition is unidentified: choose alpha=0 and expose alpha=1 sensitivity.
    """
    validate_row(row)
    m, x, t = row['parallel_n'], row['solved'], row['oracle_n']
    raw = np.array(row['epsilon'])
    z = int(round(t*raw[0]))
    q, e = x/m, z/t
    unidentified = x == 0 and z == 0
    if unidentified:
        alpha, e1 = 0.0, 0.0
    elif q <= e:
        alpha, e1 = min(1.0, q/e), e
    else:
        alpha, e1 = 1.0, (x+z)/(m+t)
    # Full oracle multinomial MLE: redistribute residual 1-e1 over its
    # empirical conditional later first-completion times (including failure).
    eps = (e1+(1-e1)*(raw-e)/(1-e)) if e < 1 else np.ones(8)
    assert np.all(np.diff(eps) >= -1e-12)
    assert np.all((eps >= -1e-12) & (eps <= 1+1e-12))
    def predict(a):
        return [sum(a*(1-a)**(k-1)*eps[K-k] for k in range(1,K+1)) for K in range(1,9)]
    return dict(alpha=alpha, epsilon1=e1, epsilon=eps.tolist(),
                boundary=q>e, unidentified=unidentified,
                prediction_unidentified_alpha0=predict(alpha),
                prediction_unidentified_alpha1=predict(1 if unidentified else alpha))


class Joint:
    def __init__(self, rows):
        if not rows:
            raise ValueError("At least one intervention row is required")
        for row in rows:
            validate_row(row)
        self.rows = rows
        self.x = np.array([r['solved'] for r in rows])[:,None]
        self.m = np.array([r['parallel_n'] for r in rows])[:,None]
        self.t = np.array([r['oracle_n'] for r in rows])
        cum = np.rint(np.array([r['epsilon'] for r in rows])*self.t[:,None]).astype(int)
        self.c = np.diff(np.c_[np.zeros(len(rows),dtype=int),cum,self.t],axis=1)
        assert np.all(self.c>=0)
        self.active = np.flatnonzero(self.c.sum(axis=0)>0)
        if 0 not in self.active or len(self.active) < 2:
            raise ValueError("R-DE requires some oracle completions at block 1 and some later/censored outcomes across the fitting cohort")
        self.h = np.arange(int((self.m-self.x).max())+1)[None,:]
        self.valid = self.h <= self.m-self.x
        f = self.m-self.x
        self.comb = np.where(self.valid, gammaln(f+1)-gammaln(self.h+1)-gammaln(np.maximum(f-self.h,0)+1),-np.inf)

    def components(self, theta):
        A,B = np.exp(theta[:2])
        d = np.zeros(9); d[self.active] = np.exp(theta[2:])
        tau = d.sum()
        dm = gammaln(tau)-gammaln(tau+self.t)
        dm += (gammaln(d[self.active]+self.c[:,self.active])-gammaln(d[self.active])).sum(axis=1)
        ap = A+self.x+self.h
        bp = B+np.maximum(self.m-self.x-self.h,0)
        e0 = d[0]+self.c[:,0,None]
        er = tau-d[0]+(self.t-self.c[:,0])[:,None]
        ep = e0+self.x
        rp = er+self.h
        terms = self.comb + betaln(ap,bp)-betaln(A,B) + betaln(ep,rp)-betaln(e0,er)
        norm = logsumexp(terms,axis=1)
        return dm+norm, np.exp(terms-norm[:,None]), ap,bp,ep,rp,d

    def objective(self,theta):
        return -float(self.components(theta)[0].sum())

    def predict(self,theta):
        _,w,ap,bp,ep,rp,d = self.components(theta)
        e = ep/(ep+rp)
        residual_den = (d[1:].sum()+self.c[:,1:].sum(axis=1))[:,None]
        curves = np.zeros((len(self.rows),8))
        for K in range(1,9):
            for k in range(1,K+1):
                ell = K-k+1
                tail = (d[1:ell].sum()+self.c[:,1:ell].sum(axis=1))[:,None]/residual_den
                eps = e+(1-e)*tail
                first = np.exp(betaln(ap+1,bp+k-1)-betaln(ap,bp))
                curves[:,K-1] += (w*first*eps).sum(axis=1)
        assert np.all(np.diff(curves,axis=1)>=-1e-10)
        assert np.all((curves>=0)&(curves<=1+1e-10))
        return curves, (w*ap/(ap+bp)).sum(axis=1)

    def fit(self):
        g = self.c.sum(axis=0)[self.active]/self.t.sum()
        starts=[]
        for ta,te in [(0.5,0.5),(2,2),(10,0.5),(0.5,10),(10,10),(50,50)]:
            initial=np.log(np.r_[ta*.6,ta*.4,te*g])
            opt=minimize(self.objective,initial,method='L-BFGS-B',bounds=[(-12,12)]*len(initial),
                         options=dict(maxiter=1500,ftol=1e-12,gtol=1e-6,maxls=40))
            starts.append(opt)
        best=min(starts,key=lambda x:x.fun)
        if not best.success or not np.isfinite(best.fun):
            raise RuntimeError(f"R-DE optimization failed: {best.message}")
        return best, [dict(nll=float(s.fun),success=bool(s.success),message=str(s.message)) for s in starts]


def posterior_n2(model,theta):
    _,w,ap,bp,ep,rp,d=model.components(theta)
    s1,_=model.predict(theta)
    den=ep+rp
    e2=ep*(ep+1)/(den*(den+1))
    e_cross=ep*rp/(den*(den+1))
    rest2=rp*(rp+1)/(den*(den+1))
    D=(d[1:].sum()+model.c[:,1:].sum(axis=1))[:,None]
    r=[None]+[(d[1:ell].sum()+model.c[:,1:ell].sum(axis=1))[:,None] for ell in range(1,5)]
    predicted=[]
    for K in range(1,5):
        second=np.zeros_like(w)
        for k in range(1,K+1):
            ell=K-k+1
            for j in range(1,K+1):
                t=K-j+1
                alpha_moment=np.exp(betaln(ap+2,bp+k+j-2)-betaln(ap,bp))
                tail_prod=(r[ell]*r[t]+r[min(ell,t)])/(D*(D+1))
                eps_prod=e2+e_cross*(r[ell]+r[t])/D+rest2*tail_prod
                second+=alpha_moment*eps_prod
        es=s1[:,K-1];es2=(w*second).sum(axis=1)
        assert np.all(es2 >= es**2-1e-10)
        assert np.all(es2 <= es+1e-10)
        pred=2*es-es2
        assert np.all(pred <= 1-(1-es)**2+1e-10)
        predicted.append(pred)
    curves=np.array(predicted).T
    assert np.all(np.diff(curves,axis=1)>=-1e-10)
    return curves


def check_posterior_n1():
    # Independent Gauss-Legendre integration for uniform priors and tiny banks.
    from numpy.polynomial.legendre import leggauss
    nodes,weights=leggauss(60);a=(nodes+1)/2;e=a.copy();weights=weights/2
    for x in range(4):
        row=dict(solved=x,parallel_n=3,oracle_n=2,epsilon=[.5]*8)
        model=Joint([row]);theta=np.zeros(4)  # active times 1 and >8
        ll=model.components(theta)[0][0]
        aa=a[:,None];ee=e[None,:];ww=weights[:,None]*weights[None,:]
        lik=(aa*ee)**x*(1-aa*ee)**(3-x)*ee*(1-ee)
        evidence=(ww*lik).sum()
        assert np.isclose(np.exp(ll),evidence,atol=1e-12)
        pred,_=model.predict(theta)
        for K in range(1,9):
            brute=(ww*lik*ee*(1-(1-aa)**K)).sum()/evidence
            assert np.isclose(pred[0,K-1],brute,atol=1e-11)
    # Nonconstant execution curve, asymmetric priors, and three time categories.
    row=dict(solved=2,parallel_n=5,oracle_n=3,epsilon=[1/3]*3+[2/3]*5)
    model=Joint([row]);theta=np.log([2,3,2,3,4])
    _,_,_,_,_,_,d=model.components(theta)
    aa=a[:,None];ee=e[None,:];ww=weights[:,None]*weights[None,:]
    # Integrate using the oracle-only conditional beta law for epsilon1.
    density_a=np.exp((2-1)*np.log(aa)+(3-1)*np.log1p(-aa)-betaln(2,3))
    density_e=np.exp((3-1)*np.log(ee)+(9-1)*np.log1p(-ee)-betaln(3,9))
    lik=(aa*ee)**2*(1-aa*ee)**3*density_a*density_e
    evidence=(ww*lik).sum();pred,_=model.predict(theta)
    for K in range(1,9):
        target=np.zeros_like(lik)
        for k in range(1,K+1):
            ell=K-k+1
            # Later category posterior masses: block4=4, >8=5.
            eps=ee+(1-ee)*(4/9 if ell>=4 else 0)
            target+=aa*(1-aa)**(k-1)*eps
        assert np.isclose(pred[0,K-1],(ww*lik*target).sum()/evidence,atol=1e-11)


def check_posterior_n2():
    # Independent 3-D quadrature over acquisition, epsilon1, and conditional
    # probability of later completion for a nonconstant execution curve.
    from numpy.polynomial.legendre import leggauss
    nodes,weights=leggauss(35);z=(nodes+1)/2;weights=weights/2
    a=z[:,None,None];e=z[None,:,None];r=z[None,None,:]
    wa=weights[:,None,None];we=weights[None,:,None];wr=weights[None,None,:]
    def pdf(x,A,B):return np.exp((A-1)*np.log(x)+(B-1)*np.log1p(-x)-betaln(A,B))
    row=dict(solved=2,parallel_n=5,oracle_n=3,epsilon=[1/3]*3+[2/3]*5)
    model=Joint([row]);theta=np.log([2,3,2,3,4])
    pred=posterior_n2(model,theta)[0]
    mass=wa*we*wr*pdf(a,2,3)*pdf(e,3,9)*pdf(r,4,5)*(a*e)**2*(1-a*e)**3
    mass/=mass.sum()
    for K in range(1,5):
        s=sum(a*(1-a)**(k-1)*(e+(1-e)*(r if K-k+1>=4 else 0)) for k in range(1,K+1))
        brute=(mass*(1-(1-s)**2)).sum()
        assert np.isclose(pred[K-1],brute,atol=1e-10),(K,pred[K-1],brute)
