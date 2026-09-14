"""Stacking experiments for the Paper 2 revision.
A: covering law, window and full-range forms, observed vs independent surrogate vs closed form.
B: pair correlations vs landing offset (clustering transition), with the pair-singular-series prediction.
"""
import numpy as np, math, sys, time, json, itertools
from collections import defaultdict
sys.argv=['x']
from ppp_gen import BitSieve, bateman_horn, legendre_vec, small_primes
t0=time.time()
MAXK=100_000
SV=BitSieve((MAXK+3)*(MAXK+3)+20_001, log=None)          # covers n^2+n+c for n<=MAXK, c<=20000
print(f"sieve to {SV.M:,} in {time.time()-t0:.0f}s", flush=True)

def witness_matrix(cs, K0, K1):
    """W[i,k] = some value of n^2+n+c_i in band [k^2,(k+1)^2) is prime; also return landing index and BH prob."""
    k=np.arange(K0,K1,dtype=np.int64); k2=k*k; k2n=(k+1)*(k+1)
    W=np.zeros((len(cs),len(k)),dtype=bool); NK=np.zeros((len(cs),len(k)),dtype=np.int64); P=np.zeros((len(cs),len(k)))
    for i,c in enumerate(cs):
        n=np.ceil((-1+np.sqrt(4.0*k2-4.0*c+1.0))/2).astype(np.int64)
        f=lambda m: m*m+m+c
        n=np.where(f(n)<k2,n+1,n); n=np.where(f(n-1)>=k2,n-1,n)        # exact landing index
        v1=f(n); v2=f(n+1); in2=v2<k2n
        w=SV.isprime(v1).copy(); w[in2]|=SV.isprime(v2[in2])
        C=bateman_horn(np.array([1]),np.array([1]),np.array([c]),np.array([1-4*c]),997)[0]
        p1=C/np.log(v1.astype(float)); p2=np.where(in2,C/np.log(v2.astype(float)),0.0)
        W[i]=w; NK[i]=n; P[i]=1-(1-p1)*(1-p2)
    return W,NK,P,k

# ---------------- Experiment A: covering law -------------------------------------------------
cs_all=np.arange(1,10_000,2,dtype=np.int64)
Cf_all=bateman_horn(np.ones(len(cs_all),dtype=np.int64),np.ones(len(cs_all),dtype=np.int64),cs_all,1-4*cs_all,997)
top=np.argsort(-Cf_all)[:100]; pool=cs_all[top]; Cpool=Cf_all[top]
Cbar=Cpool.mean()
K0=100
W,NK,P,kk=witness_matrix(pool,K0,MAXK+1)
print(f"A: pool of 100 curves n^2+n+c, odd c<1e4, C_f {Cpool.min():.2f}..{Cpool.max():.2f}, Cbar={Cbar:.3f}; bands {K0}..{MAXK}; "
      f"mean witness rate obs {W.mean():.4f} pred {P.mean():.4f}  [{time.time()-t0:.0f}s]", flush=True)
rng=np.random.default_rng(11)
grid=[1000,2000,5000,10000,20000,50000,100000]
def K_of_M(Wm, R=200):
    """mean/sd over R random orderings of the least prefix size covering every band k0..M, for each M in grid."""
    out={M:[] for M in grid}
    ncur,nb=Wm.shape
    for _ in range(R):
        perm=rng.permutation(ncur)
        firstcov=np.full(nb,ncur,dtype=np.int64)      # position of first covering curve
        for pos,i in enumerate(perm):
            m=(firstcov==ncur)&Wm[i]
            firstcov[m]=pos
        for M in grid:
            span=firstcov[:M-K0+1]
            out[M].append(np.inf if (span==ncur).any() else span.max()+1)
    return {M:(np.mean(v),np.std(v)) for M,v in out.items()}
obsK=K_of_M(W)
surK=K_of_M(rng.random(W.shape)<P)
print("A: M      log M   K observed        K independent surrogate   2(log M)^2/Cbar   log(#bands)/rho_bar(window)")
for M in grid:
    lm=math.log(M); closed=2*lm*lm/Cbar
    rhow=P[:, :M-K0+1].mean()
    print(f"   {M:>6d}  {lm:5.2f}   {obsK[M][0]:6.1f} +- {obsK[M][1]:4.1f}      {surK[M][0]:6.1f} +- {surK[M][1]:4.1f}         {closed:6.1f}          {math.log(M-K0+1)/rhow:6.1f}")
# per-band over-dispersion (relative to matched Poisson-binomial with measured per-curve marginals) in three windows
def overdisp(Wm,Pm):
    """detrended: mean_k (W_k - mu_k)^2 / mean_k sigma_k^2 with mu, sigma from the BH probabilities (bias-corrected)"""
    Pm=Pm*(Wm.mean()/Pm.mean()); mu=Pm.sum(0); s2=(Pm*(1-Pm)).sum(0); Wk=Wm.sum(0)
    return ((Wk-mu)**2).mean()/s2.mean()
for lo,hi in [(1000,4000),(10000,13000),(60000,63000)]:
    print(f"A: over-dispersion (detrended), bands {lo}..{hi}: {overdisp(W[:,lo-K0:hi-K0],P[:,lo-K0:hi-K0]):.3f}")
del W,NK,P

# ---------------- Experiment B: pair correlations and the landing offset ----------------------
cs=np.array(sorted(set(int(10001+499*j)|1 for j in range(40))),dtype=np.int64)   # spread ~2e4, all classes mod 3
CfB=bateman_horn(np.ones(len(cs),dtype=np.int64),np.ones(len(cs),dtype=np.int64),cs,1-4*cs,997)
print(f"B: pool of {len(cs)} curves, c in [{cs.min()},{cs.max()}], C_f {CfB.min():.2f}..{CfB.max():.2f}, classes mod 3: "
      f"{np.bincount(cs%3).tolist()}", flush=True)
# exact pair-singular-series prediction at offset 0 (roots mod p<=97 by brute force)
PR=[int(p) for p in small_primes(97) if p>=3]
roots={}
for c in cs:
    roots[int(c)]={p:frozenset(m for m in range(p) if (m*m+m+c)%p==0) for p in PR}
def pair_pred(ci,cj,t=0):
    r=1.0
    for p in PR:
        Ri=roots[ci][p]; Rj=frozenset((x-t)%p for x in roots[cj][p])
        Ni,Nj=len(Ri),len(Rj)
        if Ni==p or Nj==p: return float('nan')
        r*=(1-len(Ri|Rj)/p)/((1-Ni/p)*(1-Nj/p))
    return r-1
res={}
for (lo,hi) in [(2000,5000),(20000,23000),(60000,63000)]:
    Wb,NKb,Pb,kb=witness_matrix(cs,lo,hi); B=hi-lo
    pairs=list(itertools.combinations(range(len(cs)),2))
    exc=[]; pred=[]; toff=[]; cls=[]
    for i,j in pairs:
        wi,wj=Wb[i],Wb[j]
        exc.append((wi&wj).sum()*B/(wi.sum()*wj.sum())-1)
        t=NKb[j]-NKb[i]; toff.append(np.abs(t).mean())
        # prediction averaged over the actual offsets in the window (exact for the frozen regime)
        vals,counts=np.unique(t,return_counts=True)
        pred.append(sum(cnt*pair_pred(int(cs[i]),int(cs[j]),int(tv)) for tv,cnt in zip(vals,counts))/B)
        cls.append(tuple(sorted((int(cs[i])%3,int(cs[j])%3))))
    exc=np.array(exc); pred=np.array(pred); toff=np.array(toff)
    od=overdisp(Wb,Pb)
    print(f"B: bands {lo}..{hi} (band width ~{2*lo}, c-spread {cs.max()-cs.min()}): mean |offset| {toff.mean():.2f}; "
          f"pool mean pairwise excess {exc.mean():+.4f}; predicted {np.nanmean(pred):+.4f}; corr(obs,pred) {np.corrcoef(exc,pred)[0,1]:.3f}; "
          f"over-dispersion {od:.3f}")
    byc=defaultdict(list)
    for e,p,c in zip(exc,pred,cls): byc[c].append((e,p))
    for c in sorted(byc):
        a=np.array(byc[c]); print(f"     classes mod 3 {c}: observed {a[:,0].mean():+.3f}  predicted {a[:,1].mean():+.3f}  (n={len(a)})")
    res[f"{lo}-{hi}"]={'exc':exc.tolist(),'pred':pred.tolist(),'toff':toff.tolist(),'od':od}
json.dump(res,open('stack_B.json','w'))
print(f"[{time.time()-t0:.0f}s]")
