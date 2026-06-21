import math, statistics, random
from gmpy2 import is_prime, isqrt
from sympy import primerange
from sympy.functions.combinatorial.numbers import jacobi_symbol
random.seed(5)
PR=list(primerange(3,2000))
def Sig(b,c):
    D=b*b-4*c; w2=sum(1 for n in range(2) if (n*n+b*n+c)%2==0); S=(1-w2/2)/0.5
    for p in PR: S*=(1-(1 if D%p==0 else 1+jacobi_symbol(D%p,p))/p)/(1-1/p)
    return float(S)
# matched ODD-part: AO (n^2+n+c) S in [2.6,3.4] (=2*odd); HE (n^2+c,c odd) S in [1.3,1.7] (=odd)
def build(b, lo, hi, N):
    pool=[];seen=set();c=1000001
    while len(pool)<N and c<6000000:
        cc=c if c%2 else c+1; D=b*b-4*cc
        if D not in seen and lo<=Sig(b,cc)<=hi: seen.add(D);pool.append((b,cc))
        c+=2
    return pool
AO=build(1,2.6,3.4,60); HE=build(0,1.3,1.7,60)
print(f"AO (always-odd) {len(AO)} curves, mean S={statistics.mean(Sig(*x) for x in AO):.2f}")
print(f"HE (half-even)  {len(HE)} curves, mean S={statistics.mean(Sig(*x) for x in HE):.2f}  [matched odd-part]\n")
N0,N1=200000,202000
def witness(pool):
    P=[];spans=[];elig=[]
    for b,c in pool:
        w=set();el={};lo=1<<62;hi=0
        for n in range(N0,N1):
            v=n*n+b*n+c;bd=int(isqrt(v))
            if bd<lo:lo=bd
            if bd>hi:hi=bd
            el[bd]=(n%2==0)         # half-even eligible iff n even (value odd); AO always odd anyway
            if is_prime(v): w.add(bd)
        P.append(w);spans.append((lo,hi));elig.append(el)
    return P,spans,elig
PA,SA,_=witness(AO); PH,SH,EH=witness(HE)
def window(spans): return max(s[0] for s in spans)+5, min(s[1] for s in spans)-5
B0=max(window(SA)[0],window(SH)[0]); B1=min(window(SA)[1],window(SH)[1])
bands=list(range(B0,B1+1)); win=set(bands); nC=len(bands)
PA=[p&win for p in PA]; PH=[p&win for p in PH]
rhoA=statistics.mean(len(p)/nC for p in PA); rhoH=statistics.mean(len(p)/nC for p in PH)
print(f"{nC} bands.  rho_AO={rhoA:.4f}  rho_HE={rhoH:.4f}  ratio={rhoA/rhoH:.2f} (expect ~2 from density)\n")
# uncovered fraction vs K, observed and independent prediction
def uncov(P, rho_list, K, reps=25):
    obs=[];pred=[]
    for _ in range(reps):
        idx=random.sample(range(len(P)),K)
        cov=set().union(*[P[i] for i in idx]); obs.append((nC-len(cov))/nC)
        pr=1.0
        for i in idx: pr*=(1-len(P[i])/nC)
        pred.append(pr)
    return statistics.mean(obs), statistics.mean(pred)
print(f"{'K':>3}  {'AO obs/pred (excess)':>22}   {'HE obs/pred (excess)':>22}")
for K in (15,25,35,45,55):
    ao,ap=uncov(PA,None,K); he,hp=uncov(PH,None,K)
    print(f"{K:>3}  {100*ao:6.3f}%/{100*ap:6.3f}% ({ao/ap if ap>0 else 0:.2f})   {100*he:6.3f}%/{100*hp:6.3f}% ({he/hp if hp>0 else 0:.2f})")
# eligibility diagnostic for HE: per-band count of eligible curves (landing n even)
ecount={}
for el in EH:
    for bd,e in el.items():
        if bd in win and e: ecount[bd]=ecount.get(bd,0)+1
ev=[ecount.get(b,0) for b in bands]; K=len(HE)
od=statistics.pvariance(ev)/(statistics.mean(ev)*(1-statistics.mean(ev)/K))
print(f"\nHE eligibility per band: mean {statistics.mean(ev):.1f}/{K} (expect ~{K/2:.0f}), "
      f"min {min(ev)}, over-disp {od:.2f}")
print(f"  bands with < {K//4} eligible curves (bottleneck): {sum(1 for e in ev if e<K//4)} of {nC}")
