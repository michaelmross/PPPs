import math, statistics, json, numpy as np
from gmpy2 import is_prime, isqrt
rng=np.random.default_rng(7)
pool=[tuple(x) for x in json.load(open("oddpool.json"))]; K=len(pool)
def witness_matrix(N0,N1):
    P=[];spans=[]
    for b,c in pool:
        w=set();lo=1<<62;hi=0
        for n in range(N0,N1):
            v=n*n+b*n+c;bd=int(isqrt(v))
            if bd<lo:lo=bd
            if bd>hi:hi=bd
            if is_prime(v): w.add(bd)
        P.append(w);spans.append((lo,hi))
    B0=max(s[0] for s in spans)+5;B1=min(s[1] for s in spans)-5
    nC=B1-B0+1;M=np.zeros((K,nC),dtype=np.int8)
    for i,p in enumerate(P):
        for b in p:
            if B0<=b<=B1: M[i,b-B0]=1
    logm=2*sum(math.log(b) for b in range(B0,B1+1))/nC
    return M,nC,logm
def cramer_tail(M,nC,t=8,reps=60):
    real=M.sum(0);obs=int((real<=t).sum());nh=[]
    for _ in range(reps):
        c=np.zeros(nC,dtype=np.int32)
        for i in range(K): c[rng.choice(nC,int(M[i].sum()),replace=False)]+=1
        nh.append(int((c<=t).sum()))
    m=statistics.mean(nh);return obs,m,(obs/m if m>0 else float('inf'))
print(f"ALWAYS-ODD pool ({K} curves) scale-stability:\n")
print(f"{'value':>9} {'rho':>5} {'over-disp':>9} {'excessVar':>9} {'hardband real/Cramer':>20}")
for N0,N1 in [(2000,8000),(40000,46000),(200000,206000),(1000000,1006000)]:
    M,nC,lm=witness_matrix(N0,N1)
    real=M.sum(0);rho=M.sum(1)/nC
    od=real.var()/(real.mean()*(1-real.mean()/K)); ev=real.var()-sum(r*(1-r) for r in rho)
    obs,m,r=cramer_tail(M,nC)
    print(f"{math.exp(lm):9.1e} {rho.mean():.3f} {od:9.3f} {ev:9.2f}   {obs} vs {m:.0f}  ({r:.2f}x)")
