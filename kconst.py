import math, statistics, random
from gmpy2 import is_prime, isqrt
from sympy import primerange
from sympy.functions.combinatorial.numbers import jacobi_symbol
random.seed(3)
PR=list(primerange(3,2000))
def Sig(b,c):
    D=b*b-4*c; w2=sum(1 for n in range(2) if (n*n+b*n+c)%2==0); S=(1-w2/2)/0.5
    for p in PR: S*=(1-(1 if D%p==0 else 1+jacobi_symbol(D%p,p))/p)/(1-1/p)
    return float(S)
# build always-odd pool
pool=[];seen=set();c=1000001
while len(pool)<75 and c<6000000:
    cc=c if c%2 else c+1; D=1-4*cc
    if D not in seen and Sig(1,cc)>=3.0: seen.add(D);pool.append((1,cc))
    c+=2
K=len(pool); Sbar=statistics.mean(Sig(b,c) for b,c in pool)
N0,N1=200000,203000   # value ~4e10, band index M~2e5
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
win=set(range(B0,B1+1));nC=len(win);P=[p&win for p in P]
rho=[len(p)/nC for p in P]; rhobar=statistics.mean(rho)
logM=math.log(B1)
print(f"always-odd pool: {K} curves, mean S={Sbar:.2f}, band index M~{B1:.0f} (logM={logM:.2f}), {nC} bands")
print(f"measured mean rho={rhobar:.4f}   predicted S/(2 logM)={Sbar/(2*logM):.4f}\n")
# (a) independence: excess-holes obs/predicted
print("excess-holes (observed uncovered / independent prediction):")
for Ksub in (15,25,35):
    obs=[];pred=[]
    for _ in range(40):
        idx=random.sample(range(K),Ksub)
        cov=set().union(*[P[i] for i in idx]); obs.append((nC-len(cov))/nC)
        pr=1.0
        for i in idx: pr*=(1-rho[i])
        pred.append(pr)
    print(f"  K={Ksub}: obs {statistics.mean(obs)*100:.3f}%  pred {statistics.mean(pred)*100:.3f}%  ratio {statistics.mean(obs)/statistics.mean(pred):.3f}")
# (b) K to fully cover the window (random order), vs predicted log(nC)/rhobar
def Kcover():
    order=random.sample(range(K),K); uncov=set(win); used=0
    for i in order:
        used+=1; uncov-=P[i]
        if not uncov: return used
    return None
Ks=[Kcover() for _ in range(60)]; Ks=[k for k in Ks if k]
pred_win=math.log(nC)/rhobar
print(f"\nK to fully cover {nC} bands (window):  measured {statistics.mean(Ks):.1f}±{statistics.pstdev(Ks):.1f}   predicted log(nBands)/rhobar={pred_win:.1f}")
print(f"(full-range form 2(logM)^2/Sbar would give {2*logM**2/Sbar:.0f} to cover all ~{int(math.exp(logM)):,} bands up to M)")
