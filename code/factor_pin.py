import math, statistics, random
from gmpy2 import is_prime, isqrt
from sympy import primerange
from sympy.functions.combinatorial.numbers import jacobi_symbol
random.seed(13)
PR=list(primerange(3,2000))
def Cf(b,c):
    D=b*b-4*c; w2=sum(1 for n in range(2) if (n*n+b*n+c)%2==0); S=(1-w2/2)/0.5
    for p in PR: S*=(1-(1 if D%p==0 else 1+jacobi_symbol(D%p,p))/p)/(1-1/p)
    return float(S)
def build_ao(N,lo,hi):
    pool,seen,c=[],set(),1_000_001
    while len(pool)<N and c<9_000_000:
        cc=c if c%2 else c+1
        if (1-4*cc) not in seen and lo<=Cf(1,cc)<=hi: seen.add(1-4*cc);pool.append((1,cc))
        c+=2
    return pool
def build_he_spread(N,lo,hi):
    pool,binw=[],(8_000_001-1_000_001)//N
    for k in range(N):
        c=1_000_001+k*binw; c=c if c%2 else c+1
        while c<1_000_001+(k+1)*binw:
            if lo<=Cf(0,c)<=hi: pool.append((0,c));break
            c+=2
    return pool
# matched odd-part, CLEAN high-Cf AO baseline: AO Cf in [3.8,5.0] (odd 1.9-2.5), HE in [1.9,2.5]
AO=build_ao(60,3.8,5.0); HE=build_he_spread(60,1.9,2.5)
N0,N1=200000,202000
def witness(pool):
    P=[];spans=[]
    for b,c in pool:
        w=set();lo=1<<62;hi=0
        for n in range(N0,N1):
            v=n*n+b*n+c;bd=int(isqrt(v))
            if bd<lo:lo=bd
            if bd>hi:hi=bd
            if is_prime(v): w.add(bd)
        P.append(w);spans.append((lo,hi))
    return P,spans
PA,SA=witness(AO); PH,SH=witness(HE)
B0=max(max(s[0] for s in SA),max(s[0] for s in SH))+5
B1=min(min(s[1] for s in SA),min(s[1] for s in SH))-5
win=set(range(B0,B1+1));nC=len(win);PA=[p&win for p in PA];PH=[p&win for p in PH]
rA=statistics.mean(len(p)/nC for p in PA); rH=statistics.mean(len(p)/nC for p in PH)
print(f"AO Cbar={statistics.mean(Cf(*x) for x in AO):.2f} ({len(AO)} curves)  HE Cbar={statistics.mean(Cf(*x) for x in HE):.2f} ({len(HE)} curves)")
print(f"rho_AO={rA:.4f}  rho_HE={rH:.4f}  ratio={rA/rH:.2f}\n")
# AO excess-holes (confirm clean baseline) and K at 5% uncovered for both
def stats(P,Ksub,reps=60):
    o=[];pr=[]
    for _ in range(reps):
        idx=random.sample(range(len(P)),Ksub)
        o.append((nC-len(set().union(*[P[i] for i in idx])))/nC)
        q=1.0
        for i in idx: q*=(1-len(P[i])/nC)
        pr.append(q)
    return statistics.mean(o),statistics.mean(pr)
rbarA=rA; Ka=max(5,int(math.log(0.05)/math.log(1-rbarA)))
oa,pa=stats(PA,min(len(AO),Ka)); print(f"AO excess-holes at ~5%: {oa*100:.2f}%/{pa*100:.2f}% = {oa/pa:.2f}  (clean baseline check)")
data={}
for K in range(10,61,5): data[K]=(stats(PA,K)[0],stats(PH,K)[0])
def Kat(t,idx):
    pts=sorted(data.items())
    for i in range(len(pts)-1):
        k1,v1=pts[i];k2,v2=pts[i+1]
        if v1[idx]>=t>=v2[idx]: return k1+(k2-k1)*(v1[idx]-t)/(v1[idx]-v2[idx])
    return None
for tgt in (0.10,0.05):
    ka=Kat(tgt,0);kh=Kat(tgt,1)
    if ka and kh: print(f"K at {int(tgt*100)}% uncovered:  AO {ka:.0f}   HE-spread {kh:.0f}   ratio {kh/ka:.2f}")
print("\n(predict ratio -> 2.0 now that AO baseline is clean; was 1.73 at Cbar=2.98)")
