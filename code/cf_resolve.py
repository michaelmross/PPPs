import math, statistics, random
from gmpy2 import is_prime, isqrt
from sympy import primerange
from sympy.functions.combinatorial.numbers import jacobi_symbol
random.seed(11)
PR=list(primerange(3,2000))
def Cf(b,c):
    D=b*b-4*c; w2=sum(1 for n in range(2) if (n*n+b*n+c)%2==0); S=(1-w2/2)/0.5
    for p in PR: S*=(1-(1 if D%p==0 else 1+jacobi_symbol(D%p,p))/p)/(1-1/p)
    return float(S)
def build_ao(N,lo,hi,cmax=8_000_000):
    pool,seen,c=[],set(),1_000_001
    while len(pool)<N and c<cmax:
        cc=c if c%2 else c+1
        if (1-4*cc) not in seen and lo<=Cf(1,cc)<=hi: seen.add(1-4*cc);pool.append((1,cc))
        c+=2
    return pool
N0,N1=200000,204000
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
print(f"always-odd, window ~{N1-N0} bands at value ~4e10\n")
print(f"{'Cf band':>12} {'n':>3} {'Cbar':>5} {'pairwise excess':>22} {'excess-holes @~5% unc':>24}")
for lo,hi,Ntarget in [(2.6,3.4,55),(3.4,4.3,55),(4.3,7.0,45)]:
    pool=build_ao(Ntarget,lo,hi); K=len(pool)
    P,spans=witness(pool)
    B0=max(s[0] for s in spans)+5;B1=min(s[1] for s in spans)-5
    win=set(range(B0,B1+1));nC=len(win);P=[p&win for p in P]
    rho=[len(p)/nC for p in P]; Cbar=statistics.mean(Cf(*x) for x in pool)
    # mean pairwise excess: |Wi&Wj| nC/(|Wi||Wj|) - 1, over all pairs
    ex=[]
    for i in range(K):
        for j in range(i+1,K):
            int_ij=len(P[i]&P[j])
            if len(P[i]) and len(P[j]): ex.append(int_ij*nC/(len(P[i])*len(P[j]))-1)
    pe=statistics.mean(ex); pe_se=statistics.pstdev(ex)/math.sqrt(len(ex))
    # excess-holes at K giving ~5% uncovered: pick subset size so pred ~5%
    rbar=statistics.mean(rho); Ksub=min(K,max(5,int(math.log(0.05)/math.log(1-rbar))))
    obs=[];pred=[]
    for _ in range(80):
        idx=random.sample(range(K),Ksub)
        obs.append((nC-len(set().union(*[P[i] for i in idx])))/nC)
        pr=1.0
        for i in idx: pr*=(1-rho[i])
        pred.append(pr)
    eh=statistics.mean(obs)/statistics.mean(pred)
    print(f"{lo:.1f}-{hi:.1f}     {K:>3} {Cbar:5.2f}   {pe:+.4f} +- {pe_se:.4f}      "
          f"{statistics.mean(obs)*100:5.2f}%/{statistics.mean(pred)*100:5.2f}% = {eh:.2f}  (Ksub={Ksub})")
print("\n[pairwise excess ~0 => independent; excess-holes ~1 => no union/tail penalty]")
