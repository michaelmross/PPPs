import math, statistics, random
from sympy import primerange
from sympy.functions.combinatorial.numbers import jacobi_symbol
random.seed(1)
N=3_000_000
# spf sieve -> Omega
spf=list(range(N+1))
i=2
while i*i<=N:
    if spf[i]==i:
        for j in range(i*i,N+1,i):
            if spf[j]==j: spf[j]=i
    i+=1
Omega=[0]*(N+1)
for n in range(2,N+1): Omega[n]=Omega[n//spf[n]]+1
semi=[n for n in range(4,N+1) if Omega[n]==2]
print(f"semiprimes up to {N:,}: {len(semi):,}")
gaps=[semi[i+1]-semi[i] for i in range(len(semi)-1)]
# AP-triples of consecutive gaps: g1+g3 == 2*g2  -> 4 consecutive semiprimes on a quadratic, leading coeff a=(g2-g1)/2
ap=[]; monic=[]
for i in range(len(gaps)-2):
    g1,g2,g3=gaps[i],gaps[i+1],gaps[i+2]
    if g1+g3==2*g2:
        a2=g2-g1  # =2a
        ap.append(a2)
        if a2==2:  # monic
            monic.append((semi[i],g1))   # s1 and first gap g
from collections import Counter
print(f"\nAP-gap quadruples (4 consecutive semiprimes on a quadratic): {len(ap):,}  ({100*len(ap)/len(gaps):.2f}% of positions)")
print(f"  of which MONIC (common diff 2, leading coeff a=1): {len(monic):,}  ({100*len(monic)/len(ap):.1f}% of AP-quadruples)")
ac=Counter(a2//2 for a2 in ap)
print(f"  leading-coeff a distribution (top): " + ", ".join(f"a={k}:{v}" for k,v in sorted(ac.items())[:8]))
# shuffled-gap null: does semiprime gap sequence have MORE/FEWER AP-triples than chance?
def count_ap(gg):
    return sum(1 for i in range(len(gg)-2) if gg[i]+gg[i+2]==2*gg[i+1])
nulls=[]
for _ in range(6):
    sg=gaps[:]; random.shuffle(sg); nulls.append(count_ap(sg))
print(f"\nAP-quadruple count:  observed {len(ap):,}   shuffled-gap null {statistics.mean(nulls):.0f}±{statistics.pstdev(nulls):.0f}   ratio {len(ap)/statistics.mean(nulls):.3f}")
# are the MONIC quadratics special? compute S of n^2+bn+c, b=g-1, c=s1, D=(g-1)^2-4 s1
PR=list(primerange(3,500))
def Sig(b,c):
    D=b*b-4*c; w2=sum(1 for n in range(2) if (n*n+b*n+c)%2==0); S=(1-w2/2)/0.5
    for p in PR: S*=(1-(1 if D%p==0 else 1+jacobi_symbol(D%p,p))/p)/(1-1/p)
    return float(S)
Ss=[Sig(g-1, s1) for s1,g in monic[:600]]
print(f"\nmonic quadratics from strict quadruples: mean S = {statistics.mean(Ss):.2f}  (random n^2+bn+c ~ 1.0; loose semiprime-Euler ~ 3.34)")
print("examples (4 consecutive semiprimes lying on n^2+(g-1)n+s1):")
for s1,g in monic[:5]:
    vals=[s1, s1+g, s1+2*g+2, s1+3*g+6]
    print(f"  s1={s1}, gaps=({g},{g+2},{g+4}) -> {vals}, D={(g-1)**2-4*s1}, S={Sig(g-1,s1):.2f}")
