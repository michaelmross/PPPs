import math, statistics
from gmpy2 import is_prime, isqrt
from sympy import factorint, primerange
from sympy.functions.combinatorial.numbers import jacobi_symbol
PR=list(primerange(3,2000))
def Sig(b,c):
    D=b*b-4*c; w2=sum(1 for n in range(2) if (n*n+b*n+c)%2==0); S=(1-w2/2)/0.5
    for p in PR: S*=(1-(1 if D%p==0 else 1+jacobi_symbol(D%p,p))/p)/(1-1/p)
    return float(S)
def Omega(m):
    s=0
    for p,e in factorint(m).items(): s+=e
    return s
# pick three always-odd n^2+n+c curves: high / med / low S
cands={}
for c in range(41, 200000, 2):
    s=Sig(1,c)
    if 'hi' not in cands and s>4.5: cands['hi']=(c,s)
    if 'med' not in cands and 2.3<s<2.6: cands['med']=(c,s)
    if 'lo' not in cands and 1.2<s<1.45: cands['lo']=(c,s)
    if len(cands)==3: break
NMAX=14000
print(f"always-odd PPPs  n^2+n+c,  n=1..{NMAX}  (values up to ~{NMAX**2/1e8:.1f}e8)\n")
print(f"{'curve':>16} {'S':>5} {'P1%':>6} {'P2%':>6} {'P3+%':>6} {'P2/P1':>6} {'Liouville/N':>11}")
results={}
for tag in ('hi','med','lo'):
    c,s=cands[tag]
    n1=n2=n3=0; lio=0
    for n in range(1,NMAX+1):
        v=n*n+n+c
        if is_prime(v): w=1
        else: w=Omega(v)
        if w==1: n1+=1
        elif w==2: n2+=1
        else: n3+=1
        lio+= -1 if w%2 else 1
    N=NMAX
    print(f"{'n^2+n+'+str(c):>16} {s:5.2f} {100*n1/N:6.2f} {100*n2/N:6.2f} {100*n3/N:6.2f} {n2/n1:6.2f} {lio/N:+11.4f}")
    results[tag]=(c,s)
