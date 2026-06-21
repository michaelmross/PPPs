import math, statistics, random
from gmpy2 import is_prime, isqrt
from sympy import primerange
from sympy.functions.combinatorial.numbers import jacobi_symbol
random.seed(7)
PR=list(primerange(3,2000))
def Sig(b,c):
    D=b*b-4*c; w2=sum(1 for n in range(2) if (n*n+b*n+c)%2==0); S=(1-w2/2)/0.5
    for p in PR: S*=(1-(1 if D%p==0 else 1+jacobi_symbol(D%p,p))/p)/(1-1/p)
    return float(S)
# AO clustered (eligibility irrelevant); HE SPREAD across [1e6, 8e6] via bins
def build_ao(N):
    pool=[];seen=set();c=1000001
    while len(pool)<N and c<6000000:
        cc=c if c%2 else c+1
        if (1-4*cc) not in seen and 2.6<=Sig(1,cc)<=3.4: seen.add(1-4*cc);pool.append((1,cc))
        c+=2
    return pool
def build_he_spread(N):
    pool=[]; lo,hi=1_000_001, 8_000_001; binw=(hi-lo)//N
    for k in range(N):
        c0=lo+k*binw
        c=c0 if c0%2 else c0+1
        while c<c0+binw:
            if 1.3<=Sig(0,c)<=1.7: pool.append((0,c)); break
            c+=2
    return pool
AO=build_ao(60); HE=build_he_spread(60)
cs=[c for _,c in HE]
print(f"AO {len(AO)} curves (S~{statistics.mean(Sig(*x) for x in AO):.2f}, c near 1e6)")
print(f"HE-spread {len(HE)} curves (S~{statistics.mean(Sig(*x) for x in HE):.2f}, c in [{min(cs):.2e},{max(cs):.2e}], span {(max(cs)-min(cs))/4e5:.0f} band-widths)\n")
N0,N1=200000,202000
def witness(pool):
    P=[];spans=[];elig=[]
    for b,c in pool:
        w=set();el={};lo=1<<62;hi=0
        for n in range(N0,N1):
            v=n*n+b*n+c;bd=int(isqrt(v))
            if bd<lo:lo=bd
            if bd>hi:hi=bd
            el[bd]=(n%2==0)
            if is_prime(v): w.add(bd)
        P.append(w);spans.append((lo,hi));elig.append(el)
    return P,spans,elig
PA,SA,_=witness(AO); PH,SH,EH=witness(HE)
B0=max(max(s[0] for s in SA),max(s[0] for s in SH))+5
B1=min(min(s[1] for s in SA),min(s[1] for s in SH))-5
bands=list(range(B0,B1+1)); win=set(bands); nC=len(bands)
PA=[p&win for p in PA]; PH=[p&win for p in PH]
# eligibility over-dispersion for HE-spread
ec=[sum(1 for el in EH if el.get(b,False)) for b in bands]; K=len(HE)
od=statistics.pvariance(ec)/(statistics.mean(ec)*(1-statistics.mean(ec)/K))
print(f"HE-spread eligibility per band: mean {statistics.mean(ec):.1f}/{K}, min {min(ec)}, over-disp {od:.2f}  (was 60 / min 0 when clustered)\n")
def uncov(P,Ksub,reps=25):
    o=[]
    for _ in range(reps):
        idx=random.sample(range(len(P)),Ksub)
        o.append((nC-len(set().union(*[P[i] for i in idx])))/nC)
    return statistics.mean(o)
print(f"{'K':>3}  {'AO uncovered':>12}  {'HE-spread uncovered':>19}")
data={}
for Ksub in (10,20,30,40,50,60):
    a=uncov(PA,Ksub); h=uncov(PH,Ksub); data[Ksub]=(a,h)
    print(f"{Ksub:>3}  {100*a:11.3f}%  {100*h:18.3f}%")
# K at matched ~5% uncovered (interpolate)
def Kat(target, idx):
    pts=sorted(data.items())
    for i in range(len(pts)-1):
        k1,v1=pts[i]; k2,v2=pts[i+1]; y1,y2=v1[idx],v2[idx]
        if y1>=target>=y2:
            return k1+(k2-k1)*(y1-target)/(y1-y2)
    return None
ka=Kat(0.05,0); kh=Kat(0.05,1)
if ka and kh: print(f"\nK at 5% uncovered:  AO {ka:.0f}   HE-spread {kh:.0f}   ratio {kh/ka:.2f}  (predict 2.0)")
else: print(f"\n(HE-spread reaches {100*min(data[60][1],data[50][1]):.1f}% at K=60 -- {'covers' if data[60][1]<0.1 else 'plateau?'})")
