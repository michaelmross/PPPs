import numpy as np, math, sys, json
from collections import Counter
sys.argv=['x']
from ppp_gen import BitSieve, QuadScanner, small_primes
N=10**8; gmax=max(200,int(2*math.log(N)**2)+2); T=1001; M=N+(gmax//2+1)*T*T+gmax*T+10**6
scan=QuadScanner(N); sv=BitSieve(M,on_segment=scan); rec=scan.records()
p0,g1,a=rec[:,0],rec[:,1],rec[:,2]
pats=Counter(zip(a.tolist(),g1.tolist())); W=sum(pats.values())
small=[int(p) for p in small_primes(100) if p>=5]; big=[int(p) for p in small_primes(997) if p>100]
n=np.arange(1,1001)
# exact per-pattern factors for p<=100: E_r[ 1[f(n)!=0 mod p] / (1-N(r)/p) ] over allowed p0 residues r, as a function of n mod p
def pattern_factor_table(aa,gg,p):
    b=(gg-3*aa)%p
    # allowed residues: f(1..4) != 0
    fac=np.zeros(p); cnt=0
    for r in range(p):
        c=(r-aa-b)%p
        vals=[(aa*m*m+b*m+c)%p for m in range(p)]
        if any(vals[m]==0 for m in (1,2,3,4)): continue
        Nr=sum(1 for v in vals if v==0)
        if Nr==p: continue
        w=1.0/(1-Nr/p); cnt+=1
        fac+=np.array([0.0 if vals[m]==0 else w for m in range(p)])
    return fac/cnt if cnt else np.ones(p)
# first-order tail for p>100 (uniform model, symmetry-blind)
def cond_probs(p):
    w1=(p-1)/(2*p)*((p-4)*(p-5))/(p*(p-1)); wm=(p-1)/(2*p); w0=(1/p)*((p-4)/p); t=w1+wm+w0
    return wm/t,w0/t,w1/t
tail=np.ones(1000)
for p in big:
    P0,P1,P2=cond_probs(p)
    elig=P0+P1*p*(p-5)/((p-4)*(p-1))+P2*p*(p-6)/((p-4)*(p-2)); forced=P0+P1*p/(p-1)+P2*p/(p-2)
    r=n%p; tail*=np.where((r>=1)&(r<=4),forced,elig)
pred=np.zeros(1000)
for (aa,gg),k in pats.items():
    f=np.ones(1000)
    for p in small:
        f*=pattern_factor_table(aa,gg,p)[n%p]
    pred+=k*f
pred=pred/W*tail
json.dump({'pred_exact_1e8':pred.tolist()},open('profile_pred.json','w'))
prof=json.load(open('analysis.json'))['profile_1e8']
for q in prof:
    lo,hi=q['lo'],q['hi']; print(f"n={lo}..{hi}: obs {q['obs']:.4f} +- {q['se']:.4f}  first-order {q['pred']:.4f}  exact-null {pred[lo-1:hi].mean():.4f}")
