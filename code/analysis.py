import numpy as np, math, sys, time, json
from collections import Counter
sys.argv=['x']
from ppp_gen import BitSieve, QuadScanner, bateman_horn, legendre_vec, root_counts, root_counts_2, small_primes, uniform_null, gap_null
SP=[5,7,11,13,17,19,23,29,31]
out={}
def build(N):
    gmax=max(200,int(2*math.log(N)**2)+2); T=1001; M=N+(gmax//2+1)*T*T+gmax*T+10**6
    scan=QuadScanner(N); sv=BitSieve(M,on_segment=scan)
    rec=scan.records(); p0,g1,a=rec[:,0],rec[:,1],rec[:,2]
    b=g1-3*a; c=p0-a-b; D=(g1-a)**2-4*a*p0
    key=np.stack((a,b%(2*a),D),1); _,first=np.unique(key,axis=0,return_index=True)
    distinct=np.zeros(len(rec),bool); distinct[first]=True
    return sv,p0,g1,a,b,c,D,distinct
# ---------- 1e6 diagnostics
sv,p0,g1,a,b,c,D,distinct=build(10**6)
n=np.arange(1,1001,dtype=np.int64)
vals=a[:,None]*n*n+b[:,None]*n+c[:,None]; isp=sv.isprime(vals); cnt=isp.sum(1); rho=cnt/1000
Cf=bateman_horn(a,b,c,D,997)
leg={p:legendre_vec(D,p) for p in SP}; S=sum((leg[p]==-1).astype(int) for p in SP)
# uniform null histogram
un=uniform_null(SP); q=[un[p][0] for p in SP]
def pb(qs):
    d=np.array([1.0])
    for r in qs:
        nd=np.zeros(len(d)+1); nd[:-1]+=d*(1-r); nd[1:]+=d*r; d=nd
    return d
dist_u=pb(q); nd=int(distinct.sum())
out['null_q']=q; out['E_S_uniform']=float(sum(i*dist_u[i] for i in range(10)))
out['hist_obs_1e6_distinct']=[int((S[distinct]==s).sum()) for s in range(10)]
out['hist_obs_1e6_records']=[int((S==s).sum()) for s in range(10)]
out['hist_uniform_1e6_distinct']=[float(dist_u[s]*nd) for s in range(10)]
pats=Counter(zip(a[distinct].tolist(),g1[distinct].tolist()))
dist_g=np.zeros(10)
for (aa,gg),k in pats.items(): dist_g+=k*pb([gap_null(aa,gg,p)[0] for p in SP])
out['hist_gap_1e6_distinct']=dist_g.tolist()
out['E_S_gap_1e6']=float(sum(i*dist_g[i] for i in range(10))/nd)
out['E_S_obs_1e6']=float(S[distinct].mean()); out['SE_S_obs_1e6']=float(S[distinct].std()/math.sqrt(nd))
# E[N_p|cond] and top/mid/bottom root-count means
Np={p:root_counts(a,b,c,D,p) for p in [5,7,11,13,17,19]}
order=np.argsort(-rho,kind='stable'); mid=len(rho)//2
groups={'top':order[:100],'middle':order[mid-50:mid+50],'bottom':order[-100:]}
tbl={}
for p in [5,7,11,13,17,19]:
    # null E[N_p | cond]
    tot=0; s=0
    for bb in range(p):
        for cc in range(p):
            if all((m*m+bb*m+cc)%p for m in (1,2,3,4)):
                tot+=1; s+=sum(1 for m in range(p) if (m*m+bb*m+cc)%p==0)
    tbl[p]={'null':s/tot,**{g:float(Np[p][idx].mean()) for g,idx in groups.items()}}
out['Np_table']=tbl
for g,idx in groups.items():
    out[f'{g}_meanpct']=float(100*rho[idx].mean()); out[f'{g}_meanS']=float(S[idx].mean())
    out[f'{g}_all31']=float(100*np.mean(S[idx]==9)); out[f'{g}_thru19']=float(100*np.mean([all(leg[p][i]==-1 for p in SP[:6]) for i in idx]))
    out[f'{g}_meanCf']=float(Cf[idx].mean())
# threshold table with mean C_f (records)
out['threshold']=[(s,int((S==s).sum()),float(100*rho[S==s].mean()),float(100*(rho[S==s]>=0.5).mean()),float(Cf[S==s].mean())) for s in range(10) if (S==s).any()]
# correlations
out['corr']={'S':float(np.corrcoef(rho,S)[0,1]),'Cf':float(np.corrcoef(rho,Cf)[0,1]),'logCf':float(np.corrcoef(rho,np.log(Cf))[0,1]),
             'a':float(np.corrcoef(rho,a)[0,1]),'absD':float(np.corrcoef(rho,np.abs(D))[0,1])}
E997=np.zeros(len(a))
for p in small_primes(997):
    p=int(p); E997+= (root_counts_2(a,b,c) if p==2 else root_counts(a,b,c,D,p))/p
out['corr']['E997']=float(np.corrcoef(rho,E997)[0,1])
# nine-symbol linear model R^2 (in-sample and 5-fold CV grouped by curve)
X=np.column_stack([leg[p].astype(float) for p in SP]+[np.ones(len(a))])
beta,*_=np.linalg.lstsq(X,rho,rcond=None); r2=1-((rho-X@beta)**2).sum()/((rho-rho.mean())**2).sum()
out['nine_symbol_R2_insample']=float(r2)
# elite internal ranking: T(D) through 61 and through 97, and log C_f
elite=S==9; ed=elite&distinct
for lab,plist in [('T61',[37,41,43,47,53,59,61]),('T97',[37,41,43,47,53,59,61,67,71,73,79,83,89,97])]:
    Tv=sum((legendre_vec(D,p)==-1).astype(int) for p in plist)
    out[f'elite_corr_{lab}_records']=float(np.corrcoef(rho[elite],Tv[elite])[0,1]); out[f'elite_corr_{lab}_distinct']=float(np.corrcoef(rho[ed],Tv[ed])[0,1])
out['elite_corr_logCf_records']=float(np.corrcoef(rho[elite],np.log(Cf[elite]))[0,1]); out['elite_corr_logCf_distinct']=float(np.corrcoef(rho[ed],np.log(Cf[ed]))[0,1])
out['elite_n']=[int(elite.sum()),int(ed.sum())]
# BH ratio, exclude 1..4, sub-binomial scatter
pred=(Cf[:,None]/np.log(vals.astype(float))).sum(1)/1000; r=rho/pred
out['bh_1e6']={'mean':float(r.mean()),'sd':float(r.std()),'corr':float(np.corrcoef(rho,pred)[0,1])}
r5=isp[:,4:].sum(1)/((Cf[:,None]/np.log(vals[:,4:].astype(float))).sum(1)); out['bh_1e6_n5']={'mean':float(r5.mean()),'sd':float(r5.std())}
resid_sd=float((rho-pred).std()); binom_sd=float(np.sqrt((pred*(1-pred)/1000).mean()))
out['resid_sd_pp']=100*resid_sd; out['binom_sd_pp']=100*binom_sd
# 1e6 done; ---------- index profile at 1e8
del sv
sv,p0,g1,a,b,c,D,distinct=build(10**8)
Cf8=bateman_horn(a,b,c,D,997)
obs_n=np.zeros(1000); pred_n=np.zeros(1000)
for s in range(0,len(a),4000):
    e=min(s+4000,len(a)); v=a[s:e,None]*n*n+b[s:e,None]*n+c[s:e,None]
    obs_n+=sv.isprime(v).sum(0); pred_n+=(Cf8[s:e,None]/np.log(v.astype(float))).sum(0)
ratio_n=obs_n/pred_n
# prediction: prod_{p>n} [1 - delta_p], delta_p from conditioned root distribution
primes=[int(p) for p in small_primes(1000) if p>=5]
def cond_probs(p):
    w1=(p-1)/(2*p)*((p-4)*(p-5))/(p*(p-1)); wm=(p-1)/(2*p); w0=(1/p)*((p-4)/p); t=w1+wm+w0
    return wm/t,w0/t,w1/t   # P(N=0),P(N=1),P(N=2)
fac={}
for p in primes:
    P0,P1,P2=cond_probs(p)
    fac[p]=P0+P1*p*(p-5)/((p-4)*(p-1))+P2*p*(p-6)/((p-4)*(p-2))
pred_ratio=np.array([np.prod([fac[p] for p in primes if p>m]) for m in range(1,1001)])
bins=[(5,5),(6,6),(7,8),(9,10),(11,12),(13,16),(17,20),(21,30),(31,50),(51,100),(101,200),(201,500),(501,1000)]
prof=[]
for lo,hi in bins:
    o=obs_n[lo-1:hi].sum(); pr=pred_n[lo-1:hi].sum()
    prof.append({'lo':lo,'hi':hi,'obs':float(o/pr),'se':float(math.sqrt(o)/pr),'pred':float(pred_n[lo-1:hi]@pred_ratio[lo-1:hi]/pr)})
out['profile_1e8']=prof
json.dump(out,open('analysis.json','w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k not in ('profile_1e8','hist_gap_1e6_distinct','hist_uniform_1e6_distinct')},indent=0)[:3000])
print("profile:"); [print(f"  n={q['lo']}..{q['hi']}: obs {q['obs']:.4f} +- {q['se']:.4f}  pred {q['pred']:.4f}") for q in prof]
print("hist uniform", np.round(out['hist_uniform_1e6_distinct'],1)); print("hist gap", np.round(out['hist_gap_1e6_distinct'],1))
