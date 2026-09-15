"""Rebuild the cf_resolve pools of ppp_repro.py exactly, measure the pairwise witness excess in the
same window, and compare with the pair-singular-series prediction at offset 0 (no free parameter)."""
import math, statistics, itertools, time
from math import isqrt
from sympy import primerange, isprime
from sympy.functions.combinatorial.numbers import jacobi_symbol
_PR = list(primerange(3, 2000))
def Cf(b, c):
    D = b*b - 4*c
    w2 = sum(1 for n in range(2) if (n*n + b*n + c) % 2 == 0)
    S = (1 - w2/2)/0.5
    for p in _PR:
        Np = 1 if D % p == 0 else 1 + jacobi_symbol(D % p, p)
        S *= (1 - Np/p)/(1 - 1/p)
    return float(S)
def build_always_odd(N, lo, hi, c0=1_000_001, cmax=6_000_000):
    pool, seen, c = [], set(), c0
    while len(pool) < N and c < cmax:
        cc = c if c % 2 else c + 1
        D = 1 - 4*cc
        if D not in seen and lo <= Cf(1, cc) <= hi:
            seen.add(D); pool.append((1, cc))
        c += 2
    return pool
def witness_sets(pool, N0, N1):
    P, spans = [], []
    for b, c in pool:
        w, lo, hi = set(), 1 << 62, 0
        for n in range(N0, N1):
            v = n*n + b*n + c; bd = isqrt(v)
            lo = min(lo, bd); hi = max(hi, bd)
            if isprime(v): w.add(bd)
        P.append(w); spans.append((lo, hi))
    return P, spans
PRED_P = list(primerange(3, 1000))
def roots(c, p): return frozenset(m for m in range(p) if (m*m + m + c) % p == 0)
def pair_pred(ci, cj, R, pmax):
    r = 1.0
    for p in PRED_P:
        if p > pmax: break
        Ri, Rj = R[ci][p], R[cj][p]
        r *= (1 - len(Ri | Rj)/p)/((1 - len(Ri)/p)*(1 - len(Rj)/p))
    return r - 1
t0 = time.time()
print(f"{'Cf band':>8} {'n':>3} {'Cbar':>5} {'c-span':>7} {'measured excess':>17} {'v1 log':>8} {'pred p<=97':>11} {'pred p<=997':>12}")
for lo, hi, Nt, v1 in [(2.6, 3.4, 55, +0.0447), (3.4, 4.3, 55, +0.0166), (4.3, 7.0, 45, -0.0103)]:
    pool = build_always_odd(Nt, lo, hi, cmax=8_000_000); K = len(pool)
    cs = [c for _, c in pool]
    P, spans = witness_sets(pool, 200000, 204000)
    B0 = max(s[0] for s in spans) + 5; B1 = min(s[1] for s in spans) - 5
    win = set(range(B0, B1 + 1)); nC = len(win); P = [p & win for p in P]
    ex = [len(P[i] & P[j])*nC/(len(P[i])*len(P[j])) - 1 for i in range(K) for j in range(i+1, K)]
    R = {c: {p: roots(c, p) for p in PRED_P} for c in cs}
    pr97 = [pair_pred(cs[i], cs[j], R, 97) for i in range(K) for j in range(i+1, K)]
    pr997 = [pair_pred(cs[i], cs[j], R, 997) for i in range(K) for j in range(i+1, K)]
    se = statistics.pstdev(ex)/math.sqrt(len(ex))
    Cbar = statistics.mean(Cf(*x) for x in pool)
    corr = statistics.correlation(ex, pr997)
    print(f"{lo:.1f}-{hi:.1f} {K:>3} {Cbar:5.2f} {max(cs)-min(cs):>7} {statistics.mean(ex):+.4f}+-{se:.4f}   {v1:+.4f}   {statistics.mean(pr97):+.4f}     {statistics.mean(pr997):+.4f}   corr(pair) {corr:.2f}  classes mod 3 {[sum(1 for c in cs if c%3==r) for r in range(3)]}  [{time.time()-t0:.0f}s]")
