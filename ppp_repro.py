"""
ppp_repro.py
============
Reproducibility code for "Stacked Proximate-Prime Quadratics and the Covering
of Square Intervals" (continuation of Zenodo 10.5281/zenodo.20694503).

Each function below regenerates one table or claim in the paper. The shared
Bateman-Horn constant C_f (called Sig/Cf here; the predecessor's C_f) and the
pool builders are factored out so the six original scripts are not duplicated.

Dependencies: python3, gmpy2, sympy, numpy.
Run a single experiment:  python3 ppp_repro.py <name>
  names: law | multiscale | tworegime | roughness | strict | cf_resolve | factor_pin
Run everything in sequence (with timing, fast-first):
  python3 ppp_repro.py all | tee ppp_repro_log.txt
Runtimes vary: 'law' and 'tworegime' ~minutes; 'multiscale' and 'roughness'
are heavier (large n-ranges / Omega factoring). Seeds are fixed per function.

NOTE ON STATUS: these reproduce *empirical* claims. The independence of
witnesses (sec. 4) and the K(M) law (sec. 3) are heuristic, not proven.
"""

import math, statistics, json, sys
from gmpy2 import is_prime, isqrt
from sympy import primerange, factorint
from sympy.functions.combinatorial.numbers import jacobi_symbol

# ----------------------------------------------------------------------------
# Shared: Bateman-Horn constant C_f for a monic quadratic f(n)=n^2+b n+c.
# This is the predecessor's C_f (= singular series). The p=2 factor is computed
# explicitly from the root count mod 2; odd primes via the Jacobi symbol of D.
# ----------------------------------------------------------------------------
_PR = list(primerange(3, 2000))

def Cf(b, c):
    """Bateman-Horn constant of n^2 + b n + c (truncated at p<2000)."""
    D = b * b - 4 * c
    w2 = sum(1 for n in range(2) if (n * n + b * n + c) % 2 == 0)  # roots mod 2
    S = (1 - w2 / 2) / 0.5                                         # p=2 factor
    for p in _PR:
        Np = 1 if D % p == 0 else 1 + jacobi_symbol(D % p, p)     # roots mod p
        S *= (1 - Np / p) / (1 - 1 / p)
    return float(S)

# ----------------------------------------------------------------------------
# Shared: pool builders.
#   always-odd : n^2 + n + c, c odd  -> every value odd, dodges 2 (C_f doubled)
#   half-even  : n^2 + c,     c odd  -> value even at odd n, eligible only even n
# ----------------------------------------------------------------------------
def build_always_odd(N, lo, hi, c0=1_000_001, cmax=6_000_000):
    pool, seen, c = [], set(), c0
    while len(pool) < N and c < cmax:
        cc = c if c % 2 else c + 1
        D = 1 - 4 * cc
        if D not in seen and lo <= Cf(1, cc) <= hi:
            seen.add(D); pool.append((1, cc))
        c += 2
    return pool

def build_half_even(N, lo, hi, c0=1_000_001, cmax=6_000_000):
    pool, seen, c = [], set(), c0
    while len(pool) < N and c < cmax:
        if c % 2 == 1 and lo <= Cf(0, c) <= hi:
            seen.add(c); pool.append((0, c))
        c += 2
    return pool

def build_half_even_spread(N, lo, hi, c0=1_000_001, cmax=8_000_001):
    """Half-even curves with constants spread across [c0, cmax] (one per bin),
    so the 2-adic eligibility partition splits."""
    pool, binw = [], (cmax - c0) // N
    for k in range(N):
        c = c0 + k * binw
        c = c if c % 2 else c + 1
        while c < c0 + (k + 1) * binw:
            if lo <= Cf(0, c) <= hi:
                pool.append((0, c)); break
            c += 2
    return pool

def witness_sets(pool, N0, N1):
    """For each curve, the set of band indices it witnesses (prime value) in
    [N0,N1); also landing-index parity (eligibility) per band."""
    P, spans, elig = [], [], []
    for b, c in pool:
        w, el, lo, hi = set(), {}, 1 << 62, 0
        for n in range(N0, N1):
            v = n * n + b * n + c
            bd = int(isqrt(v))
            lo = min(lo, bd); hi = max(hi, bd)
            el[bd] = (n % 2 == 0)          # half-even eligible iff n even
            if is_prime(v):
                w.add(bd)
        P.append(w); spans.append((lo, hi)); elig.append(el)
    return P, spans, elig

def common_window(spans):
    return max(s[0] for s in spans) + 5, min(s[1] for s in spans) - 5

# ----------------------------------------------------------------------------
# Sec. 3 -- covering growth law K(M) ~ 2(log M)^2 / Cbar.  (kconst.py)
# Confirms rho_bar = Cbar/(2 log M) and K-to-cover = log(#bands)/rho_bar.
# ----------------------------------------------------------------------------
def law(seed=3):
    import random; random.seed(seed)
    pool = build_always_odd(75, 3.0, 1e9)            # high-C_f always-odd
    K = len(pool); Cbar = statistics.mean(Cf(*x) for x in pool)
    P, spans, _ = witness_sets(pool, 200000, 203000)
    B0, B1 = common_window(spans); win = set(range(B0, B1 + 1)); nC = len(win)
    P = [p & win for p in P]
    rho = [len(p) / nC for p in P]; rhobar = statistics.mean(rho)
    logM = math.log(B1)
    print(f"always-odd {K} curves, Cbar={Cbar:.2f}, M~{B1:.0f} (logM={logM:.2f}), {nC} bands")
    print(f"measured rho_bar={rhobar:.4f}   predicted Cbar/(2 logM)={Cbar/(2*logM):.4f}")
    def Kcover():
        order = random.sample(range(K), K); uncov = set(win)
        for u, i in enumerate(order, 1):
            uncov -= P[i]
            if not uncov: return u
        return None
    Ks = [k for k in (Kcover() for _ in range(60)) if k]
    print(f"K to cover {nC} bands: measured {statistics.mean(Ks):.1f}+-{statistics.pstdev(Ks):.1f}"
          f"   predicted log(nbands)/rho_bar={math.log(nC)/rhobar:.1f}")
    print(f"full-range form 2(logM)^2/Cbar = {2*logM**2/Cbar:.0f} curves to cover all bands up to M")

# ----------------------------------------------------------------------------
# Sec. 4 -- always-odd independence, scale-stability table.  (odd_multiscale.py)
# Uses the fixed 40-curve oddpool.json shipped alongside.
# ----------------------------------------------------------------------------
def multiscale(seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    pool = [tuple(x) for x in json.load(open("oddpool.json"))]; K = len(pool)
    def wmat(N0, N1):
        P, spans = [], []
        for b, c in pool:
            w, lo, hi = set(), 1 << 62, 0
            for n in range(N0, N1):
                v = n * n + b * n + c; bd = int(isqrt(v))
                lo = min(lo, bd); hi = max(hi, bd)
                if is_prime(v): w.add(bd)
            P.append(w); spans.append((lo, hi))
        B0 = max(s[0] for s in spans) + 5; B1 = min(s[1] for s in spans) - 5
        nC = B1 - B0 + 1; M = np.zeros((K, nC), dtype=np.int8)
        for i, p in enumerate(P):
            for b in p:
                if B0 <= b <= B1: M[i, b - B0] = 1
        logm = 2 * sum(math.log(b) for b in range(B0, B1 + 1)) / nC
        return M, nC, logm
    def cramer_tail(M, nC, t=8, reps=60):
        real = M.sum(0); obs = int((real <= t).sum()); nh = []
        for _ in range(reps):
            c = np.zeros(nC, dtype=np.int32)
            for i in range(K): c[rng.choice(nC, int(M[i].sum()), replace=False)] += 1
            nh.append(int((c <= t).sum()))
        m = statistics.mean(nh); return obs, m, (obs / m if m > 0 else float('inf'))
    print(f"{'value':>9} {'rho':>5} {'over-disp':>9} {'excessVar':>9} {'hardband real/Cramer':>20}")
    for N0, N1 in [(2000, 8000), (40000, 46000), (200000, 206000), (1000000, 1006000)]:
        M, nC, lm = wmat(N0, N1)
        real = M.sum(0); rho = M.sum(1) / nC
        od = real.var() / (real.mean() * (1 - real.mean() / K))
        ev = real.var() - sum(r * (1 - r) for r in rho)
        obs, m, r = cramer_tail(M, nC)
        print(f"{math.exp(lm):9.1e} {rho.mean():.3f} {od:9.3f} {ev:9.2f}   {obs} vs {m:.0f} ({r:.2f}x)")

# ----------------------------------------------------------------------------
# Sec. 5 -- two-regime half-even: clustered (50% ceiling) vs spread (factor ~2).
# (halfeven_check.py + spread_check.py)
# ----------------------------------------------------------------------------
def tworegime(seed=7):
    import random; random.seed(seed)
    AO = build_always_odd(60, 2.6, 3.4)
    HEc = build_half_even(60, 1.3, 1.7)            # clustered (constants near 1e6)
    HEs = build_half_even_spread(60, 1.3, 1.7)     # spread across [1e6, 8e6]
    for tag, pool in (("AO", AO), ("HE-clustered", HEc), ("HE-spread", HEs)):
        cs = [c for _, c in pool]
        print(f"{tag:>13}: {len(pool)} curves, Cbar={statistics.mean(Cf(*x) for x in pool):.2f}, "
              f"c-span {(max(cs)-min(cs))/4e5:.0f} band-widths")
    def analyse(pool, ref_spans=None):
        P, spans, EH = witness_sets(pool, 200000, 202000)
        return P, spans, EH
    PA, SA, _ = analyse(AO); PC, SC, EC = analyse(HEc); PS, SS, ES = analyse(HEs)
    B0 = max(s[0] for grp in (SA, SC, SS) for s in grp) + 5
    B1 = min(s[1] for grp in (SA, SC, SS) for s in grp) - 5
    bands = list(range(B0, B1 + 1)); win = set(bands); nC = len(bands)
    PA = [p & win for p in PA]; PC = [p & win for p in PC]; PS = [p & win for p in PS]
    def overdisp(EH, K):
        ec = [sum(1 for el in EH if el.get(b, False)) for b in bands]
        return statistics.mean(ec), min(ec), statistics.pvariance(ec) / (statistics.mean(ec) * (1 - statistics.mean(ec) / K))
    mc, minc, odc = overdisp(EC, len(HEc)); ms, mins, ods = overdisp(ES, len(HEs))
    print(f"\neligibility over-dispersion:  clustered {odc:.2f} (min {minc})   spread {ods:.2f} (min {mins})")
    def uncov(P, Ksub, reps=25):
        return statistics.mean(
            (nC - len(set().union(*[P[i] for i in random.sample(range(len(P)), Ksub)]))) / nC
            for _ in range(reps))
    print(f"\n{'K':>3} {'AO unc':>8} {'HE-clust':>9} {'HE-spread':>10}")
    for K in (15, 25, 35, 45, 55):
        print(f"{K:>3} {100*uncov(PA,K):7.2f}% {100*uncov(PC,K):8.2f}% {100*uncov(PS,K):9.2f}%")

# ----------------------------------------------------------------------------
# Sec. 6 -- roughness: Omega-distribution (P1/P2/P3+) vs C_f.  (semiq1.py)
# ----------------------------------------------------------------------------
def roughness():
    def Omega(m): return sum(e for _, e in factorint(m).items())
    cands = {}
    for c in range(41, 200000, 2):
        s = Cf(1, c)
        if 'hi' not in cands and s > 4.5: cands['hi'] = (c, s)
        if 'med' not in cands and 2.3 < s < 2.6: cands['med'] = (c, s)
        if 'lo' not in cands and 1.2 < s < 1.45: cands['lo'] = (c, s)
        if len(cands) == 3: break
    NMAX = 14000
    print(f"{'curve':>16} {'C_f':>5} {'P1%':>6} {'P2%':>6} {'P3+%':>6} {'P2/P1':>6} {'Liouville/N':>11}")
    for tag in ('hi', 'med', 'lo'):
        c, s = cands[tag]; n1 = n2 = n3 = lio = 0
        for n in range(1, NMAX + 1):
            v = n * n + n + c
            w = 1 if is_prime(v) else Omega(v)
            if w == 1: n1 += 1
            elif w == 2: n2 += 1
            else: n3 += 1
            lio += -1 if w % 2 else 1
        print(f"{'n^2+n+'+str(c):>16} {s:5.2f} {100*n1/NMAX:6.2f} {100*n2/NMAX:6.2f} "
              f"{100*n3/NMAX:6.2f} {n2/n1:6.2f} {lio/NMAX:+11.4f}")

# ----------------------------------------------------------------------------
# Sec. 6 -- strict reading: 4 consecutive semiprimes on a quadratic.  (strict.py)
# Rate vs shuffled-gap null; monic fraction; C_f of resulting curves.
# ----------------------------------------------------------------------------
def strict(seed=1, N=3_000_000):
    import random; random.seed(seed)
    spf = list(range(N + 1)); i = 2
    while i * i <= N:
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j: spf[j] = i
        i += 1
    Om = [0] * (N + 1)
    for n in range(2, N + 1): Om[n] = Om[n // spf[n]] + 1
    semi = [n for n in range(4, N + 1) if Om[n] == 2]
    gaps = [semi[i + 1] - semi[i] for i in range(len(semi) - 1)]
    ap, monic = [], []
    for i in range(len(gaps) - 2):
        g1, g2, g3 = gaps[i], gaps[i + 1], gaps[i + 2]
        if g1 + g3 == 2 * g2:
            ap.append(g2 - g1)
            if g2 - g1 == 2: monic.append((semi[i], g1))
    def count_ap(gg): return sum(1 for i in range(len(gg) - 2) if gg[i] + gg[i + 2] == 2 * gg[i + 1])
    nulls = []
    for _ in range(6):
        sg = gaps[:]; random.shuffle(sg); nulls.append(count_ap(sg))
    print(f"semiprimes<{N:,}: {len(semi):,}   AP-quadruples: {len(ap):,} "
          f"({100*len(ap)/len(gaps):.2f}% of positions), monic {100*len(monic)/len(ap):.1f}%")
    print(f"observed {len(ap):,} vs shuffled-null {statistics.mean(nulls):.0f}+-{statistics.pstdev(nulls):.0f} "
          f"(ratio {len(ap)/statistics.mean(nulls):.3f})")
    Ss = [Cf(g - 1, s1) for s1, g in monic[:600]]
    print(f"resulting monic quadratics: mean C_f = {statistics.mean(Ss):.2f} (random ~1.0)")

def cf_resolve(seed=11):
    """Remark on the low-C_f residual: mean pairwise witness excess and
    excess-holes for always-odd pools across C_f bands. Excess decays to 0 as
    C_f rises (real, not noise)."""
    import random; random.seed(seed)
    print(f"{'Cf band':>10} {'n':>3} {'Cbar':>5} {'pairwise excess':>20} {'excess-holes':>13}")
    for lo, hi, Nt in [(2.6, 3.4, 55), (3.4, 4.3, 55), (4.3, 7.0, 45)]:
        pool = build_always_odd(Nt, lo, hi, cmax=8_000_000); K = len(pool)
        P, spans, _ = witness_sets(pool, 200000, 204000)
        B0, B1 = common_window(spans); win = set(range(B0, B1 + 1)); nC = len(win)
        P = [p & win for p in P]; rho = [len(p) / nC for p in P]
        Cbar = statistics.mean(Cf(*x) for x in pool)
        ex = [len(P[i] & P[j]) * nC / (len(P[i]) * len(P[j])) - 1
              for i in range(K) for j in range(i + 1, K) if len(P[i]) and len(P[j])]
        pe = statistics.mean(ex); se = statistics.pstdev(ex) / math.sqrt(len(ex))
        rbar = statistics.mean(rho); Ks = min(K, max(5, int(math.log(0.05) / math.log(1 - rbar))))
        o = [(nC - len(set().union(*[P[i] for i in random.sample(range(K), Ks)]))) / nC for _ in range(80)]
        pr = statistics.mean(
            math.prod(1 - rho[i] for i in random.sample(range(K), Ks)) for _ in range(80))
        print(f"{lo:.1f}-{hi:.1f} {K:>3} {Cbar:5.2f}   {pe:+.4f}+-{se:.4f}   "
              f"{statistics.mean(o)/pr:.2f}")

def factor_pin(seed=13):
    """Remark on the factor: clean high-C_f always-odd baseline matched at equal
    odd-part to a spread half-even pool gives covering ratio ~2.0 (vs 1.73 at a
    contaminated low-C_f baseline)."""
    import random; random.seed(seed)
    AO = build_always_odd(60, 3.8, 5.0, cmax=9_000_000)
    HE = build_half_even_spread(60, 1.9, 2.5)
    PA, SA, _ = witness_sets(AO, 200000, 202000)
    PH, SH, _ = witness_sets(HE, 200000, 202000)
    B0 = max(s[0] for s in SA + SH) + 5; B1 = min(s[1] for s in SA + SH) - 5
    win = set(range(B0, B1 + 1)); nC = len(win)
    PA = [p & win for p in PA]; PH = [p & win for p in PH]
    rA = statistics.mean(len(p) / nC for p in PA); rH = statistics.mean(len(p) / nC for p in PH)
    print(f"AO Cbar={statistics.mean(Cf(*x) for x in AO):.2f}  HE Cbar={statistics.mean(Cf(*x) for x in HE):.2f}"
          f"  rho ratio={rA/rH:.2f}")
    def unc(P, K):
        return statistics.mean(
            (nC - len(set().union(*[P[i] for i in random.sample(range(len(P)), K)]))) / nC
            for _ in range(60))
    data = {K: (unc(PA, K), unc(PH, K)) for K in range(10, 61, 5)}
    def Kat(t, idx):
        pts = sorted(data.items())
        for i in range(len(pts) - 1):
            (k1, v1), (k2, v2) = pts[i], pts[i + 1]
            if v1[idx] >= t >= v2[idx]:
                return k1 + (k2 - k1) * (v1[idx] - t) / (v1[idx] - v2[idx])
    for t in (0.10, 0.05):
        ka, kh = Kat(t, 0), Kat(t, 1)
        if ka and kh: print(f"K at {int(t*100)}% uncovered: AO {ka:.0f}  HE-spread {kh:.0f}  ratio {kh/ka:.2f}")

def run_all():
    """Run every experiment in sequence (fast -> slow), with section headers and
    per-experiment timing. Capture stdout to a file for the paper appendix, e.g.
        python3 ppp_repro.py all | tee ppp_repro_log.txt
    """
    import time
    # (target name, paper section, function) -- ordered fastest-first
    order = [
        ("law",        "sec.3  K(M) covering law",          law),
        ("tworegime",  "sec.5  two-regime half-even",       tworegime),
        ("strict",     "sec.6  4 semiprimes on a quadratic", strict),
        ("cf_resolve", "sec.4  low-C_f residual",            cf_resolve),
        ("factor_pin", "sec.5  factor-2 at clean baseline",  factor_pin),
        ("roughness",  "sec.6  Omega-distribution vs C_f",  roughness),
        ("multiscale", "sec.4  always-odd independence",    multiscale),
    ]
    t_all = time.time()
    for name, where, fn in order:
        print(f"\n{'='*72}\n# {name}   [{where}]\n{'='*72}")
        t0 = time.time()
        fn()
        print(f"[{name} done in {time.time()-t0:.0f}s]")
    print(f"\n{'='*72}\nall experiments done in {time.time()-t_all:.0f}s")

if __name__ == "__main__":
    TARGETS = {"law": law, "multiscale": multiscale, "tworegime": tworegime,
               "roughness": roughness, "strict": strict,
               "cf_resolve": cf_resolve, "factor_pin": factor_pin, "all": run_all}
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    TARGETS.get(arg, lambda: print(__doc__))()
