#!/usr/bin/env python3
"""
ppp_gen.py -- generator and diagnostics for proximate-prime polynomials (PPPs).

A PPP is the quadratic f(n) = a n^2 + b n + c through four consecutive primes
p0 < p1 < p2 < p3 at n = 1, 2, 3, 4.  It exists iff the gaps g1, g2, g3 form an
arithmetic progression (g1 + g3 = 2 g2), and then

    2a = g2 - g1,   b = g1 - 3a,   c = p0 - a - b,   D = b^2 - 4ac = (g1 - a)^2 - 4 a p0.

Pipeline
  1. segmented sieve of Eratosthenes (numpy, odd numbers only, bit-packed, ~M/16 bytes)
     up to M = N + margin, the margin chosen so every value f(n), n <= terms, is covered;
  2. on-the-fly detection of AP-gap quadruples with p0 < N (vectorised per segment);
  3. per-record diagnostics: Legendre signature and nonresidue score S over the primes
     in --sp, root counts N_2, N_3, N_5, shift-alias flag, Bateman-Horn constant C_f in
     the single-polynomial normalisation prod_p (1 - N_p/p)/(1 - 1/p) over p <= --P,
     the BH predicted prime count sum_n C_f / log f(n), and the observed prime count;
  4. null models for S:
       uniform     (b, c) uniform mod p conditioned on f(1..4) != 0 mod p;
       gap-matched p0 uniform over the residue classes allowed by the record's own
                   (a, g1), i.e. four primes in the same gap pattern, not necessarily
                   consecutive (analytic, exact);
       sampled     the same null realised with random primes p0 (--sampled-null K).

Usage
  python3 ppp_gen.py 1000000                      # the 10^6 dataset, ppp1e+06.csv + summary
  python3 ppp_gen.py 100000000 --out ppp1e8       # 10^8 (~2 min, ~30 MB sieve)
  python3 ppp_gen.py 1000000 --index-start 0      # reproduce the n = 0..999 counts of
                                                  # naturalnumbers.org/PPP1000000.txt
Options
  --terms T           values f(n) counted (default 1000)
  --index-start s     first index counted, s or s+1..; 1 = paper's definition (default 1)
  --P P               Euler-product cutoff for C_f (default 997)
  --sp list           primes for the Legendre signature (default 5,...,31)
  --keep-negative     keep quadruples with decreasing gaps (a < 0); default a > 0 only
  --sampled-null K    draw up to K random non-consecutive quadruples per (a, g1) pattern
  --no-counts         skip prime counting and C_f (fast S-only run)
  --catalog "2,0;1,1" scan every odd c < N for the given forms a m^2 + b m + c, rank by C_f,
                      and report how many of the top curves the PPP construction caught
  --seg n             odd numbers per sieve segment (default 2^23)
Requires numpy only.  Values f(n) <= 1 never count as prime.
"""
import argparse
import math
import sys
import time
from collections import Counter, defaultdict

import numpy as np


# ----------------------------------------------------------------------------- sieve
def small_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, math.isqrt(n) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s)


class BitSieve:
    """Bit-packed primality table for odd numbers <= M, built segment by segment.

    on_segment(primes) is called with the primes of each segment in increasing
    order (2 included in the first one) and may return False to stop the calls.
    """

    def __init__(self, M, seg_odds=1 << 23, on_segment=None, log=None):
        self.M = int(M)
        nbits = (self.M >> 1) + 1                       # index i <-> odd number 2i+1
        nbits = ((nbits + 7) // 8) * 8
        seg_odds = ((seg_odds + 7) // 8) * 8
        self.packed = np.zeros(nbits // 8, dtype=np.uint8)
        sp = small_primes(math.isqrt(self.M) + 1)
        sp = [int(p) for p in sp if p > 2]
        cutoff = (self.M + 1) // 2                       # first index of a number > M
        want_primes = on_segment is not None
        t0 = time.time()
        for i_lo in range(0, nbits, seg_odds):
            i_hi = min(i_lo + seg_odds, nbits)
            seg = np.ones(i_hi - i_lo, dtype=bool)
            lo_num, hi_num = 2 * i_lo + 1, 2 * i_hi - 1
            for p in sp:
                if p * p > hi_num:
                    break
                s = max(p * p, ((lo_num + p - 1) // p) * p)
                if s % 2 == 0:
                    s += p
                seg[(s - 1) // 2 - i_lo::p] = False
            if i_lo == 0:
                seg[0] = False                           # 1 is not prime
            if cutoff - i_lo < len(seg):
                seg[max(cutoff - i_lo, 0):] = False      # padding beyond M
            self.packed[i_lo // 8:i_hi // 8] = np.packbits(seg)
            if want_primes:
                primes = 2 * (i_lo + np.flatnonzero(seg)) + 1
                if i_lo == 0:
                    primes = np.concatenate(([2], primes))
                if on_segment(primes) is False:
                    want_primes = False
            if log and (i_lo // seg_odds) % 10 == 9:
                log(f"  sieve {hi_num / self.M:5.1%}  {time.time() - t0:.0f}s")

    def isprime(self, v):
        """Vectorised primality for an int64 array of any sign (values > M raise)."""
        v = np.asarray(v, dtype=np.int64)
        out = np.zeros(v.shape, dtype=bool)
        if v.size == 0:
            return out
        if v.max() > self.M:
            raise ValueError(f"value {v.max()} exceeds sieve bound {self.M}")
        odd = ((v & 1) == 1) & (v > 1)
        i = v[odd] >> 1
        out[odd] = ((self.packed[i >> 3] >> (7 - (i & 7))) & 1).astype(bool)
        out |= (v == 2)
        return out


# ----------------------------------------------------------------------------- quadruples
class QuadScanner:
    """Collects (p0, g1, a) for consecutive-prime quadruples with AP gaps and p0 < N."""

    def __init__(self, N, keep_negative=False):
        self.N = N
        self.keep_negative = keep_negative
        self.carry = np.zeros(0, dtype=np.int64)
        self.chunks = []

    def __call__(self, primes):
        P = np.concatenate((self.carry, primes.astype(np.int64)))
        if len(P) >= 4:
            g = np.diff(P)
            g1, g2, g3 = g[:-2], g[1:-1], g[2:]
            p0 = P[:-3]
            cond = (g1 + g3 == 2 * g2) & (g2 != g1) & (((g2 - g1) & 1) == 0) & (p0 < self.N)
            if not self.keep_negative:
                cond &= (g2 > g1)
            idx = np.flatnonzero(cond)
            if idx.size:
                self.chunks.append(np.stack((p0[idx], g1[idx], (g2[idx] - g1[idx]) // 2), axis=1))
        self.carry = P[-3:]
        # every quadruple with p0 < N has p3 < N + 3 * maxgap; 10^6 is a safe stop margin
        return not (primes.size and int(primes[-1]) > self.N + 10 ** 6)

    def records(self):
        if not self.chunks:
            return np.zeros((0, 3), dtype=np.int64)
        return np.concatenate(self.chunks)


# ----------------------------------------------------------------------------- local arithmetic
_LEG = {}


def legendre_vec(D, p):
    """(D/p) for an int64 array D and odd prime p: -1, 0, +1 as int8 (table lookup)."""
    if p not in _LEG:
        t = np.full(p, -1, dtype=np.int8)
        t[(np.arange(1, p, dtype=np.int64) ** 2) % p] = 1
        t[0] = 0
        _LEG[p] = t
    return _LEG[p][D % p]


def root_counts(a, b, c, D, p):
    """N_p = #{n mod p : f(n) = 0}, odd prime p, arrays a, b, c, D."""
    N = np.empty(len(a), dtype=np.int16)
    nd = (a % p) != 0
    N[nd] = 1 + legendre_vec(D[nd], p)
    dg = ~nd
    if dg.any():                                          # f is linear mod p
        bb, cc = b[dg] % p, c[dg] % p
        N[dg] = np.where(bb != 0, 1, np.where(cc != 0, 0, p))
    return N


def root_counts_2(a, b, c):
    return np.where(((a + b) & 1) == 1, 1, np.where((c & 1) == 1, 0, 2)).astype(np.int16)


def roots_mod_p_scalar(a, b, c, p):
    return sum(1 for n in range(p) if (a * n * n + b * n + c) % p == 0)


def bateman_horn(a, b, c, D, P):
    """C_f = prod_{p <= P} (1 - N_p/p) / (1 - 1/p), single-polynomial normalisation."""
    with np.errstate(divide="ignore"):
        logC = np.log1p(-root_counts_2(a, b, c) / 2.0) - math.log1p(-0.5)
        for p in small_primes(P):
            p = int(p)
            if p == 2:
                continue
            logC += np.log1p(-root_counts(a, b, c, D, p) / p) - math.log1p(-1.0 / p)
    return np.exp(logC)


# ----------------------------------------------------------------------------- nulls
def uniform_null(sp):
    """P((D/p) = -1) and P(N_p = 0) for (b, c) uniform mod p given f(1..4) != 0 mod p."""
    out = {}
    for p in sp:
        tot = leg = root = 0
        for b in range(p):
            for c in range(p):
                if all((n * n + b * n + c) % p for n in (1, 2, 3, 4)):
                    tot += 1
                    D = b * b - 4 * c
                    leg += (D % p != 0 and pow(D % p, (p - 1) // 2, p) != 1)
                    root += (roots_mod_p_scalar(1, b, c, p) == 0)
        out[p] = (leg / tot, root / tot)
    return out


_GAP = {}


def gap_null(a, g1, p):
    """For pattern (a, g1) and prime p: P((D/p) = -1), P(N_p = 0) with p0 uniform over the
    residue classes that keep f(1), f(2), f(3), f(4) nonzero mod p."""
    key = (a % p, g1 % p, p)
    if key not in _GAP:
        aa, gg = key[0], key[1]
        bad = {0, (-gg) % p, (-2 * gg - 2 * aa) % p, (-3 * gg - 6 * aa) % p}
        allowed = [r for r in range(p) if r not in bad]
        b = (gg - 3 * aa) % p
        leg = root = 0
        for r in allowed:
            c = (r - aa - b) % p
            D = (gg - aa) ** 2 - 4 * aa * r
            leg += (D % p != 0 and pow(D % p, (p - 1) // 2, p) != 1)
            root += (roots_mod_p_scalar(aa, b, c, p) == 0)
        _GAP[key] = (leg / len(allowed), root / len(allowed))
    return _GAP[key]


def fmt_poly(a, b, c):
    a, b, c = int(a), int(b), int(c)
    sa = "n^2" if a == 1 else ("-n^2" if a == -1 else f"{a}n^2")
    sb = "" if b == 0 else ("+n" if b == 1 else ("-n" if b == -1 else f"{b:+d}n"))
    return f"{sa}{sb}{c:+d}"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("N", type=float, help="upper bound for p0 (e.g. 1e6, 100000000)")
    ap.add_argument("--terms", type=int, default=1000)
    ap.add_argument("--index-start", type=int, default=1)
    ap.add_argument("--P", type=int, default=997)
    ap.add_argument("--sp", default="5,7,11,13,17,19,23,29,31")
    ap.add_argument("--keep-negative", action="store_true")
    ap.add_argument("--sampled-null", type=int, default=0, metavar="K")
    ap.add_argument("--no-counts", action="store_true")
    ap.add_argument("--seg", type=int, default=1 << 23)
    ap.add_argument("--out", default=None, help="output prefix (default ppp<N>)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--catalog", default=None, metavar="FORMS",
                    help='also scan every odd c < N for the forms "a,b;a,b;..." (e.g. "2,0;1,1;2,2"), rank by '
                         'C_f, and report which of the top curves the PPP construction caught')
    ap.add_argument("--catalog-top", type=int, default=100)
    args = ap.parse_args()

    N = int(args.N)
    sp = [int(x) for x in args.sp.split(",")]
    out = args.out or f"ppp{args.N:.0e}".replace("+", "")
    logf = open(out + "_summary.txt", "w")

    def log(s=""):
        print(s)
        logf.write(s + "\n")
        logf.flush()

    t0 = time.time()
    # margin: every f(n), n <= terms, must lie below the sieve bound.  Gaps below N are
    # far smaller than 2 (log N)^2 for every computed range, so a <= (log N)^2 and
    # f(terms) <= N + (log N)^2 terms^2 + 2 (log N)^2 terms.
    gmax = max(200, int(2 * math.log(N) ** 2) + 2)
    T = args.terms + max(args.index_start, 1)
    M = N + (gmax // 2 + 1) * T * T + gmax * T + 10 ** 6
    log(f"ppp_gen  N={N:,}  sieve bound M={M:,}  ({M / 16 / 2 ** 20:.0f} MB packed)")

    scan = QuadScanner(N, args.keep_negative)
    sieve = BitSieve(M, seg_odds=args.seg, on_segment=scan, log=log)
    log(f"sieve done in {time.time() - t0:.0f}s")

    rec = scan.records()
    if len(rec) == 0:
        log("no quadruples found")
        return
    p0, g1, a = rec[:, 0], rec[:, 1], rec[:, 2]
    b = g1 - 3 * a
    c = p0 - a - b
    D = (g1 - a) ** 2 - 4 * a * p0
    assert np.all(D == b * b - 4 * a * c)
    p1, p2, p3 = p0 + g1, p0 + 2 * g1 + 2 * a, p0 + 3 * g1 + 6 * a
    assert np.all(sieve.isprime(p1)) and np.all(sieve.isprime(p2)) and np.all(sieve.isprime(p3))
    n_rec = len(rec)

    # shift aliases: f(n+t) has the same a and D and b shifted by 2at
    key = np.stack((a, b % (2 * np.abs(a)), D), axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    distinct = np.zeros(n_rec, dtype=bool)
    distinct[first] = True

    leg = {p: legendre_vec(D, p) for p in sp}
    Np = {p: root_counts(a, b, c, D, p) for p in sp}
    S_leg = sum((leg[p] == -1).astype(np.int16) for p in sp)
    S_root = sum((Np[p] == 0).astype(np.int16) for p in sp)
    N2 = root_counts_2(a, b, c)
    N3 = root_counts(a, b, c, D, 3)
    N5 = root_counts(a, b, c, D, 5)
    K = len(sp)

    log(f"records {n_rec:,}  distinct curves {int(distinct.sum()):,}  "
        f"shift-aliases {n_rec - int(distinct.sum()):,}")
    log("a-values (records): " + " ".join(f"{k}:{v}" for k, v in sorted(Counter(a.tolist()).items())))
    log(f"N_2 max {N2.max()}  N_3 max {N3.max()}  N_5 max {N5.max()}  "
        f"D<0 all: {bool(np.all(D < 0))}  (D/5)=+1 records: {int((leg[5] == 1).sum())} "
        f"(all with 5|a: {bool(np.all((a[leg[5] == 1] % 5) == 0))})")

    def s_table(S, mask, label):
        h = Counter(S[mask].tolist())
        log(f"  S_leg histogram {label}: " + " ".join(f"{s}:{h.get(s, 0)}" for s in range(K + 1))
            + f"   mean {S[mask].mean():.3f} (SE {S[mask].std() / math.sqrt(mask.sum()):.3f})"
            + f"   S={K}: {h.get(K, 0)} ({100 * h.get(K, 0) / mask.sum():.2f}%)")

    log("Legendre nonresidue score S_leg over p in " + ",".join(map(str, sp)))
    s_table(S_leg, np.ones(n_rec, bool), "(records) ")
    s_table(S_leg, distinct, "(distinct)")
    log(f"  root-count score S_root (N_p = 0): mean records {S_root.mean():.3f}, distinct "
        f"{S_root[distinct].mean():.3f}, S_root={K}: {int((S_root[distinct] == K).sum())} distinct")

    # nulls
    un = uniform_null(sp)
    log("null models for S_leg (S_root in parentheses):")
    log(f"  uniform (b,c) given f(1..4)!=0:  E[S] {sum(v[0] for v in un.values()):.3f} "
        f"({sum(v[1] for v in un.values()):.3f}),  P(S={K}) "
        f"{100 * np.prod([v[0] for v in un.values()]):.2f}% ({100 * np.prod([v[1] for v in un.values()]):.2f}%)")
    pats = Counter(zip(a[distinct].tolist(), g1[distinct].tolist()))
    ES = EK = ESr = EKr = 0.0
    for (aa, gg), k in pats.items():
        q = [gap_null(aa, gg, p) for p in sp]
        ES += k * sum(x[0] for x in q)
        ESr += k * sum(x[1] for x in q)
        EK += k * float(np.prod([x[0] for x in q]))
        EKr += k * float(np.prod([x[1] for x in q]))
    nd = int(distinct.sum())
    log(f"  gap-matched analytic (distinct curves): E[S] {ES / nd:.3f} ({ESr / nd:.3f}),  "
        f"P(S={K}) {100 * EK / nd:.2f}% ({100 * EKr / nd:.2f}%)  -> expected S={K} count {EK:.1f}")
    log(f"  observed (distinct): mean S {S_leg[distinct].mean():.3f} +- "
        f"{S_leg[distinct].std() / math.sqrt(nd):.3f},  S={K} count {int((S_leg[distinct] == K).sum())}")

    if args.sampled_null:
        rng = np.random.default_rng(args.seed)
        lo = max(10 ** 5, N // 10)
        tot_w = tot_mean = tot_var = tot_k = 0.0
        n_pat = 0
        for (aa, gg), k in pats.items():
            want = min(k, args.sampled_null)
            got = []
            for _ in range(400):
                q = rng.integers(lo // 2, N // 2, size=1 << 17) * 2 + 1
                ok = (sieve.isprime(q) & sieve.isprime(q + gg) & sieve.isprime(q + 2 * gg + 2 * aa)
                      & sieve.isprime(q + 3 * gg + 6 * aa))
                got.extend(q[ok].tolist())
                if len(got) >= want:
                    break
            if not got:
                continue
            got = np.array(got[:want], dtype=np.int64)
            Dq = (gg - aa) ** 2 - 4 * aa * got
            Sq = sum((legendre_vec(Dq, p) == -1).astype(np.int16) for p in sp)
            tot_w += k
            tot_mean += k * Sq.mean()
            tot_var += (k ** 2) * Sq.var() / len(Sq)
            tot_k += k * (Sq == K).mean()
            n_pat += 1
        log(f"  sampled non-consecutive null ({n_pat} patterns, <= {args.sampled_null} draws each, "
            f"weighted by pattern frequency): mean S {tot_mean / tot_w:.3f} +- "
            f"{math.sqrt(tot_var) / tot_w:.3f},  P(S={K}) {100 * tot_k / tot_w:.2f}%")

    # prime counts and Bateman-Horn
    cnt = pred = Cf = None
    if not args.no_counts:
        t1 = time.time()
        Cf = bateman_horn(a, b, c, D, args.P)
        n = np.arange(args.index_start, args.index_start + args.terms, dtype=np.int64)
        cnt = np.zeros(n_rec, dtype=np.int64)
        pred = np.zeros(n_rec)
        # the interpolated values n = 1..4 are prime by construction; report the ratio without them
        skip4 = 5 - args.index_start if 1 <= args.index_start <= 4 and args.terms > 5 - args.index_start else None
        cnt5 = np.zeros(n_rec, dtype=np.int64)
        pred5 = np.zeros(n_rec)
        B = 2000
        for s in range(0, n_rec, B):
            e = min(s + B, n_rec)
            vals = a[s:e, None] * n * n + b[s:e, None] * n + c[s:e, None]
            pos = vals > 1
            isp = np.zeros(vals.shape, dtype=bool)
            isp[pos] = sieve.isprime(vals[pos])
            cnt[s:e] = isp.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                lv = np.log(np.where(pos, vals, 2.0))
            terms_n = np.where(pos, 1.0, 0.0) / lv
            pred[s:e] = terms_n.sum(axis=1) * Cf[s:e]
            if skip4 is not None:
                cnt5[s:e] = isp[:, skip4:].sum(axis=1)
                pred5[s:e] = terms_n[:, skip4:].sum(axis=1) * Cf[s:e]
        rho = cnt / args.terms
        prho = pred / args.terms
        ratio = rho / prho
        log(f"prime counts over n = {args.index_start}..{args.index_start + args.terms - 1}  "
            f"({time.time() - t1:.0f}s)")
        log(f"  prime %: min {100 * rho.min():.1f}  max {100 * rho.max():.1f}  "
            f"mean {100 * rho.mean():.2f} (records) {100 * rho[distinct].mean():.2f} (distinct)")
        log(f"  C_f (exponent 1, p <= {args.P}): min {Cf.min():.2f} mean {Cf.mean():.2f} max {Cf.max():.2f}")
        log(f"  Bateman-Horn check: obs/pred mean {ratio.mean():.4f} sd {ratio.std():.4f}  "
            f"corr(rho, pred) {np.corrcoef(rho, prho)[0, 1]:.4f}  "
            f"(the first four values are prime by construction: +{400 * (1 - rho.mean()) / args.terms:.2f} pp expected)")
        if skip4 is not None:
            r5 = cnt5 / pred5
            log(f"  same check excluding n = 1..4: obs/pred mean {r5.mean():.4f} sd {r5.std():.4f}  "
                f"(a deficit here is expected: for p > n the value f(n) sits in a root-eligible class mod p,"
                f" so its divisibility probability is E[N_p]/(p-4), not E[N_p]/p; the effect decays like 4/(n log n))")
        log(f"  corr(rho, S_leg) {np.corrcoef(rho, S_leg)[0, 1]:.3f}  corr(rho, C_f) "
            f"{np.corrcoef(rho, Cf)[0, 1]:.3f}  corr(rho, log C_f) {np.corrcoef(rho, np.log(Cf))[0, 1]:.3f}")
        log("  threshold table (records): S  count  mean%  frac>=50%  mean C_f")
        for s_ in range(K + 1):
            m = S_leg == s_
            if m.any():
                log(f"    {s_}  {int(m.sum()):8d}  {100 * rho[m].mean():6.2f}  {100 * (rho[m] >= 0.5).mean():6.2f}  {Cf[m].mean():6.2f}")
        # scale-free leaderboard: C_f against the ERH ceiling 2 e^gamma log log |D| (always-odd normalisation)
        oc = np.argsort(-Cf, kind="stable")[:10]
        log("  top 10 by C_f (scale-free), with C_f / (2 e^gamma log log |D|), the asymptotic ERH ceiling:")
        for i in oc:
            ceil = 2 * math.exp(0.5772156649) * math.log(math.log(abs(int(D[i]))))
            log(f"    {fmt_poly(a[i], b[i], c[i])}  p0={p0[i]}  C_f={Cf[i]:.3f}  ratio {Cf[i] / ceil:.2f}  "
                f"S={S_leg[i]}  {cnt[i]}/{args.terms}  {'alias' if not distinct[i] else ''}")
        order = np.argsort(-rho, kind="stable")
        mid = n_rec // 2
        for label, idx in (("top", order[:100]), ("middle", order[max(mid - 50, 0):mid + 50]), ("bottom", order[-100:])):
            allnr = np.mean(S_leg[idx] == K)
            log(f"  {label:6s}100: mean% {100 * rho[idx].mean():.2f}  mean S {S_leg[idx].mean():.2f}  "
                f"all-nonres {100 * allnr:.0f}%")
        i = int(order[0])
        log(f"  best by count: {fmt_poly(a[i], b[i], c[i])}  {cnt[i]}/{args.terms}  S={S_leg[i]}  C_f={Cf[i]:.2f}")

    if args.catalog:
        ppD = set(D.tolist())
        nn = np.arange(1, args.terms + 1, dtype=np.int64)
        for form in args.catalog.split(";"):
            fa, fb = (int(x) for x in form.split(","))
            if (fa + fb) % 2:
                log(f"catalog {fa}m^2{fb:+d}m+c skipped: a+b must be even for always-odd values")
                continue
            keep = max(20 * args.catalog_top, 2000)
            best_c = np.zeros(0, dtype=np.int64)
            best_l = np.zeros(0)
            chunk = 1 << 25
            screen = [int(p) for p in small_primes(97) if p > 2]
            for lo in range(1, N, 2 * chunk):                     # odd c in [lo, lo + 2*chunk)
                cc = np.arange(lo, min(lo + 2 * chunk, N), 2, dtype=np.int64)
                Dc = fb * fb - 4 * fa * cc
                logC = np.zeros(len(cc))
                for p in screen:
                    if fa % p == 0:
                        Npc = np.where(fb % p != 0, 1, np.where(cc % p != 0, 0, p))
                    else:
                        l = legendre_vec(Dc, p)
                        Npc = np.where(l == 0, 1, 1 + l)
                    logC += np.log1p(-Npc / p) - math.log1p(-1.0 / p)
                sel = np.argpartition(-logC, min(keep, len(logC) - 1))[:keep]
                best_c = np.concatenate((best_c, cc[sel]))
                best_l = np.concatenate((best_l, logC[sel]))
                if len(best_c) > keep:
                    sel = np.argpartition(-best_l, keep)[:keep]
                    best_c, best_l = best_c[sel], best_l[sel]
            ct = best_c
            Dt = fb * fb - 4 * fa * ct
            Cft = bateman_horn(np.full(len(ct), fa), np.full(len(ct), fb), ct, Dt, args.P)
            o = np.argsort(-Cft)[:args.catalog_top]
            ct, Cft, Dt = ct[o], Cft[o], Dt[o]
            vals = fa * nn[None, :] ** 2 + fb * nn[None, :] + ct[:, None]
            cnt_t = sieve.isprime(vals).sum(axis=1)
            inppp = np.array([int(d) in ppD for d in Dt])
            rec = (f"  C_f >= {Cf.max():.2f} (the PPP record C_f): {int((Cft >= Cf.max()).sum())}"
                   if Cf is not None else "")
            log(f"catalog {fmt_poly(fa, fb, 0)[:-2]}+c, odd c < N, top {len(ct)} by C_f: "
                f"{int(inppp.sum())} are PPPs (D matches a record).{rec}")
            for i in range(min(10, len(ct))):
                log(f"    c={ct[i]:<9d} C_f={Cft[i]:.3f}  primes(n=1..{args.terms})={cnt_t[i]}  "
                    f"{'PPP' if inppp[i] else '-'}")
            with open(f"{out}_catalog_{fa}_{fb}.csv", "w") as f:
                f.write("c,D,C_f,primes,is_ppp\n")
                for i in range(len(ct)):
                    f.write(f"{ct[i]},{Dt[i]},{Cft[i]:.4f},{cnt_t[i]},{int(inppp[i])}\n")

    # csv
    with open(out + ".csv", "w") as f:
        cols = ["p0", "p1", "p2", "p3", "a", "b", "c", "D", "g1", "S_leg", "S_root", "N2", "N3", "N5", "distinct"]
        if cnt is not None:
            cols += ["primes", "pct", "C_f", "bh_pred"]
        f.write(",".join(cols) + "\n")
        for i in range(n_rec):
            row = [p0[i], p1[i], p2[i], p3[i], a[i], b[i], c[i], D[i], g1[i], S_leg[i], S_root[i],
                   N2[i], N3[i], N5[i], int(distinct[i])]
            if cnt is not None:
                row += [cnt[i], f"{100 * cnt[i] / args.terms:.1f}", f"{Cf[i]:.4f}", f"{pred[i]:.1f}"]
            f.write(",".join(map(str, row)) + "\n")
    log(f"wrote {out}.csv and {out}_summary.txt  ({time.time() - t0:.0f}s total)")
    logf.close()


if __name__ == "__main__":
    main()
