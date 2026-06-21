# Reproducibility code — Stacked Proximate-Prime Quadratics and the Covering of Square Intervals

Code behind the tables and claims in *Stacked Proximate-Prime Quadratics and the
Covering of Square Intervals*, a continuation of *Congruence Entropy and
Prime-Rich Proximate-Prime Quadratics*.

All results are **empirical/heuristic** — confirmed by computation, not proven.
In particular, the independence of witnesses (§4) and the covering law (§3) are
heuristic. The shared Bateman–Horn constant is written `C_f` (the predecessor's
`C_f`; the singular series), and `Cbar` is its arithmetic mean over a stack.

## Dependencies

Python 3 with:

```
pip install gmpy2 sympy numpy
```

## Quick start

Run the whole suite in one command (fast experiments first, with section
headers and per-experiment timing), capturing a log for the appendix:

```
python3 ppp_repro.py all | tee ppp_repro_log.txt
```

Total runtime is about **1 minute** on a normal machine. Run a single
experiment instead:

```
python3 ppp_repro.py law          # or: tworegime | strict | roughness | multiscale
python3 ppp_repro.py              # no argument -> prints the module docstring
```

`oddpool.json` must sit in the same directory (the `multiscale` target reads it).

## What each target reproduces

| target | paper | reproduces | ~time |
|---|---|---|---|
| `law` | §3 | $K(M)\approx 2(\log M)^2/\overline{C}$: measured $\overline\rho$ vs $\overline C/(2\log M)$, and curves-to-cover vs $\log(\\#\text{bands})/\overline\rho$ | 10 s |
| `tworegime` | §5 | clustered vs constant-spread half-even: eligibility over-dispersion (60 vs 0.03) and the 50% ceiling vs recovery | 27 s |
| `strict` | §6 | four consecutive semiprimes on a quadratic — rate (5.87%), monic fraction, shuffled-gap null (ratio ≈1.007), resulting $C_f\approx2.74$ | 15 s |
| `roughness` | §6 | $P_1/P_2/P_3^+$ split vs $C_f$ (Euler 39.6/48.2/12.3, etc.) | 3 s |
| `multiscale` | §4 | always-odd independence to $10^{12}$: over-dispersion ≈1, hard-band tail ≈1.00× Cramér | 2 s |
| `cf_resolve` | §4 | low-$C_f$ residual: mean pairwise witness excess across $C_f$ bands (+0.045 → −0.010, decays to 0 by $\overline C\approx5$) | 60 s |
| `factor_pin` | §5 | the half-even factor is 2: clean high-$C_f$ baseline gives covering ratio 1.92–1.98 (vs 1.73 at a low-$C_f$ baseline) | 40 s |

## Files

**`ppp_repro.py`** — the consolidated module. Read this one. The shared pieces
are factored to the top:

- `Cf(b, c)` — Bateman–Horn constant of $n^2+bn+c$ (Euler product truncated at
  $p<2000$; `p=2` factor from the explicit root count).
- `build_always_odd`, `build_half_even`, `build_half_even_spread` — pool builders
  ($n^2+n+c$ dodges 2; $n^2+c$ does not; the third spreads constants so the
  2-adic eligibility partition splits).
- `witness_sets`, `common_window` — per-curve band-witness sets and landing
  parity.

Each experiment is then a named function (`law`, `tworegime`, `strict`,
`roughness`, `multiscale`) called by the dispatcher at the bottom.

**Original scripts** (the literal code each table came from; the consolidated
module deduplicates these):

| script | corresponds to target |
|---|---|
| `kconst.py` | `law` |
| `cf_resolve.py` | `cf_resolve` |
| `factor_pin.py` | `factor_pin` |
| `halfeven_check.py`, `spread_check.py` | `tworegime` |
| `strict.py` | `strict` |
| `semiq1.py` | `roughness` |
| `odd_multiscale.py` | `multiscale` |

**`oddpool.json`** — the fixed 40-curve always-odd pool ($n^2+n+c$, constants
near $10^6$) that the `multiscale` table was computed on. `odd_multiscale.py`
and the `multiscale` target both read it.

## Caveats

- **`C_f` precision.** The Euler product is truncated at $p<2000$. This is fine
  for the relative comparisons the paper makes; do not quote an absolute `C_f`
  to more than ~2 figures from this code without extending the truncation.
- **Seeds.** Each function fixes its RNG seed, so runs are deterministic. The
  `law` covering count fluctuates by a few curves across seeds (reported with a
  standard deviation).
- **`oddpool.json` provenance.** The multiscale pool is a specific 40-curve set;
  the always-odd independence result does not depend on the particular pool, but
  the exact table numbers do.
- **Heavy reruns.** `law` and `tworegime` were smoke-tested against the module;
  `multiscale` and `roughness` reproduce the paper numbers exactly (verified).
  For a submission, archive one clean `python3 ppp_repro.py all` log alongside
  the paper.
