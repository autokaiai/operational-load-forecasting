# The evaluation protocol

How every number in this repository was produced, and how the rule that decided what shipped was
arrived at. This is the part of the project that is worth reading. The model is a blend; the
protocol is the thing that stopped the blend from being assembled out of noise.

Two halves:

1. **[The gapped CV design](#1-the-gapped-cv-design)** — what is measured, on which rows.
2. **[The admission bar](#2-the-admission-bar)** — how a difference between two measurements is
   turned into a decision, and the two earlier bars that were retired because they were measuring
   the wrong quantity.

---

## 1. The gapped CV design

### The split shape drives everything

The evaluation set opens a **336-hour covariate-absent gap** after the last observed target. The
model sees history up to hour *T*, is asked for hours *T+337 … T+672*, and is given the known-future
planning covariates for those hours — but nothing at all for the 336 hours in between.

A contiguous rolling-origin CV does not measure that. It measures forecasting from a warm start, and
it will rank models that lean on lag-1 persistence far above where they belong on the real task. So
the whole protocol is built around the gap rather than around convenience:

| | |
|---|---|
| Cutoffs | `3648`, `3312`, `2976` (hour index into the 4320 labelled hours) |
| Forecast length per window | **672** steps |
| Scored | the **last 336** only — the FAR block |
| Unscored | the first 336 — the gap the model must traverse blind |
| Scored block | 96 series × 336 hours × 3 windows = 96,768 rows |
| **Scored rows (headline)** | the block's last 112 hours: 96 × 112 × 3 = **32,256** — see below |
| Metric | pooled WAPE — `sum|y - ŷ| / sum|y|` over all scored rows at once, not a mean of per-series WAPEs |
| Loss | L1 (MAE), because WAPE's numerator is a sum of absolute errors |

Pooling rather than averaging matters: WAPE is a ratio of sums, and a mean of per-window ratios is a
different statistic with different weighting. Every pooled number here is the ratio, computed once
over all 32,256 rows.

`src/eval/protocol.py` is the implementation; `src/eval/splits.py` carves the blocks.

### The near/far axis, and why only one of them counts

Each 672-step window contains two disjoint 336-hour blocks:

* **FAR** — steps 337–672. The gap has been traversed. **This is the graded regime, and every
  weight in the shipped model was fitted and scored on it.**
* **NEAR** — steps 1–336. Gap zero, a warm start. This is what the *public leaderboard* scores.

They are not comparable, and the difference is not small. The two are kept apart mechanically rather
than by convention: `regime` defaults to `"far"` at every one of the six entry points that thread it
through, unknown values raise rather than falling back, and a near run with the gap covariates
withheld is refused outright — under `far` those blanked rows are the unscored gap, under `near`
they are exactly the block being graded, so the run would silently measure a scenario that does not
exist. `tests/test_block_regime.py` pins all of it.

The consequence for reading this repository: **the public leaderboard was treated as a pipeline
check, never as a selection signal.** It is externally scored, which internal CV is not, so it is
the only evidence that the artifact runs at all in someone else's hands — and that is exactly what
it was used for. See [W1 in `method.md`](method.md#w1--a-cv-to-holdout-offset-is-a-reproducibility-monitor-not-just-an-accuracy-gap), where
the *stability of the offset* between the two turned out to be the most useful diagnostic in the
project.

### Blend weights are fitted on rows they are not scored on

Within the scored FAR block, position `blk` runs 0–335. Weights are fitted on `blk < 224` and
scored on `blk >= 224`, pooled across the three windows. **That is the row-set behind every headline
number in this repository**, standalone members included, and it is why the scored count is 32,256
and not 96,768 — `src/eval/protocol.py` makes `regime` a required argument with no default for
exactly this reason, having found three different row-sets all being called "the score". No
reported blend number was fitted on the
rows it is quoted against.

### The seed policy is asymmetric, and deliberately so

* **Tuning, ablations, every A/B: single seed 42.** A paired A/B on a shared seed already controls
  seed variance — both arms carry the same draw — so multi-seeding each comparison costs 5× for no
  gain in resolution.
* **Finalists only: the frozen five**, `892, 7739, 6545, 4388, 4330`, drawn once and reproducibly
  via `np.random.default_rng(42).integers(0, 10000, size=5)` and never re-drawn.

The tuple is stored **in draw order**, not sorted, and that is load-bearing: the documented
compute-tight fallback is "the first 3 of the set", which is a prefix only in draw order
(drawn → 892, 7739, 6545; sorted → 892, 4330, 4388 — a different subset). `frozen_seeds(n)` is the
only sanctioned way to take fewer than five, and `verify_frozen_seeds()` re-derives the tuple from
its documented RNG call as a tripwire.

Never select the best-scoring seed. That overfits the evaluation set, and it is the reason the set
is frozen rather than chosen.

---

## 2. The admission bar

*Written 2026-07-30 as a standalone note and reproduced here nearly whole — before the members
it governs had been measured.*

### The question the bar has to answer

A candidate member is admitted to the ensemble only if it improves pooled WAPE by more than noise.
That intent is not in dispute. The whole difficulty is the second half of the sentence: **noise
measured how?** Two of our three admission rules got this wrong in the same way, and it is worth
recording because the failure is generic — it is not a LightGBM problem or an ensembling problem.

### Two rules, one shared mistake

| Rule | Bar | Status |
|---|---|---|
| 19% relative improvement | `(pair − three) / pair > 0.19` | **retired** |
| 1 × cross-window σ | `pooled < baseline − σ_window(baseline)` | **retired** |
| improvement + 2-of-3 windows + orthogonality | see *The rule we adopt* | **adopted** |

The 19% figure was a *solo-score sub-window* noise floor — the spread you see when you score one
model on different slices. It was then applied as a bar on the *relative gain of a paired three-way
blend over a two-way blend*. Those are different quantities, and paired comparisons on identical
windows are substantially tighter. No third member in a saturated blending problem could ever clear
it.

The σ-rule is the same error in a milder form. `σ_window` is the standard deviation of a model's
per-window WAPEs. It answers: *how much does this model's score move when I change the evaluation
block?* But an A/B is run on **identical rows** — both arms see the same units, the same hours, the
same window difficulty. The question the bar should ask is: *how much would this **difference** move
if I had drawn a different sample of units?* Window-difficulty variance is common to both arms and
largely cancels out of the difference; charging the candidate for it is charging it for noise the
comparison design already removed.

### The right estimator, and the numbers

The pooled WAPE is a ratio of sums over 32,256 scored rows grouped into **288 unit×window blocks**.
The uncertainty of a *difference* of two such ratios is estimated directly by a **paired cluster
bootstrap**: resample whole `(unit, window)` blocks with replacement, apply the *same* resample to
both arms, recompute both pooled WAPEs, take the difference.

Whole blocks rather than individual rows, because hourly residuals within a series are strongly
autocorrelated and a row-level resample would badly understate the standard error. Pairing is by
construction, since both arms see the same draw.

The calibration below used 4,000 resamples. `N_BOOTSTRAP` in `src/eval/protocol.py` defaults to
**2,000**, which is ample for a standard error, and that is what every recorded `results/ab_*.json`
carries — the CI tails are the thing that would want more, which is why it is a parameter.

Measured on the cached three-window cube, TFT as baseline:

| Quantity | Value | What it measures |
|---|---|---|
| TFT pooled WAPE | 0.1463 | per-window 0.1368 / 0.1546 / 0.1502 |
| `σ_window(TFT)` — **the old bar** | 0.0076 | dispersion of TFT's score across windows |
| **Paired bootstrap SE of the delta** | **0.0028** | uncertainty of the *difference*, same rows |
| σ of per-window paired deltas (n=3) | 0.0155 | *not* a noise estimate — see below |

So the old bar was **≈ 2.7× stricter than the comparison's actual uncertainty**. In relative terms
it demanded a 5.2% improvement from a single lever.

| Bar | Required Δ WAPE | Relative | Candidate must beat |
|---|---|---|---|
| 1 × σ_window *(old)* | 0.0076 | 5.2% | 0.1387 |
| 1 × paired SE | 0.0028 | 1.9% | 0.1435 |
| 2 × paired SE | 0.0056 | 3.8% | 0.1407 |

The useful part: switching to the correct estimator lets you be **stricter in the multiplier** — a
2σ bar rather than 1σ — and still end up with a *lower and better-justified* threshold than before.
The bar stops being arbitrary. "2 × the paired standard error" is a statement about the sampling
distribution of the quantity actually being tested; "1 × the cross-window spread" was not.

### Why the per-window delta spread is not the bar

Tempting shortcut: compute the three per-window deltas and take their standard deviation. Here that
gives 0.0155 — *larger* than the unpaired σ it was meant to replace. That is not a failure of
pairing; it is real information. TFT beats LightGBM in **all three** windows, but by margins that
range over a factor of 23:

| Window | cutoff | TFT | LGBM | Δ (TFT − LGBM) |
|---|---|---|---|---|
| W2 | 2976 | 0.1368 | 0.1588 | −0.0220 |
| W1 | 3312 | 0.1546 | 0.1864 | −0.0318 |
| W0 | 3648 | 0.1502 | 0.1516 | **−0.0014** |

So the spread of deltas conflates sampling noise with genuine window-to-window variation in the
*size* of the effect, and with n = 3 it estimates neither well. Keep it as a **heterogeneity
diagnostic** — "this member is only competitive in some windows" is decision-relevant — but never as
the noise bar. The blend weights corroborate it independently: the fitted three-way simplex gives
LightGBM **0.25 / 0.00 / 0.10** across W0/W1/W2, zero weight in exactly the window where it is
furthest behind.

### The rule adopted

Two stages with different jobs, running the **same three checks** on different data. **No check
drops a member automatically**; every one is advisory and the call stays human.

**Stage 1 — Screening.** Every lever, single seed 42, minutes of CPU. Question: *is this worth
paying for a multi-seed confirmation?* A false positive costs one cheap confirmation run; a false
negative loses a real improvement permanently — which is precisely how a genuine +1.1% blend gain
came to be recorded as "below bar". The costs are asymmetric, so the screen is permissive.

**Stage 2 — Admission.** Final ensemble members only, averaged over the frozen five seeds.
Question: *does this go in the ensemble?* This is the decision of record.

The three checks:

1. **Pooled improvement.** The candidate's pooled WAPE beats the baseline's. **Any improvement
   counts — there is no σ threshold to clear.** This is not a published hypothesis test, it is a
   choice of ensemble: for a convex blend, a member with positive expected gain is worth having, and
   the weight search decides how much to trust it. Taking the positive-expectation option is the
   correct decision under a proper scoring rule even when the gain is not significant.
2. **Majority of windows — at least 2 of 3 must improve.** Guards the one pathology a pooled number
   cannot see: a large win in a single window carrying a candidate that is *worse* everywhere else.
   Not a significance test — P(≥2 of 3) = 0.5 under a coin-flip null — it removes a failure mode.
3. **Orthogonality.** Error correlation against every incumbent below 0.95. A candidate that merely
   reproduces an existing member adds fitting risk without adding information, however good it looks
   alone. The final member pool sits at 0.77–0.96.

**Which question is being asked matters, and check 3 is where the two diverge.** The rule above was
written for **member admission** — *does this go into the blend alongside the incumbents?* The same
machinery gets reused for a different question: **replacement** — *is variant B a better version of
member A?* For a replacement, orthogonality is not merely irrelevant, it is inverted: a lever that
changes one hyperparameter *should* produce near-identical errors, and a low error correlation would
be the surprising result.

Concretely, `lgbm_es` vs `lgbm` passes checks 1 and 2 and misses on 3, because it is the same model
at a measured round count rather than a hand-picked one. That miss carries no information about
whether to adopt it. **So: replacement levers are judged on checks 1–2; orthogonality applies when
the survivor is proposed as an ensemble member.** Say that plainly rather than let a "2 of 3 checks
pass" line imply a marginal result.

Reported alongside, **never gated on**: the paired bootstrap delta with its 95% CI, the per-window
deltas and their spread, and at Stage 2 the per-seed `mean ± std`. The CI is what tells a reader
whether a gain is *demonstrated* or merely *observed* — a distinction the pass/fail flags cannot
carry.

Sanity check against the requirement that set this off: a 0.5 p.p. improvement (Δ = 0.005, 3.4%
relative) holding in all three windows passes both stages comfortably. Under the 2 × paired SE bar
first proposed here it would have been *rejected*, since 0.005 < 0.0056. That is why the magnitude
threshold was dropped rather than merely lowered.

### Two caveats stated rather than hidden

**More seeds is not more data.** Averaging over the frozen five removes training stochasticity —
measured at σ = 0.0005 for LightGBM, tiny — but leaves the dominant noise source completely
untouched: which 96 units and which three windows we happened to evaluate on, paired SE 0.0028,
roughly **six times larger**. A small multi-seed win is therefore an **expected-value** argument, not
a **statistical** one, and `mean ± std` must not be allowed to imply a precision the design does not
have.

**Multiplicity.** The programme runs on the order of ten independent levers, so at a nominal 5%
false-positive rate one should expect roughly **one spurious "win" by chance**. Stage 2 is the
guard: a screening false positive costs one confirmation run, and the confirmation is run on data
the screen did not see.

### The final bar — four criteria, not three

For the last round of candidates, after the screening stage was over, the bar was tightened. A
candidate shipped only if **all four** held, on the scored rows, against the incumbent **re-derived
in the same frame** (never quoted from a previous run):

1. **Δ pooled WAPE > 0** with a paired cluster bootstrap **CI excluding zero**.
2. **3 of 3 windows won** — not 2 of 3. Past the screening stage, this is the last artifact.
3. **Survives the weight refit** — the full simplex is refitted with the candidate in the pool and
   the candidate still carries weight ≥ 0.05.
4. **Selection is declared.** Any lane that searches many variants states how many, because the CI
   is conditional on that search.

Two extra rules for levers that add *parameters* rather than *models*:

* **Structured-weight lanes must clear the bar under two instruments** — the standing
  `blk<224 / blk>=224` split **and** leave-one-window-out (fit on two cutoffs, score on the third).
  A gain that appears under one and not the other is capacity, not signal.
* **Shrinkage is part of the candidate.** Report the shrinkage coefficient λ toward the global
  weights alongside the number; λ→1 collapsing to the incumbent is a legitimate, reportable result.

### Why the discarded bars are reported at all

Both discarded bars were plausible-looking numbers applied to the wrong quantity, and both were
silently *conservative* — they rejected real improvements rather than admitting fake ones, so
nothing looked broken from the outside. A member measured at a genuine +1.1% pooled gain with an
error correlation of 0.77 against TFT was recorded as "below bar". Selection rules deserve the same
scrutiny as models, and an honest account of how an ensemble was assembled has to include the
criterion that decided what got in.

---

## Running it

The bar is executable, not just documented:

```bash
python -m scripts.member_admission --candidate <member> --baseline <member>
```

It writes a `results/ab_<candidate>_vs_<baseline>.json` carrying the three checks, the paired
bootstrap (`delta`, `se`, `ci95`, `delta_in_se`, `n_blocks`, `n_boot`), the per-window deltas and
their spread, and the baseline's window σ. Eleven of those artifacts are in
[`results/`](../results); they are tabulated in [`results.md`](results.md#3-the-ab-record).
