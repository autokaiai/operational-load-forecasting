# Results

**Where each number comes from is stated per section, because they do not all come from the same
place.** Sections [1](#1-baselines), [2a/2b](#2-the-architecture-screen), [3](#3-the-ab-record),
[5](#5-clean-room-verification) and [6](#6-what-is-left-in-the-residual) are re-derived from the
artifacts in [`results/`](../results) and from `bundle.json` inside the graded checkpoint.
[Section 2c](#2c-the-arc-on-one-instrument) is the exception: it is the project's summary table and
several of its rows are **transcribed from the write-up rather than re-derived here**, because the
prediction cubes behind them are training outputs that are not shipped. Every such row is marked.
Section 1's right-hand column is recomputed from `train.csv` and is marked too.

> **The caveat that rides with every internal number.** All CV figures are **pooled far-regime
> WAPE** over the `blk >= 224` rows of the scored block — its final 112 hours — across three gapped
> windows (cutoffs 3648 / 3312 / 2976): 96 × 112 × 3 = 32,256 rows, self-scored against labels
> carved from the public 4320 labelled hours. Blend weights are fitted on `blk < 224`, so this is
> the honest half. **Two tables below are deliberate exceptions and say so in their own headers.** The **public leaderboard
> scores the near block** (gap 0); the private grade scores far. **The two are not comparable**, and
> only the leaderboard rows were scored by anyone but us. See
> [`protocol.md`](protocol.md#the-nearfar-axis-and-why-only-one-of-them-counts).

Contents: [1. Baselines](#1-baselines) · [2. The architecture screen](#2-the-architecture-screen) ·
[2c. The arc, on one instrument](#2c-the-arc-on-one-instrument) ·
[3. The A/B record](#3-the-ab-record) · [4. Leaderboard progression](#4-leaderboard-progression) ·
[5. Clean-room verification](#5-clean-room-verification) ·
[6. What is left in the residual](#6-what-is-left-in-the-residual)

---

## 1. Baselines

**Exception to the standing caveat, and it matters.** `results/baselines.json` was produced by the
course's own `baselines.py` on the **contiguous** rolling CV — three 336-hour blocks with *no gap*.
The report quotes those figures and flags them as "no gap, so conservative". They are not on the
instrument the rest of this repository uses, so both are given:

| baseline | contiguous, no gap (`baselines.json`) | gapped, `blk >= 224` — the headline instrument |
|---|---|---|
| naive last value | 0.54709 | **0.45919** |
| lag-24 repeat | 0.46338 | 0.38740 |
| lag-168 repeat | 0.47918 | 0.37729 |
| seasonal mean | 0.32691 | **0.31734** |

The right-hand column is what every model number here should be read against; it is recomputed from
`train.csv` with the course's baseline definitions unchanged, and the left-hand column reproduces
`baselines.json` to five decimals, which is the check that the recomputation is faithful.

**Both columns are self-scored.** A third set exists — the same four baselines as the course scores
them on the leaderboard's own block — and it is in
[§4](#the-baselines-as-the-leaderboard-scores-them). It is near-regime and externally scored, so it
belongs with the leaderboard numbers rather than here, and the three sets are not interchangeable.

**The two columns disagree about which baseline is better, and the disagreement is informative.**
Contiguously, lag-168 loses to lag-24 — over 336 hours the weekly repeat is asked to carry two full
cycles and it drifts. Across the 336-hour gap that reverses: lag-24 is extrapolating a day that is
now two weeks stale, while the weekly phase is still the right phase. **A baseline ranking read off
the wrong regime would have pointed the whole project at the wrong seasonality.**

---

## 2. The architecture screen

Twenty-three architectures were trained. **They do not all carry the same kind of number, and
mixing them into one column would be wrong**, so they are in two tables.

### 2a. Scored on the graded regime (13)

`gapped_eval` — h=672, only the last 336 steps scored, the FAR block.

> **Second exception, and the one most likely to mislead.** `src/train.py` calls
> `run_gapped_eval`, which evaluates at **one** cutoff (3648) over the **full** 336-hour block, and
> supplies the gap's covariates locally — its own docstring calls this "an *optimistic* gap
> estimate". So these 13 numbers are comparable **to each other**, and *not* to the 3-window
> `blk >= 224` numbers everywhere else. The plain TFT is `0.13605` here and `0.15138` on the
> headline instrument; both are correct measurements of different things. Where this document
> quotes a member against a blend, it uses the headline instrument.

| model | WAPE | MAE | MSE | RMSE | MAPE | sMAPE | params |
|---|---|---|---|---|---|---|---|
| **tft** | **0.13605** | 1.4531 | 6.7223 | 2.5927 | 0.14843 | 0.14879 | 1,649,911 |
| lstm_combo | 0.16742 | 1.7882 | 9.8666 | 3.1411 | 0.21157 | 0.18491 | 897,793 |
| lstm_vw_wide | 0.16881 | 1.8029 | 9.9940 | 3.1613 | 0.21146 | 0.18976 | — |
| lstm_wide | 0.16997 | 1.8154 | 10.0282 | 3.1667 | 0.20996 | 0.19206 | 897,793 |
| lstm_sched | 0.17073 | 1.8235 | 10.0917 | 3.1767 | 0.21006 | 0.19055 | 235,905 |
| lstm_vw | 0.17308 | 1.8486 | 10.2504 | 3.2016 | 0.20996 | 0.19537 | — |
| lstm_drop | 0.17380 | 1.8563 | 10.2736 | 3.2053 | 0.21144 | 0.19752 | 897,793 |
| lstm | 0.18018 | 1.9244 | 10.6613 | 3.2652 | 0.21409 | 0.20532 | 235,905 |
| lstm_recurrent | 0.18107 | 1.9339 | 9.4799 | 3.0789 | 0.23026 | 0.19691 | 20,736 |
| bitcn | 0.18541 | 1.9803 | 11.0806 | 3.3288 | 0.21795 | 0.20058 | 35,233 |
| tsmixerx | 0.25825 | 2.7582 | 16.3102 | 4.0386 | 0.31035 | 0.28330 | 2,087,968 |
| timesnet | 0.30307 | 3.2370 | 20.4248 | 4.5194 | 0.37387 | 0.33312 | 5,113,801 |
| nhits | 0.43330 | 4.6279 | 36.5313 | 6.0441 | 0.50649 | 0.60846 | 28,908,084 |

**The TFT wins by a margin nothing else in the screen comes close to** — 0.136 against a next-best
0.167, a gap larger than the entire spread of the seven LSTM variants. The reason is visible in the
task: 22 covariates, most of them known-future, and the TFT is the only architecture here with a
variable-selection network and an explicit future-covariate channel. That is why every subsequent
lane took the TFT as its supervised architecture. Note what the screen does *not* settle: the
largest weight in the shipped blend went to a fine-tuned foundation model, not to this winner.

**Parameter count predicts nothing.** N-HiTS at 28.9M parameters is the worst model in the table;
`lstm_recurrent` at 20,736 parameters beats it by a factor of 2.4 on WAPE. On a 96-series panel with
4,320 hours of history, capacity is not the binding constraint — and that observation is what the
[capacity probes in `method.md`](method.md#w6--the-budget-went-to-models-not-to-features) later
confirmed the expensive way.

**The h=336 contiguous CV is in the same files** (`cross_validation.cv_metrics_pooled`) and is a
**different regime** — no gap to traverse. It is not mixed into the column above. Where the two
disagree they disagree informatively: `lstm_recurrent` scores 0.18107 gapped against 0.22565
contiguous — one of the very few models the gap *helps*, because its contiguous fit leans hardest on
persistence.

### 2b. Trained, not carried to gapped evaluation (10)

These files carry **`train_wape` only**. **`train_wape` is an in-sample fit score. It is not a
generalisation estimate and it must never be read in the same column as the table above.** They are
listed for completeness — the screen ran them, and omitting them silently would misrepresent how
wide it was.

| model | train_wape (in-sample — *not* a score) | params |
|---|---|---|
| informer | 0.18483 | 349,409 |
| fedformer | 0.19326 | 691,201 |
| autoformer | 0.19988 | 297,985 |
| mlp | 0.21106 | 26,858,832 |
| kan | 0.25251 | 129,039,360 |
| nbeatsx | 0.28808 | 42,247,890 |
| tide | 0.32240 | 7,845,260 |
| dlinear | 0.33462 | 339,360 |
| patchtst | 0.34238 | 3,117,523 |
| itransformer | — | 6,736,720 |

`itransformer.json` carries neither — that run never completed.

For scale on why in-sample numbers cannot be compared across the boundary: the TFT's own
`train_wape` is **0.12993** against its gapped **0.13605**. A 0.006 train-to-gapped gap on the one
model where both exist is not enough to license reading the left column as if it were the right one.


### 2c. The arc, on one instrument

Every row here is the **same 32,256 scored rows** — the `blk >= 224` block of the three gapped
windows — so the differences are attributable. If you read one table, read this one.

> **Provenance, per row.** `†` = re-derived here from a tracked artifact. `‡` = recomputed from
> `train.csv`. Unmarked rows are **transcribed from the project's write-up**: they were measured on
> member prediction cubes produced during training, which are ~100 MB of intermediate output and are
> not shipped. They are reported because the table is incomplete and misleading without them, and
> marked because this repository does not otherwise ask you to take a number on trust.

| | far WAPE | note |
|---|---|---|
| naive last value / lag-168 repeat `‡` | 0.4592 / 0.3773 | the required minimum baseline |
| seasonal mean `‡` | 0.3173 | the required stronger baseline |
| ridge on the 29 known-future channels | 0.1724 | control: the covariates alone support ≈0.17 |
| Chronos-2 zero-shot / LoRA FT / full FT | 0.1858 / 0.1602 / 0.1325 | full FT is 17 % under LoRA — but batch differs too |
| BiTCN / LSTM / CatBoost | 0.1805 / 0.1653 / 0.1586 | screened, not admitted |
| LightGBM untuned / tuned + interp `†` | 0.16281 / 0.15200 | the shipped tree; `results/ab_lgbm_best_vs_lgbm.json`, `weights.json` |
| **TFT (plain)** | **0.15138** | the same TFT, without the channel |
| convex 0.70·TFT + 0.30·FT-Chronos | 0.1427 | the exposé's ship candidate |
| **TFT cascade** — 1 seed `†` / seed-mean / 5-seed SWA bag | **0.13429** / 0.1391 ± 0.0028 / 0.13334 | that draw is ≈1.7σ lucky, which is why a bag ships; `results/ab_tft_cascade_matched_vs_tft_cascade.json` |
| two-member blend 0.24·tree + 0.76·bag `†` | 0.13163 | an earlier submission — rung 2 today; `weights.json` |
| three-member blend, + the full fine-tune `†` | 0.12722 | +0.0044, CI [+0.0035, +0.0054], 3/3; `results/fp/1a_error_map.json` |
| + cross-series aggregates `†` | 0.12029 | +0.0069, CI [+0.0060, +0.0079], 3/3; `weights.json` |
| **SHIPPED, + amplitude recalibration γ=1.04** `†` | **0.11807** | +0.0022, 3/3; weights 0.472 / 0.341 / 0.186; `weights.json` |

**The largest single step is the plain TFT to the cascade.** Against the cascade's *seed-mean*
(0.1391 ± 0.0028) that is **+8.1 % relative** from one extra input column; against the single draw
shown in the table, +11.3 %. The averaged figure is the one to quote — that draw sits ≈1.7σ on the
favourable side, which is why a five-seed bag ships. Either way nothing else in the table moves the
number as far, including every architecture substitution and the entire hyperparameter search.

**Absolutes are quoted seed-averaged, and one member is a deliberate exception.** The cascade's
best single draw (0.13429) sits ≈1.7σ on the favourable side of its own mean, so a five-seed bag
ships instead. The fine-tuned Chronos-2 is single-seed on purpose — a seed costs 456 MB of hosted
weights against a cross-seed σ of 0.0023 — and **the draw that shipped is the worst of the three
that were run**. The TFT cascade's cross-seed σ is ≈0.0055 per window, twice the paired bootstrap
SE, so two differently seeded configurations under ≈0.005 apart are not resolvable at one seed. No
paired comparison here is affected: both sides of every A/B carry the same draw.

---

## 3. The A/B record


Eleven `results/ab_*.json`, each the full output of `scripts/member_admission.py`: the three checks,
the paired cluster bootstrap over 288 unit×window blocks, the per-window deltas and their spread.
**This is the best evidence in the repository that the selection protocol was real rather than
retrospective.** The rule they implement is in [`protocol.md`](protocol.md#2-the-admission-bar).

Δ is *candidate minus baseline* with the sign flipped so **positive = better**. `nSE` is the delta
in units of its own paired standard error. `chk` counts the three advisory checks. All at Stage 1
(screen, seed 42), `n_blocks = 288`, `n_boot = 2000`.

| candidate vs baseline | regime | cand | base | Δ | SE | 95% CI | nSE | windows | chk |
|---|---|---|---|---|---|---|---|---|---|
| lgbm_s24_unitcat vs lgbm | full | 0.15848 | 0.16581 | **+0.00733** | 0.00088 | [+0.00561, +0.00902] | +8.3 | 3/3 | 2/3 |
| lgbm_s24_unitcat vs lgbm_stride24 | full | 0.15848 | 0.16204 | **+0.00357** | 0.00067 | [+0.00231, +0.00491] | +5.3 | 3/3 | 2/3 |
| lgbm_stride24 vs lgbm_es | late | 0.15933 | 0.16200 | **+0.00267** | 0.00060 | [+0.00149, +0.00384] | +4.4 | 3/3 | 2/3 |
| lgbm_es vs lgbm | late | 0.16200 | 0.16292 | +0.00093 | 0.00016 | [+0.00061, +0.00126] | +5.6 | 2/3 | 2/3 |
| lgbm_cascade vs lgbm_s24_unitcat | full | 0.15782 | 0.15848 | +0.00066 | 0.00054 | [−0.00038, +0.00170] | +1.2 | 2/3 | 2/3 |
| lgbm_s24_wlag vs lgbm_stride24 | full | 0.16184 | 0.16204 | +0.00021 | 0.00029 | [−0.00036, +0.00077] | +0.7 | 2/3 | 2/3 |
| lgbm_wlag vs lgbm_es | late | 0.16219 | 0.16200 | −0.00019 | 0.00053 | [−0.00117, +0.00086] | −0.4 | 1/3 | 0/3 |
| lgbm_s24_recency vs lgbm_stride24 | full | 0.16561 | 0.16204 | −0.00357 | 0.00035 | [−0.00428, −0.00289] | −10.1 | 0/3 | 0/3 |
| lgbm vs lgbm_stride24 | full | 0.16581 | 0.16204 | −0.00377 | 0.00048 | [−0.00469, −0.00277] | −7.8 | 0/3 | 0/3 |
| lgbm_s24_norm vs lgbm_stride24 | full | 0.16821 | 0.16204 | −0.00617 | 0.00101 | [−0.00816, −0.00429] | −6.1 | 0/3 | 0/3 |
| tft_cascade_matched vs tft_cascade | full | 0.15278 | 0.14030 | **−0.01248** | 0.00081 | [−0.01405, −0.01092] | −15.4 | 0/3 | 0/3 |

Three things this table is here to show.

**The bar rejects as often as it admits.** Five of eleven are losses, and four of those have CIs
excluding zero — they are *demonstrated* losses, not ambiguous ones. `lgbm_s24_norm` and
`lgbm_s24_recency` were plausible ideas (per-unit normalisation; recency-weighted samples) that the
bar killed cleanly.

**`chk` of 2/3 is not a marginal result, and reading it as one would be an error.** Every LightGBM
row above is a *replacement* question — is variant B a better version of member A — where the third
check, orthogonality, is inverted: a one-hyperparameter change *should* produce correlated errors.
`lgbm_s24_unitcat` at +8.3 SE and 3/3 windows reads "2 of 3 checks pass" purely because it correlates
with the model it replaces. It is the shipped tree.

**The largest single effect in the table is a negative one.** `tft_cascade_matched` — the
matched-lead cascade, where the covariate's forecast lead is aligned with the target's — is worse by
0.01248 at 15.4 SE, losing all three windows. The train/inference mismatch it was designed to remove
turns out to be load-bearing. See [W5 in `method.md`](method.md#w5--two-nulls-that-looked-certain).

---

## 4. Leaderboard progression

**Externally scored**, on the public validation split (the NEAR block, gap 0). The board reports
WAPE the usual way, as a percentage; this repository's internal figures are the same quantity
written as a fraction. `offset` is stated in **percentage points**, `public_LB − 100 × CV_far` — the
gap between an externally scored near number and our own internal far number.

### The baselines, as the leaderboard scores them

The course publishes its four baselines on the same board, so this is the **only anchor in this
repository that neither we nor our code produced**. Same block, same scorer; the internal tables
elsewhere write the same quantity as a fraction:

| entry | WAPE | ours is |
|---|---|---|
| naive last value | 48.098 | **74.8 % better** |
| lag-24 repeat | 46.820 | 74.1 % better |
| lag-168 repeat | 46.699 | 74.0 % better |
| seasonal mean | 34.409 | **64.7 % better** |
| **`v-prod-1.1` — shipped** | **12.141** | — |

**These are a third set of baseline numbers and must not be mixed with the two in
[§1](#1-baselines).** They are near-regime, externally scored, on the validation block; §1's are
far-regime and contiguous-CV, self-scored on blocks carved from `train.csv`. The correspondence is
loose by construction — the naive baseline is 48.098 here and 45.919 on our far instrument — and
nothing in this repository depends on the two agreeing.

**Per-member near and far, side by side.** `results/member_metrics_near_far.json` carries pooled and
per-window WAPE for `tft`, `tft_cascade`, `chronos_ft` and `lgbm` split into near (steps 1–336) and
far (steps 337–672). It is the clearest single view of the axis this whole section is about.
**Caveat, and it is the same one as everywhere else in this project:** that file was produced by an
earlier harness (its `source` field reads `blend_sandbox/full` — the full 336-hour block, 96,768
rows, from a working directory not shipped here), so its absolutes are a
fourth row-set and are *not* comparable to the `blk >= 224` numbers elsewhere here — plain TFT is
0.1486 in it. Read it for the near-vs-far contrast within a member, not across tables.

**One thing worth reading across the three sets.** The order of the two seasonal baselines is not
stable: lag-168 clearly beats lag-24 on the far instrument (0.37729 vs 0.38740), they are within
0.12 points of each other here, and lag-24 is ahead on our contiguous CV blocks. Which seasonality a
naive forecaster should lean on depends on how far the origin sits from the scored block — and a
conclusion drawn on one of these instruments does not carry to the others. That is the near/far
argument in its cheapest possible form, with no model involved.

| # | tag | public LB | CV far | offset | note |
|---|---|---|---|---|---|
| 1 | `v-b-0.1` | 13.190 | — | — | early near-specialist |
| 2 | `v-candidate` | 13.483 | — | — | hand-built CSV, not a runnable artifact |
| 3 | ~~`v-final`~~ | ~~15.794~~ | — | — | **VOID** — the bag-never-averaged defect |
| 4 | `v-final-2` | 13.622 | 0.13163 | — | first archive-produced output |
| 5 | `v-final-3` | 13.066 | 0.12722 | +0.344 | |
| 6 | `v-prod-1-0` | 12.632 | 0.12029 | **+0.603** | ← ~2× historical: **the defect signal** |
| 7 | **`v-prod-1.1`** | **12.141** | **0.11807** | +0.334 | ← **the shipped model**, offset restored |

**The offset column is the point of this table**, and it is the finding described in full at
[W1](method.md#w1--a-cv-to-holdout-offset-is-a-reproducibility-monitor-not-just-an-accuracy-gap). A
CV-to-holdout offset that is *stable across revisions* is evidence the shipped object matches the
measured one. When it nearly doubled at `v-prod-1-0`, that was not noise — it was a degraded member.
Aligning the two paths restored the offset to within 0.01 of its historical value and moved the
board score by 0.491 points, of which the amplitude recalibration explains ~0.22 and the alignment
~0.27.

### What the offset measures, and what it does not

The column is read for its **stability**, not its level — and the distinction is worth making
explicit, because the level points the opposite way to what the near/far axis alone would predict.
Far is the harder regime, so a far CV number should sit *above* a near leaderboard number. It sits
below: 0.11807 against 12.141.

That is not a contradiction; it means the offset is a composite, and its components are nameable:

* **The headline slice is the easier part of the block.** Scoring `blk >= 224` — the block's final
  112 hours — is systematically easier than scoring all 336. On the naive baseline alone, with no
  model involved, that slice scores **0.45919 against 0.48929** for the full block: about 6 % easier
  by construction.
* **The blocks are different stretches of time.** The CV windows are carved from inside the 4,320
  labelled hours; the leaderboard scores 4,320–4,655, a genuinely later period with its own
  difficulty.
* **Selection pressure.** Members, imputation surfaces, blend weights and γ were all chosen while
  looking at these same three windows. The `blk < 224 / blk >= 224` split holds out the *weight fit*;
  it cannot hold out the design decisions.

**None of that weakens the diagnostic, because the diagnostic is the change.** A stable offset says
the object being shipped is the object that was measured, whatever its absolute size; that is what
caught a degraded member at `v-prod-1-0` and what confirmed the fix at `v-prod-1.1`. Naming the
components is what stops the level being read as an accuracy claim in its own right — the internal
numbers and the leaderboard measure different blocks, different slices and different regimes, and
only their *relationship over time* is informative.

Row 3 is kept deliberately. `v-final` scored 15.794 — *worse than the naive baseline's neighbourhood
and worse than the model three revisions earlier* — because the five-seed bag shipped one seed of
five while producing a schema-perfect CSV that every check in the artifact passed.

### Two numbers for the same blend

`bundle.json` records `cv_pooled_wape: 0.12029`; this repository's headline is `0.11807`. Both are
correct and the difference is exact:

```
0.12029 − 0.00222 = 0.11807
```

The `bundle.json` figure is the blend **before** amplitude recalibration; 0.00222 is gamma's
measured gain (SE 0.00017, 13.1 SE, 3/3 windows). The bundle records the weight fit; gamma is
applied downstream in `predict.py` and is not part of the weight fit, which is why the two numbers
differ and why neither is a typo. Both are in [`weights.json`](../weights.json).

---

## 5. Clean-room verification

`results/cleanroom/summary.json`. **Seven** runs of the graded command in a container built from
the archive alone, each checked against the submission contract rather than against a score. Six ran
on a GPU lane; the seventh is a CPU lane, listed last, and is the only run that did not reach its
expected rung.

| scenario | rc | wall (s) | peak GPU (MiB) | rung | rows | non-finite | negative | scorer |
|---|---|---|---|---|---|---|---|---|
| `graded_literal` | 0 | 168.8 | 7,684 | 1 | 32,256 | 0 | 0 | ACCEPTED |
| `graded_literal` (2nd lane) | 0 | 195.1 | 7,684 | 1 | 32,256 | 0 | 0 | ACCEPTED |
| `board` | 0 | 158.4 | 7,684 | 1 | 32,256 | 0 | 0 | ACCEPTED |
| `hostile` | 0 | 173.1 | 7,684 | 1 | 32,256 | 0 | 0 | ACCEPTED |
| `nofullft` | 0 | 109.0 | 5,722 | **2** | 32,256 | 0 | 0 | ACCEPTED |
| `offline` | 0 | 58.7 | 2,164 | **2** | 32,256 | 0 | 0 | ACCEPTED |
| `graded_literal` (CPU lane) | 0 | 387.8 | — | **4** | 32,256 | 0 | 0 | ACCEPTED |

What each scenario is for:

* **`graded_literal`** — the private test's shape, the literal graded command, input dir mounted
  read-only. Run on two independent lanes; **both produced sha256 `293114b3f257f53a…`**, which is
  the reproducibility claim stated as a hash rather than as an adjective.
* **`board`** — the public leaderboard's shape (gap 0). Reproduced sha256 `43e1860e345cfb5e…`, the
  CSV that scored 12.141.
* **`hostile`** — shuffled forecast index, extra columns, 15% extra NaNs, no `metadata.json`. Still
  rung 1, still row-for-row aligned to the index's own order.
* **`nofullft`** — the fine-tune unreachable. Drops to rung 2 with its **own** measured weights and
  raises the degraded banner.
* **`offline`** — an air-gapped box: both weight fetches must fail and the CSV must still land.
* **`graded_literal` (CPU lane)** — the same command with no CUDA device. It is the **one run in
  the artifact where `rung_as_expected` is `false`**: `neuralforecast` raises rather than falling
  back to CPU, so every rung carrying the neural bag drops out and the run lands on rung 4, the tree
  alone. It is listed rather than omitted because a table of contract checks that quietly drops its
  one non-conforming row is not a contract check. The banner fired, the CSV was valid, and the
  behaviour is now documented in the README's "Run it" section.

Every run: schema exact, order matching the forecast index, timestamps byte-verbatim, 0 duplicate /
missing / extra keys, input dir unwritten, and accepted by the leaderboard's own scorer with
`allow_padding=False`. The three degraded runs raised the banner; the four nominal runs did not.

The wall-clock and memory columns carry a practical point of their own: **the offline arm is 2.9×
faster and needs 3.6× less memory**, because it skips both Chronos-2 forward passes. A degraded run
is not just less accurate — it is visibly cheaper, which is a second, independent signal that
something was skipped.

---

## 6. What is left in the residual

`results/fp/1c_residual.json` and `results/fp/1a_error_map.json`. Method, and why the instrument
matters more than the number, at
[method.md W7](method.md#w7--how-much-predictive-value-is-left-and-how-you-measure-that).

**The direct test.** A LightGBM on `y − blend` against 34 covariate features, applied shrunk. If
the residual were noise, nothing could beat predicting zero out of sample. Something can — and the
size of the gain shrinks monotonically as the instrument gets more honest:

| instrument | base blend | stacked | delta |
|---|---|---|---|
| standing `blk < 224 / blk >= 224` split | 0.12722 | 0.12322 | +0.00400 |
| leave-one-window-out | 0.13049 | 0.12769 | +0.00279 |
| forward chaining, far rows only | — | — | +0.00265 |
| **forward chaining + 336 h embargo, near and far** | **0.13162** | **0.13049** | **+0.00113** |

**+0.00113, CI [+0.00086, +0.00139], 3/3 windows** — roughly 1 % **of the base it was measured
against**. All four admission criteria hold. Per fold the gain tracks training-set size (+0.00053 /
+0.00038 / +0.00245 at cutoffs 2976 / 3312 / 3648), and the leakage account — that absolute-time
features manufactured it — was **tested and rejected** by ablating `trend` and `_blk`, after which
the gain survives.

> **Read the base column, not just the delta.** The shipped blend is **0.11807**. The stacker's best
> absolute output, on its most optimistic instrument, is **0.12322** — *worse than the model that
> shipped*. It improved an earlier blend, one without the cross-series aggregates (+0.00693) and
> without the amplitude recalibration (+0.00222). Both of those are panel-level corrections acting
> on the same kind of structure a tree over cross-sectional covariates would find, so overlap is the
> expected case. **Headroom against the shipped model is very likely smaller than +0.00113 and was
> never measured.** Why it was not shipped, in full, at
> [W7](method.md#why-it-was-not-shipped).

**Where the error is not.** `1a_error_map.json`, four measurements over the same 32,256 rows:

| | |
|---|---|
| top 10 units' share of pooled error | **15.7 %** (11.1 % of volume) — diffuse, against a 40 % threshold set in advance |
| member ranking across windows | Kendall τ **+0.775** — stable |
| member ranking across lead bands | τ(first, last) **+0.926**, order unchanged — **flat in lead** |
| statics vs per-unit error | log(unit WAPE) ~ capacity + zone: **R² = 0.007** |

Two planned lanes were closed on this evidence before either ran, and a third — structured blend
weights — later kept about 3 % of its own oracle ceiling, which is what the diffuse error map
predicted.

**What is missing from this section**, and stated rather than quietly omitted: the classical
residual diagnostics — residual *median* per unit and per lead band, ACF at lags 1 / 24 / 168,
spread against fitted level, residual against each covariate — were **not run**. They are minutes of
CPU and they answer a narrower but more interpretable question than the stacker does. The checklist,
and why *mean*-zero and normality are the wrong targets under a WAPE objective, is at
[W7](method.md#the-residual-checklist-we-did-not-run).
