# Method

What the problem is, what shipped, and — at greater length — what did not work and what that
measured.

* [1. The problem, and why the split shape drives everything](#1-the-problem-and-why-the-split-shape-drives-everything)
* [2. The shipped system, member by member](#2-the-shipped-system-member-by-member)
* [3. Findings](#3-findings)
  * [W1 — a CV-to-holdout offset is a reproducibility monitor, not just an accuracy gap](#w1--a-cv-to-holdout-offset-is-a-reproducibility-monitor-not-just-an-accuracy-gap)
  * [W2 — serialised state can be correct in one reader and silently wrong in another](#w2--serialised-state-can-be-correct-in-one-reader-and-silently-wrong-in-another)
  * [W3 — the blend was under-dispersed, and one scalar fixes it](#w3--the-blend-was-under-dispersed-and-one-scalar-fixes-it)
  * [W4 — neither large lever transfers, and one variable explains both](#w4--neither-large-lever-transfers-and-one-variable-explains-both)
  * [W5 — two nulls that looked certain](#w5--two-nulls-that-looked-certain)
  * [W6 — the budget went to models, not to features](#w6--the-budget-went-to-models-not-to-features)
  * [W7 — how much predictive value is left, and how you measure that](#w7--how-much-predictive-value-is-left-and-how-you-measure-that)
* [4. Three lessons, and what this work does not establish](#4-three-lessons-and-what-this-work-does-not-establish)
* [5. Decision ledger](#5-decision-ledger)

Related: [`protocol.md`](protocol.md) · [`results.md`](results.md) · [`references.md`](references.md)

The evaluation protocol these all run against is in [`protocol.md`](protocol.md); the tables are in
[`results.md`](results.md).

---

## 1. The problem, and why the split shape drives everything

Forecast **336 hourly steps (14 days)** of a strictly positive "operational load index" for **96
units**, from 4,320 labelled hours of history plus 22 covariates — 19 time-varying and 3 static.
The model's known-future channel is 29 columns: those 19 plus 10 derived missingness indicators. Primary metric **WAPE**
(`sum|y − ŷ| / sum|y|`), pooled over all scored rows.

The data has three properties that decided most of the design:

**Many of the covariates are known-future.** `demand_forecast`, `staffing_forecast`,
`maintenance_known`, `queue_pressure_forecast`, `network_pressure_forecast`, `event_load_forecast`
and others are supplied *for the rows being predicted*. A model with an explicit future-covariate
channel is not a stylistic preference here; it is the difference between using the given
information and discarding it. That is most of why the TFT wins the architecture screen by 0.03
WAPE — see [`results.md`](results.md#2a-scored-on-the-graded-regime-13).

**~4.5% of the covariate cells are NaN, and the NaNs persist into the future rows.** Imputation is
therefore an *inference-time* requirement, not a training convenience. It also turns out to be two
separate problems with two different answers ([§2](#the-two-imputation-surfaces)).

**The private split opens a 336-hour gap.** History ends at hour *T*; the scored block is
*T+337 … T+672*. Nothing about hours *T+1 … T+336* is observed. This is the single most consequential
fact about the task, because a contiguous rolling-origin CV — the default thing to build — does not
measure it. It measures forecasting from a warm start, and it ranks persistence-leaning models far
above where they belong.

So the CV was built to mirror the split rather than to be convenient: three cutoffs, 672 steps each,
**only the last 336 scored**. Blend weights are fitted on the first 224 hours of that block and
every headline number is scored on the remaining 112 — 96 series x 112 hours x 3 windows =
**32,256 scored rows**. The full design, the near/far axis, and the
seed policy are in [`protocol.md`](protocol.md#1-the-gapped-cv-design).

### The dataset is deliberately anonymised, and that has consequences

96 unnamed `unit_*` series; covariates called `queue_pressure_forecast`, `shock_risk`,
`upstream_quality_forecast` with no units, no documented generating process, and no statement of
what a "zone" or a "unit" physically is. Feature engineering is normally driven by domain semantics
— you build a feature because you know what the quantity means. Here there is no domain to reason
from. [W6](#w6--the-budget-went-to-models-not-to-features) is about what that did to how the budget
was spent, and about why the conclusion drawn from it at the time was the wrong one.

---

## 2. The shipped system, member by member

A convex blend of three members, each carrying the same three cross-series aggregate covariates,
followed by one scalar recalibration:

```
0.4724 * Chronos-2, fully fine-tuned (2000 steps)
0.3413 * TFT on a Chronos-2 cascade covariate, 5-seed SWA bag
0.1863 * LightGBM (lgbm_s24_unitcat)
------
       then  pred' = unit_mean + 1.04 * (pred - unit_mean),  clipped at 0
```

Pooled far-regime CV: **0.11807**. Against 0.13163 for the two-member predecessor and **0.4592** for
the naive last-value baseline on the same rows.

The weights are fitted by **Frank–Wolfe on the simplex**, which is exact here rather than
approximate: pooled WAPE is convex in the weights, so the linear-minimisation step has a vertex
solution and the iterates converge to the global optimum of the convex combination. They are fitted
on `blk < 224` and evaluated on `blk >= 224`, pooled over the three windows. The exact weights, the checkpoint hash, and every fallback rung are
recorded in [`weights.json`](../weights.json); the training config of record is
[`configs/tft_chronos_swa_shipped.yaml`](../configs/tft_chronos_swa_shipped.yaml).

### The members

### The same model in three roles

A pretrained forecaster can enter a system in three places, and this project measured Chronos-2 in
**all three of them**:

| role | what it is | result |
|---|---|---|
| **forecaster** | predict the target directly, zero-shot | **0.1858** — *worse* than the supervised TFT at 0.15138 |
| **covariate** | its forecast handed to the TFT as a known-future channel | **+8.1 %** seed-averaged — 0.15138 → 0.1391; the best single draw reaches 0.13429 (+11.3 %) |
| **blend member** | fine-tuned on this data, combined at fitted weights | **0.1325 solo**, and the largest weight of the three (0.4724) |

**Quote the seed-averaged figure, not the best draw.** 0.13429 is a *single* cascade fit, and this
project's own seed accounting puts it about **1.7σ on the favourable side** of its seed-mean
(0.1391 ± 0.0028, cross-seed σ ≈ 0.0055 per window). Against that mean the channel is worth
**+8.1 %**, and that is the number this document uses. The lucky draw is reported where the seed
discussion belongs — it is also why a five-seed bag ships rather than one fit. Note the asymmetry
honestly: only the cascade side has a measured seed distribution, so the plain-TFT baseline (0.15138)
is itself one draw.

**What this does not establish.** The comparison is paired against *our own* TFT, so it measures what
the channel adds **to this model** — not that the channel is necessary to reach this accuracy. A
differently-built supervised model, with a different input representation, could plausibly reach a
similar level without it. [W4](#w4--neither-large-lever-transfers-and-one-variable-explains-both) is
the same boundary from the other side: on a second panel the same channel is worth −0.5 %.

*Provenance of the headline pair:* **0.13429 is re-derived from a tracked artifact**
(`results/ab_tft_cascade_matched_vs_tft_cascade.json`, `regimes.late.tft_cascade`). Its baseline
0.15138 is weaker: it is *recorded* in a shipped artifact
(`results/addl_dataset_metrics.json`, `m5.skill.our_reference.tft`) as the reference point for the
transfer comparison, but that block transcribes the value rather than computing it. The measurement
itself lives on the plain-TFT prediction cube from the same run, which is training output and is not
shipped. Both numbers were recomputed together, on the same rows and the same `regime="late"` slice,
when this document was written.

One model, three roles, three different answers — and the ordering is the finding. **The role in
which it is weakest as a model is the role in which it is worth most.** As a forecaster it loses to
the TFT by 22.7 % relative; as that TFT's covariate it is the best thing that happened to the
project. What a covariate is worth is not its accuracy but its **complementarity** — see
[W4](#w4--neither-large-lever-transfers-and-one-variable-explains-both), where the same measurement
on M5 returns −0.5 % and the difference is explained.

The line between roles two and three is the project's standing rule: *a model fitted on our data
enters only as a blend member, never as a covariate.* Only the **frozen, zero-shot** model is
allowed into the covariate channel, which is what keeps the cascade honest by construction rather
than by inspection.

**`chronos_full_ft` (0.4724) — a full fine-tune of Chronos-2 on this dataset.** Not LoRA: the whole
455.8 MB model rather than a 4.6 MB adapter. It is the best single member and it earned the largest
weight. It is fetched from a public Hugging Face repo at a **pinned revision** at inference, because
455.8 MB does not fit a 200 MB submission archive.

**`cascade_bag` (0.3413) — a TFT conditioned on a foundation model's forecast.** A frozen, zero-shot
Chronos-2 forecast of the target across the horizon is handed to the TFT as an extra *known-future
covariate* (`chronos2_forecast`) that its variable-selection network learns when to trust. That
channel alone is worth **+8.1% relative** seed-averaged (0.15138 → 0.1391) and was the largest
single gain in the project — the same TFT, the same three windows, one config key apart. The best
single draw reaches 0.13429 (+11.3 %); see
[the three roles](#the-same-model-in-three-roles) for why the averaged figure is the one quoted.

The channel is **generated at inference, never shipped as data** — the evaluation timeframe may
differ, and baking data into a checkpoint is disallowed. It is conditioned on the target history the
bundle carries, and the context ends at the last observed hour, so it is causally honest by
construction rather than by inspection.

The member is a **5-seed bag** over the frozen seeds, each with **checkpoint averaging** (SWA) over
the last 25% of its own run. The averaging is ours (`src/models/swa.py`), not Lightning's:
Lightning's `StochasticWeightAveraging` is epoch-based and asserts `max_epochs is not None`, which
under neuralforecast's step-based training is always `None`. The window is a *fraction* of the run
rather than a fixed step count, because the three windows early-stop at wall times spanning 2.19×
and a fixed last-250-step window covered ~52% of the shortest run and ~24% of the longest — the
measured delta tracked that fraction on 3/3 windows. 0.25 is the SWA paper's own convention, chosen
as an external anchor rather than read off our best window.

**`tree` (0.1863) — LightGBM, `lgbm_s24_unitcat`.** `objective regression_l1` (L1 on the raw target
minimises the WAPE numerator exactly), `num_leaves 63`, `min_child_samples 200`,
`feature_fraction 0.8`, `bagging_fraction 0.8`, `lambda_l2 1.0`, `origin_stride 24`, unit id as a
native categorical. It **runs first, on purpose**: a serialised booster, a design matrix, one
predict call — no download, no network, no GPU. It is the one member that cannot fail for an
environmental reason, so running it before the neural bag means a catastrophic failure there still
leaves a shippable forecast.

### Two components the weights above do not name

**Cross-series aggregates (`xs_cap_*`), carried by all three members.** Three extra known-future
covariates: the capacity-weighted cross-sectional mean of `queue_pressure_forecast`,
`network_pressure_forecast` and `shock_risk`, weighted by each unit's `nominal_capacity`. Rebuilt at
inference from the covariates the input dir supplies — never stored. Worth **+0.00693** on the blend
(SE 0.00047, 14.7 SE, 3/3 windows). The 96 units share a common operational factor; a per-unit model
cannot see it, and this is the channel through which it does.

Three columns, no semantics required, available from day one. See
[W6](#w6--the-budget-went-to-models-not-to-features).

**Amplitude recalibration (gamma = 1.04).** Worth **+0.00222**. WAPE is minimised by the
**conditional median**, and averaging members shrinks predicted dispersion toward each unit's mean —
so a blend is systematically under-dispersed, and under an absolute loss that shrinkage is a cost
rather than the free variance reduction it would be under a squared loss.

**The mechanism was checked from a second direction rather than assumed.** If under-dispersion is
the cause, γ's value must scale inversely with how dispersed a member already is — and it does: the
five-seed bag realises **0.870** of the target's own within-unit spread and gains **+0.0022**, while
a single unshrunk model at **0.947** gains only **+0.0011**. The gain plateaus over γ ∈ [1.02, 1.08]
(1.04 ships) and survives a weight refit. A **per-series** γ is a null — +0.00002 for 96 fitted
parameters — so the over-smoothing is a global property of blending, not a per-unit calibration
error, which is why one scalar captures all of it. Full account at
[W3](#w3--the-blend-was-under-dispersed-and-one-scalar-fixes-it).

### The two imputation surfaces

The scattered ~4.5% NaNs get **different treatment per member**, and this is a measured decision
rather than an oversight:

| member | imputation | effect |
|---|---|---|
| tree | linear interpolation | **+0.0040** (3/3 windows) |
| cascade | the stored per-series median | interpolation *cost* it **−0.0094** |

`predict.py` therefore builds **one covariate frame per member**. Interpolation is available at
inference because it needs no history — it reads the observed neighbours either side of an isolated
NaN, which a bare 336-row future block carries — while the block edges are covered by a train-fitted
median table travelling inside the checkpoint. So inference imputes exactly as training did without
baking any data in. Measured on the real NaN pattern, `interp` reconstructs at 0.2171 against the
median's 0.5060, because real NaN runs are isolated single hours.

### The budget this was built under

Training ran on **Modal's free monthly quota of $30** — 37 hours on an L4 at $0.80/h, or 14 on an
A100-40GB at $2.10/h — on an L4 wherever it fitted and an A100 wherever memory or wall clock
required it, the shipped fit included. That is the constraint every de-scoping decision in this
document refers to, and it is why the selection protocol is built around cheap screens and a
pre-declared stop rule rather than around large sweeps.

### The degradation ladder

Two sets of Chronos-2 weights are fetched at inference. If either is unreachable the run **steps
down a defined ladder rather than failing**, and every rung uses its **own measured weights**:

| rung | when | CV pooled |
|---|---|---|
| **1.** fine-tune + cascade bag + tree | both repos reachable | **0.11807** |
| **2.** cascade bag + tree | the fine-tune is unreachable | 0.13163 |
| **3.** blanked cascade channel + tree | zero-shot Chronos-2 also unreachable | ~0.14 |
| **4.** tree alone | the neural bag fails outright | 0.15200 |

Rung 2 restores exactly the previously-submitted two-member model at the weights it was scored at,
rather than re-spreading rung 1's three weights over two members — which would be a mixture nothing
has ever evaluated. Trading a measured quantity for an invented one is the failure the ladder exists
to avoid.

Rung 3's blanked channel is not an improvised value: it is the stored per-series median with
`chronos2_forecast_missing = 1`, which is the **exact state the model already met during training**,
where the rolled covariate leaves a 288-hour warm-up prefix resolved the same way. The
variable-selection network has seen this input and has learned to distrust the channel there.

The ladder was clean-room verified at **gaps 0, 168, 336 and 672**, and at spans beyond the
checkpoint's own 672 steps — where it rolls the context forward and warns in its log that the
request exceeded the trained horizon rather than silently truncating.

**A degraded rung is announced on stdout and stderr, names the missing member, and states the
accuracy cost.** That is not politeness. Rungs 2–4 emit a perfectly well-formed CSV of the right
shape with plausible values, so nothing downstream can tell that a member went missing — and this
project has already been bitten by exactly that, when a dead revision pin silently dropped a run to
rung 2 and the output looked entirely normal.

---

## 3. Findings

Seven results that are *mechanisms* rather than scores. Ordered by how much weight they deserve.

### W1 — a CV-to-holdout offset is a reproducibility monitor, not just an accuracy gap

**The finding.** The gap between an externally scored holdout and internal CV carries information
that neither number carries alone. Hold the offset up across revisions and it monitors whether the
object you ship is still the object you measured — a property no unit test can assert, because it
is a property of the *relationship between two pipelines* rather than of either one.

**What made that concrete here.** `scripts/submission_cascade.py` is a **second implementation** of
the CV fit — it builds its own `NeuralForecast` rather than calling `registry.build_nf` — and the
two had drifted on three modelling axes:

| axis | CV (`src/eval/cv.py`) | submission (before) |
|---|---|---|
| `local_scaler_type` | robust | **none** (never passed) |
| checkpoint averaging | on | **off** (the config it drove carries no `swa` block) |
| `batch_size` | **16** (`cv.py` halves it) | 32 (only `windows_batch_size` was halved) |

`batch_size` is the one axis this project never swept, which is exactly why it stayed invisible.

**The leaderboard offset is what exposed it, and it is the cleanest diagnostic in the project.**

```
v-final-3    LB 13.066   CV 0.12722   offset +0.344
v-prod-1-0   LB 12.632   CV 0.12029   offset +0.603   <- ~2x the historical value
v-prod-1.1   LB 12.141   CV 0.11807   offset +0.334   <- restored
```

A CV-to-holdout offset that is **stable across revisions** is evidence that the shipped object
matches the measured one. When it nearly doubled, that was not noise — it was a degraded member.

The *level* of the offset is a separate matter and carries no accuracy claim: the two numbers are
computed on different blocks, a different slice of each block, and different regimes. What those
components are, and why the far CV reads below the near leaderboard when the axis alone predicts the
reverse, is set out in
[results.md](results.md#what-the-offset-measures-and-what-it-does-not).
Aligning the two paths restored the offset to within 0.01 of its historical value and moved the
board score by **0.491 points**, of which gamma explains ~0.22 and the alignment ~0.27.

**The methodological point.** A held-out score is not only a measure of accuracy; the *stability of
its offset* against internal CV is a measure of whether the pipeline is reproducible. No test could
have caught this, because **both paths were individually correct** — the defect existed only in the
relationship between them, which is precisely the class of thing unit tests do not see. Watching the
offset is free and would have caught it a revision earlier.

The response is [`configs/tft_chronos_swa_shipped.yaml`](../configs/tft_chronos_swa_shipped.yaml):
the shipped hyperparameters transcribed from `bundle.json`, so the artifact of record exists in
tracked form and cannot drift from a second implementation again.

### W2 — serialised state can be correct in one reader and silently wrong in another

**The finding.** A framework that stores state in one representation and converts it at an API
boundary is safe only for readers that go through that boundary. Any second reader that touches the
stored array directly inherits the other representation, silently, and every downstream consumer
still returns plausible numbers. The guard that generalises is a **round-trip assertion on the
stored artifact**, not a check on either code path.

**What made that concrete here.** Fixing the first axis of W1 — enabling `local_scaler_type` on the
submission path — produced a checkpoint whose *inference* path was wrong in exactly that way.

`NeuralForecast` stores the **scaled** temporal array and inverts it only inside `predict()`. The
inference path reads `dataset.temporal` **directly**. Measured on the resulting checkpoint:

```
y                        mean  0.200   min -3.119     (raw: 9.913 / 0.164)
queue_pressure_forecast  mean  0.068                  (raw: 5.071)
nominal_capacity         exactly 0.000                (a per-series constant: (x−median)/MAD → 0)
```

That frame is the **Chronos cascade context**, the **tree's lag and rolling features**, and the
source of the capacity weights. The largest member — weight 0.4724 — would have been conditioned on
a **zero-crossing target its fine-tune never saw**, and would have returned plausible numbers.

**Only one guard caught it, and by luck of construction.** The capacity-weighted aggregate divides
by the summed capacity weights, all of which were now exactly zero, so `0/0` raised. Chronos and the
tree had no equivalent check and would have failed **silently**. Worth stating plainly: the project's
error detection here was one accidental division, not a designed check.

The fix inverts the stored array through `coreforecast`'s own scaler statistics and is verified by
round-trip: y returns to 9.9128 / 5.5482 / 0.1640 and all 39 scaled columns match to ≤ 4e-6 —
including the 17 with `scale == 0` (a mostly-zero binary has MAD 0), where `is_weekend` recovers two
distinct values at mean 0.2833 against the raw 0.2833. `tests/test_history_unscale.py` fails on the
un-inverted frame.

**CV was never affected**: `src/eval/cv.py` never reads the stored array, and `predict()` inverts,
so CV predictions were always in raw units.

### W3 — the blend was under-dispersed, and one scalar fixes it

Consensus plots showed compression at the extremes: **+9.1% against consensus in the lowest
quintile, −3.4% in the highest**. That is the signature of averaging. Averaging K
imperfectly-correlated forecasts shrinks the average toward the mean, while **WAPE is minimised by
the conditional median**, so any blend is biased against the metric it is scored on.

The correction is one line:

```
pred' = unit_mean + gamma * (pred − unit_mean)        gamma = 1.04
```

**+0.00222, SE 0.00017, 13.1 SE, 3/3 windows**, on a plateau over [1.02, 1.08]. Fitted independently
on our own fit region it returns **1.040 to three decimals**. The anchor is the mean of the model's
*own* prediction over the forecast block, so no labels are read and the step is legal at any gap.

**Per-series gamma is a null** — +0.00002 for 96 extra parameters, spread 0.995–1.105. The
over-smoothing is a **global property of the blending operation**, so one scalar is the right model
rather than a simplification.

**The confirming measurement, from a different direction.** Gamma's gain scales inversely with a
member's dispersion:

| member | dispersion (sd_pred / sd_y) | gamma gain |
|---|---|---|
| 5-seed TFT bag | 0.8701 (an average of five) | +0.00222 |
| robust_es | 0.8878 | +0.00193 |
| noscale_full | 0.9474 (single, unshrunk) | +0.00113 |

**And the uncomfortable corollary: bagging buys member accuracy by destroying what a blend pays
for.** The 5-seed bag is the **best** member (0.12812) and the **worst** ingredient — most correlated
with the other two (0.921 / 0.802) and most under-dispersed (0.870). A single unshrunk model is a
worse forecaster but a better blend component. Ensembling at two levels is not free; the outer level
pays for what the inner level removed.

### W4 — neither large lever transfers, and one variable explains both

```
Chronos-2 cascade covariate:  +11.3%     on ours     −0.5%   on M5     (single seed both sides)
cross-series aggregates:       +9–14%    on ours     −0.7%   on M5
```

Both of the project's largest gains fail to transfer to a second panel.

**The panel and its control** (`results/addl_dataset_metrics.json`). M5: 500 series stratified
across (department, store), split
`1885 | gap 28 | score 28`, single seed, the architecture unmodified apart from frequency and
horizon. The manipulation is isolated by a **negative control** — the covariate-free baselines
return **bit-identical predictions** on the panel built with the cross-series aggregate columns and
the panel built without them. A model that reads no covariates cannot respond to a covariate column,
so identity is the *required* outcome here and a difference would have been the bug. That is what
licenses reading the seven covariate-using arms as a real effect rather than as run-to-run noise.

The architecture does fit that data — a Chronos-2 fine-tuned on M5 itself reaches 0.7148 against
0.7209 zero-shot. What fails to carry over is not the method but **the value of these inputs**.

**The explanatory variable was measured before the transfer run rather than fitted afterwards:**

| panel | known-future covariates | vary **cross-sectionally** |
|---|---|---|
| ours | 13 | **8** (5 are global) |
| M5 | 8 | **1** (2 usable once `snap_own` is constructed from `state_id`) |

Seven of M5's eight known-future covariates are published per *date* and are byte-identical across
series, so their cross-series mean **is a column the model already has**. There is almost no common
factor left to extract, and two extra near-empty features make a tree slightly worse.

**Both levers are properties of *this* panel's structure — covariate-rich, fully observed, strongly
co-moving, 96 units — not general facts about multivariate forecasting.** A write-up showing only
the wins would claim a generality the evidence does not support, and the cross-sectional degeneracy
count is a cheap, pre-registerable statistic that says in advance whether either lever is worth
trying on a new panel.

*(A second additional dataset, ECL, was dropped as contaminated before it produced any number. Its
loader is not in this repository. The contamination detector was written before the run and came
back clean on M5.)*

### W5 — two nulls that looked certain

**Exactly-recoverable NaNs.** Five of 13 signals are byte-identical across all 96 units at every
timestamp — cross-unit spread exactly `0.00e+00`. The NaN mask is per-unit. So 55,967 imputed rows
across three signals are **exactly recoverable from sibling units**, verified on the horizon block
too (84–86 observed units per timestamp minimum, zero all-missing). Recovery is exact, not
approximate.

**And it is worth nothing on the shipped configuration: −0.00014, 1 of 3 windows.** It helps a bare
tree (+0.00249) and vanishes once the cross-series aggregates are present — because an aggregate of
a *globally shared* signal already reconstructs the missing value implicitly. We were already
getting the information; we simply were not getting it exactly. An exactness that no downstream
model can use is not an improvement.

**Removing the Chronos covariate for diversity.** The reasoning was that a no-Chronos TFT would be
decorrelated enough to earn a place as a fourth member. Measured: **−0.00006, 0.4 SE**. Its error
correlation with the Chronos member is 0.8978 against the bag's 0.9211, and **0.9730 with the bag
itself** — dropping a covariate worth 8–11% in accuracy barely decorrelates the model at all. The
channel changes how *well* the model predicts and almost nothing about *where* it errs.

**The target transform, which was the plausible route to a large number.** Instead of handing the
Chronos forecast to the TFT as a covariate, train the TFT on `y − chronos2_forecast` and add the
channel back at inference — the same information, packaged so the network cannot ignore it. Exactly
one config key differs from `tft_cascade`; backbone, frame, horizon, gap fill, arm and seed are
identical, and the control is **the same seed's own fit** (`casc_s892`) rather than the five-seed
bag, so the bag's variance reduction is not charged to the candidate.

**It is worse: −0.00294, CI [−0.00537, −0.00057], 1 of 3 windows** (`results/fp/1g_residual_target.json`).
The blend refit agrees a third way — with both in the pool the residual member takes weight 0.0129,
below the 0.05 floor. **So the gating is the mechanism, not the packaging.** Removing the need for
the variable-selection network to calibrate the channel at all makes the model worse; what it learns
about *when to trust* the Chronos forecast is doing real work that a target transform cannot
reproduce. Read alongside the matched-lead result below, the two bracket the same conclusion from
opposite sides.

**A third, larger negative worth naming here**: the matched-lead cascade, where the covariate's
forecast lead is aligned with the target's, is worse by **0.01248 at 15.4 SE, 0 of 3 windows**
(`results/ab_tft_cascade_matched_vs_tft_cascade.json`). The train/inference mismatch it was built to
remove is load-bearing.

**A learned combiner over the members was tried and lost.** An MLP meta-learner over four members
scored 0.1337 against 0.1247 for the fixed convex blend on a temporal holdout
(`results/ensemble_metalearner.json`) — worse by 0.009. A combiner that cannot beat a reweighting is
not a ship candidate, and it is the reason the blend stayed convex and fixed.

Other lanes that returned measured nulls and are retained here as evidence rather than deleted: a
**selective state-space (Mamba) encoder** replacing the TFT's two LSTM encoders at 98% of their
parameter count (worse by 0.02610, 17.9 SE, CI excluding zero, in both regimes —
`src/models/mamba_tft.py`, `configs/tft_chronos_mamba.yaml`); **cross-variate ("inverted")
attention** inside the TFT at both insertion points (−0.00986 and −0.00356, CIs excluding zero); a
**hyperparameter search** that gave back all of its +7.3% in-window gain on the two unseen windows;
and **quantile heads**, cut before fitting rather than left as an unsupported claim.

### W5b — two predictions recorded in advance, and falsified

Both were written down before the run, which is the only reason they count as results rather than
as post-hoc stories.

**Better gap covariates are worth nothing — including perfect ones.** The prediction was that the
gain would be *monotone in reconstruction quality*: reconstruct the 13 planning signals withheld
across the gap more accurately, forecast better. A zero-shot reconstruction cuts reconstruction
WAPE by **38 %** and buys **zero**. The ceiling arm — handed the **true** withheld covariates —
scores **0.0006 below** the flat-median incumbent, with a CI excluding zero. *Handing the model the
answer made it slightly worse.*

The obvious objection is that the model simply ignores these columns, and it is ruled out rather
than assumed: the **bad** end of the axis does respond (tiling the last observed week is clearly
worse, 0/3), so the columns are demonstrably read. A third arm closes the family — the 55,967
exactly recoverable values of [W5](#w5--two-nulls-that-looked-certain), worth −0.0001.

**The conclusion is a distinction worth carrying:** *reconstruction quality and forecast quality are
not the same axis.* The gap block is never scored, and the model already holds a substitute for it,
so improving a surface nobody grades cannot move a number.

**Feature engineering that duplicates what an architecture already computes is harmful, not merely
useless.** The prediction was that exponentially weighted moving averages of the per-unit planning
signals would pay **most** on Chronos-2 (a tokenising model, assumed to read covariates pointwise)
and **least** on the TFT (whose encoder already walks the sequence). Over three folds:

```
tree      +0.0137
cascade   −0.0005  (0.4 SE)   as predicted
Chronos   −0.0029  (4.0 SE)   OPPOSITE SIGN, and not free
```

A step-budget confound is impossible — both arms ran the identical schedule. What settled it was
reading the loader: `build_train_inputs` passes Chronos-2 the **entire covariate history** as past
covariates, so it, like the TFT, already computes what the feature supplies. The tree, which sees
only origin-anchored lags of the target, is **the only member that cannot see covariate history at
all** — and that single asymmetry predicts all three signs, where the original hypothesis predicted
two and mispredicted the third.

**The replacement hypothesis is better than the original**, and it was reached by inspecting the
data path rather than by fitting an explanation to the result.

### W6 — the budget went to models, not to features

**The honest retrospective.** Every lane opened before the final sprint — Mamba encoder, TFT
hyperparameter sweep, checkpoint averaging, cross-variate TFT, N-HiTS, meta-learner, activation sweep, quantile
head — is an **architecture or ensembling** lever. **Not one was feature construction.** The final
sprint then bought **+0.00693** from a *three-column* feature block, after three model families'
worth of hyperparameter search had returned nothing outside the noise band.

**But the reason is not only poor prioritisation, and both halves belong in the account.** As
[§1](#the-dataset-is-deliberately-anonymised-and-that-has-consequences) notes, the dataset carries no
semantics. Feature engineering is normally driven by domain meaning; here the usual generator of
hypotheses is unavailable and the candidate space collapses to what can be derived mechanically from
the panel's shape.

**That is an explanation, not an excuse, and the evidence says so.** The levers that eventually paid
needed no semantics at all — a capacity-weighted cross-sectional mean, a centred rolling window, an
hour-of-week profile are properties of a **panel**, not of a domain. They were available from day
one. The planning record shows cross-series aggregates being cut **without ever being measured**, by
analogy to a different family that genuinely was null. **A test never run was scored as if it had
failed** — a reasoning failure, not a data-vagueness failure.

**The transferable lesson.** When the data carries no semantics, the correct response is to
enumerate the structure-driven feature families **exhaustively and cheaply** (the tree screens one in
about 30 seconds) *before* spending GPU on architecture — precisely **because** the domain prior is
missing, not despite it. Vagueness raises the value of a cheap systematic screen; it does not lower
it.

The capacity probes make the same point from the hardware side: widening the cascade's TFT to
`hidden_size 128`, and BiTCN at 1× and 2× the TFT's width, were all negative or null. And
[`results.md`](results.md#2a-scored-on-the-graded-regime-13) shows a 28.9M-parameter N-HiTS losing to
a 20,736-parameter LSTM by a factor of 2.4. Capacity was never the binding constraint.

---

### W7 — how much predictive value is left, and how you measure that

Lesson 2 below says *read the errors before optimising further*. This is that reading, done late,
with the answer it produced. Both artifacts are in `results/fp/`.

#### How you find out whether signal remains

There is a direct test. **Fit a model on the residual.** If `y - blend` is pure noise, no learner
can beat predicting zero out of sample. If a learner *can* beat zero, reliably, then whatever it
found is structure the blend failed to capture.

**What that establishes, exactly.** A positive result is real evidence of exploitable structure — a
**lower** bound, and only *conditional on a model class, a feature set and an instrument*. A
negative result is much weaker: "a gradient-boosted tree on these 34 features found nothing" is not
"there is nothing". This test tells you whether to keep looking; it does not tell you how much is
there.

It is also a **complement to the classical residual diagnostics, not a replacement** — see
["the residual checklist we did not run"](#the-residual-checklist-we-did-not-run) below, which is
the part of this that was skipped.

Two caveats make the difference between an answer and an artefact:

1. **The residual model must never see, in training, information it would not have at deployment.**
   A stacker fitted and scored inside the same block will find "signal" that is memorisation of
   rows adjacent to the ones it is scored on. Here that is handled by **forward chaining with a
   336-hour embargo**: fold *k* trains only on data ending before the gap preceding its own scored
   block. The three folds train on 32,256 / 96,768 / 161,280 rows respectively, and the training
   universe extends across the near block as well as the far one — which matters, because the
   earliest fold has no far-regime training rows at all and its delta is the smallest of the three
   (+0.00053 against +0.00245 for the latest).
2. **Absolute-time features can manufacture a gain.** `trend` and `_blk` were the two largest
   contributors by split gain, which is exactly what a leaking stacker would look like. So the
   hypothesis was **tested rather than argued**: a second arm with both features ablated. The gain
   survives (+0.00447 on the standing split against +0.00400 with them), so the leakage account is
   **rejected** — the structure is in the covariates, not in the clock.

#### The measurement

A LightGBM (`objective: l1`, 400 rounds, `num_leaves 31`, `min_data_in_leaf 200`) on `y - blend`
against 34 features: the calendar block, the 19 known-future covariates, their 10 missingness
indicators, the statics, `_spread` (the disagreement between the three members) and `_blk`. The
correction is applied shrunk, `y_hat = blend + gamma * residual_hat`, with `gamma` selected on an
inner split from a seven-point grid.

**The number depends on the instrument, and it depends on it monotonically:**

| instrument | delta |
|---|---|
| the standing `blk < 224 / blk >= 224` split | +0.00400 |
| leave-one-window-out | +0.00279 |
| forward chaining, far rows only | +0.00265 |
| **forward chaining + embargo, near and far — the deployment-faithful one** | **+0.00113** |

**That monotone decay is itself the finding.** Every step from top to bottom removes a way for the
stacker to see something it would not have at inference, and every step costs it accuracy. A single
instrument would have reported a number 3.5x too large, and reported it with a CI excluding zero.

#### The answer

**Under the honest instrument: +0.00113, CI [+0.00086, +0.00139], 3/3 windows** — roughly 1 % of
the base it was measured against. It clears all four admission criteria.

#### The caveat that matters most, and it is about the baseline

**This gain was never measured against the model that shipped.** The base blends in the artifact
are 0.12722 (standing split) and 0.13162 (forward chaining). The shipped blend is **0.11807**. So:

```
best stacked output, most optimistic instrument   0.12322
the model that actually shipped                   0.11807
```

**The residual-corrected model never beat the shipped system in absolute terms.** It improved an
*earlier* blend — one without the cross-series aggregates (+0.00693) and without the amplitude
recalibration (+0.00222). Those two are **panel-level corrections acting on exactly the kind of
structure a tree over cross-sectional covariates would find**, so overlap is the expected case
rather than a risk. The headroom against the shipped model is very likely smaller than +0.00113,
and it was never measured. Read the number as *what was left in the blend as of that lane*, not as
*what is left today*.

*(The artifact is internally inconsistent on one point: its `members` field lists `casc_bag5`, while
its summary says the base used `casc_s892`. That should be resolved before either figure is quoted
with confidence.)*

#### Why it was not shipped

Three reasons, in order of weight:

1. **The baseline problem above.** Admitting a component on a gain measured against a different,
   weaker blend would break this project's own rule that every rung carries its own measured
   weights.
2. **A residual model that helps is evidence of signal; it is not automatically the way to capture
   it.** If the structure is real, feeding those features to the *members* — where they interact
   with everything else — usually beats bolting on a post-hoc corrector. And note the stacker's
   second-largest feature by gain is `_spread`, the disagreement between members: that information
   only exists *after* the members run. It is not signal a better model could have used. It is
   signal only a stacker can use, which is a narrower and less interesting thing.
3. **Schedule.** A fourth ladder rung, a rebuilt archive and two more clean-room runs against a
   hard stop. Recorded as a schedule decision rather than dressed up as a modelling one.

What the CI is conditional on: two feature sets x three instruments were searched.

#### The residual checklist we did not run

The classical diagnostics were **skipped**, and their absence is a real hole rather than a
deliberate omission. They are minutes of CPU, they are interpretable in a way a stacker is not, and
they answer a narrower question the stacker's 34-dimensional search can hide. Recorded here as the
checklist a next project should run **first**:

| check | what it would catch | status |
|---|---|---|
| **Residual *median* ≈ 0**, per unit and per lead band | point-forecast bias | **not run** |
| **ACF / Ljung–Box at lags 1, 24, 168** | leftover daily or weekly structure — the classical "is it white noise" test | **not run** |
| **Residual spread vs fitted level** | heteroscedasticity, and whether γ should be level-dependent rather than one global scalar | **not run** |
| **Residual vs each covariate** | a covariate the members under-use; the interpretable complement to the stacker | **not run** |
| Error concentration, ranking stability, lead-band ordering, statics R² | *where* the error is | run — below |

**Two traps in that list, and the first is why "check the residuals are zero-mean Gaussian noise"
is the wrong instruction here.**

**Mean-zero is the wrong statistic under this loss.** WAPE is minimised by the conditional
**median**, not the mean. The target is strictly positive and right-skewed (0.16–53), so a
*correctly specified* median-optimal forecast leaves residuals whose **mean is above zero** — by
construction. A zero-mean test would flag the right model as biased. The property to check is that
the residual **median** is near zero.

**Normality is not a target at all.** It buys parametric inference — Gaussian standard errors and
prediction intervals. Every interval in this project comes from a
[paired cluster bootstrap](protocol.md#2-the-admission-bar), which is distribution-free precisely so
that assumption is not needed, and the deliverable is a point forecast rather than an interval.
Non-normal residuals here are not a defect.

**What the classical checks would have added.** An ACF sees the residual's own past — one dimension.
The stacker sees 34, including covariates, cross-series state and interactions, so it strictly
dominates on coverage. But it returns *one number*, and a number cannot tell you that the leftover
structure is a 168-hour echo. Two of the four rows above were in effect answered sideways — γ is a
dispersion calibration, and the lead-band decomposition below is a residual-structure check — but
neither was framed or reported as a residual diagnostic, which is how they came to be missing from
the record.

#### The structural half of the answer

A residual model tells you *how much* is left. `results/fp/1a_error_map.json` tells you **where it
is not**, which is what stops you spending a week on the wrong lever. Four measurements, about two
minutes of CPU:

| question | measurement | consequence |
|---|---|---|
| Is the error concentrated in a few units? | top 10 units own **15.7 %** of pooled error (11.1 % of the volume); top 20 own 28.5 % | **diffuse** — against a 40 % threshold set in advance. Per-unit blend weights have almost nothing to exploit, and a fitted version later kept ~3 % of its own oracle ceiling |
| Does the member ranking move across windows? | Kendall tau **+0.775** | stable — the three-window CIs read as intended |
| Does it move with forecast lead? | tau(first band, last band) **+0.926** over 6 x 56 h bands, **order unchanged** | **flat in lead.** Lead-conditioned weighting is dead — later confirmed independently when a shrinkage fit chose lambda = 0.98 and moved the number by -0.00000 |
| Do the statics explain per-unit error? | log(unit WAPE) ~ `nominal_capacity + zone_sin + zone_cos`: **R2 = 0.007** | the three statics the neural members see explain essentially nothing; a learned unit embedding would have something to learn |

The lead-band result is the sharpest: the error *level* rises and falls across lead (band 0, lead
337-392, is hardest for every member at 0.1508 blended) but the member **order** never changes.
Lead-time conditioning can only pay if *which member to trust* changes with lead, and it does not.
Two planned lanes were closed on this evidence before either was run.

*(Scope note: `1a_error_map.json` also carries per-window WAPE for 20 members on the final-push
member frame. Those are a different object from the arc in [`results.md`](results.md#2c-the-arc-on-one-instrument)
and should not be read across to it; this document cites the file only for the four decomposition
statistics above.)*

## 4. Three lessons, and what this work does not establish

### The lessons

**1. Measure the condition you will ship, not a proxy for it.** Five times in this project a
conclusion drawn from indirect evidence was contradicted by the direct measurement. The most
expensive: the cross-series-aggregate family was set aside early *by analogy* to a different family
that genuinely was null — so a test that was never run entered the record as though it had failed,
and it was later worth **+0.00693**. The near/far axis is the same failure in a different costume:
the leaderboard is a proxy, the far regime is the condition, and only one of them is graded.

**2. Read the errors before optimising further.** The question worth asking early is whether the
residuals still hold structure a better architecture could reach, or whether the model class is at
its practical optimum and the remaining signal sits in the inputs. **We did not ask it, and kept
tuning.** When it was finally asked, the answer was the second: a tree with access to every
covariate extracts only **+0.00113** from the residual under a deployment-faithful instrument — and
that against an earlier blend, not the one that shipped — while the error map says the remaining
error is diffuse across units, flat in lead, and unexplained by the statics. An error map costs about two minutes of CPU and repriced three lanes before any of
them ran. Full method and numbers at
[W7](#w7--how-much-predictive-value-is-left-and-how-you-measure-that).

**The concrete, portable version of this lesson** is the checklist at
[the residual checklist we did not run](#the-residual-checklist-we-did-not-run): residual *median*
per unit and per lead band, ACF at lags 1 / 24 / 168, residual spread against fitted level, and
residual against each covariate. All four are minutes of CPU and all four were skipped here. Run
them **before** the model search, not after it — and note that the instruction most people carry,
*"check the residuals are zero-mean Gaussian noise"*, is wrong for a WAPE objective: the optimum is
the conditional **median**, so a correctly specified model on a right-skewed positive target leaves
residuals with a mean above zero, and normality is not required by anything in the pipeline.

**3. Leave real time for the data.** Feature construction is a data-mining concern rather than
something an architectures course sets out to teach, which is part of why it was not planned for.
The honest reason it came late is that the data is synthetic and anonymised: no column is explained,
so the usual generator of feature hypotheses — understanding what a variable *is* — was unavailable,
and architecture search was substituted for it. But the lever that eventually paid needs **no
semantics at all**: a capacity-weighted cross-sectional mean is a property of a panel, not of a
domain. **Opacity raises the value of a cheap systematic screen over the input space rather than
lowering it.**

### Limitations

Stated plainly, because the findings above are worth only as much as their boundaries.

* **The transfer result is one corpus at one seed.** M5 is a single panel and a single draw. It can
  separate "these levers are general" from "these levers are not", but it cannot separate the
  cross-sectional-degeneracy explanation from the many other differences between two datasets. It is
  carried here as current thinking, not as demonstration.
* **The fine-tuned member is single-seed.** A seed costs 456 MB of hosted weights against a
  cross-seed σ of 0.0023. The draw that shipped is the worst of the three that were run, which
  bounds the risk in the favourable direction but does not remove it.
* **The two effects of the train/inference lead mismatch cannot be separated** within the designs
  available here. Matching the lead *requires* longer-lead — therefore worse — forecasts, so the
  matched arm hands the variable-selection network a uniformly poorer channel. The confound is
  intrinsic, and the claim is bounded to the matched designs that were reachable.
* **The imputation asymmetry has a mechanism that remains untested.** Interpolation helps the tree
  and costs the cascade; the proposed explanation — that better in-sample covariates lead the network
  to down-gate the Chronos channel — needs the variable-selection gate weights, which are overwritten
  by every forward pass and would have had to ride a training run. The adopted split rests on the
  measured −0.0094, not on the explanation.
* **Every internal number is self-scored.** Only the leaderboard rows were scored by anyone else,
  and they measure the near regime, which is not the graded one.
* **Selection was run against seed 42 and confirmed on a frozen five-seed set.** Cross-seed σ for the
  TFT cascade is ≈0.0055 per window — twice the paired bootstrap SE — so two differently seeded
  configurations under ≈0.005 apart are not resolvable at one seed. No paired comparison here is
  affected, since both sides of every A/B carry the same draw.

---

## 5. Decision ledger

One line per phase, in the order they ran. Every entry is a decision that was made and can be
argued with. The numbering is this document's own — the working repository used internal
sprint codes, which meant nothing outside it.

| # | phase | headline |
|---|---|---|
| **1** | **Evaluation protocol** | The gapped CV protocol, the admission bar, and the cascade-leak arc: found leaky, fenced, measured honest (the leak was worth 0.0018 of a 0.0209 gap). |
| **2** | **The standing rule** | The rule that shaped everything after it: **a model fitted on our data enters only as a blend member, never as a covariate.** |
| **3** | **Tree squeeze** | LightGBM squeezed; tree frozen at `lgbm_s24_unitcat` (+4.2%, 0.16281 → 0.15600, 3/3). Tuning bought accuracy and spent diversity, so the blend did not pay. |
| **4** | **Foundation-model screen** | Seven foundation candidates screened as members *and* as covariates, four cascade variants trained, **nothing admitted**. Cluster structure is set by the inputs. |
| **5** | **Imputation** | Imputation is two surfaces. `interp` adopted for the **tree only** (+0.00401, 3/3); it costs the cascade −0.00937. The gap fill is a null — even *real* covariates lose. |
| **6** | **Matched-lead cascade** | Matched-lead cascade: a decisive **loss** (−0.01455, 0/3). The train/inference mismatch is load-bearing. The control came back bit-identical. |
| **7** | **Hyperparameter search** | The TPE hyperparameter search: a **null** — all of the search window's +7.3% given back on the two unseen windows. Backbone cross-seed σ ≈ 0.0055. WAPE-aligned loss measured and rejected. |
| **8** | **Seed bagging** | 5-seed cascade bagging worth **+0.00355** for zero extra GPU. The convex surface is flat at its optimum. Checkpoints must be persisted, never rebuilt — the same seed on a different card is as far away as a fresh seed. |
| **9** | **First ship-path dry run** | Ship-path dry run and the first submission containing LightGBM. Established the near/far rule. |
| **10** | **Quantile head** | **CUT.** The quantile claim is retired rather than left standing — no quantile head was ever fitted. |
| **11** | **Transfer to a second panel** | M5; ECL dropped as contaminated. **The cascade does not transfer**: +11.3% here, −0.5% there, single-seed both sides. Contamination detector written in advance, comes back clean. → [W4](#w4--neither-large-lever-transfers-and-one-variable-explains-both) |
| **12** | **The artifact** | Three defects found in it, including **the bag never being averaged** — one seed of five, 2.3 WAPE points, and a schema-perfect CSV. Ladder, offline arm, clean room. |
| **13** | **Full fine-tune** | The full fine-tune this project had not run: best single model, admitted as a third member, 0.13163 → **0.12722**. Width probes negative. A seed gate found `--seed` never reached the trainer. |
| **14** | **Fine-tuning the covariate models** | No package ships a trainer for them. Zero-shot they all earn weight 0.000. |
| **15** | **Mamba encoder** | The TFT's two LSTM encoders replaced by a **selective state-space (Mamba)** stack at 98% of their parameter count. **Worse by 0.02610** (17.9 SE, CI excluding zero), in both regimes, after early-stopping at ~849 of 5000 steps. The recurrence is not the binding constraint. → [W5](#w5--two-nulls-that-looked-certain) |
| **16** | **Cross-variate attention** | "Inverted" attention inside the TFT. Both insertion points are worse than the same network without them (−0.00986 and −0.00356, CIs excluding zero); the multivariate wrapper costs a further 0.0114. |
| **17** | **Checkpoint averaging (SWA)** | Lightning's callback cannot run under neuralforecast (epoch-based; `max_steps` leaves `max_epochs = None`). Run 1 invalidated by a fixed-step window confounded with run length; refit with a fraction. +0.00156 at member level (4.37 SE, 5/5 seeds), +0.00047 in blend — 3 of 4 criteria, so it was **deferred**. It was subsequently **shipped** as part of `v-prod-1.1`, when aligning the submission path to the CV path ([W1](#w1--a-cv-to-holdout-offset-is-a-reproducibility-monitor-not-just-an-accuracy-gap)) made the CV member and the shipped object the same thing. |
| **18** | **Representation sprint** | Opened late, and only at this stage: every earlier lane had attacked the *model*, so input representation was the one large untried axis left. Primary arm: cross-series aggregates, 12 strategies, **zero of them previously present in the codebase**. → [W6](#w6--the-budget-went-to-models-not-to-features) |
