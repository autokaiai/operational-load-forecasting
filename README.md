# operational-load-forecasting

[![CI](https://github.com/autokaiai/operational-load-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/autokaiai/operational-load-forecasting/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-docs)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**336-hour-ahead forecasting of an "operational load index" across 96 hourly series with 22
covariates, graded by WAPE.** History ends at hour *T*; the scored block is *T+337 … T+672*. The 336
hours in between are never observed.

**The result this repository is organised around:** a zero-shot foundation model is *worse* than the
supervised model it advises — yet handing its forecast to that model as a known-future covariate is
worth **+8.1 %** seed-averaged, the largest single gain measured here. The same model appears in
three roles and gives a different answer in each.

**What shipped is larger than that finding.** A three-member convex blend — a fully fine-tuned
Chronos-2, a five-seed checkpoint-averaged TFT conditioned on the covariate above, and a LightGBM
booster — followed by one amplitude constant, behind a frozen CLI that degrades down four measured
rungs when hosted weights or a GPU are unavailable, and verified in a clean room against the
submission contract.

**Released with its evidence:** 23 architectures screened, 11 A/Bs under a pre-declared admission
bar, 7 clean-room runs against the submission contract, an error decomposition, a residual stacker,
a transfer panel on a second dataset — and the measured negatives kept rather than deleted. 488
tests, none of which needs a GPU, a network or the checkpoint. A course project at TU Darmstadt
(Deep Learning: Architectures & Methods, SoSe 2026).

---

## The bar the task set

The brief names two baselines to beat — the naive last value as the minimum, the seasonal mean as
the real bar — and the course publishes all four on the leaderboard. These and our submission are
scored on the public validation split by the same scorer, so this is the one table here that does
not rest on our own evaluation code — ours from the shipped model, fitted on all 4,320 labelled
hours. Lower is better.

| | WAPE | |
|---|---|---|
| naive last value | 48.098 | **the minimum to clear**, named in the brief |
| lag-24 repeat | 46.820 | |
| lag-168 repeat | 46.699 | |
| seasonal mean | 34.409 | **the stronger bar**, named in the brief |
| **this system** | **12.141** | **74.8 % below naive · 64.7 % below the seasonal mean** |

---

## The progression

Every row below is **the same 32,256 scored rows** — the `blk >= 224` block of three gapped CV
windows. Read top to bottom, it is the project's arc. Row-by-row provenance, including which figures
are transcribed rather than re-derived from a tracked artifact, is in
[`docs/results.md`](docs/results.md#2c-the-arc-on-one-instrument).

> **Units.** Every internal figure in this repository is WAPE as a **fraction** — `0.118` = 11.8 % —
> which is how the artifacts in [`results/`](results) and inside the checkpoint store it. The
> leaderboard above reports the same quantity as a percentage. Deltas quoted in the text are on the
> fraction scale too, so `0.026` is **2.6 WAPE points**, not 0.026 of one.

| | far WAPE |
|---|---|
| naive last value | 0.4592 |
| seasonal mean — the course's stronger required baseline | 0.3173 |
| ridge on the 29 known-future channels — how much the covariates carry alone | 0.1724 |
| TFT, best of 23 screened architectures | 0.15138 |
| ↳ + a frozen zero-shot Chronos-2 forecast as a known-future covariate | 0.13429 |
| ↳ + checkpoint averaging and a 5-seed bag | 0.13334 |
| ↳ + a fully fine-tuned Chronos-2 as a third member, at fitted convex weights | 0.12722 |
| ↳ + cross-series aggregates | 0.12029 |
| **↳ + amplitude recalibration — shipped** | **0.11807** |

The single largest step in the table is the plain TFT to the cascade: **+8.1 % relative
seed-averaged** from one extra input column (the best single draw shown, 0.13429, is ≈1.7σ lucky
against its seed-mean of 0.1391 ± 0.0028 — which is why a five-seed bag ships). Even at the averaged
figure it beats any architecture substitution and the entire hyperparameter search. The two cheapest
levers in the project — that column and the cross-series aggregates — arrived last.

**Public leaderboard: 12.141** — the externally scored number, in the near regime. Progression across
all seven submissions, and what its offset against internal CV diagnosed, in
[`docs/results.md`](docs/results.md#4-leaderboard-progression).

---

## The system

![Top: the graded split — history to a cutoff, a 336-hour gap with no covariates, then the 336-hour
scored block whose first 224 hours fit the blend weights and whose last 112 are scored. Below: the
input dir and checkpoint feed cross-series aggregates and two per-member covariate frames, which
feed three models in parallel; both Chronos-2 weight sets come from Hugging Face at pinned
revisions, the same model in two roles. The convex blend is rescaled by one amplitude constant and
written as CSV, with a four-rung degradation ladder on the right.](docs/architecture.svg)

```
0.4724 * Chronos-2, fully fine-tuned on this dataset (not LoRA)
0.3413 * TFT conditioned on a zero-shot Chronos-2 forecast, 5-seed checkpoint-averaged bag
0.1863 * LightGBM, direct multi-horizon, L1 objective
------
       then  pred' = unit_mean + 1.04 * (pred - unit_mean),  clipped at 0
```

The weights are a convex simplex fit on rows they are not scored on. They travel inside the
checkpoint; [`weights.json`](weights.json) is the tracked record of them, and
[`configs/tft_chronos_swa_shipped.yaml`](configs/tft_chronos_swa_shipped.yaml) is the training
config transcribed from the graded artifact rather than reconstructed.

Member-by-member detail: [`docs/method.md`](docs/method.md#2-the-shipped-system-member-by-member).

---

## Four things here worth reading

**1. The same foundation model in three roles gives three different answers — and the ordering is
the finding.** Zero-shot Chronos-2 as a *forecaster* scores 0.1858, worse than the supervised TFT's
0.15138. That same forecast handed to that same TFT as a *known-future covariate* is worth
**+8.1 %** seed-averaged, the largest single step in the project. Fine-tuned as a *blend member* it
earns the largest weight of the three. The role in which it is weakest as a model is the role in which it is
worth most, because what a covariate is worth is its complementarity, not its accuracy.
→ [the three roles](docs/method.md#the-same-model-in-three-roles)

**2. The two cheapest levers arrived last, and that is the honest retrospective.** 23 architectures
screened, a 32-trial search over nine dimensions, a Mamba encoder, cross-variate attention, a second
boosting library — and what actually moved the number was a three-column cross-sectional feature
(+0.00693) and one scalar (+0.00222), both available on day one. Every null is kept here as
evidence.
→ [W6](docs/method.md#w6--the-budget-went-to-models-not-to-features) ·
[the three lessons](docs/method.md#4-three-lessons-and-what-this-work-does-not-establish)

**3. Two predictions were recorded in advance and then falsified — including by handing the model
the answer.** The ceiling arm, given the *true* covariates withheld across the gap, scores 0.0006
**below** the incumbent with a CI excluding zero; a 38 % better reconstruction buys nothing.
Reconstruction quality and forecast quality are not the same axis. And a feature predicted to help
Chronos-2 most hurt it at 4.0 SE — the replacement explanation, found by reading the loader, predicts
all three signs where the original predicted two.
→ [W5b](docs/method.md#w5b--two-predictions-recorded-in-advance-and-falsified)

**4. A CV-to-holdout offset is a reproducibility monitor, not just an accuracy gap.** The submission
path and the CV path were two implementations of one fit and had silently drifted on three axes —
one of them `batch_size`, the single axis never swept. No unit test could catch it, because *both
paths were individually correct*; the defect lived only in the relationship between them. What
exposed it was the offset nearly doubling from +0.344 to +0.603 across one revision. Watching it is
free.
→ [W1](docs/method.md#w1--a-cv-to-holdout-offset-is-a-reproducibility-monitor-not-just-an-accuracy-gap)

**And an answer to the question the retrospective raises.** *Is there predictive value left in the
residual?* A tree with access to every covariate extracts **+0.00113** from it under forward
chaining with a 336-hour embargo — against **+0.00400** for the same measurement under a single
in-block split. **The instrument matters more than the number**, and the leakage account was tested
and rejected rather than argued. The honest reading is narrower still: that gain is measured against
an *earlier* blend, and the stacker's best absolute output (0.12322) is worse than the model that
shipped (0.11807).
→ [W7](docs/method.md#w7--how-much-predictive-value-is-left-and-how-you-measure-that)

**And the apparatus that makes all of it believable:** a gapped CV design that mirrors the private
split, and an executable admission bar — a paired cluster bootstrap over 288 unit×window blocks with
four criteria declared before each run.
→ [`docs/protocol.md`](docs/protocol.md)

---

## What did not work

Kept in this repository as evidence, not deleted. A results-only repo would discard the more useful
half of what was measured.

| lever | result | |
|---|---|---|
| Selective state-space (Mamba) encoder replacing the TFT's LSTM encoders | **worse by 0.02610** (17.9 SE, CI excluding zero), both regimes | [W5](docs/method.md#w5--two-nulls-that-looked-certain) |
| Cross-variate ("inverted") attention inside the TFT | worse at both insertion points (−0.00986, −0.00356) | [ledger 16](docs/method.md#5-decision-ledger) |
| Matched-lead cascade — aligning the covariate's forecast lead with the target's | **worse by 0.01248** at 15.4 SE, 0/3 windows | [results §3](docs/results.md#3-the-ab-record) |
| Exact NaN recovery from sibling units (55,967 rows, recovery is *exact*) | **−0.00014**, 1/3 windows — the aggregates already carried it | [W5](docs/method.md#w5--two-nulls-that-looked-certain) |
| TPE hyperparameter search over the cascade's TFT | **null** — all of the +7.3% in-window gain given back on the two unseen windows | [ledger 7](docs/method.md#5-decision-ledger) |
| Transfer of the two largest levers to a second panel | −0.5% and −0.7% | [W4](docs/method.md#w4--neither-large-lever-transfers-and-one-variable-explains-both) |
| Quantile head | cut before fitting — the claim is retired rather than left standing | [ledger 10](docs/method.md#5-decision-ledger) |
| Residual-target cascade — the same information, packaged as a target transform | **worse by 0.00294**, 1/3 windows; takes weight 0.0129 in a refit | [W5](docs/method.md#w5--two-nulls-that-looked-certain) |
| MLP meta-learner over the members, instead of a fixed convex blend | 0.1337 vs **0.1247** — worse by 0.009 on a temporal holdout | [W5](docs/method.md#w5--two-nulls-that-looked-certain) |

And the retrospective the evidence forces:
[**the budget went to models, not to features**](docs/method.md#w6--the-budget-went-to-models-not-to-features).
Every lane opened before the final sprint was an architecture or ensembling lever; not one was
feature construction. The final sprint then bought +0.00693 from a *three-column* feature block.

---

## Run it

```bash
pip install -r requirements-inference.txt
./scripts/verify_checkpoint.sh           # optional: confirms ./checkpoint.pt is the graded artifact

python predict.py \
    --input_dir   /data/input \
    --output_file /output/predictions.csv \
    --checkpoint  checkpoint.pt
```

**`checkpoint.pt` ships with this repository** — 55 MB, sha256
`13e2f2d8…8e0d295`, the graded artifact byte for byte. It holds five TFT weight files, the LightGBM
booster, the per-series history the models condition on, and `bundle.json`, which carries the blend
weights that `predict.py` deliberately does not hardcode. The two Chronos-2 weight sets are the only
things fetched at run time, from public Hugging Face repos at pinned revisions.

`--input_dir` is read, never written. It needs a `forecast_index_*.csv` (the authoritative
`(series_id, timestamp)` rows) and a `*_input.csv` (the known-future covariates). **No target
history is needed** — the model's conditioning window travels inside the checkpoint. Output is
exactly `series_id,timestamp,prediction`, one row per index row, in the index's own order, clipped
at 0 and never NaN.

**The horizon is derived at runtime**, never hardcoded: `gap = first_forecast_hour −
last_observed_hour − 1`, `h = gap + span`. The same checkpoint serves the gap-0 and gap-336 shapes.

**A CUDA GPU is needed for the full model.** `neuralforecast` raises rather than falling back to
CPU, so without one every rung carrying the neural bag drops out and the run degrades to rung 4 (the
tree alone, 0.1520) — announced loudly, never silent. Peak GPU, measured in the clean room:

| rung | peak GPU | wall clock |
|---|---|---|
| 1 — all three members | **7,684 MiB** | 169 s |
| 2 — fine-tune unreachable | 5,722 MiB | 109 s |
| 3 — cascade channel blanked (offline) | 2,164 MiB | 59 s |

A degraded run is not only less accurate, it is visibly cheaper — a second, independent signal that
something was skipped.

**No network?** The run still writes a complete, valid CSV. Both Chronos-2 fetches degrade down a
[defined ladder](docs/method.md#the-degradation-ladder) rather than raising. `DISABLE_CASCADE=1` or
`DISABLE_FULLFT=1` exercises a lower rung deliberately.

---

## Reproduce it

Data: `train.csv` from
[`AIML-TUDA/dlam-ts-project-data-2026`](https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026)
into `data/raw/`.

```bash
pip install -r requirements.txt

# 1. the cascade covariate — zero-shot Chronos-2, rolled backward in 336h blocks so every block's
#    context ends before the block it forecasts. Writes a provenance sidecar the member runner
#    verifies before training (a leaky covariate scores BETTER, so it fails closed).
python -m src.models.chronos2_oof --train-region --out data/derived/train_chronos.csv

# 2. the neural member — five TFT backbones on all 4320 labelled hours, at the frozen seeds.
python -m scripts.submission_cascade --seeds 5 \
    --train-frame   data/derived/train_chronos.csv \
    --horizon-frame data/derived/validation_input_chronos.csv \
    --save-checkpoint checkpoints/cascade_bag5.pt

# 3. the tree member
python -m scripts.fit_submission_tree --out checkpoints/submission_tree.json

# 4. the fine-tuned Chronos member — a FULL fine-tune, so the artifact is the whole 455.8 MB model
#    rather than a 4.6 MB adapter. ~13 min on one L4, peak 13.7 GB VRAM.
python -m src.models.chronos2_finetune --config configs/chronos2_fullft.yaml \
    --cut-idx 4320 --no-eval --output-dir checkpoints/chronos2_fullft_sub

# 5. fold everything into one checkpoint.pt and assemble the archive
python -m scripts.make_submission --smoke
```

Step 2 needs ~24 GB of GPU (the TFT at h=672 peaks near 20 GiB in training); steps 1 and 3 are
minutes; step 4 is CPU-only.

**The saved weights are the artifact, and they are not rebuildable.** The same seed on a different
GPU model produces a materially different fit — as far apart as a fresh seed — so the bag is *moved*
into the archive rather than regenerated.

To re-run the selection protocol itself:

```bash
python -m scripts.member_admission --candidate <member> --baseline <member>   # the bar, executable
python -m scripts.ensemble_full_sweep --out results/sweep.json                # the weight search
```

---

## Repo map

```
predict.py                    the frozen inference contract — CLI and output schema do not change
checkpoint.pt                 the graded artifact, 55 MB — 5 TFT weights, the booster, history, bundle.json
weights.json                  the shipped blend, in tracked form (predict.py reads the checkpoint)
configs/                      41 experiment configs, incl. tft_chronos_swa_shipped.yaml (the artifact of record)
src/                          39 modules
  data/                       loading, calendar closed forms, imputation, gap fill, cross-series blocks
  eval/                       protocol.py — the gapped CV and the admission bar; splits.py, cv.py
  models/                     registry, members, the three shipped members, and the measured nulls
scripts/                      the reproduction path and the executable admission bar
tests/                        36 files — no GPU, no network, no checkpoint required
results/                      the curated evidence: 23 architecture runs, 11 A/Bs, the clean room,
                              the error map, the residual stacker, the M5 panel
docs/                         method.md · protocol.md · results.md · references.md · architecture.svg
```

Where to start reading: [`docs/method.md`](docs/method.md) if you care what happened,
[`docs/protocol.md`](docs/protocol.md) if you care how the numbers were decided,
[`docs/results.md`](docs/results.md) for every number re-derived from `results/`,
[`docs/references.md`](docs/references.md) for the work this builds on, and
[`src/eval/protocol.py`](src/eval/protocol.py) if you would rather read the code.

## Tests

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # skip the ~3 GB CUDA wheels
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q
```

**No test needs a GPU, a network, or the shipped checkpoint** — the neural tests force
`accelerator: "cpu"`, the one `resolve_source` call asserts it returns `None` offline, and the CLI
test trains a tiny model and writes its own checkpoint. The handful that need the dataset skip
cleanly, because `data/` is not in git. CI runs the whole suite on a stock runner.

---

## Provenance and credit

This repository is a **curated snapshot** of a four-person course project, cut from the state that
produced the graded submission. It carries the code and the evidence, not the working history.

**Kai Wöllstein designed the system, led architecture and modelling, and set the technical
direction.**

The graded project was submitted by a group of four — Kai Wöllstein, Jenny Kraft, Herai Hench,
Fedir Tykhonov — who are named in [`CITATION.cff`](CITATION.cff) and hold joint copyright in the
underlying work.

**AI assistance.** The code and documentation here were produced with AI assistance, as declared in
the course submission. Every design decision, every measurement and every claim in this repository
was directed and reviewed by Kai Wöllstein, who is answerable for them.

## Licence

| | |
|---|---|
| Code | **MIT** — [`LICENSE`](LICENSE) |
| `docs/` and this README | **CC BY 4.0** — [`LICENSE-docs`](LICENSE-docs) |

Copyright in both is held jointly by Kai Wöllstein, Jenny Kraft, Herai Hench and Fedir Tykhonov.

**Not ours to license**, and used under their own terms: the dataset
([`AIML-TUDA/dlam-ts-project-data-2026`](https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026)),
[`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2), and the fine-tuned weights derived
from it.
