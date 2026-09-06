# References

Work this project builds on, screens against, or measures. Ordered as they are cited across
[`method.md`](method.md), [`protocol.md`](protocol.md) and [`results.md`](results.md).

## The supervised member, and how it is trained

The shipped blend has no single backbone: its largest weight (0.4724) goes to a fine-tuned
foundation model and the second (0.3413) to the TFT below. The two sections that follow are peers,
not a main model and an accessory.

* **[TFT]** B. Lim, S. Ö. Arık, N. Loeff, T. Pfister. *Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting.* International Journal of Forecasting 37(4), 2021.
  [arXiv:1912.09363](https://arxiv.org/abs/1912.09363) — the architecture the **cascade member**
  (blend weight 0.3413) is built on, and the best of the 23 screened. Chosen for the
  variable-selection network and the explicit known-future channel, which is what a task with 19
  known-future covariates is asking for.
* **[SWA]** P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, A. G. Wilson. *Averaging Weights
  Leads to Wider Optima and Better Generalization.* UAI 2018.
  [arXiv:1803.05407](https://arxiv.org/abs/1803.05407) — the source of the last-25 %-of-run
  convention used by `src/models/swa.py`, taken as an external anchor rather than tuned here.
* **[RevIN]** T. Kim et al. *Reversible Instance Normalization for Accurate Time-Series Forecasting
  against Distribution Shift.* ICLR 2022 — swept as a live search dimension; a measured null.
* **[Deep ensembles]** B. Lakshminarayanan, A. Pritzel, C. Blundell. *Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017.
  [arXiv:1612.01474](https://arxiv.org/abs/1612.01474) — the prior behind seed bagging.

## Foundation models

Chronos-2 appears in the shipped system twice — fine-tuned as the **largest-weighted member**
(0.4724), and frozen as the covariate channel the TFT above conditions on.

* **[Chronos-2]** A. F. Ansari, O. Shchur, J. Küken, A. Auer, B. Han, P. Mercado, et al. *Chronos-2:
  From Univariate to Universal Forecasting.* 2025.
  [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) — used in all three roles; see
  [method.md](method.md#the-same-model-in-three-roles).
* **[Chronos]** A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, et al. *Chronos: Learning the Language
  of Time Series.* TMLR 2024. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)
* **[TiRex]** A. Auer, P. Podest, D. Klotz, S. Böck, G. Klambauer, S. Hochreiter. *TiRex: Zero-Shot
  Forecasting Across Long and Short Horizons.* 2025.
  [arXiv:2505.23719](https://arxiv.org/abs/2505.23719) — screened, weight 0.000.
* **[TimesFM]** A. Das, W. Kong, R. Sen, Y. Zhou. *A Decoder-only Foundation Model for Time-series
  Forecasting.* ICML 2024. [arXiv:2310.10688](https://arxiv.org/abs/2310.10688) — screened.
* **[Moirai]** G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, D. Sahoo. *Unified Training of
  Universal Time Series Forecasting Transformers.* ICML 2024.
  [arXiv:2402.02592](https://arxiv.org/abs/2402.02592) — screened.
* **[TabPFN-TS]** S. B. Hoo, S. Müller, D. Salinas, F. Hutter. *From Tables to Time: Extending
  TabPFN-v2 to Time Series Forecasting.* 2025.
  [arXiv:2501.02945](https://arxiv.org/abs/2501.02945) — screened.
* **[Toto 2.0]** E. Khwaja, C. Lettieri, G. Woo, E. Belouadah, et al. *Toto 2.0: Time Series
  Forecasting Enters the Scaling Era.* 2026.
  [arXiv:2605.20119](https://arxiv.org/abs/2605.20119) — screened.
* **[LoRA]** E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — the adapter the shipped member does *not*
  use; a full fine-tune scores 0.1325 against LoRA's 0.1602.
* **[DoRA]** S.-Y. Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024.
  [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) — cut on evidence, not on cost.

## Screened architecture families

* **[DLinear]** A. Zeng, M. Chen, L. Zhang, Q. Xu. *Are Transformers Effective for Time Series
  Forecasting?* AAAI 2023. [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)
* **[PatchTST]** Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam. *A Time Series is Worth 64 Words.*
  ICLR 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
* **[iTransformer]** Y. Liu et al. *iTransformer: Inverted Transformers Are Effective for Time Series
  Forecasting.* ICLR 2024. [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) — also the source of
  the cross-variate attention arm, which is a measured negative here.
* **[TiDE]** A. Das, W. Kong, A. Leach, S. Mathur, R. Sen, R. Yu. *Long-term Forecasting with TiDE.*
  TMLR 2023. [arXiv:2304.08424](https://arxiv.org/abs/2304.08424)
* **[TSMixer]** V. Ekambaram, A. Jati, N. Nguyen, P. Sinthong, J. Kalagnanam. *TSMixer: Lightweight
  MLP-Mixer Model for Multivariate Time Series Forecasting.* KDD 2023.
  [arXiv:2306.09364](https://arxiv.org/abs/2306.09364)
* **[xLSTM-Mixer]** M. Kraus, F. Divo, D. S. Dhami, K. Kersting. *xLSTM-Mixer: Multivariate Time
  Series Forecasting by Mixing via Scalar Memories.* 2024.
  [arXiv:2410.16928](https://arxiv.org/abs/2410.16928) — included as an independent covariate-fusion
  check from outside our own screen.
* **[BiTCN]** O. Sprangers, S. Schelter, M. de Rijke. *Parameter-efficient deep probabilistic
  forecasting.* International Journal of Forecasting 39(1), 2023, 332–345.
  [arXiv:2112.02905](https://arxiv.org/abs/2112.02905) — the bidirectional TCN behind
  `neuralforecast`'s `BiTCN`. Screened at 1x and 2x the TFT's width; accurate enough to look
  promising and too correlated with the tree to earn blend weight (err-corr 0.929 / 0.963).
* **[LSTM]** S. Hochreiter, J. Schmidhuber. *Long Short-Term Memory.* Neural Computation 9(8), 1997,
  1735–1780 — seven variants in the screen, best of them 0.16742.
* **[Mamba]** A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
  2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — the selective state-space stack that
  replaced the TFT's two LSTM encoders in `src/models/mamba_tft.py`. **Worse by 0.02610**, which is
  the largest measured negative in the project.
* **[N-HiTS]** C. Challú, K. G. Olivares, B. N. Oreshkin, F. Garza, M. Mergenthaler-Canseco,
  A. Dubrawski. *NHITS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI 2023.
  [arXiv:2201.12886](https://arxiv.org/abs/2201.12886) — 28.9M parameters and the worst model in the
  screen, which is the capacity argument in one row.
* **[NBEATSx]** K. G. Olivares, C. Challú, G. Marcjasz, R. Weron, A. Dubrawski. *Neural basis
  expansion analysis with exogenous variables.* International Journal of Forecasting 39(2), 2023.
  [arXiv:2104.05522](https://arxiv.org/abs/2104.05522) — screened.
* **[Informer]** H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, W. Zhang. *Informer: Beyond
  Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI 2021.
  [arXiv:2012.07436](https://arxiv.org/abs/2012.07436) — screened.
* **[Autoformer]** H. Wu, J. Xu, J. Wang, M. Long. *Autoformer: Decomposition Transformers with
  Auto-Correlation for Long-Term Series Forecasting.* NeurIPS 2021.
  [arXiv:2106.13008](https://arxiv.org/abs/2106.13008) — screened.
* **[FEDformer]** T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, R. Jin. *FEDformer: Frequency Enhanced
  Decomposed Transformer for Long-term Series Forecasting.* ICML 2022.
  [arXiv:2201.12740](https://arxiv.org/abs/2201.12740) — screened.
* **[TimesNet]** H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, M. Long. *TimesNet: Temporal 2D-Variation
  Modeling for General Time Series Analysis.* ICLR 2023.
  [arXiv:2210.02186](https://arxiv.org/abs/2210.02186) — screened.
* **[KAN]** Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, M. Tegmark.
  *KAN: Kolmogorov–Arnold Networks.* 2024.
  [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) — screened at 129M parameters.

## Trees

* **[LightGBM]** G. Ke et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.*
  NeurIPS 2017 — the shipped third member.
* **[CatBoost]** L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, A. Gulin. *CatBoost:
  Unbiased Boosting with Categorical Features.* NeurIPS 2018.
  [arXiv:1706.09516](https://arxiv.org/abs/1706.09516) — screened as a second boosting library; not
  admitted.

## Evaluation, and the statistics the admission bar rests on

* **[Forecast accuracy measures]** R. J. Hyndman, A. B. Koehler. *Another look at measures of
  forecast accuracy.* International Journal of Forecasting 22(4), 2006, 679–688 — why a
  scale-dependent, volume-weighted ratio like WAPE behaves differently from a per-series mean, which
  is the distinction behind the MAD-scaling mismatch in
  [method.md](method.md#w6--the-budget-went-to-models-not-to-features).
* **[Quantile regression]** R. Koenker, G. Bassett. *Regression Quantiles.* Econometrica 46(1),
  1978, 33–50 — why WAPE is minimised by the conditional median, why pinball at q=0.5 is L1, and
  why "check the residuals are zero-mean Gaussian" is the wrong instruction under this loss.
* **[Time-series CV]** C. Bergmeir, J. M. Benítez. *On the use of cross-validation for time series
  predictor evaluation.* Information Sciences 191, 2012, 192–213 — the case for blocked rather than
  random folds, which is what [`protocol.md`](protocol.md) implements.
* **[Purging and embargoing]** M. López de Prado. *Advances in Financial Machine Learning.* Wiley,
  2018, ch. 7 — the embargo between train and score that the residual stacker's forward chaining
  uses ([W7](method.md#w7--how-much-predictive-value-is-left-and-how-you-measure-that)).
* **[Bootstrap]** B. Efron. *Bootstrap Methods: Another Look at the Jackknife.* Annals of Statistics
  7(1), 1979, 1–26.
* **[Block bootstrap]** H. R. Künsch. *The Jackknife and the Bootstrap for General Stationary
  Observations.* Annals of Statistics 17(3), 1989, 1217–1241 — resampling whole blocks rather than
  rows, which is what makes the 288 unit×window clusters the right unit here.
* **[Kendall's tau]** M. G. Kendall. *A New Measure of Rank Correlation.* Biometrika 30(1/2), 1938,
  81–93 — the member-ranking stability statistic in `results/fp/1a_error_map.json`.
* **[Ljung–Box]** G. M. Ljung, G. E. P. Box. *On a measure of lack of fit in time series models.*
  Biometrika 65(2), 1978, 297–303 — the classical residual-autocorrelation test, listed in
  [the checklist this project did not run](method.md#the-residual-checklist-we-did-not-run).

## Combination and search

* **[Frank–Wolfe]** M. Frank, P. Wolfe. *An algorithm for quadratic programming.* Naval Research
  Logistics Quarterly 3(1–2), 1956, 95–110 — how the convex blend weights are fitted on the simplex;
  exact here because pooled WAPE is convex in the weights.
* **[Stacked generalization]** D. H. Wolpert. *Stacked generalization.* Neural Networks 5(2), 1992,
  241–259.
* **[Stacked regressions]** L. Breiman. *Stacked Regressions.* Machine Learning 24, 1996, 49–64 —
  the constrained non-negative combination this project's blend is a case of; the residual stacker
  and the MLP meta-learner are the two arms that went further and were not admitted.
* **[TPE]** J. Bergstra, R. Bardenet, Y. Bengio, B. Kégl. *Algorithms for Hyper-Parameter
  Optimization.* NeurIPS 2011 — the search algorithm behind the 32-trial sweep that returned a null.
* **[Ridge]** A. E. Hoerl, R. W. Kennard. *Ridge Regression: Biased Estimation for Nonorthogonal
  Problems.* Technometrics 12(1), 1970, 55–67 — the linear control at 0.1724 that fixes how much
  the planning covariates carry on their own.

## Method and tooling

* **[Optuna]** T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama. *Optuna: A Next-generation
  Hyperparameter Optimization Framework.* KDD 2019.
  [arXiv:1907.10902](https://arxiv.org/abs/1907.10902) — the TPE search that returned a null.
* **[neuralforecast]** K. G. Olivares, C. Challú, F. Garza, M. Mergenthaler Canseco, A. Dubrawski.
  *NeuralForecast.* 2022. <https://github.com/Nixtla/neuralforecast> — the training stack, pinned at
  3.1.9.
* **[Modal]** Modal Labs, *Pricing*, 2026. <https://modal.com/pricing> — the compute this ran on.

## Data

* **[M5]** S. Makridakis, E. Spiliotis, V. Assimakopoulos. *M5 Accuracy Competition: Results,
  Findings, and Conclusions.* International Journal of Forecasting 38(4), 2022, 1346–1364 — the
  second panel, on which neither of this project's two main levers reproduces.
* **[ECL]** A. Trindade. *ElectricityLoadDiagrams20112014.* UCI Machine Learning Repository, 2015 —
  dropped as contaminated rather than left unrun; see
  [method.md](method.md#w4--neither-large-lever-transfers-and-one-variable-explains-both).
