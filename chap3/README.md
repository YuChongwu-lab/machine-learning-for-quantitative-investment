# Chapter 3: Linear Models and Regularization (R)

Both sections in this chapter are based on `data/data_ml.RData` .

## Running

```bash
Rscript R/install_packages.R          # install dependencies before the first run
Rscript chap3/3.6/3.6_pb_valuation.R
```

The scripts automatically locate the repo's top-level `data/` directory, so they
can be run from any working directory. 

## Experiments

| File | Content |
|---|---|
| [3.6/3.6_pb_valuation.R](3.6/3.6_pb_valuation.R) | PB relative valuation: OLS -> regression diagnostics -> VIF multicollinearity -> BP heteroskedasticity test -> FGLS -> 2SLS -> quantile regression. Uses the single 2016-07-31 cross section |
| [3.7/3.7_predictive_regression.R](3.7/3.7_predictive_regression.R) | Predictive regression: Ridge, Lasso, ElasticNet (alpha=0.25/0.75), caret cross-validation, quantile lasso (tau=0.2/0.4/0.6/0.8), multi-layer group lasso. Uses 2010-2018 data, split chronologically at 2018-01-15 |
| [3.7/3.7_predictive_regression_single_month.R](3.7/3.7_predictive_regression_single_month.R) | A **single-period variant** of the script above, used to check the effect of month-by-month normalization |



## A note on the `vip` package

`vip` (used by `3.7_predictive_regression.R` and
`3.7_predictive_regression_single_month.R` for variable-importance plots)
was removed from CRAN on 2026-07-08. `R/install_packages.R` now falls back
to installing it from the CRAN archive automatically; no action is needed
beyond running that script.

