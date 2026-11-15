# Chapter 4: Tree Models (R)

## Running

```bash
Rscript R/install_packages.R          # install dependencies before the first run
Rscript chap4/4.7/4.7_cart.R
```

## Experiments

| File | Content | Data |
|---|---|---|
| [4.6/4.6_cart_mob_structural_break.R](4.6/4.6_cart_mob_structural_break.R) | Analyzes structural breaks in PB relative valuation using CART and MOB (model-based recursive partitioning) | `data_ml.RData` (2016-07-31 cross section) |
| [4.7/4.7_cart.R](4.7/4.7_cart.R) | CART regression tree, with pruning and variable importance | the analyst stock-price-synchronicity panel |
| [4.7/4.7_ctree.R](4.7/4.7_ctree.R) | Conditional inference tree (two settings: depth-limited and default) | same as above |
| [4.7/4.7_random_forest.R](4.7/4.7_random_forest.R) | Random forest and conditional-inference forest, with caret cross-validated tuning | same as above |
| [4.7/4.7_lightgbm.R](4.7/4.7_lightgbm.R) | LightGBM, with a 50-combination random hyperparameter search | same as above |

`4.7_lightgbm.R`'s search results are distributed alongside the script:
[4.7/hyperparameter_search_results.csv](4.7/hyperparameter_search_results.csv).
