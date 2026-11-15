# R dependency installation script for Chapters 3 and 4 of
# "Machine Learning for Quantitative Investment"
#
# Usage:
#     Rscript R/install_packages.R


packages <- c(
  # --- 3.6 PB relative valuation: OLS / FGLS / 2SLS / quantile regression ---
  "lmtest",       # bptest heteroskedasticity test
  "car",          # vif variance inflation factor (the script calls vif() and needs this package loaded)
  "quantreg",     # quantile regression
  "ivreg",        # two-stage least squares

  # --- 3.7 Predictive regression: Ridge / Lasso / ElasticNet / quantile lasso / multi-layer group lasso ---
  "tidyverse",    # includes dplyr, tibble, ggplot2
  "lubridate",
  "glmnet",
  "caret",
  "vip",          # variable importance
  "rqPen",        # quantile lasso
  "MLGL",         # multi-layer group lasso

  # --- 4.6 CART and MOB structural-break analysis ---
  "rpart",
  "party",        # mob model-based recursive partitioning
  "partykit",     # as.party (the script calls as.party() and needs this package loaded)

  # --- 4.7 Tree-model comparison: CART / CTree / random forest / LightGBM ---
  "readstata13",  # reads .dta files
  "randomForest",
  "lightgbm"
)

missing <- packages[!(packages %in% rownames(installed.packages()))]

if (length(missing) > 0) {
  cat("Installing the following missing packages:\n  ", paste(missing, collapse = ", "), "\n\n", sep = "")
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  cat("All dependencies are already installed.\n\n")
}

# vip was removed from CRAN on 2026-07-08 ; a plain install.packages("vip") will report "not
# available for this version of R". If it's still missing, fall back to
# installing the last available version from the CRAN archive.
if (!("vip" %in% rownames(installed.packages()))) {
  cat("vip has been removed from CRAN; trying to install from the archive...\n")
  if (!("yardstick" %in% rownames(installed.packages()))) {
    install.packages("yardstick", repos = "https://cloud.r-project.org")
  }
  install.packages(
    "https://cloud.r-project.org/src/contrib/Archive/vip/vip_0.4.6.tar.gz",
    repos = NULL, type = "source"
  )
}

still_missing <- packages[!(packages %in% rownames(installed.packages()))]
if (length(still_missing) > 0) {
  cat("\nThe following packages failed to install and need manual attention:\n  ",
      paste(still_missing, collapse = ", "), "\n", sep = "")
  cat("lightgbm and MLGL require a system-level build toolchain on some platforms.\n")
}

cat("\n---------------- Environment record (keep this on file) ----------------\n")
print(sessionInfo())
