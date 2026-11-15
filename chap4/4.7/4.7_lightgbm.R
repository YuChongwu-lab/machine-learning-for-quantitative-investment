# ---- Bootstrap: locate the repo root and load the data-path resolver ----
.bootstrap_start <- local({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg[1])))
  } else {
    getwd()
  }
})
.repo_dir <- .bootstrap_start
while (!file.exists(file.path(.repo_dir, "R", "paths.R"))) {
  .parent <- dirname(.repo_dir)
  if (identical(.parent, .repo_dir)) {
    stop("Could not locate R/paths.R: please run this script from inside the repo.", call. = FALSE)
  }
  .repo_dir <- .parent
}
.qml_repo_root <- .repo_dir  # hand off to paths.R, so it doesn't depend on the working directory
source(file.path(.repo_dir, "R", "paths.R"))
rm(list = intersect(c(".bootstrap_start", ".repo_dir", ".parent", ".qml_repo_root"),
                    ls(all.names = TRUE)))
# ---- End of bootstrap ----

library(lightgbm)
library(readstata13)

#=====1. Data preprocessing===========================================
sps <- readstata13::read.dta13(data_file("analysts_stock_price_synchronicity.dta"))
sps <- sps[, 1:25]
sps <- sps[, -11]

set.seed(42)
train.id <- sample.int(nrow(sps), size = floor(0.7 * nrow(sps)))
sps.train <- sps[train.id, ]
sps.test  <- sps[-train.id, ]

factor_features <- c("crosslist","neg","exp","star","broker",
                     "rec","ir","nstar","disp","nwb","ill")
for (ff in factor_features) {
  if (ff %in% names(sps.train)) {
    sps.train[[ff]] <- factor(sps.train[[ff]])
    sps.test[[ff]]  <- factor(sps.test[[ff]], levels = levels(sps.train[[ff]]))
  }
}

sps.train.clean <- na.omit(sps.train)
keep_cols <- names(sps.train.clean)
test_cc <- complete.cases(sps.test[, keep_cols])
sps.test.clean <- sps.test[test_cc, keep_cols, drop = FALSE]

target <- "synch2"
feature_names <- setdiff(keep_cols, target)
train_matrix <- data.matrix(sps.train.clean[, feature_names])
test_matrix  <- data.matrix(sps.test.clean[, feature_names])

# Disable feature pre-filtering
train_data <- lgb.Dataset(
  data = train_matrix,
  label = sps.train.clean[[target]],
  params = list(feature_pre_filter = FALSE)
)

valid_data <- lgb.Dataset(
  data = test_matrix,
  label = sps.test.clean[[target]],
  params = list(feature_pre_filter = FALSE)
)

#=====2. Hyperparameter search space (random search)=======================
set.seed(123)

param_space <- list(
  num_leaves = c(15, 31, 63, 127, 255),
  max_depth = c(-1, 5, 7, 10, 15),
  min_data_in_leaf = c(10, 20, 50, 100),
  learning_rate = c(0.01, 0.03, 0.05, 0.1, 0.2),
  lambda_l1 = c(0, 0.1, 0.5, 1, 5),
  lambda_l2 = c(0, 0.1, 0.5, 1, 5),
  min_gain_to_split = c(0, 0.1, 0.5, 1),
  feature_fraction = c(0.6, 0.7, 0.8, 0.9, 1.0),
  bagging_fraction = c(0.6, 0.7, 0.8, 0.9, 1.0),
  bagging_freq = c(1, 3, 5, 7)
)

n_iter <- 50
param_grid <- data.frame(
  num_leaves = sample(param_space$num_leaves, n_iter, replace = TRUE),
  max_depth = sample(param_space$max_depth, n_iter, replace = TRUE),
  min_data_in_leaf = sample(param_space$min_data_in_leaf, n_iter, replace = TRUE),
  learning_rate = sample(param_space$learning_rate, n_iter, replace = TRUE),
  lambda_l1 = sample(param_space$lambda_l1, n_iter, replace = TRUE),
  lambda_l2 = sample(param_space$lambda_l2, n_iter, replace = TRUE),
  min_gain_to_split = sample(param_space$min_gain_to_split, n_iter, replace = TRUE),
  feature_fraction = sample(param_space$feature_fraction, n_iter, replace = TRUE),
  bagging_fraction = sample(param_space$bagging_fraction, n_iter, replace = TRUE),
  bagging_freq = sample(param_space$bagging_freq, n_iter, replace = TRUE)
)

results <- data.frame()
cat("Starting the hyperparameter search...\n")

for (i in 1:n_iter) {
  params <- list(
    objective = "regression",
    metric = "mse",
    boosting_type = "gbdt",
    num_leaves = param_grid$num_leaves[i],
    max_depth = param_grid$max_depth[i],
    min_data_in_leaf = param_grid$min_data_in_leaf[i],
    learning_rate = param_grid$learning_rate[i],
    lambda_l1 = param_grid$lambda_l1[i],
    lambda_l2 = param_grid$lambda_l2[i],
    min_gain_to_split = param_grid$min_gain_to_split[i],
    feature_fraction = param_grid$feature_fraction[i],
    bagging_fraction = param_grid$bagging_fraction[i],
    bagging_freq = param_grid$bagging_freq[i],
    feature_pre_filter = FALSE,
    verbosity = -1
  )

  cv_res <- lgb.cv(
    params = params,
    data = train_data,
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = -1
  )

  results <- rbind(results, data.frame(
    iter = i,
    cv_mse = cv_res$best_score,
    best_iter = cv_res$best_iter,
    param_grid[i, ]
  ))

  if (i %% 10 == 0) {
    cat(sprintf("Completed %d/%d searches, current best MSE: %.6f\n",
                i, n_iter, min(results$cv_mse)))
  }
}

# Find the best parameters
best_idx <- which.min(results$cv_mse)
best_params <- as.list(results[best_idx, ])

cat("\n========== Best parameters ==========\n")
print(best_params)
cat("Best CV MSE:", best_params$cv_mse, "\n")
cat("Best iteration count:", best_params$best_iter, "\n")

#=====3. Retrain with the best parameters===========================
final_params <- list(
  objective = "regression",
  metric = "mse",
  boosting_type = "gbdt",
  num_leaves = best_params$num_leaves,
  max_depth = best_params$max_depth,
  min_data_in_leaf = best_params$min_data_in_leaf,
  learning_rate = best_params$learning_rate,
  lambda_l1 = best_params$lambda_l1,
  lambda_l2 = best_params$lambda_l2,
  min_gain_to_split = best_params$min_gain_to_split,
  feature_fraction = best_params$feature_fraction,
  bagging_fraction = best_params$bagging_fraction,
  bagging_freq = best_params$bagging_freq,
  feature_pre_filter = FALSE,
  verbosity = -1
)

set.seed(42)
final_model <- lgb.train(
  params = final_params,
  data = train_data,
  nrounds = as.numeric(best_params$best_iter) * 2,
  valids = list(valid = valid_data),
  early_stopping_rounds = 30
)

final_pred <- predict(final_model, test_matrix)
final_mse <- mean((sps.test.clean[[target]] - final_pred)^2)
cat("\nFinal test-set MSE:", final_mse, "\n")

#=====4. Visualize the search results=============================
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

hist(results$cv_mse, breaks = 30, col = "skyblue", border = "white",
     main = "CV MSE Distribution", xlab = "MSE", ylab = "Frequency")
abline(v = best_params$cv_mse, col = "red", lwd = 2, lty = 2)

plot(results$learning_rate, results$cv_mse, pch = 19, col = rgb(0,0,1,0.5),
     main = "Learning Rate vs MSE", xlab = "Learning Rate", ylab = "CV MSE")

plot(results$num_leaves, results$cv_mse, pch = 19, col = rgb(0,1,0,0.5),
     main = "Num Leaves vs MSE", xlab = "Num Leaves", ylab = "CV MSE")

top10 <- head(results[order(results$cv_mse), ], 10)
barplot(top10$cv_mse, names.arg = top10$iter, col = "coral",
        main = "Top 10 Parameter Sets", xlab = "Iteration", ylab = "CV MSE")

#=====5. Variable importance (final model)========================

lgb.imp <- lgb.importance(final_model, percentage = TRUE)

# Show only the top 15 variables to avoid overcrowding
top_n <- 15
lgb.imp.top <- head(lgb.imp, top_n)

# Plot settings: 3 side-by-side panels, larger font and left margin
par(mfrow = c(1, 3),
    mar = c(5, 8, 3, 2),
    cex.main = 1.2)

# Plot the three horizontal bar charts
barplot(lgb.imp.top$Frequency,
        names.arg = lgb.imp.top$Feature,
        horiz = TRUE, las = 1, col = "steelblue",
        cex.names = 1.5, xlab = "Frequency (%)")
title("Feature Frequency", line = 1)

barplot(lgb.imp.top$Gain,
        names.arg = lgb.imp.top$Feature,
        horiz = TRUE, las = 1, col = "coral",
        cex.names = 1.5, xlab = "Gain (%)")
title("Feature Gain", line = 1)

barplot(lgb.imp.top$Cover,
        names.arg = lgb.imp.top$Feature,
        horiz = TRUE, las = 1, col = "lightgreen",
        cex.names =1.5, xlab = "Cover (%)")
title("Feature Cover", line = 1)


write.csv(results, "hyperparameter_search_results.csv", row.names = FALSE)
cat("\nSearch results saved to hyperparameter_search_results.csv\n")
