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

#=========4.7 CTree vs CART Random Forest with Cross-Validation=========================

# Load the required packages
library(readstata13)
library(partykit)
library(randomForest)
library(caret)

#=====1. Data preprocessing===========================================
sps <- readstata13::read.dta13(data_file("analysts_stock_price_synchronicity.dta"))
sps <- sps[, 1:25]
sps <- sps[, -11]  # drop synch, use synch2 instead

# Split into train/test sets
set.seed(42)
train.id <- sample.int(nrow(sps), size = floor(0.7 * nrow(sps)))
sps.train <- sps[train.id, ]
sps.test  <- sps[-train.id, ]

# Convert categorical variables to factors first, then clean up missing values
factor_features <- c("crosslist","neg","exp","star","broker","rec","ir","nstar","disp","nwb","ill")
for (ff in factor_features) {
  if (ff %in% names(sps.train)) {
    sps.train[[ff]] <- factor(sps.train[[ff]])
    sps.test[[ff]]  <- factor(sps.test[[ff]], levels = levels(sps.train[[ff]]))
  }
}

# Drop missing values from the training set
sps.train.clean <- na.omit(sps.train)
test_cc <- complete.cases(sps.test[, names(sps.train.clean)])
sps.test.cc <- sps.test[test_cc, , drop = FALSE]

#=====2. Random forest (CART base learner)===================================
set.seed(42)
sps.rf <- randomForest(synch2 ~ ., data = sps.train.clean, importance = TRUE)
print(sps.rf)
which.min(sps.rf$mse)
varImpPlot(sps.rf, scale = TRUE)

# Test-set prediction and MSE
rf.pred <- predict(sps.rf, newdata = sps.test.cc)
rf.mse  <- mean((sps.test.cc$synch2 - rf.pred)^2)
cat("RF (CART-based) test MSE:", rf.mse, "\n")

#=====3. Random forest (CTree base learner)===================================
set.seed(42)
sps.crf <- partykit::cforest(synch2 ~ ., data = sps.train.clean)
vi.crf <- partykit::varimp(sps.crf)
barplot(sort(vi.crf, decreasing = TRUE),
        horiz = TRUE, las = 1, cex.names = 0.7,
        main = "cForest Variable Importance")

# Prediction and MSE
crf.pred <- predict(sps.crf, newdata = sps.test.cc, OOB = FALSE)
crf.mse  <- mean((sps.test.cc$synch2 - crf.pred)^2)
cat("cForest (CTree-based) test MSE:", crf.mse, "\n")

#=====4. Cross-validation: CART base learner (Random Forest)=====================
set.seed(42)
cv.ctrl <- trainControl(method = "cv", number = 5)

rf.cv <- train(
  synch2 ~ .,
  data = sps.train.clean,
  method = "rf",
  trControl = cv.ctrl,
  tuneLength = 3,
  importance = TRUE
)

print(rf.cv)
plot(rf.cv)
#varImpPlot(rf.cv, scale = TRUE)
cat("CART-based RF CV RMSE:", min(rf.cv$results$RMSE), "\n")

#=====5. Cross-validation: CTree base learner (cForest)==========================
set.seed(42)
cf.cv <- train(
  synch2 ~ .,
  data = sps.train.clean,
  method = "cforest",
  trControl = cv.ctrl,
  tuneLength = 3
)

print(cf.cv)
plot(cf.cv)
cat("CTree-based cForest CV RMSE:", min(cf.cv$results$RMSE), "\n")

#=====5.1 Variable importance and test MSE for the cross-validated models=========================
# CART RF
rf.final <- rf.cv$finalModel
rf.imp <- importance(rf.final)
varImpPlot(rf.final, main = "Variable Importance - CART RF (CV Best Model)")
rf.cv.pred <- predict(rf.cv, newdata = sps.test.cc)
rf.cv.mse  <- mean((sps.test.cc$synch2 - rf.cv.pred)^2)
cat("CART RF (CV-best) Test MSE:", rf.cv.mse, "\n")

# cForest
cf.final <- cf.cv$finalModel
vi.cf <- party::varimp(cf.final, conditional = FALSE)
barplot(sort(vi.cf, decreasing = TRUE),
        horiz = TRUE, las = 1, cex.names = 0.7,
        main = "Variable Importance - CTree cForest (CV Best Model)")
cf.cv.pred <- predict(cf.cv, newdata = sps.test.cc)
cf.cv.mse  <- mean((sps.test.cc$synch2 - cf.cv.pred)^2)
cat("CTree cForest (CV-best) Test MSE:", cf.cv.mse, "\n")


#=====6. Model comparison====================================================
barplot(
  c(min(rf.cv$results$RMSE), min(cf.cv$results$RMSE)),
  names.arg = c("CART RF", "CTree cForest"),
  col = "steelblue",
  main = "Cross-Validation RMSE Comparison",
  ylab = "RMSE (lower is better)"
)
