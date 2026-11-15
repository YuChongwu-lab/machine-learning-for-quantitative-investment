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

###=======Section 3.7: predictive regression (using 2009-2019 data)========

#======== Load the required packages =================
library(tidyverse) # Activate the data science package
library(lubridate) # Activate the date management package
library(glmnet)# Ridge and plain lasso
library(caret)#hyperparameter tuning
library(vip)#variable importance
library(rqPen)#quantile lasso
library(MLGL)#multi-layer group lasso#load the related packages


#================1. Prepare the data====================
# Data preprocessing
load(data_file("data_ml.RData")) # Load the data

data_ml <- data_ml %>%
  filter(date > "2009-12-31", # Keep the date with sufficient data points
         date < "2019-01-01") %>%
  arrange(stock_id, date) # Order the data

features <- colnames(data_ml[3:95]) # Keep the feature's column names (hard-coded, beware!)

separation_date <- as.Date("2018-01-15")
training_sample <- filter(data_ml, date < separation_date)
testing_sample <- filter(data_ml, date >= separation_date)

y_train <- training_sample$R1M_Usd # Dependent variable
x_train <- training_sample %>% # Predictors
  dplyr::select(all_of(features)) %>% as.matrix()

x_test <- testing_sample %>% # Predictors
  dplyr::select(all_of(features)) %>% as.matrix()
y_test <- testing_sample$R1M_Usd


# =================2. Ridge, Lasso, and Elastic Net ====================================

ridge <- glmnet(x_train, y_train, # Model
                       family = "gaussian", alpha = 0)
lasso <- glmnet(x_train, y_train, # Model
                family = "gaussian", alpha = 1)
elnet0.25 <- glmnet(x_train, y_train, # Model
                         family = "gaussian", alpha = 0.25)
elnet0.75 <- glmnet(x_train, y_train, # Model
                             family = "gaussian", alpha = 0.75)
# Plot the solution paths
par(mfrow = c(2, 2))
plot(ridge, main = "Ridge penalty\n\n")
plot(lasso, main = "Lasso penalty\n\n")
plot(elnet0.25, main = "Elastic Net0.25\n\n")
plot(elnet0.75, main = "Elastic Net0.75\n\n")

# Hyperparameter tuning
set.seed(42)
ridge_cv <-  cv.glmnet( x = x_train,  y = y_train,  alpha = 0)
lasso_cv <-  cv.glmnet( x = x_train,  y = y_train,  alpha = 1)

glmnet_cv <- train(  x = x_train,  y = y_train,  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# Get the best parameters
glmnet_cv$bestTune

# Plot variable importance
vip(glmnet_cv, num_features = 20, geom = "point")

#prediction
mean((predict(ridge, x_test) - y_test)^2) # MSE
mean(predict(ridge, x_test) * y_test > 0) # Hit ratio

mean((predict(lasso, x_test) - y_test)^2) # MSE
mean(predict(lasso, x_test) * y_test > 0) # Hit ratio

mean((predict(elnet0.25, x_test) - y_test)^2) # MSE
mean(predict(elnet0.25, x_test) * y_test > 0) # Hit ratio
mean((predict(elnet0.75, x_test) - y_test)^2) # MSE
mean(predict(elnet0.75, x_test) * y_test > 0) # Hit ratio

mean((predict(glmnet_cv, x_test) - y_test)^2) # MSE
mean(predict(glmnet_cv, x_test) * y_test > 0) # Hit ratio



# ==============3. Quantile Lasso =======================================
# Compute quantile lasso at the 0.2/0.4/0.6/0.8 quantiles
rqlasso0.2 <- rq.pen(x_train, y_train,tau=0.2,penalty = "LASSO",alg="huber")
rqlasso0.4 <- rq.pen(x_train, y_train,tau=0.4,penalty = "LASSO",alg="huber")
rqlasso0.6 <- rq.pen(x_train, y_train,tau=0.6,penalty = "LASSO",alg="huber")
rqlasso0.8 <- rq.pen(x_train, y_train,tau=0.8,penalty = "LASSO",alg="huber")

# Plot the solution paths
par(mfrow = c(2, 2))
plot(rqlasso0.2, main = "Quantile LASSO0.2\n\n")
plot(rqlasso0.4, main = "Quantile LASSO0.4\n\n")
plot(rqlasso0.6, main = "Quantile LASSO0.6\n\n")
plot(rqlasso0.8, main = "Quantile LASSO0.8\n\n")

# Hyperparameter tuning
set.seed(42)
taus <- seq(0.2, 0.8, 0.2)
rqlasso_cv<- rq.gq.pen.cv(x=x_train, y=y_train, tau=taus)

print(rqlasso_cv)
summary(rqlasso_cv)

# Prediction
predict(rqlasso_cv,x_test)
mean((predict(rqlasso_cv, x_test) - y_test)^2) # MSE
mean(predict(rqlasso_cv, x_test) * y_test > 0) # Hit ratio


# ================4. Multi-Layer Group LASSO (MLGL) ==================================
set.seed(42)
hc <- bootstrapHclust(x_train, frac = 1, method = "average", B = 50)
groupWeight <- computeGroupSizeWeight(hc, sizeMax = 30)
set.seed(42)
mlglasso <- fullProcess(x_train, y_train, hc = hc, fractionSampleMLGL = 0.5, weightSizeGroup = groupWeight)

summary(mlglasso)
plot(mlglasso)
print(mlglasso)


### Predict using the selected variables
### Select the variables
# 1) Use the step k in res closest to lambdaOpt to build a variable-group detail table
lam_star <- mlglasso$lambdaOpt[1]
lambda_seq <- mlglasso$res$lambda
k <- which.min(abs(lambda_seq - lam_star))

var_idx_k  <- mlglasso$res$var[[k]]    # the "column indices" of the variables selected at this lambda
grp_id_k   <- mlglasso$res$group[[k]]  # the group IDs, aligned with var_idx_k

sel_df_k <- tibble::tibble(
  var_index = var_idx_k,
  var_name  = features[var_idx_k],
  group_id  = grp_id_k
)

# 2) Keep only variables within the groups fullProcess actually selected (Option A)
final_groups <- mlglasso$selectedGroups
sel_features_by_grp <- unique(dplyr::filter(sel_df_k, group_id %in% final_groups)$var_name)
sel_features_by_grp

## 3) Also prepare a fallback, Option B: use fullProcess$var directly (may be empty)
sel_idx_fp <- mlglasso$var
sel_features_fp <- if (is.null(sel_idx_fp) || length(sel_idx_fp) == 0) character(0) else features[sel_idx_fp]
sel_features_fp

## 4) Choose the final variable names: prefer Option A ("by group"), fall back to Option B if empty
selected_features <- if (length(sel_features_by_grp) > 0) sel_features_by_grp else sel_features_fp

length(selected_features); selected_features[1:min(10, length(selected_features))]


### Predict using the selected variables
# 1) Prepare the train/test data (keep as a data.frame, with column names = feature names)
train_df <- training_sample %>% dplyr::select(all_of(selected_features))
test_df  <- testing_sample  %>% dplyr::select(all_of(selected_features))

# 2) Fit: use a data.frame (with column names)
mlgl_fit <- lm(y_train ~ ., data = cbind(y_train = y_train, train_df))

# 3) Predict: newdata must be a data.frame, with column names matching training
pred <- predict(mlgl_fit, newdata = test_df)

# 4) Evaluate
mse <- mean((pred - y_test)^2)
hit <- mean(pred * y_test > 0)

mse; hit
