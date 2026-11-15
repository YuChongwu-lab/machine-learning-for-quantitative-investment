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

#=========4.7 CART=========================

# Load the required packages
library(readstata13)
library(rpart)
library(partykit)


#=====1. Data preprocessing===========================================
sps<-readstata13::read.dta13(data_file("analysts_stock_price_synchronicity.dta"))
sps <- sps[,1:25]
sps <- sps[,-11]# drop synch, use synch2 instead, which is computed using the 2001 industry classification standard, excluding the effect of industry-classification changes
str(sps)


# Split into train/test sets
set.seed(42)
train.id <- sample.int(nrow(sps), size = floor(0.7 * nrow(sps)))
sps.train <- sps[train.id, ]
sps.test <- sps[-train.id, ]


# Convert the categorical variables in the train/test sets to factors
factor_features <- c("crosslist","neg", "exp", "star","broker","rec", "ir","nstar","disp","nwb","ill" )
sps.train[factor_features] <-lapply(sps.train[factor_features],factor)
sps.test[factor_features] <-lapply(sps.test[factor_features],factor)



##===========2. Compare how CART handles missing values=========================
# Without dropping missing values from the training sample
sps.cart <- rpart(sps.train$synch2 ~.,data=sps.train,method = "anova", na.action = na.rpart)
print(sps.cart)
plot(as.party(sps.cart))

# With missing values dropped from the training sample
sps.train.clean <- na.omit(sps.train)
sps.cart.clean <- rpart(sps.train.clean$synch2 ~.,data=sps.train.clean,method = "anova")
print(sps.cart.clean)
plot(as.party(sps.cart.clean))

# Plot variable importance
par(mfrow=c(1,2))
# Variable importance without dropping missing values
vi.cart <- sort(sps.cart$variable.importance, decreasing = FALSE)
vi.cart <- vi.cart / max(vi.cart)*100 # scale to sum to 1
dotchart(vi.cart,xlab = "Importance Score (Scaled)")
# Variable importance with missing values dropped
vi.cart.clean <- sort(sps.cart.clean$variable.importance, decreasing = FALSE)
vi.cart.clean <- vi.cart.clean / max(vi.cart.clean)*100 # scale to sum to 1
dotchart(vi.cart.clean,xlab = "Importance Score (Scaled)")

# Compare the two methods' performance on the test set
# Drop missing values from the test sample
sps.test.clean <- na.omit(sps.test)

cart.predictions <- predict(sps.cart, newdata = sps.test.clean)
mean((sps.test.clean$synch2-cart.predictions)^2)

cart.clean.predictions <- predict(sps.cart.clean, newdata = sps.test.clean)
mean((sps.test.clean$synch2-cart.clean.predictions)^2)
