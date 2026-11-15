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

#=========4.7 cTree=========================

# Load the required packages
library(readstata13)
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
sps.train.clean <- na.omit(sps.train)
sps.test.clean <- na.omit(sps.test)


# Convert the categorical variables in the train/test sets to factors
factor_features <- c("crosslist","neg", "exp", "star","broker","rec", "ir","nstar","disp","nwb","ill" )
sps.train[factor_features] <-lapply(sps.train[factor_features],factor)
sps.test[factor_features] <-lapply(sps.test[factor_features],factor)


#================2. CTree================================================
# Limit the max tree depth to 3, min split sample size to 20
ctree.control <- ctree_control(
  maxdepth = 3,         # limit the tree depth
  minsplit = 20         # lower the minimum split sample size 
sps.ctree <- partykit::ctree(sps.train.clean$synch2 ~.,data=sps.train.clean,control =ctree.control)
print(sps.ctree)
plot(sps.ctree)
sps.ctree.predictions <- predict(sps.ctree, newdata = sps.test.clean)
mean((sps.test.clean$synch2-sps.ctree.predictions)^2)

# Default settings
sps.ctree.full<- partykit::ctree(sps.train.clean$synch2 ~.,data=sps.train.clean)
print(sps.ctree.full)
png("ctree_plot.png", width = 2800, height = 1800)
plot(sps.ctree.full, cex = 0.3)
dev.off()
sps.ctree.full.predictions <- predict(sps.ctree.full, newdata = sps.test.clean)
mean((sps.test.clean$synch2-sps.ctree.full.predictions)^2)

# Plot variable importance
par(mfrow=c(1,2))
# Max tree depth = 3, min split sample size = 20
vi.ctree <- sort(varimp(sps.ctree),decreasing = FALSE)
vi.ctree <-vi.ctree/max(vi.ctree)*100
dotchart(vi.ctree,xlab = "Importance Score (Scaled)")
# Default settings
vi.ctree.full <- sort(varimp(sps.ctree.full),decreasing = FALSE)
vi.ctree.full <-vi.ctree.full/max(vi.ctree.full)*100
dotchart(vi.ctree.full,xlab = "Importance Score (Scaled)")
