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


# Load the required packages
library(party)     # mob / mob_control, model-based recursive partitioning (pinned via party::)
library(partykit)  # as.party() for rpart -> party; loaded after `party` to win method dispatch
library(rpart)     # CART
library(vip)       # variable importance


# Load the data
load(data_file("data_ml.RData"))
data_pb <-  subset(data_ml,date == "2016-07-31")

# CART
CART <- rpart(Pb ~., data = data_pb)
plotcp(CART)
plot(as.party(CART))
CP <- min(CART$cptable[11,])
CART_Prune <- prune(CART,cp=CP)
plot(as.party(CART_Prune))

vip(CART_Prune,num_features =40, bar=FALSE)

# MOB
ctrl <-party:: mob_control(alpha = 0.05, bonferroni = TRUE, minsplit = 100,objfun = deviance,verbose = TRUE)
pb_mob <- party::mob(Pb ~ Dps+Eps_Basic_Gr + Roe +Vol1Y_Usd +Mom_5M_Usd|Roe +Roce +Roa, data = data_pb, control = ctrl, model = linearModel)

plot(pb_mob)
coef(pb_mob)
summary(pb_mob,node=3)
summary(pb_mob,node=4)
summary(pb_mob,node=6)
summary(pb_mob,node=8)
summary(pb_mob,node=11)
summary(pb_mob,node=12)
summary(pb_mob,node=14)
summary(pb_mob,node=15)
