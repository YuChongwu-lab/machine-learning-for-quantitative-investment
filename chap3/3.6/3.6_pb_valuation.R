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

###### Section 3.6: OLS, WLS, 2SLS, and quantile regression applied to and compared on the PB relative-valuation model #####


library(lmtest)    # bptest for the heteroskedasticity test
library(car)       # vif for the variance inflation factor 
library(quantreg)  # rq for quantile regression
library(ivreg)     # two-stage least squares


# Load the data
load(data_file("data_ml.RData"))

data_pb <-  subset(data_ml,date == "2016-07-31")
# Note: the FGLS and 2SLS sections below rely on attach() to put data_pb's
# columns on the search path (those regression calls have no data= argument),
# so this attach() cannot be removed.
attach(data_pb)

## OLS regression and diagnostics

# OLS regression
model1 <- lm(Pb ~ Dps+Eps_Basic_Gr + Roe +Vol1Y_Usd +Mom_5M_Usd , data = data_pb)
summary(model1)

# OLS regression diagnostics
par(mfrow = c(3,2))
plot(model1,which =1: 6)

# Test for multicollinearity via the variance inflation factor
vif(model1)

# Test for heteroskedasticity via the BP test
bptest(model1)


# FGLS regression
resid <- residuals(model1)
var_model1 <- lm(resid^2 ~Dps+Eps_Basic_Gr + Roe + Vol1Y_Usd +Mom_5M_Usd )
wts <- 1/sqrt(fitted(var_model1))
pb_gls <-lm(Pb ~ Dps+Eps_Basic_Gr + Roe + Vol1Y_Usd +Mom_5M_Usd , weights = wts)
summary(pb_gls)


## 2SLS regression
pb_ivreg <- ivreg(Pb ~ Dps+Eps_Basic_Gr + Roe + Vol1Y_Usd + Mom_5M_Usd| Dps+Eps_Basic_Gr + Roe + Mkt_Cap_3M_Usd + Mom_5M_Usd,data = data_pb)
summary(pb_ivreg)


## Quantile regression
pb_quantile <- rq(Pb ~ Dps+Eps_Basic_Gr + Roe + Mkt_Cap_3M_Usd +Mom_5M_Usd, tau = c(0.05,0.25,0.5,0.75,0.95),data = data_pb)
summary(pb_quantile)

# Visualize the quantile regression
pb_quantile_plot<- rq(Pb ~ Dps+Eps_Basic_Gr + Roe + Mkt_Cap_3M_Usd +Mom_5M_Usd, tau = seq(0.05,0.95,by = 0.05),data = data_pb)
d <- summary(pb_quantile_plot)
plot(d)

# Paired with the attach() above, to avoid polluting the search path for the rest of the session
detach(data_pb)
