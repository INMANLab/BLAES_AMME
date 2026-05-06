##############################################################################
#  IED Spread -> Memory GLMM
#
#  Two separate models (to avoid collinearity):
#    1. memory ~ n_regions_c  + (1 | patient_id)
#    2. memory ~ n_channels_c + (1 | patient_id)
#
#  Saves: odds ratios, diagnostics (ICC, R², AIC/BIC), LRT vs null
##############################################################################

if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")
if (!require("performance")) install.packages("performance",  repos = "https://cloud.r-project.org")

library(lme4)
library(lmerTest)
library(broom.mixed)
library(performance)

script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg)))
    else getwd()
  }
)
out_dir <- file.path(script_dir, "outputs", "ied_timing_memory")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

save_tidy <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  return(td)
}

save_diagnostics <- function(model, filename) {
  vc <- as.data.frame(VarCorr(model))
  rand_var <- vc$vcov[1]
  rand_sd  <- vc$sdcor[1]
  icc_val  <- rand_var / (rand_var + pi^2 / 3)
  aic_val  <- AIC(model)
  bic_val  <- BIC(model)
  loglik_val <- as.numeric(logLik(model))
  dev_val  <- deviance(model)
  nobs_val <- nobs(model)
  ngrps_val <- ngrps(model)[[1]]
  r2 <- tryCatch(r2_nakagawa(model), error = function(e) {
    list(R2_marginal = NA, R2_conditional = NA)
  })
  diag_df <- data.frame(
    metric = c("ICC_latent", "rand_intercept_var", "rand_intercept_sd",
               "AIC", "BIC", "logLik", "deviance", "nobs", "ngrps",
               "R2_marginal", "R2_conditional"),
    value = c(icc_val, rand_var, rand_sd,
              aic_val, bic_val, loglik_val, dev_val, nobs_val, ngrps_val,
              r2$R2_marginal, r2$R2_conditional)
  )
  write.csv(diag_df, file.path(out_dir, filename), row.names = FALSE)
  return(diag_df)
}

save_lrt <- function(m_reduced, m_full, filename) {
  lr <- anova(m_reduced, m_full)
  lrt_df <- as.data.frame(lr)
  lrt_df$model <- c("Reduced", "Full")
  write.csv(lrt_df, file.path(out_dir, filename), row.names = FALSE)
  return(lrt_df)
}

# Load data
dat <- read.csv(file.path(out_dir, "ied_spread_trials.csv"),
                stringsAsFactors = FALSE)
dat$patient_id <- factor(dat$patient_id)
dat$n_regions_c  <- as.numeric(scale(dat$n_regions, scale = FALSE))
dat$n_channels_c <- as.numeric(scale(dat$n_channels, scale = FALSE))

cat(sprintf("Data: %d trials, %d patients\n", nrow(dat), nlevels(dat$patient_id)))
cat(sprintf("Memory rate: %.1f%%\n\n", mean(dat$memory) * 100))

# ── Null (intercept-only) model ──
cat("=== NULL MODEL ===\n")
m_null <- glmer(memory ~ 1 + (1 | patient_id),
                data = dat, family = binomial, control = ctrl)
print(summary(m_null))
save_diagnostics(m_null, "spread_null_diagnostics.csv")

# ── Model 1: Region spread ──
cat("\n=== REGION SPREAD MODEL ===\n")
m_region <- glmer(memory ~ n_regions_c + (1 | patient_id),
                  data = dat, family = binomial, control = ctrl)
print(summary(m_region))
save_tidy(m_region, "spread_region_odds_ratios.csv")
save_diagnostics(m_region, "spread_region_diagnostics.csv")
save_lrt(m_null, m_region, "lrt_spread_region_vs_null.csv")

# ── Model 2: Channel spread ──
cat("\n=== CHANNEL SPREAD MODEL ===\n")
m_channel <- glmer(memory ~ n_channels_c + (1 | patient_id),
                   data = dat, family = binomial, control = ctrl)
print(summary(m_channel))
save_tidy(m_channel, "spread_channel_odds_ratios.csv")
save_diagnostics(m_channel, "spread_channel_diagnostics.csv")
save_lrt(m_null, m_channel, "lrt_spread_channel_vs_null.csv")

cat("\n===== DONE =====\n")
