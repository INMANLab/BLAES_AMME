##############################################################################
#  Power x IED Timing Windows MLM - RETRIEVAL phase
#
#  Mirror of power_ied_mlm.R but for retrieval (phase3). Per-region power
#  (BLA = amygdala, HPC) on retrieved old-item IED trials. Tests whether
#  trial-level power predicts subsequent memory (remembered vs forgotten),
#  controlling for IED timing windows and prior stimulation.
#
#  Retrieval differences from encoding:
#    - IED windows are BeforeImg + DuringImg only (no during-stim/after-image:
#      there is no stimulation at retrieval).
#    - The stim analog is prior_stim (stimulated at STUDY, S vs NS), coded
#      stim = 1 for S. It is NOT confounded with any window, so the power x
#      prior_stim model is fit on all old-item trials (no window restriction).
#
#  Models per region and band (theta, slow gamma); prefix "powret":
#    A:  memory ~ pow + stim + (1|patient)                [all retrieval trials]
#    T:  memory ~ pow + ied_before_image + ied_during_image + (1|patient)
#    TI: memory ~ pow * (ied_before_image + ied_during_image) + (1|patient)
#    TF: memory ~ pow + ied_before_image + ied_during_image + stim + (1|patient)
#    AI: memory ~ pow * stim + (1|patient)                [old-item IED trials]
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
out_dir <- file.path(dirname(script_dir), "IED", "ied_timing_memory")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

save_tidy <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  td
}

save_diagnostics <- function(model, filename) {
  vc <- as.data.frame(VarCorr(model))
  rand_var <- vc$vcov[1]; rand_sd <- vc$sdcor[1]
  icc_val  <- rand_var / (rand_var + pi^2 / 3)
  r2 <- tryCatch(r2_nakagawa(model),
                 error = function(e) list(R2_marginal = NA, R2_conditional = NA))
  diag_df <- data.frame(
    metric = c("ICC_latent", "rand_intercept_var", "rand_intercept_sd",
               "AIC", "BIC", "logLik", "deviance", "nobs", "ngrps",
               "R2_marginal", "R2_conditional"),
    value = c(icc_val, rand_var, rand_sd,
              AIC(model), BIC(model), as.numeric(logLik(model)),
              deviance(model), nobs(model), ngrps(model)[[1]],
              r2$R2_marginal, r2$R2_conditional))
  write.csv(diag_df, file.path(out_dir, filename), row.names = FALSE)
  diag_df
}

save_lrt <- function(m_reduced, m_full, filename) {
  lr <- anova(m_reduced, m_full)
  lrt_df <- as.data.frame(lr)
  lrt_df$model <- c("Reduced", "Full")
  write.csv(lrt_df, file.path(out_dir, filename), row.names = FALSE)
  lrt_df
}

bands <- c("theta", "slow_gamma")
band_labels <- c("Theta (4-8 Hz)", "Slow Gamma (30-55 Hz)")
regions <- c("BLA", "HPC")

##############################################################################
# PART 1: ALL RETRIEVAL POWER TRIALS -> MEMORY
##############################################################################
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("PART 1: POWER -> MEMORY (ALL RETRIEVAL TRIALS)\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

pw_all <- read.csv(file.path(out_dir, "retrieval_power_all.csv"),
                   stringsAsFactors = FALSE)

for (reg in regions) {
  sub <- pw_all[pw_all$region == reg, ]
  sub$patient_id <- factor(sub$patient_id)
  if (nrow(sub) < 20 || nlevels(sub$patient_id) < 3) next
  cat(sprintf("\n--- %s: %d trials, %d patients ---\n",
              reg, nrow(sub), nlevels(sub$patient_id)))
  for (b in bands) sub[[paste0("pow_", b, "_z")]] <- as.numeric(scale(sub[[paste0("pow_", b)]]))
  for (i in seq_along(bands)) {
    b <- bands[i]; zcol <- paste0("pow_", b, "_z")
    m <- tryCatch(
      glmer(as.formula(paste0("memory ~ ", zcol, " + stim + (1 | patient_id)")),
            data = sub, family = binomial, control = ctrl),
      error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      save_tidy(m, sprintf("powret_A_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("powret_A_%s_%s_diagnostics.csv", reg, b))
    }
  }
}

##############################################################################
# PART 2: POWER x IED TIMING WINDOWS -> MEMORY (OLD-ITEM IED TRIALS)
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse = ""), "\n")
cat("PART 2: POWER x IED TIMING WINDOWS (RETRIEVAL OLD-ITEM IED TRIALS)\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

ied_pw <- read.csv(file.path(out_dir, "retrieval_power_ied_merged.csv"),
                   stringsAsFactors = FALSE)

for (reg in regions) {
  sub <- ied_pw[ied_pw$region == reg, ]
  pc <- table(sub$patient_id)
  sub <- sub[sub$patient_id %in% names(pc)[pc >= 5], ]
  sub$patient_id <- factor(sub$patient_id)
  if (nrow(sub) < 20 || nlevels(sub$patient_id) < 3) {
    cat(sprintf("  Skipping %s: %d trials, %d patients (insufficient)\n",
                reg, nrow(sub), nlevels(sub$patient_id)))
    next
  }
  cat(sprintf("\n--- %s: %d IED trials, %d patients (>=5 each) ---\n",
              reg, nrow(sub), nlevels(sub$patient_id)))
  cat(sprintf("  Memory rate: %.1f%%\n", mean(sub$memory) * 100))
  for (v in c("ied_before_image", "ied_during_image")) {
    cat(sprintf("    %s: %d/%d (%.1f%%)\n", v, sum(sub[[v]]), nrow(sub),
                sum(sub[[v]]) / nrow(sub) * 100))
  }
  for (b in bands) sub[[paste0("pow_", b, "_z")]] <- as.numeric(scale(sub[[paste0("pow_", b)]]))

  # Part 2a: Main effects with timing windows
  t_models <- list()
  for (i in seq_along(bands)) {
    b <- bands[i]; zcol <- paste0("pow_", b, "_z")
    cat(sprintf("\n  MODEL T_%s_%s\n", reg, b))
    m <- tryCatch(
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " + ied_before_image + ied_during_image + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl),
      error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      save_tidy(m, sprintf("powret_T_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("powret_T_%s_%s_diagnostics.csv", reg, b))
      t_models[[b]] <- m
    }
  }

  # Part 2b: Interaction models
  for (i in seq_along(bands)) {
    b <- bands[i]; zcol <- paste0("pow_", b, "_z")
    cat(sprintf("\n  MODEL TI_%s_%s\n", reg, b))
    m <- tryCatch(
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " * ied_before_image + ",
        zcol, " * ied_during_image + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl),
      error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      save_tidy(m, sprintf("powret_TI_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("powret_TI_%s_%s_diagnostics.csv", reg, b))
      if (!is.null(t_models[[b]])) {
        save_lrt(t_models[[b]], m, sprintf("lrt_ret_TI_vs_T_%s_%s.csv", reg, b))
      }
    }
  }

  # Part 2c: Focused model (windows + prior_stim covariate)
  for (i in seq_along(bands)) {
    b <- bands[i]; zcol <- paste0("pow_", b, "_z")
    cat(sprintf("\n  MODEL TF_%s_%s\n", reg, b))
    m <- tryCatch(
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " + ied_before_image + ied_during_image + stim + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl),
      error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      save_tidy(m, sprintf("powret_TF_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("powret_TF_%s_%s_diagnostics.csv", reg, b))
    }
  }

  # Part 2d: Power x prior_stim interaction (all old-item IED trials)
  for (i in seq_along(bands)) {
    b <- bands[i]; zcol <- paste0("pow_", b, "_z")
    cat(sprintf("\n  MODEL AI_%s_%s\n", reg, b))
    m <- tryCatch(
      glmer(as.formula(paste0("memory ~ ", zcol, " * stim + (1 | patient_id)")),
            data = sub, family = binomial, control = ctrl),
      error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      save_tidy(m, sprintf("powret_AI_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("powret_AI_%s_%s_diagnostics.csv", reg, b))
    }
  }
}

cat("\n\n===== RETRIEVAL ALL DONE =====\n")
