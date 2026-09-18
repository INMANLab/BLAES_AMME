##############################################################################
#  Power x IED Timing Windows MLM - Encoding Only
#
#  Per-region power (BLA = amygdala, HPC) on IED trials only.
#  Tests whether trial-level power predicts subsequent memory,
#  controlling for IED timing windows and stimulation.
#
#  Streamlined to theta + slow gamma bands and the after-image + during-stim
#  IED timing windows only (before-image / during-image windows and HFA dropped;
#  spread and 3-way models dropped).
#
#  Models per region and band (theta, slow gamma):
#    Part 1: All power trials (no IED data needed)
#      A:  memory ~ pow_band_z + stim + (1|patient_id)
#    Part 2: IED+power merged trials
#      T:  memory ~ pow_band_z + ied_after_image + ied_during_stim + (1|patient_id)
#      TI: memory ~ pow_band_z * (ied_after_image + ied_during_stim) + (1|patient_id)
#      TF: memory ~ pow_band_z + ied_during_stim + ied_after_image + stim + (1|patient_id)
#      SO: TI model fit within stim trials only (sensitivity)
#      AI: memory ~ pow_band_z * stim + (1|patient_id), after-image IED trials only
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
# Outputs/data live in the sibling IED/ied_timing_memory dir (scripts were
# moved into mlm_code/ during the topic-dir reorg, but the data did not move).
out_dir <- file.path(dirname(script_dir), "IED", "ied_timing_memory")
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
  # ICC via latent-variable approach for binary GLMM (logit link)
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

# Streamlined: theta + slow gamma only (drop HFA), amygdala (BLA) + HPC only.
bands <- c("theta", "slow_gamma")
band_labels <- c("Theta (4-8 Hz)", "Slow Gamma (30-55 Hz)")
regions <- c("BLA", "HPC")

##############################################################################
# PART 1: ALL POWER TRIALS -> MEMORY
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 1: POWER -> MEMORY (ALL TRIALS)\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

pw_all <- read.csv(file.path(out_dir, "power_all.csv"), stringsAsFactors = FALSE)

for (reg in regions) {
  sub <- pw_all[pw_all$region == reg, ]
  sub$patient_id <- factor(sub$patient_id)

  if (nrow(sub) < 20 || nlevels(sub$patient_id) < 3) {
    cat(sprintf("  Skipping %s: too few data\n", reg))
    next
  }

  cat(sprintf("\n--- %s: %d trials, %d patients ---\n", reg, nrow(sub), nlevels(sub$patient_id)))

  for (b in bands) {
    col <- paste0("pow_", b)
    sub[[paste0(col, "_z")]] <- as.numeric(scale(sub[[col]]))
  }

  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")
    cat(sprintf("\n  MODEL A_%s_%s: %s (All Trials)\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0("memory ~ ", zcol, " + stim + (1 | patient_id)")),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      print(summary(m))
      save_tidy(m, sprintf("pow_A_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("pow_A_%s_%s_diagnostics.csv", reg, b))
    }
  }
}

##############################################################################
# PART 2: POWER x IED TIMING WINDOWS -> MEMORY (IED TRIALS ONLY)
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 2: POWER x IED TIMING WINDOWS (IED TRIALS ONLY)\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

ied_pw <- read.csv(file.path(out_dir, "power_ied_merged.csv"), stringsAsFactors = FALSE)

for (reg in regions) {
  sub <- ied_pw[ied_pw$region == reg, ]
  # Filter patients with >= 5 trials
  pc <- table(sub$patient_id)
  kp <- names(pc)[pc >= 5]
  sub <- sub[sub$patient_id %in% kp, ]
  sub$patient_id <- factor(sub$patient_id)

  if (nrow(sub) < 20 || nlevels(sub$patient_id) < 3) {
    cat(sprintf("  Skipping %s: %d trials, %d patients (insufficient)\n",
                reg, nrow(sub), nlevels(sub$patient_id)))
    next
  }

  cat(sprintf("\n--- %s: %d IED trials, %d patients (>=5 each) ---\n",
              reg, nrow(sub), nlevels(sub$patient_id)))
  cat(sprintf("  Memory rate: %.1f%%\n", mean(sub$memory) * 100))

  cat("  Timing window distribution:\n")
  for (v in c("ied_after_image", "ied_during_stim")) {
    n1 <- sum(sub[[v]])
    cat(sprintf("    %s: %d/%d (%.1f%%)\n", v, n1, nrow(sub),
                n1 / nrow(sub) * 100))
  }

  for (b in bands) {
    col <- paste0("pow_", b)
    sub[[paste0(col, "_z")]] <- as.numeric(scale(sub[[col]]))
  }

  # Part 2a: Main effects with timing windows
  t_models <- list()  # store for LRT comparison with TI models
  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")

    cat(sprintf("\n  MODEL T_%s_%s: %s + Timing Windows\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " + ied_after_image + ied_during_stim + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      print(summary(m))
      save_tidy(m, sprintf("pow_T_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("pow_T_%s_%s_diagnostics.csv", reg, b))
      t_models[[b]] <- m
    }
  }

  # Part 2b: Interaction models
  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")

    cat(sprintf("\n  MODEL TI_%s_%s: %s x Timing Interactions\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " * ied_after_image + ",
        zcol, " * ied_during_stim + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      print(summary(m))
      save_tidy(m, sprintf("pow_TI_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("pow_TI_%s_%s_diagnostics.csv", reg, b))
      # LRT: interaction model (TI) vs main-effects model (T)
      if (!is.null(t_models[[b]])) {
        save_lrt(t_models[[b]], m, sprintf("lrt_TI_vs_T_%s_%s.csv", reg, b))
      }
    }
  }

  # Part 2b-robustness: the power x IED-during-stim interaction estimate depends
  # on which other window interactions are in the model (the windows co-occur on
  # trials, so the interaction terms are correlated). Fit two extra
  # specifications for sensitivity reporting alongside the 2-window TI model:
  #   TI4 = power x all four windows (before/during-image + after-image + during-stim)
  #   TID = power x during-stim only (the headline term in isolation)
  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")

    cat(sprintf("\n  MODEL TI4_%s_%s: %s x ALL FOUR windows\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " * ied_before_image + ", zcol, " * ied_during_image + ",
        zcol, " * ied_after_image + ", zcol, " * ied_during_stim + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      save_tidy(m, sprintf("pow_TI4_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("pow_TI4_%s_%s_diagnostics.csv", reg, b))
    }

    cat(sprintf("\n  MODEL TID_%s_%s: %s x during-stim only\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " * ied_during_stim + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      save_tidy(m, sprintf("pow_TID_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("pow_TID_%s_%s_diagnostics.csv", reg, b))
    }
  }

  # Part 2c: Focused model - during-stim + after-image windows + stim covariate
  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")

    cat(sprintf("\n  MODEL TF_%s_%s: %s + During-Stim + After-Image + Stim\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " + ied_during_stim + ied_after_image + stim + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      print(summary(m))
      save_tidy(m, sprintf("pow_TF_%s_%s_odds_ratios.csv", reg, b))
      save_diagnostics(m, sprintf("pow_TF_%s_%s_diagnostics.csv", reg, b))
    }
  }

  # Part 2d: Stim sensitivity - Power x IED timing within STIM TRIALS ONLY (SO)
  # Sensitivity check that the Power x IED-during-stim interaction is not an
  # artifact of stim vs. sham differences. Same z-scoring as the full-sample
  # models above (sub already carries pow_*_z); restrict to stim trials.
  sub_stim <- sub[sub$stim == 1, ]
  sub_stim$patient_id <- factor(sub_stim$patient_id)
  if (nrow(sub_stim) >= 20 && nlevels(sub_stim$patient_id) >= 3) {
    cat(sprintf("\n  --- %s stim-only subsample: %d trials, %d patients ---\n",
                reg, nrow(sub_stim), nlevels(sub_stim$patient_id)))
    for (i in seq_along(bands)) {
      b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")
      cat(sprintf("\n  MODEL SO_%s_%s: %s x IED timing (stim trials only)\n", reg, b, bl))
      m <- tryCatch({
        glmer(as.formula(paste0(
          "memory ~ ", zcol, " * ied_after_image + ",
          zcol, " * ied_during_stim + (1 | patient_id)"
        )), data = sub_stim, family = binomial, control = ctrl)
      }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
      if (!is.null(m)) {
        print(summary(m))
        save_tidy(m, sprintf("pow_SO_%s_%s_odds_ratios.csv", reg, b))
        save_diagnostics(m, sprintf("pow_SO_%s_%s_diagnostics.csv", reg, b))
      }
    }
  }

  # Part 2e: Power x Stim interaction on AFTER-IMAGE IED trials (AI)
  # Tests whether stimulation moderates the power-memory relationship on the
  # timing window with adequate stim/sham representation (after-image). The
  # during-stim window is confounded with stim by design, so it cannot be used.
  sub_ai <- ied_pw[ied_pw$region == reg & ied_pw$ied_after_image == 1, ]
  pc_ai <- table(sub_ai$patient_id)
  kp_ai <- names(pc_ai)[pc_ai >= 5]
  sub_ai <- sub_ai[sub_ai$patient_id %in% kp_ai, ]
  sub_ai$patient_id <- factor(sub_ai$patient_id)
  if (nrow(sub_ai) >= 20 && nlevels(sub_ai$patient_id) >= 3) {
    for (b in bands) {
      sub_ai[[paste0("pow_", b, "_z")]] <- as.numeric(scale(sub_ai[[paste0("pow_", b)]]))
    }
    cat(sprintf("\n  --- %s after-image subsample: %d trials, %d patients ---\n",
                reg, nrow(sub_ai), nlevels(sub_ai$patient_id)))
    for (i in seq_along(bands)) {
      b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")
      cat(sprintf("\n  MODEL AI_%s_%s: %s x Stim (after-image trials)\n", reg, b, bl))
      m <- tryCatch({
        glmer(as.formula(paste0(
          "memory ~ ", zcol, " * stim + (1 | patient_id)"
        )), data = sub_ai, family = binomial, control = ctrl)
      }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
      if (!is.null(m)) {
        print(summary(m))
        save_tidy(m, sprintf("pow_AI_%s_%s_odds_ratios.csv", reg, b))
        save_diagnostics(m, sprintf("pow_AI_%s_%s_diagnostics.csv", reg, b))
      }
    }
  }
}

cat("\n\n===== ALL DONE =====\n")
