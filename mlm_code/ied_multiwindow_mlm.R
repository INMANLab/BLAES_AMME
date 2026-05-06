##############################################################################
#  IED Multi-Window Dose-Response MLM
#
#  Tests whether the NUMBER of IED timing windows on a trial (dose) predicts
#  subsequent memory, beyond the patient random intercept. Runs separately
#  for encoding (4 possible windows) and retrieval (2 possible windows).
#
#  Models (encoding):
#    E1  memory ~ n_windows               + (1 | patient_id)     # continuous dose
#    E2  memory ~ factor(n_windows)       + (1 | patient_id)     # omnibus across levels
#    E3  memory ~ n_windows + stim        + (1 | patient_id)     # control for stim
#    E4  memory ~ n_windows * stim        + (1 | patient_id)     # does stim modulate dose effect
#
#  Models (retrieval):
#    R1  memory ~ n_windows               + (1 | patient_id)
#    R2  memory ~ factor(n_windows)       + (1 | patient_id)
#
#  Outputs in outputs/ied_timing_memory/:
#    encoding_multiwindow_mlm_coefs.csv
#    retrieval_multiwindow_mlm_coefs.csv
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
  if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
  if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")
  if (!require("performance")) install.packages("performance",  repos = "https://cloud.r-project.org")
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
  library(performance)
})

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
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

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

enc_csv <- file.path(script_dir, "IED",
                     "AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv")
ret_csv <- file.path(script_dir, "IED",
                     "AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv")

yn_to_num <- function(v) as.integer(v == "Y")

collapse_trials <- function(df, timing_cols, group_cols) {
  df <- df[df$MemoryOutcome %in% c("remembered", "forgotten"), ]
  for (c in timing_cols) df[[c]] <- yn_to_num(df[[c]])
  agg <- aggregate(df[, timing_cols, drop = FALSE],
                   by = df[, group_cols, drop = FALSE],
                   FUN = function(x) as.integer(any(x == 1)))
  agg$n_windows <- rowSums(agg[, timing_cols, drop = FALSE])
  agg$memory <- as.integer(agg$MemoryOutcome == "remembered")
  agg$patient_id <- factor(agg$Patient)
  agg
}

fit_and_report <- function(formula, data, label, diag_file = NULL) {
  cat("\n------", label, "------\n")
  cat("Formula:", deparse(formula), "\n")
  m <- tryCatch(glmer(formula, data = data, family = binomial, control = ctrl),
                error = function(e) { cat("FAILED:", conditionMessage(e), "\n"); NULL })
  if (is.null(m)) return(list(tidy = NULL, model = NULL))
  print(summary(m)$coefficients)
  td <- broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  td$model <- label
  if (!is.null(diag_file)) save_diagnostics(m, diag_file)
  list(tidy = td, model = m)
}

##############################################################################
# ENCODING
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n", sep="")
cat("ENCODING MULTI-WINDOW MLM\n")
cat(paste(rep("=", 70), collapse=""), "\n", sep="")

enc_raw <- read.csv(enc_csv, stringsAsFactors = FALSE)
enc <- collapse_trials(enc_raw,
                       timing_cols = c("BeforeImgITI", "DuringImg", "AfterImgITI", "DuringStim"),
                       group_cols  = c("Patient", "Trial", "MemoryOutcome", "StimCond"))
enc$stim <- factor(enc$StimCond, levels = c("NS", "S"))
cat("Encoding trials:", nrow(enc),
    "| patients:", nlevels(enc$patient_id),
    "| remembered:", sum(enc$memory),
    "| forgotten:", sum(1 - enc$memory), "\n")
cat("n_windows distribution:\n"); print(table(enc$n_windows))

enc_tidys <- list()

# Fit null model first so all LRTs compare against it
m_null <- glmer(memory ~ 1 + (1 | patient_id), data = enc, family = binomial, control = ctrl)
save_diagnostics(m_null, "mw_enc_null_diagnostics.csv")

# E1: continuous dose
cat("\n------ E1: n_windows (continuous) ------\n")
m_e1 <- glmer(memory ~ n_windows + (1 | patient_id), data = enc, family = binomial, control = ctrl)
print(summary(m_e1)$coefficients)
enc_tidys[[1]] <- { td <- broom.mixed::tidy(m_e1, effects="fixed", conf.int=TRUE, exponentiate=TRUE); td$model <- "E1: n_windows (continuous)"; td }
save_diagnostics(m_e1, "mw_E1_diagnostics.csv")
save_lrt(m_null, m_e1, "lrt_mw_E1_vs_null.csv")

# E2: factor dose (omnibus LRT against null)
cat("\n------ E2: factor(n_windows) ------\n")
m_fac <- glmer(memory ~ factor(n_windows) + (1 | patient_id),
               data = enc, family = binomial, control = ctrl)
print(summary(m_fac)$coefficients)
cat("\nE2 omnibus LRT (factor n_windows vs intercept-only):\n")
print(anova(m_null, m_fac))
save_diagnostics(m_fac, "mw_E2_diagnostics.csv")
save_lrt(m_null, m_fac, "lrt_mw_E2_vs_null.csv")
enc_tidys[[2]] <- { td <- broom.mixed::tidy(m_fac, effects="fixed", conf.int=TRUE, exponentiate=TRUE); td$model <- "E2: factor(n_windows)"; td }

# E3: dose + stim
cat("\n------ E3: n_windows + stim ------\n")
m_e3 <- glmer(memory ~ n_windows + stim + (1 | patient_id), data = enc, family = binomial, control = ctrl)
print(summary(m_e3)$coefficients)
enc_tidys[[3]] <- { td <- broom.mixed::tidy(m_e3, effects="fixed", conf.int=TRUE, exponentiate=TRUE); td$model <- "E3: n_windows + stim"; td }
save_diagnostics(m_e3, "mw_E3_diagnostics.csv")

# E4: dose x stim interaction
cat("\n------ E4: n_windows * stim ------\n")
m_e4 <- glmer(memory ~ n_windows * stim + (1 | patient_id), data = enc, family = binomial, control = ctrl)
print(summary(m_e4)$coefficients)
enc_tidys[[4]] <- { td <- broom.mixed::tidy(m_e4, effects="fixed", conf.int=TRUE, exponentiate=TRUE); td$model <- "E4: n_windows * stim"; td }
save_diagnostics(m_e4, "mw_E4_diagnostics.csv")
save_lrt(m_e3, m_e4, "lrt_mw_E4_vs_E3.csv")

enc_out <- do.call(rbind, enc_tidys)
write.csv(enc_out, file.path(out_dir, "encoding_multiwindow_mlm_coefs.csv"), row.names = FALSE)
cat("\nSaved", file.path(out_dir, "encoding_multiwindow_mlm_coefs.csv"), "\n")

##############################################################################
# RETRIEVAL
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n", sep="")
cat("RETRIEVAL MULTI-WINDOW MLM\n")
cat(paste(rep("=", 70), collapse=""), "\n", sep="")

ret_raw <- read.csv(ret_csv, stringsAsFactors = FALSE)
ret <- collapse_trials(ret_raw,
                       timing_cols = c("BeforeImgITI", "DuringImgITI"),
                       group_cols  = c("Patient", "Trial", "MemoryOutcome"))
cat("Retrieval trials:", nrow(ret),
    "| patients:", nlevels(ret$patient_id),
    "| remembered:", sum(ret$memory),
    "| forgotten:", sum(1 - ret$memory), "\n")
cat("n_windows distribution:\n"); print(table(ret$n_windows))

ret_tidys <- list()

m_null_r <- glmer(memory ~ 1 + (1 | patient_id), data = ret, family = binomial, control = ctrl)
save_diagnostics(m_null_r, "mw_ret_null_diagnostics.csv")

# R1: continuous dose
cat("\n------ R1: n_windows (continuous) ------\n")
m_r1 <- glmer(memory ~ n_windows + (1 | patient_id), data = ret, family = binomial, control = ctrl)
print(summary(m_r1)$coefficients)
ret_tidys[[1]] <- { td <- broom.mixed::tidy(m_r1, effects="fixed", conf.int=TRUE, exponentiate=TRUE); td$model <- "R1: n_windows (continuous)"; td }
save_diagnostics(m_r1, "mw_R1_diagnostics.csv")
save_lrt(m_null_r, m_r1, "lrt_mw_R1_vs_null.csv")

# R2: factor dose
cat("\n------ R2: factor(n_windows) ------\n")
m_fac_r <- glmer(memory ~ factor(n_windows) + (1 | patient_id),
                 data = ret, family = binomial, control = ctrl)
print(summary(m_fac_r)$coefficients)
cat("\nR2 omnibus LRT (factor n_windows vs intercept-only):\n")
print(anova(m_null_r, m_fac_r))
save_diagnostics(m_fac_r, "mw_R2_diagnostics.csv")
save_lrt(m_null_r, m_fac_r, "lrt_mw_R2_vs_null.csv")
ret_tidys[[2]] <- { td <- broom.mixed::tidy(m_fac_r, effects="fixed", conf.int=TRUE, exponentiate=TRUE); td$model <- "R2: factor(n_windows)"; td }

# R3: individual timing windows
cat("\n------ R3: BeforeImgITI + DuringImgITI ------\n")
m_r3 <- glmer(memory ~ BeforeImgITI + DuringImgITI + (1 | patient_id),
              data = ret, family = binomial, control = ctrl)
print(summary(m_r3)$coefficients)
ret_tidys[[3]] <- { td <- broom.mixed::tidy(m_r3, effects="fixed", conf.int=TRUE, exponentiate=TRUE); td$model <- "R3: timing windows"; td }
save_diagnostics(m_r3, "mw_R3_diagnostics.csv")
save_lrt(m_null_r, m_r3, "lrt_mw_R3_vs_null.csv")

ret_out <- do.call(rbind, ret_tidys)
write.csv(ret_out, file.path(out_dir, "retrieval_multiwindow_mlm_coefs.csv"), row.names = FALSE)
cat("\nSaved", file.path(out_dir, "retrieval_multiwindow_mlm_coefs.csv"), "\n")

cat("\nDone.\n")
