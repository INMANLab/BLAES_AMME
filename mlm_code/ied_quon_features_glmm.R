##############################################################################
#  Quon Figure 2A-style GLMM: IED features -> memory (encoding)
#
#  Replaces the Mann-Whitney / point-biserial tests in the original panel
#  with a single trial-level binomial GLMM (random intercept for patient)
#  containing all IED features as simultaneous predictors. The White Matter
#  (W-G) feature is dropped from the predictor list (per request), but ALL
#  IED rows are retained regardless of tissue type. All four encoding
#  timing windows are kept.
#
#  Trial-level feature construction (per Patient + Trial, among trials with
#  >= 1 gray-matter IED):
#    ied_rate        = mean Rate(within trials) across rows (constant per trial)
#    ch_spread       = max ChannelSpread across rows
#    reg_spread      = max RegionSpread across rows
#    any_R_ied       = 1 if any IED on R hemisphere (vs none -> 0)
#    before_img      = 1 if any row with BeforeImgITI == "Y"
#    during_img      = 1 if any row with DuringImg    == "Y"
#    after_img       = 1 if any row with AfterImgITI  == "Y"
#    during_stim     = 1 if any row with DuringStim   == "Y"
#    memory          = 1 if MemoryOutcome == "remembered"
#
#  Fits one joint multivariable model:
#    memory ~ ied_rate_z + any_R_ied + ch_spread_z + reg_spread_z +
#             before_img + during_img + after_img + during_stim +
#             (1 | patient_id)
#  Continuous features are z-scored before entering the model so the log-
#  odds beta is interpretable as "per 1 SD of feature" while adjusting for
#  the other features. Binary features contrast Yes vs. No (or R vs. none).
#
#  Outputs: outputs/ied_Quon_paper/figure2_glmm_coefs.csv
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
  if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
  if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
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
out_dir <- file.path(script_dir, "outputs", "ied_Quon_paper")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

enc_csv <- file.path(script_dir, "IED",
                     "AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv")

cat("Reading", enc_csv, "\n")
raw <- read.csv(enc_csv, stringsAsFactors = FALSE)
cat("Raw rows:", nrow(raw), "\n")

# Keep trials with a memory label (no tissue filter; all IEDs retained)
raw <- raw[raw$MemoryOutcome %in% c("remembered", "forgotten"), ]
cat("After memory filter:", nrow(raw), "rows\n")

yn <- function(v) as.integer(v == "Y")
raw$BeforeImgITI_num <- yn(raw$BeforeImgITI)
raw$DuringImg_num    <- yn(raw$DuringImg)
raw$AfterImgITI_num  <- yn(raw$AfterImgITI)
raw$DuringStim_num   <- yn(raw$DuringStim)
raw$hem_R            <- as.integer(raw$Hemisphere == "R")

# Collapse to per-trial ---------------------------------------------------
agg <- aggregate(
  cbind(BeforeImgITI_num, DuringImg_num, AfterImgITI_num, DuringStim_num,
        hem_R, ChannelSpread, RegionSpread, Rate.within.trials.) ~
    Patient + Trial + MemoryOutcome,
  data = raw,
  FUN = function(x) c(any_y = as.numeric(any(x > 0)),
                      max_v = max(x, na.rm = TRUE),
                      mean_v = mean(x, na.rm = TRUE))
)

# aggregate() returns matrix columns when FUN returns a vector; flatten them.
trial <- data.frame(
  patient_id  = factor(agg$Patient),
  trial       = agg$Trial,
  memory      = as.integer(agg$MemoryOutcome == "remembered"),
  before_img  = agg$BeforeImgITI_num[, "any_y"],
  during_img  = agg$DuringImg_num[,    "any_y"],
  after_img   = agg$AfterImgITI_num[,  "any_y"],
  during_stim = agg$DuringStim_num[,   "any_y"],
  any_R_ied   = agg$hem_R[,            "any_y"],
  ch_spread   = agg$ChannelSpread[,    "max_v"],
  reg_spread  = agg$RegionSpread[,     "max_v"],
  ied_rate    = agg$Rate.within.trials.[, "mean_v"]
)

cat("Trials (encoding, memory-labeled, all tissues):", nrow(trial),
    "| patients:", nlevels(trial$patient_id), "\n")
cat("Remembered:", sum(trial$memory), " Forgotten:", sum(1 - trial$memory), "\n")

zscore <- function(x) as.numeric(scale(x))

trial$ied_rate_z   <- zscore(trial$ied_rate)
trial$ch_spread_z  <- zscore(trial$ch_spread)
trial$reg_spread_z <- zscore(trial$reg_spread)

# -------------------------------------------------------------------------
# Feature list -> one joint multivariable model
# -------------------------------------------------------------------------
features <- list(
  list(name = "IED Rate",              term = "ied_rate_z",   kind = "continuous (per 1 SD)"),
  list(name = "Hemisphere (R vs. none)", term = "any_R_ied",  kind = "binary (any R IED vs. none)"),
  list(name = "Channel Spread",        term = "ch_spread_z",  kind = "continuous (per 1 SD)"),
  list(name = "Region Spread",         term = "reg_spread_z", kind = "continuous (per 1 SD)"),
  list(name = "Before Image",          term = "before_img",   kind = "binary (Y vs. N)"),
  list(name = "During Image",          term = "during_img",   kind = "binary (Y vs. N)"),
  list(name = "After Image",           term = "after_img",    kind = "binary (Y vs. N)"),
  list(name = "During Stim",           term = "during_stim",  kind = "binary (Y vs. N)")
)

# Print per-feature counts for the report note
for (f in features) {
  if (f$term %in% c("before_img", "during_img", "after_img",
                    "during_stim", "any_R_ied")) {
    cat(f$name, ": Y/N counts = ",
        sum(trial[[f$term]] == 1), "/", sum(trial[[f$term]] == 0), "\n", sep="")
  } else {
    cat(f$name, ": n =", sum(!is.na(trial[[f$term]])), "\n")
  }
}

full_fml <- memory ~ ied_rate_z + any_R_ied + ch_spread_z + reg_spread_z +
                     before_img + during_img + after_img + during_stim +
                     (1 | patient_id)

cat("\nFitting joint multivariable GLMM:\n")
cat(deparse(full_fml), "\n")

m_full <- glmer(full_fml, data = trial, family = binomial, control = ctrl)
cat("\n--- Joint model summary ---\n")
print(summary(m_full)$coefficients)

# Singularity / convergence diagnostics
cat("\nRandom-effects variance:\n")
print(VarCorr(m_full))
cat("isSingular:", isSingular(m_full), "\n")

tidy_full <- broom.mixed::tidy(m_full, effects = "fixed",
                               conf.int = TRUE, conf.method = "Wald",
                               exponentiate = FALSE)

# Map each term back to its human-readable feature name + kind
term_to_feature <- setNames(
  vapply(features, function(f) f$name, character(1)),
  vapply(features, function(f) f$term, character(1))
)
term_to_kind <- setNames(
  vapply(features, function(f) f$kind, character(1)),
  vapply(features, function(f) f$term, character(1))
)

tidy_full$feature <- ifelse(tidy_full$term == "(Intercept)",
                            "(Intercept)",
                            term_to_feature[tidy_full$term])
tidy_full$kind <- ifelse(tidy_full$term == "(Intercept)",
                         "intercept",
                         term_to_kind[tidy_full$term])

# Per-feature sample counts (feature=1 vs feature=0 for binaries)
n_yes <- integer(nrow(tidy_full))
n_no  <- integer(nrow(tidy_full))
for (i in seq_len(nrow(tidy_full))) {
  term_i <- tidy_full$term[i]
  if (term_i == "(Intercept)") { n_yes[i] <- nrow(trial); n_no[i] <- NA; next }
  if (term_i %in% c("before_img", "during_img", "after_img",
                    "during_stim", "any_R_ied")) {
    n_yes[i] <- sum(trial[[term_i]] == 1)
    n_no[i]  <- sum(trial[[term_i]] == 0)
  } else {
    n_yes[i] <- sum(!is.na(trial[[term_i]]))
    n_no[i]  <- NA_integer_
  }
}
tidy_full$n_yes <- n_yes
tidy_full$n_no  <- n_no

out <- tidy_full

# Also provide exponentiated OR columns alongside the log-odds
out$or        <- exp(out$estimate)
out$or_ci_low <- exp(out$conf.low)
out$or_ci_hi  <- exp(out$conf.high)

out_path <- file.path(out_dir, "figure2_glmm_coefs.csv")
write.csv(out, out_path, row.names = FALSE)
cat("\nSaved", out_path, "\n")

# Also persist the trial-level frame for downstream plotting / sanity checks
tlvl_path <- file.path(out_dir, "figure2_glmm_trial_features.csv")
write.csv(trial, tlvl_path, row.names = FALSE)
cat("Saved", tlvl_path, "\n")


##############################################################################
#  RETRIEVAL
#
#  Same joint multivariable GLMM approach on retrieval-phase IED data.
#  Retrieval timing has only two windows (BeforeImgITI, DuringImgITI);
#  the "During Stim" and "After Image" features do not exist for retrieval.
#  Only remembered / forgotten old items are included (new_item excluded).
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n", sep="")
cat("RETRIEVAL: JOINT GLMM\n")
cat(paste(rep("=", 70), collapse=""), "\n", sep="")

ret_csv <- file.path(script_dir, "IED",
                     "AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv")
ret_raw <- read.csv(ret_csv, stringsAsFactors = FALSE)
cat("Retrieval raw rows:", nrow(ret_raw), "\n")

ret_raw <- ret_raw[ret_raw$MemoryOutcome %in% c("remembered", "forgotten"), ]
cat("After memory filter:", nrow(ret_raw), "rows\n")

ret_raw$BeforeImgITI_num <- yn(ret_raw$BeforeImgITI)
ret_raw$DuringImgITI_num <- yn(ret_raw$DuringImgITI)
ret_raw$hem_R            <- as.integer(ret_raw$Hemisphere == "R")

ret_agg <- aggregate(
  cbind(BeforeImgITI_num, DuringImgITI_num,
        hem_R, ChannelSpread, RegionSpread, Rate.within.trials.) ~
    Patient + Trial + MemoryOutcome,
  data = ret_raw,
  FUN = function(x) c(any_y = as.numeric(any(x > 0)),
                      max_v = max(x, na.rm = TRUE),
                      mean_v = mean(x, na.rm = TRUE))
)

ret_trial <- data.frame(
  patient_id  = factor(ret_agg$Patient),
  trial       = ret_agg$Trial,
  memory      = as.integer(ret_agg$MemoryOutcome == "remembered"),
  before_img  = ret_agg$BeforeImgITI_num[, "any_y"],
  during_img  = ret_agg$DuringImgITI_num[, "any_y"],
  any_R_ied   = ret_agg$hem_R[,           "any_y"],
  ch_spread   = ret_agg$ChannelSpread[,   "max_v"],
  reg_spread  = ret_agg$RegionSpread[,    "max_v"],
  ied_rate    = ret_agg$Rate.within.trials.[, "mean_v"]
)

cat("Retrieval trials (all tissues, remembered/forgotten):",
    nrow(ret_trial),
    "| patients:", nlevels(ret_trial$patient_id), "\n")
cat("Remembered:", sum(ret_trial$memory),
    " Forgotten:", sum(1 - ret_trial$memory), "\n")

ret_trial$ied_rate_z   <- zscore(ret_trial$ied_rate)
ret_trial$ch_spread_z  <- zscore(ret_trial$ch_spread)
ret_trial$reg_spread_z <- zscore(ret_trial$reg_spread)

ret_features <- list(
  list(name = "IED Rate",              term = "ied_rate_z",   kind = "continuous (per 1 SD)"),
  list(name = "Hemisphere (R vs. none)", term = "any_R_ied",  kind = "binary (any R IED vs. none)"),
  list(name = "Channel Spread",        term = "ch_spread_z",  kind = "continuous (per 1 SD)"),
  list(name = "Region Spread",         term = "reg_spread_z", kind = "continuous (per 1 SD)"),
  list(name = "Before Image",          term = "before_img",   kind = "binary (Y vs. N)"),
  list(name = "During Image",          term = "during_img",   kind = "binary (Y vs. N)")
)

for (f in ret_features) {
  if (f$term %in% c("before_img", "during_img", "any_R_ied")) {
    cat(f$name, ": Y/N counts = ",
        sum(ret_trial[[f$term]] == 1), "/",
        sum(ret_trial[[f$term]] == 0), "\n", sep="")
  } else {
    cat(f$name, ": n =", sum(!is.na(ret_trial[[f$term]])), "\n")
  }
}

ret_full_fml <- memory ~ ied_rate_z + any_R_ied + ch_spread_z + reg_spread_z +
                         before_img + during_img +
                         (1 | patient_id)
cat("\nFitting retrieval joint multivariable GLMM:\n")
cat(deparse(ret_full_fml), "\n")

m_ret <- tryCatch(glmer(ret_full_fml, data = ret_trial, family = binomial,
                        control = ctrl),
                  error = function(e) { cat("FAILED:", conditionMessage(e), "\n"); NULL })
if (!is.null(m_ret)) {
  cat("\n--- Retrieval joint model summary ---\n")
  print(summary(m_ret)$coefficients)
  cat("\nRandom-effects variance:\n")
  print(VarCorr(m_ret))
  cat("isSingular:", isSingular(m_ret), "\n")

  ret_tidy <- broom.mixed::tidy(m_ret, effects = "fixed",
                                conf.int = TRUE, conf.method = "Wald",
                                exponentiate = FALSE)

  ret_term_to_feature <- setNames(
    vapply(ret_features, function(f) f$name, character(1)),
    vapply(ret_features, function(f) f$term, character(1))
  )
  ret_term_to_kind <- setNames(
    vapply(ret_features, function(f) f$kind, character(1)),
    vapply(ret_features, function(f) f$term, character(1))
  )
  ret_tidy$feature <- ifelse(ret_tidy$term == "(Intercept)", "(Intercept)",
                             ret_term_to_feature[ret_tidy$term])
  ret_tidy$kind <- ifelse(ret_tidy$term == "(Intercept)", "intercept",
                          ret_term_to_kind[ret_tidy$term])

  ret_n_yes <- integer(nrow(ret_tidy))
  ret_n_no  <- integer(nrow(ret_tidy))
  for (i in seq_len(nrow(ret_tidy))) {
    term_i <- ret_tidy$term[i]
    if (term_i == "(Intercept)") {
      ret_n_yes[i] <- nrow(ret_trial); ret_n_no[i] <- NA; next
    }
    if (term_i %in% c("before_img", "during_img", "any_R_ied")) {
      ret_n_yes[i] <- sum(ret_trial[[term_i]] == 1)
      ret_n_no[i]  <- sum(ret_trial[[term_i]] == 0)
    } else {
      ret_n_yes[i] <- sum(!is.na(ret_trial[[term_i]]))
      ret_n_no[i]  <- NA_integer_
    }
  }
  ret_tidy$n_yes <- ret_n_yes
  ret_tidy$n_no  <- ret_n_no
  ret_tidy$or        <- exp(ret_tidy$estimate)
  ret_tidy$or_ci_low <- exp(ret_tidy$conf.low)
  ret_tidy$or_ci_hi  <- exp(ret_tidy$conf.high)

  ret_out_path <- file.path(out_dir, "figure2_glmm_coefs_retrieval.csv")
  write.csv(ret_tidy, ret_out_path, row.names = FALSE)
  cat("\nSaved", ret_out_path, "\n")

  ret_tlvl_path <- file.path(out_dir, "figure2_glmm_trial_features_retrieval.csv")
  write.csv(ret_trial, ret_tlvl_path, row.names = FALSE)
  cat("Saved", ret_tlvl_path, "\n")
}
