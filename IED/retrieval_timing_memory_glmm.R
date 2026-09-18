##############################################################################
#  Retrieval IED Timing x Memory GLMMs
#
#  Binomial GLMMs (patient random intercept) at RETRIEVAL, one row per
#  retrieved old item (remembered = 1 / forgotten = 0).
#
#  Model A - timing main effects:
#     memory ~ BeforeImg + DuringImg + (1 | patient_id)
#     FDR family 1 = the two timing-window main effects {BeforeImg, DuringImg}.
#
#  Model B - prior-stimulation interaction:
#     memory ~ BeforeImg * prior_stim + DuringImg * prior_stim + (1|patient_id)
#     FDR family 2 = the two interactions {BeforeImg:stim, DuringImg:stim}.
#
#  Predictors are IED present (Y) vs absent (N) in each window; prior_stim is
#  whether the item was stimulated at study (S) vs not (NS), reference = NS.
#
#  Input  : IED/ied_timing_memory/retrieval_timing_memory_trials.csv
#  Outputs: IED/ied_timing_memory/
#     retrieval_timing_maineffects_coefs.csv
#     retrieval_timing_interaction_coefs.csv
#     retrieval_timing_maineffects_diagnostics.csv
#     retrieval_timing_interaction_diagnostics.csv
#     retrieval_timing_lrt.csv
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
  if (!require("lmerTest"))    install.packages("lmerTest",    repos = "https://cloud.r-project.org")
  if (!require("broom.mixed")) install.packages("broom.mixed", repos = "https://cloud.r-project.org")
  if (!require("performance")) install.packages("performance", repos = "https://cloud.r-project.org")
  library(lme4)
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
out_dir <- file.path(script_dir, "ied_timing_memory")   # script lives in IED/
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

##############################################################################
# Data
##############################################################################
dat <- read.csv(file.path(out_dir, "retrieval_timing_memory_trials.csv"),
                stringsAsFactors = FALSE)
dat$BeforeImg  <- factor(dat$BeforeImg,  levels = c("N", "Y"))
dat$DuringImg  <- factor(dat$DuringImg,  levels = c("N", "Y"))
dat$prior_stim <- factor(dat$prior_stim, levels = c("NS", "S"))
dat$patient_id <- factor(dat$Patient)

cat("Retrieval timing GLMM data:\n")
cat("  trials:", nrow(dat),
    "| patients:", nlevels(dat$patient_id),
    "| remembered:", sum(dat$memory),
    "| forgotten:", sum(1 - dat$memory), "\n")
cat("  BeforeImg (Y):", sum(dat$BeforeImg == "Y"),
    "| DuringImg (Y):", sum(dat$DuringImg == "Y"),
    "| prior_stim (S):", sum(dat$prior_stim == "S"), "\n\n")

bh <- function(p) p.adjust(p, method = "BH")

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

tidy_or <- function(model, label) {
  td <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE,
                          exponentiate = TRUE)
  td$model <- label
  td
}

##############################################################################
# Model A - timing main effects
##############################################################################
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("MODEL A: memory ~ BeforeImg + DuringImg + (1 | patient_id)\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
m_null <- glmer(memory ~ 1 + (1 | patient_id), data = dat,
                family = binomial, control = ctrl)
m_main <- glmer(memory ~ BeforeImg + DuringImg + (1 | patient_id),
                data = dat, family = binomial, control = ctrl)
print(summary(m_main)$coefficients)

main_td <- tidy_or(m_main, "A: timing main effects")
# FDR family 1: the two timing-window main effects
fam1 <- main_td$term %in% c("BeforeImgY", "DuringImgY")
main_td$q_fdr <- NA_real_
main_td$q_fdr[fam1] <- bh(main_td$p.value[fam1])
main_td$fdr_family <- ifelse(fam1, "timing_main_effects", NA)
write.csv(main_td, file.path(out_dir, "retrieval_timing_maineffects_coefs.csv"),
          row.names = FALSE)
save_diagnostics(m_main, "retrieval_timing_maineffects_diagnostics.csv")

##############################################################################
# Model B - prior-stimulation interaction
##############################################################################
cat("\n", paste(rep("=", 70), collapse = ""), "\n", sep = "")
cat("MODEL B: memory ~ BeforeImg*prior_stim + DuringImg*prior_stim + (1|patient_id)\n")
cat(paste(rep("=", 70), collapse = ""), "\n", sep = "")
m_int  <- glmer(memory ~ BeforeImg * prior_stim + DuringImg * prior_stim +
                  (1 | patient_id), data = dat, family = binomial, control = ctrl)
# additive (no-interaction) model for the joint LRT of the two interactions
m_add  <- glmer(memory ~ BeforeImg + DuringImg + prior_stim + (1 | patient_id),
                data = dat, family = binomial, control = ctrl)
print(summary(m_int)$coefficients)

int_td <- tidy_or(m_int, "B: prior-stim interaction")
# FDR family 2: the two interaction terms (term order can be flipped by R,
# e.g. "prior_stimS:DuringImgY", so match any prior_stim interaction).
fam2 <- grepl(":", int_td$term) & grepl("prior_stim", int_td$term)
int_td$q_fdr <- NA_real_
int_td$q_fdr[fam2] <- bh(int_td$p.value[fam2])
int_td$fdr_family <- ifelse(fam2, "stim_interactions", NA)
write.csv(int_td, file.path(out_dir, "retrieval_timing_interaction_coefs.csv"),
          row.names = FALSE)
save_diagnostics(m_int, "retrieval_timing_interaction_diagnostics.csv")

##############################################################################
# Omnibus likelihood-ratio tests (for the report)
##############################################################################
lrt_main <- anova(m_null, m_main)   # do the timing windows add anything?
lrt_int  <- anova(m_add,  m_int)    # do the two interactions jointly add?
lrt_df <- rbind(
  data.frame(test = "Timing main effects vs. null (2 df)",
             Chisq = lrt_main$Chisq[2], Df = lrt_main$Df[2],
             p.value = lrt_main$`Pr(>Chisq)`[2]),
  data.frame(test = "Prior-stim interactions vs. additive (2 df)",
             Chisq = lrt_int$Chisq[2], Df = lrt_int$Df[2],
             p.value = lrt_int$`Pr(>Chisq)`[2]))
write.csv(lrt_df, file.path(out_dir, "retrieval_timing_lrt.csv"), row.names = FALSE)
cat("\nOmnibus LRTs:\n"); print(lrt_df)

cat("\nDone. Outputs in", out_dir, "\n")
