##############################################################################
#  During-Stimulation IEDs x CONTINUOUS memory modulation (GLMM)
#
#  Companion to the categorical responder analysis
#  (build_during_stim_responder_inferential_report.py). Same trial set
#  (stimulated, IED-positive encoding trials), but the 4-level responder factor
#  is replaced by the continuous memory-modulation score (avg_stim_dprime_diff,
#  patient-level, centered).
#
#  Binomial GLMM, patient random intercept, one row per stimulated IED-positive
#  trial (forgotten = 1 / remembered = 0):
#
#     forgotten ~ during_stim * dprime_diff_c + (1 | Patient)
#
#  Key term = during_stim:dprime_diff_c -> does a patient's continuous memory
#  modulation moderate the During-Stimulation forgetting effect? The during_stim
#  main term is the During-Stim effect at the MEAN modulation (because dprime is
#  centered).
#
#  Input  : OUTPUTS/ied_timing_memory/during_stim_continuous_trials.csv
#  Outputs: OUTPUTS/ied_timing_memory/
#     during_stim_continuous_glmm_coefs.csv
#     during_stim_continuous_glmm_diagnostics.csv
#     during_stim_continuous_glmm_lrt.csv
#     during_stim_continuous_glmm_simpleslopes.csv
##############################################################################

suppressPackageStartupMessages({
  library(lme4)
  library(broom.mixed)
  library(performance)
})

script_dir <- {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg))) else getwd()
}
root    <- normalizePath(file.path(script_dir, ".."))
out_dir <- file.path(root, "OUTPUTS", "ied_timing_memory")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

##############################################################################
# Data
##############################################################################
dat <- read.csv(file.path(out_dir, "during_stim_continuous_trials.csv"),
                stringsAsFactors = FALSE)
dat$Patient      <- factor(dat$Patient)
dprime_mean      <- mean(dat$dprime_diff)
dprime_sd        <- sd(dat$dprime_diff)
dat$dprime_diff_c <- dat$dprime_diff - dprime_mean   # patient-level, centered

cat("During-Stim continuous-modulation GLMM data:\n")
cat("  trials:", nrow(dat),
    "| patients:", nlevels(dat$Patient),
    "| forgotten:", sum(dat$forgotten),
    "| during_stim trials:", sum(dat$during_stim), "\n")
cat(sprintf("  d' diff (patient-level): mean = %+.3f, SD = %.3f\n\n",
            dprime_mean, dprime_sd))

bh <- function(p) p.adjust(p, method = "BH")

tidy_or <- function(model, label) {
  td <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE,
                          exponentiate = TRUE)
  td$model <- label
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
               "R2_marginal", "R2_conditional",
               "dprime_mean", "dprime_sd"),
    value = c(icc_val, rand_var, rand_sd,
              AIC(model), BIC(model), as.numeric(logLik(model)),
              deviance(model), nobs(model), ngrps(model)[[1]],
              r2$R2_marginal, r2$R2_conditional,
              dprime_mean, dprime_sd))
  write.csv(diag_df, file.path(out_dir, filename), row.names = FALSE)
  diag_df
}

##############################################################################
# Models
##############################################################################
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("GLMM: forgotten ~ during_stim * dprime_diff_c + (1 | Patient)\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

m_add <- glmer(forgotten ~ during_stim + dprime_diff_c + (1 | Patient),
               data = dat, family = binomial, control = ctrl)
m_int <- glmer(forgotten ~ during_stim * dprime_diff_c + (1 | Patient),
               data = dat, family = binomial, control = ctrl)
print(summary(m_int)$coefficients)

coefs <- tidy_or(m_int, "during_stim x dprime_diff (continuous)")
# FDR family: the during_stim main effect and its interaction with modulation
fam <- coefs$term %in% c("during_stim", "during_stim:dprime_diff_c")
coefs$q_fdr <- NA_real_
coefs$q_fdr[fam] <- bh(coefs$p.value[fam])
coefs$fdr_family <- ifelse(fam, "during_stim_continuous", NA)
write.csv(coefs, file.path(out_dir, "during_stim_continuous_glmm_coefs.csv"),
          row.names = FALSE)
save_diagnostics(m_int, "during_stim_continuous_glmm_diagnostics.csv")

##############################################################################
# LRT: does the modulation x during_stim interaction add anything?
##############################################################################
lrt <- anova(m_add, m_int)
lrt_df <- data.frame(
  test    = "during_stim:dprime_diff interaction vs. additive (1 df)",
  Chisq   = lrt$Chisq[2], Df = lrt$Df[2], p.value = lrt$`Pr(>Chisq)`[2])
write.csv(lrt_df, file.path(out_dir, "during_stim_continuous_glmm_lrt.csv"),
          row.names = FALSE)

##############################################################################
# Simple slopes: During-Stim forgetting effect (OR) at low / mean / high
# modulation (mean and +/- 1 SD of the patient-level d' difference).
##############################################################################
beta  <- fixef(m_int)
V     <- vcov(m_int)
b_ds  <- "during_stim"
b_int <- "during_stim:dprime_diff_c"
levels_c <- c(`Low (-1 SD)` = -dprime_sd, `Mean` = 0, `High (+1 SD)` = dprime_sd)
ss_rows <- lapply(names(levels_c), function(nm) {
  cc <- levels_c[[nm]]
  L <- setNames(numeric(length(beta)), names(beta))
  L[b_ds]  <- 1
  L[b_int] <- cc
  est <- sum(L * beta)
  se  <- sqrt(as.numeric(t(L) %*% V %*% L))
  z   <- est / se
  data.frame(modulation = nm, dprime_diff = dprime_mean + cc,
             logOR = est, SE = se, z = z,
             p = 2 * pnorm(-abs(z)),
             OR = exp(est), CI_low = exp(est - 1.96 * se),
             CI_high = exp(est + 1.96 * se))
})
simple <- do.call(rbind, ss_rows)
write.csv(simple, file.path(out_dir, "during_stim_continuous_glmm_simpleslopes.csv"),
          row.names = FALSE)

##############################################################################
# Console summary
##############################################################################
fmt <- function(x) sprintf("%.3f", x)
cat("\n--- Fixed effects (OR scale) ---\n")
print(coefs[, c("term", "estimate", "conf.low", "conf.high", "p.value", "q_fdr")])
cat("\n--- Interaction LRT ---\n"); print(lrt_df)
cat("\n--- During-Stim forgetting OR at low/mean/high modulation ---\n")
print(simple[, c("modulation", "dprime_diff", "OR", "CI_low", "CI_high", "z", "p")])
cat("\nDone. Outputs in", out_dir, "\n")
