##############################################################################
# Refit the EC-HPC coherence two-band GLMM and save:
#   - per-subject mean accuracy / mean band coherence (for scatter overlay)
#   - GLMM-predicted P(remembered) along a slow-gamma-coherence grid for
#     stim/no-stim, with model-based 95% CI from the fixed-effects vcov.
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4")) install.packages("lme4", repos = "https://cloud.r-project.org")
  library(lme4)
})

script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    a <- commandArgs(trailingOnly = FALSE)
    f <- grep("--file=", a, value = TRUE)
    if (length(f) > 0) dirname(normalizePath(sub("--file=", "", f)))
    else getwd()
  }
)

out_dir <- file.path(script_dir, "outputs", "h1c_imbalanced_retrieval_mlm")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)

coh_csv <- file.path(script_dir, "outputs", "csvs",
                    "combined_retrieval_coherence_all_mlmr_input.csv")
coh <- read.csv(coh_csv, stringsAsFactors = FALSE, check.names = FALSE)
coh <- coh[coh$yes_or_no %in% c("yes", "no") & coh$trial_type != "new", ]

freq_cols <- sort(grep("^diff_Freq_", names(coh), value = TRUE))
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))

coh$theta <- rowMeans(coh[, freq_cols[freqs >= THETA[1] & freqs <= THETA[2]],
                          drop = FALSE], na.rm = TRUE)
coh$slow_gamma <- rowMeans(coh[, freq_cols[freqs >= SLOW_GAMMA[1] & freqs <= SLOW_GAMMA[2]],
                               drop = FALSE], na.rm = TRUE)

coh$Accuracy <- ifelse(coh$yes_or_no == "yes", 1, 0)
coh$StimCond <- factor(ifelse(coh$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

pair_code <- "EC_HPC"
focus_band <- "slow_gamma_c"

sub <- coh[coh$Region == pair_code, ]
sub$Patient <- factor(sub$Patient)
sub$theta_c <- sub$theta - mean(sub$theta, na.rm = TRUE)
sub$slow_gamma_c <- sub$slow_gamma - mean(sub$slow_gamma, na.rm = TRUE)

cat(sprintf("\n%s coherence: %d trials, %d patients\n",
            pair_code, nrow(sub), nlevels(sub$Patient)))

m <- glmer(
  Accuracy ~ StimCond + theta_c + slow_gamma_c +
               theta_c:StimCond + slow_gamma_c:StimCond + (1 | Patient),
  data = sub, family = binomial, control = ctrl
)

# Per-subject means
per_subj <- aggregate(
  cbind(Accuracy, theta_c, slow_gamma_c) ~ Patient + StimCond,
  data = sub, FUN = mean
)
per_subj$n_trials <- as.integer(table(
  sub$Patient, sub$StimCond
)[cbind(as.character(per_subj$Patient), as.character(per_subj$StimCond))])
write.csv(per_subj,
          file.path(out_dir, sprintf("coherence_twoband_%s_per_subject.csv",
                                     pair_code)),
          row.names = FALSE)

# Prediction grid + model-based CI
beta <- fixef(m)
V <- vcov(m)
band_quant <- quantile(sub[[focus_band]], probs = c(0.02, 0.98), na.rm = TRUE)
grid_band <- seq(band_quant[1], band_quant[2], length.out = 200)

pred_rows <- list()
for (stim_val in c(0, 1)) {
  cond <- if (stim_val == 0) "nostim" else "stim"
  X <- cbind(
    `(Intercept)` = 1,
    StimCondstim = stim_val,
    theta_c = 0,
    slow_gamma_c = grid_band,
    `StimCondstim:theta_c` = 0,
    `StimCondstim:slow_gamma_c` = stim_val * grid_band
  )
  X <- X[, names(beta), drop = FALSE]
  eta <- as.numeric(X %*% beta)
  se_eta <- sqrt(rowSums((X %*% V) * X))
  p_hat <- plogis(eta)
  p_lo <- plogis(eta - 1.96 * se_eta)
  p_hi <- plogis(eta + 1.96 * se_eta)
  pred_rows[[length(pred_rows) + 1]] <- data.frame(
    stim = cond, band = grid_band,
    p_hat = p_hat, p_lo = p_lo, p_hi = p_hi
  )
}
preds <- do.call(rbind, pred_rows)
write.csv(preds,
          file.path(out_dir, sprintf("coherence_twoband_%s_predictions.csv",
                                     pair_code)),
          row.names = FALSE)
cat(sprintf("  Wrote per-subject + predictions for %s.\n", pair_code))

# Print slopes for sanity
sg_main <- log(0.3273)   # slow_gamma_c
sg_int  <- log(6.9249)   # StimCondstim:slow_gamma_c
cat(sprintf("\n  No-stim slow-gamma slope (log-odds): %+.4f (OR = %.4f)\n",
            sg_main, exp(sg_main)))
cat(sprintf("  Stim    slow-gamma slope (log-odds): %+.4f (OR = %.4f)\n",
            sg_main + sg_int, exp(sg_main + sg_int)))

cat("\n===== DONE =====\n")
