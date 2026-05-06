##############################################################################
# Refit the PRC-theta and EC-slow-gamma two-band GLMMs and save:
#   - per-subject mean accuracy / mean band power (for scatter overlay)
#   - GLMM-predicted P(remembered) along a band-power grid for stim/no-stim,
#     with model-based 95% CI from the fixed-effects covariance matrix.
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

out_dir <- file.path(script_dir, "outputs", "imbalanced_retrieval_mlm")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)

pw_csv <- file.path(script_dir, "outputs", "csvs",
                    "combined_retrieval_power_all_mlmr_input.csv")
pw <- read.csv(pw_csv, stringsAsFactors = FALSE, check.names = FALSE)
pw <- pw[pw$yes_or_no %in% c("yes", "no") & pw$trial_type != "new", ]

freq_cols <- sort(grep("^diff_Freq_", names(pw), value = TRUE))
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))

pw$theta <- rowMeans(pw[, freq_cols[freqs >= THETA[1] & freqs <= THETA[2]],
                        drop = FALSE], na.rm = TRUE)
pw$slow_gamma <- rowMeans(pw[, freq_cols[freqs >= SLOW_GAMMA[1] & freqs <= SLOW_GAMMA[2]],
                             drop = FALSE], na.rm = TRUE)

pw$Accuracy <- ifelse(pw$yes_or_no == "yes", 1, 0)
pw$StimCond <- factor(ifelse(pw$trial_type == "nostim", "nostim", "stim"),
                      levels = c("nostim", "stim"))

fit_and_save <- function(region_code, focus_band) {
  sub <- pw[pw$Region == region_code, ]
  sub$Patient <- factor(sub$Patient)
  sub$theta_c <- sub$theta - mean(sub$theta, na.rm = TRUE)
  sub$slow_gamma_c <- sub$slow_gamma - mean(sub$slow_gamma, na.rm = TRUE)

  cat(sprintf("\n%s: %d trials, %d patients\n",
              region_code, nrow(sub), nlevels(sub$Patient)))

  m <- glmer(
    Accuracy ~ StimCond + theta_c + slow_gamma_c +
                 theta_c:StimCond + slow_gamma_c:StimCond + (1 | Patient),
    data = sub, family = binomial, control = ctrl
  )

  # Per-subject means (scatter overlay)
  per_subj <- aggregate(
    cbind(Accuracy, theta_c, slow_gamma_c) ~ Patient + StimCond,
    data = sub, FUN = mean
  )
  per_subj$n_trials <- as.integer(table(
    sub$Patient, sub$StimCond
  )[cbind(as.character(per_subj$Patient), as.character(per_subj$StimCond))])
  write.csv(per_subj,
            file.path(out_dir, sprintf("power_twoband_%s_per_subject.csv",
                                       region_code)),
            row.names = FALSE)

  # GLMM-predicted P(remembered) over a focus-band grid, with model-based CI.
  beta <- fixef(m)
  V <- vcov(m)
  band_quant <- quantile(sub[[focus_band]], probs = c(0.02, 0.98),
                         na.rm = TRUE)
  grid_band <- seq(band_quant[1], band_quant[2], length.out = 200)

  pred_rows <- list()
  for (stim_val in c(0, 1)) {
    cond <- if (stim_val == 0) "nostim" else "stim"
    # Build a design matrix matching the model's term order.
    # Terms: (Intercept), StimCondstim, theta_c, slow_gamma_c,
    #        StimCondstim:theta_c, StimCondstim:slow_gamma_c
    # Hold the other band at 0 (mean).
    if (focus_band == "theta_c") {
      X <- cbind(
        `(Intercept)` = 1,
        StimCondstim = stim_val,
        theta_c = grid_band,
        slow_gamma_c = 0,
        `StimCondstim:theta_c` = stim_val * grid_band,
        `StimCondstim:slow_gamma_c` = 0
      )
    } else {
      X <- cbind(
        `(Intercept)` = 1,
        StimCondstim = stim_val,
        theta_c = 0,
        slow_gamma_c = grid_band,
        `StimCondstim:theta_c` = 0,
        `StimCondstim:slow_gamma_c` = stim_val * grid_band
      )
    }
    X <- X[, names(beta), drop = FALSE]
    eta <- as.numeric(X %*% beta)
    se_eta <- sqrt(rowSums((X %*% V) * X))
    p_hat <- plogis(eta)
    p_lo <- plogis(eta - 1.96 * se_eta)
    p_hi <- plogis(eta + 1.96 * se_eta)
    pred_rows[[length(pred_rows) + 1]] <- data.frame(
      stim = cond,
      band = grid_band,
      p_hat = p_hat,
      p_lo = p_lo,
      p_hi = p_hi
    )
  }
  preds <- do.call(rbind, pred_rows)
  write.csv(preds,
            file.path(out_dir, sprintf("power_twoband_%s_predictions.csv",
                                       region_code)),
            row.names = FALSE)
  cat(sprintf("  Wrote per-subject + predictions for %s.\n", region_code))
}

fit_and_save("PRC", "theta_c")
fit_and_save("EC",  "slow_gamma_c")

cat("\n===== DONE =====\n")
