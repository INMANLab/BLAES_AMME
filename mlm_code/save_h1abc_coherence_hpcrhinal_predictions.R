##############################################################################
# Refit the per-pair single-band coherence GLMMs from the h1abc full retrieval
# pipeline for the HPCrhinal and HippSubRhinal scopes. Saves predictions for
# theta and slow gamma. Mirrors run_h1abc_full_retrieval.R exactly:
#
#   Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)
#
# band_c is RAW (uncentered). HPCrhinal needs ALLHPC_EC/ALLHPC_PRC built by
# averaging trial-matched source pairs (ALLHPC_PAIR_SOURCES). HippSubRhinal
# uses raw pair labels from the input. EC_PRC is dropped from the BLAMTL family
# upstream (insufficient subjects); we still attempt it here and skip on
# convergence/min-N failure to mirror the FDR family contents.
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4"))  install.packages("lme4",  repos = "https://cloud.r-project.org")
  if (!require("dplyr")) install.packages("dplyr", repos = "https://cloud.r-project.org")
  library(lme4); library(dplyr)
})

REPO_ROOT <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_IN    <- file.path(REPO_ROOT, "OUTPUTS", "csvs")
STATS_BASE <- file.path(REPO_ROOT, "OUTPUTS", "retrieval_memory_reports",
                        "stats", "h1abc_full_retrieval_mlm")

THETA      <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)
BANDS      <- list(theta = THETA, slow_gamma = SLOW_GAMMA)
ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

ALLHPC_PAIR_SOURCES <- list(
  ALLHPC_EC  = c("EC_HPC",  "CA_EC",  "DG_EC"),
  ALLHPC_PRC = c("HPC_PRC", "CA_PRC", "DG_PRC")
)

SCOPES <- list(
  HPCrhinal     = list(pairs = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
                       out   = file.path(STATS_BASE, "coherence_HPCrhinal_GLMM")),
  HippSubRhinal = list(pairs = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC",
                                  "DG_PRC", "HPC_PRC", "EC_PRC"),
                       out   = file.path(STATS_BASE, "coherence_HippSubRhinal_GLMM"))
)

dat <- read.csv(file.path(CSV_IN, "combined_retrieval_coherence_all_mlmr_input.csv"),
                stringsAsFactors = FALSE, check.names = FALSE)
dat <- dat[dat$yes_or_no %in% c("yes", "no") & dat$trial_type != "new", ]
dat$Patient[dat$Patient == "BJH033"] <- "BJH032"
dat$Accuracy <- ifelse(dat$yes_or_no == "yes", 1L, 0L)
dat$StimCond <- factor(ifelse(dat$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

freq_cols <- grep("^diff_Freq_", names(dat), value = TRUE)

build_allhpc_pair <- function(d, target_label, source_pairs) {
  meta_cols <- c("Patient", "Region", "trial_type", "yes_or_no",
                 "Accuracy", "StimCond")
  sub <- d[d$Region %in% source_pairs, c(meta_cols, freq_cols)]
  if (nrow(sub) == 0) return(NULL)
  sub <- sub %>% group_by(Patient, Region) %>%
    mutate(trial_id = row_number()) %>% ungroup()
  agg <- sub %>%
    group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy, StimCond) %>%
    summarise(across(all_of(freq_cols), ~ mean(.x, na.rm = TRUE)),
              .groups = "drop") %>%
    mutate(Region = target_label) %>% select(-trial_id)
  as.data.frame(agg)
}

for (target in names(ALLHPC_PAIR_SOURCES)) {
  add <- build_allhpc_pair(dat, target, ALLHPC_PAIR_SOURCES[[target]])
  if (!is.null(add)) dat <- bind_rows(dat, add)
}

freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
for (band_name in names(BANDS)) {
  rng <- BANDS[[band_name]]
  cols <- freq_cols[freqs >= rng[1] & freqs <= rng[2]]
  dat[[band_name]] <- rowMeans(dat[, cols, drop = FALSE], na.rm = TRUE)
}


fit_and_save <- function(pair, band_name, out_dir) {
  sub <- dat[dat$Region == pair & is.finite(dat[[band_name]]), ]
  if (nrow(sub) < 30 || length(unique(sub$Patient)) < 4) {
    cat(sprintf("  SKIP %s / %s (n=%d, patients=%d)\n",
                pair, band_name, nrow(sub), length(unique(sub$Patient))))
    return(invisible(NULL))
  }
  sub$Patient <- factor(sub$Patient)
  sub$band_c  <- sub[[band_name]]

  cat(sprintf("\n%s / %s: %d trials, %d patients\n",
              pair, band_name, nrow(sub), nlevels(sub$Patient)))

  m <- tryCatch(
    glmer(Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient),
          data = sub, family = binomial, control = ctrl_glmer),
    error = function(e) { cat("  glmer failed:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(m)) return(invisible(NULL))

  beta <- fixef(m); V <- vcov(m)
  band_quant <- quantile(sub$band_c, probs = c(0.02, 0.98), na.rm = TRUE)
  grid_band <- seq(band_quant[1], band_quant[2], length.out = 200)

  pred_rows <- list()
  for (stim_val in c(0, 1)) {
    cond <- if (stim_val == 0) "nostim" else "stim"
    X <- cbind(
      `(Intercept)`           = 1,
      band_c                  = grid_band,
      StimCondstim            = stim_val,
      `band_c:StimCondstim`   = stim_val * grid_band
    )
    X <- X[, names(beta), drop = FALSE]
    eta    <- as.numeric(X %*% beta)
    se_eta <- sqrt(rowSums((X %*% V) * X))
    pred_rows[[length(pred_rows) + 1]] <- data.frame(
      stim = cond, band = grid_band,
      p_hat = plogis(eta),
      p_lo  = plogis(eta - 1.96 * se_eta),
      p_hi  = plogis(eta + 1.96 * se_eta)
    )
  }
  preds <- do.call(rbind, pred_rows)
  out_path <- file.path(out_dir,
                        sprintf("coherence_%s_%s_predictions.csv",
                                pair, band_name))
  write.csv(preds, out_path, row.names = FALSE)
  cat(sprintf("  -> %s\n", basename(out_path)))
}

for (scope_name in names(SCOPES)) {
  cat(sprintf("\n=== %s ===\n", scope_name))
  scope <- SCOPES[[scope_name]]
  dir.create(scope$out, showWarnings = FALSE, recursive = TRUE)
  for (p in scope$pairs) for (b in names(BANDS)) fit_and_save(p, b, scope$out)
}
cat("\n===== DONE =====\n")
