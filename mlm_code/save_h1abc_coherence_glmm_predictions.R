##############################################################################
# Refit the per-pair single-band coherence GLMMs from the h1abc full retrieval
# pipeline (BLAMTL family: BLA_ALLHPC, BLA_EC, BLA_PRC) and save predictions
# for plotting. Mirrors run_h1abc_full_retrieval.R exactly:
#
#   Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)
#
# where band_c is the RAW (uncentered) band coherence. Generates predictions
# for theta and slow gamma. BLA_ALLHPC is built by averaging trial-matched
# BLA_HPC, BLA_CA, BLA_DG rows (matching ALLHPC_PAIR_SOURCES). CIs are
# model-based 95% from vcov(glmer).
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4"))  install.packages("lme4",  repos = "https://cloud.r-project.org")
  if (!require("dplyr")) install.packages("dplyr", repos = "https://cloud.r-project.org")
  library(lme4); library(dplyr)
})

REPO_ROOT <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_IN    <- file.path(REPO_ROOT, "OUTPUTS", "csvs")
OUT_DIR   <- file.path(REPO_ROOT, "OUTPUTS", "retrieval_memory_reports",
                       "stats", "h1abc_full_retrieval_mlm",
                       "coherence_BLAMTL_GLMM")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

THETA      <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)
PAIRS      <- c("BLA_ALLHPC", "BLA_EC", "BLA_PRC")
ALLHPC_SOURCES <- c("BLA_HPC", "BLA_CA", "BLA_DG")
BANDS      <- list(theta = THETA, slow_gamma = SLOW_GAMMA)
ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

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
dat <- bind_rows(dat, build_allhpc_pair(dat, "BLA_ALLHPC", ALLHPC_SOURCES))

freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
for (band_name in names(BANDS)) {
  rng <- BANDS[[band_name]]
  cols <- freq_cols[freqs >= rng[1] & freqs <= rng[2]]
  dat[[band_name]] <- rowMeans(dat[, cols, drop = FALSE], na.rm = TRUE)
}


fit_and_save <- function(pair, band_name) {
  sub <- dat[dat$Region == pair & is.finite(dat[[band_name]]), ]
  sub$Patient <- factor(sub$Patient)
  sub$band_c  <- sub[[band_name]]

  cat(sprintf("\n%s / %s: %d trials, %d patients\n",
              pair, band_name, nrow(sub), nlevels(sub$Patient)))

  m <- glmer(
    Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient),
    data = sub, family = binomial, control = ctrl_glmer
  )

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
  out_path <- file.path(OUT_DIR,
                        sprintf("coherence_%s_%s_predictions.csv",
                                pair, band_name))
  write.csv(preds, out_path, row.names = FALSE)
  cat(sprintf("  -> %s\n", basename(out_path)))
}

for (p in PAIRS) for (b in names(BANDS)) fit_and_save(p, b)
cat("\n===== DONE =====\n")
