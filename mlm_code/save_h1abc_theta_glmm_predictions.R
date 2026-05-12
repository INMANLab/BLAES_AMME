##############################################################################
# Refit the per-region single-band power GLMMs from the h1abc full retrieval
# pipeline (BLAMTL family: BLA, ALLHPC, EC, PRC) and save predictions for
# plotting. Mirrors run_h1abc_full_retrieval.R exactly:
#
#   Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)
#
# where band_c is the RAW (uncentered) band power. Generates predictions for
# theta and slow gamma. ALLHPC is built by averaging trial-matched HPC, CA, DG
# rows. CIs are model-based 95% from vcov(glmer).
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
                       "power_BLAMTL_GLMM")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

THETA      <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)
HPC_SUB    <- c("HPC", "CA", "DG")
REGIONS    <- c("BLA", "ALLHPC", "EC", "PRC")
BANDS      <- list(theta = THETA, slow_gamma = SLOW_GAMMA)
ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

pw <- read.csv(file.path(CSV_IN, "combined_retrieval_power_all_mlmr_input.csv"),
               stringsAsFactors = FALSE, check.names = FALSE)
pw <- pw[pw$yes_or_no %in% c("yes", "no") & pw$trial_type != "new", ]
pw$Patient[pw$Patient == "BJH033"] <- "BJH032"
pw$Accuracy <- ifelse(pw$yes_or_no == "yes", 1L, 0L)
pw$StimCond <- factor(ifelse(pw$trial_type == "nostim", "nostim", "stim"),
                      levels = c("nostim", "stim"))

freq_cols <- grep("^diff_Freq_", names(pw), value = TRUE)

build_allhpc <- function(d) {
  meta_cols <- c("Patient", "Region", "trial_type", "yes_or_no",
                 "Accuracy", "StimCond")
  sub <- d[d$Region %in% HPC_SUB, c(meta_cols, freq_cols)]
  if (nrow(sub) == 0) return(NULL)
  sub <- sub %>% group_by(Patient, Region) %>%
    mutate(trial_id = row_number()) %>% ungroup()
  agg <- sub %>%
    group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy, StimCond) %>%
    summarise(across(all_of(freq_cols), ~ mean(.x, na.rm = TRUE)),
              .groups = "drop") %>%
    mutate(Region = "ALLHPC") %>% select(-trial_id)
  as.data.frame(agg)
}
pw <- bind_rows(pw, build_allhpc(pw))

freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
for (band_name in names(BANDS)) {
  rng <- BANDS[[band_name]]
  cols <- freq_cols[freqs >= rng[1] & freqs <= rng[2]]
  pw[[band_name]] <- rowMeans(pw[, cols, drop = FALSE], na.rm = TRUE)
}


fit_and_save <- function(region, band_name) {
  sub <- pw[pw$Region == region & is.finite(pw[[band_name]]), ]
  sub$Patient <- factor(sub$Patient)
  sub$band_c  <- sub[[band_name]]   # uncentered, matching run_h1abc_full_retrieval.R

  cat(sprintf("\n%s / %s: %d trials, %d patients\n",
              region, band_name, nrow(sub), nlevels(sub$Patient)))

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
                        sprintf("power_%s_%s_predictions.csv",
                                region, band_name))
  write.csv(preds, out_path, row.names = FALSE)
  cat(sprintf("  -> %s\n", basename(out_path)))
}

for (r in REGIONS) for (b in names(BANDS)) fit_and_save(r, b)
cat("\n===== DONE =====\n")
