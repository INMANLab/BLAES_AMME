##############################################################################
# Refit per-pair single-band PAC GLMMs from the h1abc full retrieval pipeline
# for all 4 scopes (BLAMTL, HPCrhinal, HippSubBLA, HippSubRhinal) and save
# predictions for plotting. Mirrors run_h1abc_full_retrieval.R exactly:
#
#   Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)
#
# band_c is RAW (uncentered) PAC slow gamma. PAC slow gamma range is 30-50 Hz
# (PAC_SLOW_GAMMA in run_h1abc_full_retrieval.R), distinct from the 30-55 Hz
# range used for power/coherence. ALLHPC pairs are aggregated from their HPC
# subfield source pairs. Pairs with < MIN_PATIENTS or < MIN_TRIALS are
# silently skipped to mirror the FDR family contents.
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

PAC_SLOW_GAMMA <- c(30, 50)
MIN_PATIENTS <- 4
MIN_TRIALS   <- 30
ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# Mirrors SCOPES + ALLHPC_PAIR_SOURCES from run_h1abc_full_retrieval.R
SCOPES <- list(
  BLAMTL = list(
    pairs = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    out   = file.path(STATS_BASE, "pac_BLAMTL_GLMM")
  ),
  HPCrhinal = list(
    pairs = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
    out   = file.path(STATS_BASE, "pac_HPCrhinal_GLMM")
  ),
  HippSubBLA = list(
    pairs = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    out   = file.path(STATS_BASE, "pac_HippSubBLA_GLMM")
  ),
  HippSubRhinal = list(
    pairs = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"),
    out   = file.path(STATS_BASE, "pac_HippSubRhinal_GLMM")
  )
)

ALLHPC_PAIR_SOURCES <- list(
  BLA_ALLHPC = c("BLA_HPC", "BLA_CA", "BLA_DG"),
  ALLHPC_EC  = c("EC_HPC",  "CA_EC",  "DG_EC"),
  ALLHPC_PRC = c("HPC_PRC", "CA_PRC", "DG_PRC")
)

dat <- read.csv(file.path(CSV_IN, "combined_retrieval_pac_all_mlmr_input.csv"),
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
sg_cols <- freq_cols[freqs >= PAC_SLOW_GAMMA[1] & freqs <= PAC_SLOW_GAMMA[2]]
dat$slow_gamma <- rowMeans(dat[, sg_cols, drop = FALSE], na.rm = TRUE)


fit_and_save <- function(pair, out_dir) {
  sub <- dat[dat$Region == pair & is.finite(dat$slow_gamma), ]
  if (nrow(sub) < MIN_TRIALS || length(unique(sub$Patient)) < MIN_PATIENTS) {
    cat(sprintf("  SKIP %s (n=%d, patients=%d)\n",
                pair, nrow(sub), length(unique(sub$Patient))))
    return(invisible(NULL))
  }
  sub$Patient <- factor(sub$Patient)
  sub$band_c  <- sub$slow_gamma

  cat(sprintf("\n%s: %d trials, %d patients\n",
              pair, nrow(sub), nlevels(sub$Patient)))

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
                        sprintf("pac_%s_slow_gamma_predictions.csv", pair))
  write.csv(preds, out_path, row.names = FALSE)
  cat(sprintf("  -> %s\n", basename(out_path)))
}

for (scope_name in names(SCOPES)) {
  cat(sprintf("\n=== %s ===\n", scope_name))
  scope <- SCOPES[[scope_name]]
  dir.create(scope$out, showWarnings = FALSE, recursive = TRUE)
  for (p in scope$pairs) fit_and_save(p, scope$out)
}
cat("\n===== DONE =====\n")
