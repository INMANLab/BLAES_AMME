##############################################################################
# Refit per-region/pair single-band GLMMs for the h1abc encoding pipeline and
# save predictions for plotting. Covers HPCrhinal and HippSubRhinal scopes,
# all three measures (power, coherence, pac). Mirrors run_h1abc_full_retrieval.R
# exactly:
#
#   Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)
#
# band_c is RAW (uncentered). Predictions are written next to the coefs/anova
# CSVs in OUTPUTS/encoding_memory_reports/stats/h1abc_full_encoding_mlm/
# <measure>_<scope>_GLMM/. ALLHPC pairs/regions are built by averaging
# trial-matched HPC subfield rows. Panels below MIN_PATIENTS / MIN_TRIALS are
# silently skipped to mirror the FDR family contents.
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4"))  install.packages("lme4",  repos = "https://cloud.r-project.org")
  if (!require("dplyr")) install.packages("dplyr", repos = "https://cloud.r-project.org")
  library(lme4); library(dplyr)
})

REPO_ROOT <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_IN    <- file.path(REPO_ROOT, "OUTPUTS", "csvs")
STATS_BASE <- file.path(REPO_ROOT, "OUTPUTS", "encoding_memory_reports",
                        "stats", "h1abc_full_encoding_mlm")

THETA          <- c(4.88, 7.81)
SLOW_GAMMA     <- c(30.27, 54.69)
PAC_SLOW_GAMMA <- c(30, 50)

MIN_PATIENTS <- 4
MIN_TRIALS   <- 30
ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

HPC_SUB <- c("HPC", "CA", "DG")

ALLHPC_PAIR_SOURCES <- list(
  ALLHPC_EC  = c("EC_HPC",  "CA_EC",  "DG_EC"),
  ALLHPC_PRC = c("HPC_PRC", "CA_PRC", "DG_PRC")
)

# Scope definitions mirror run_h1abc_full_retrieval.R for HPCrhinal +
# HippSubRhinal (no BLA, no ALLHPC in HippSubRhinal, EC_PRC only in HPCrhinal).
SCOPE_POWER <- list(
  HPCrhinal     = c("ALLHPC", "EC", "PRC"),
  HippSubRhinal = c("CA", "DG", "HPC", "EC", "PRC")
)
SCOPE_COH <- list(
  HPCrhinal     = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
  HippSubRhinal = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC")
)
SCOPE_PAC <- SCOPE_COH


# ── Data loaders ────────────────────────────────────────────────────────────

prep_base <- function(d) {
  d <- d[d$yes_or_no %in% c("yes", "no") & d$trial_type != "new", ]
  d$Patient[d$Patient == "BJH033"] <- "BJH032"
  d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
  d$StimCond <- factor(ifelse(d$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))
  d
}

build_allhpc_power <- function(d, freq_cols) {
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

build_allhpc_pair <- function(d, freq_cols, target_label, source_pairs) {
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

add_bands <- function(d, freq_cols, bands) {
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
  for (band_name in names(bands)) {
    rng  <- bands[[band_name]]
    cols <- freq_cols[freqs >= rng[1] & freqs <= rng[2]]
    d[[band_name]] <- rowMeans(d[, cols, drop = FALSE], na.rm = TRUE)
  }
  d
}


# ── GLMM fit + prediction grid ──────────────────────────────────────────────

fit_and_save <- function(sub, band_name, out_path) {
  if (nrow(sub) < MIN_TRIALS ||
      length(unique(sub$Patient)) < MIN_PATIENTS) {
    cat(sprintf("  SKIP %s (n=%d, patients=%d)\n",
                basename(out_path), nrow(sub), length(unique(sub$Patient))))
    return(invisible(NULL))
  }
  sub$Patient <- factor(sub$Patient)
  sub$band_c  <- sub[[band_name]]

  cat(sprintf("  %s: %d trials, %d patients\n",
              basename(out_path), nrow(sub), nlevels(sub$Patient)))

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
  write.csv(preds, out_path, row.names = FALSE)
}


# ── Power ───────────────────────────────────────────────────────────────────

run_power <- function() {
  d <- read.csv(file.path(CSV_IN, "combined_encoding_power_all_mlmr_input.csv"),
                stringsAsFactors = FALSE, check.names = FALSE)
  d <- prep_base(d)
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  extra <- build_allhpc_power(d, freq_cols)
  if (!is.null(extra)) d <- bind_rows(d, extra)
  d <- add_bands(d, freq_cols, list(theta = THETA, slow_gamma = SLOW_GAMMA))

  for (scope_name in names(SCOPE_POWER)) {
    out_dir <- file.path(STATS_BASE, sprintf("power_%s_GLMM", scope_name))
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    cat(sprintf("\n=== power | %s ===\n", scope_name))
    for (region in SCOPE_POWER[[scope_name]]) {
      sub_region <- d[d$Region == region, ]
      for (band_name in c("theta", "slow_gamma")) {
        out_path <- file.path(out_dir,
          sprintf("power_%s_%s_predictions.csv", region, band_name))
        fit_and_save(sub_region[is.finite(sub_region[[band_name]]), ],
                     band_name, out_path)
      }
    }
  }
}


# ── Coherence ───────────────────────────────────────────────────────────────

run_coherence <- function() {
  d <- read.csv(file.path(CSV_IN, "combined_encoding_coherence_all_mlmr_input.csv"),
                stringsAsFactors = FALSE, check.names = FALSE)
  d <- prep_base(d)
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  for (target in names(ALLHPC_PAIR_SOURCES)) {
    add <- build_allhpc_pair(d, freq_cols, target, ALLHPC_PAIR_SOURCES[[target]])
    if (!is.null(add)) d <- bind_rows(d, add)
  }
  d <- add_bands(d, freq_cols, list(theta = THETA, slow_gamma = SLOW_GAMMA))

  for (scope_name in names(SCOPE_COH)) {
    out_dir <- file.path(STATS_BASE, sprintf("coherence_%s_GLMM", scope_name))
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    cat(sprintf("\n=== coherence | %s ===\n", scope_name))
    for (pair in SCOPE_COH[[scope_name]]) {
      sub_pair <- d[d$Region == pair, ]
      for (band_name in c("theta", "slow_gamma")) {
        out_path <- file.path(out_dir,
          sprintf("coherence_%s_%s_predictions.csv", pair, band_name))
        fit_and_save(sub_pair[is.finite(sub_pair[[band_name]]), ],
                     band_name, out_path)
      }
    }
  }
}


# ── PAC (slow gamma only) ───────────────────────────────────────────────────

run_pac <- function() {
  d <- read.csv(file.path(CSV_IN, "combined_encoding_pac_all_mlmr_input.csv"),
                stringsAsFactors = FALSE, check.names = FALSE)
  d <- prep_base(d)
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  for (target in names(ALLHPC_PAIR_SOURCES)) {
    add <- build_allhpc_pair(d, freq_cols, target, ALLHPC_PAIR_SOURCES[[target]])
    if (!is.null(add)) d <- bind_rows(d, add)
  }
  d <- add_bands(d, freq_cols, list(slow_gamma = PAC_SLOW_GAMMA))

  for (scope_name in names(SCOPE_PAC)) {
    out_dir <- file.path(STATS_BASE, sprintf("pac_%s_GLMM", scope_name))
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    cat(sprintf("\n=== pac | %s ===\n", scope_name))
    for (pair in SCOPE_PAC[[scope_name]]) {
      sub_pair <- d[d$Region == pair & is.finite(d$slow_gamma), ]
      out_path <- file.path(out_dir,
        sprintf("pac_%s_slow_gamma_predictions.csv", pair))
      fit_and_save(sub_pair, "slow_gamma", out_path)
    }
  }
}


run_power()
run_coherence()
run_pac()
cat("\n===== ENCODING H1ABC PREDICTIONS DONE =====\n")
