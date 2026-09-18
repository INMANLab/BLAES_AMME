##############################################################################
# Two sensitivity analyses for the encoding stim-only slow-gamma main effects:
#   (1) Within-subject trial-permutation test (1000 iters)
#       - Shuffles yes_or_no within each Patient
#       - Refits Accuracy ~ band_c + (1 | Patient)
#       - Records the band_c z-statistic
#       - Preserves per-subject marginal memory rate (imbalance baked into null)
#   (2) Balanced-subsample sensitivity (200 iters)
#       - For each subject, randomly downsamples the majority class to match
#         the minority class
#       - Refits same GLMM, records OR, z, p
#
# Run on both effects:
#   - HPC-PRC (ALLHPC_PRC) coherence slow-gamma (encoding, stim only)
#       slow-gamma band: 30.27 - 54.69 Hz; ALLHPC built from HPC_PRC+CA_PRC+DG_PRC
#   - EC-PRC PAC slow-gamma (encoding, stim only)
#       slow-gamma band: 30 - 50 Hz; raw pair (no aggregation)
#
# Output (all in OUTPUTS/encoding_memory_reports/stim_effect/):
#   <effect>_permutation_null_z.csv     observed_z + 1000 null z-statistics
#   <effect>_downsample_estimates.csv   200 balanced-subsample fits (OR, z, p)
#
# Total compute estimate: ~30-45 min on a laptop. Saves incrementally so
# partial results survive an interrupt.
##############################################################################

suppressPackageStartupMessages({
  library(lme4); library(dplyr); library(tidyr)
})

REPO <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_DIR <- file.path(REPO, "OUTPUTS", "csvs")
OUT <- file.path(REPO, "OUTPUTS", "encoding_memory_reports", "stim_effect")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

set.seed(42)

N_PERM <- 1000
N_DOWNSAMPLE <- 200
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA          <- c(4.88, 7.81)        # not used here but kept for clarity
SLOW_GAMMA_COH <- c(30.27, 54.69)      # matches run_h1abc_maineffect.R
SLOW_GAMMA_PAC <- c(30, 50)

HPC_PRC_SOURCES <- c("HPC_PRC", "CA_PRC", "DG_PRC")


# ── Data prep ────────────────────────────────────────────────────────────────

prep_coh_allhpc_prc <- function() {
  d <- read.csv(file.path(CSV_DIR, "combined_encoding_coherence_all_mlmr_input.csv"),
                stringsAsFactors = FALSE, check.names = FALSE)
  d <- d[d$yes_or_no %in% c("yes", "no") &
         d$trial_type %in% c("nostim", "stim"), ]
  d$Patient[d$Patient == "BJH033"] <- "BJH032"
  d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  sub <- d[d$Region %in% HPC_PRC_SOURCES, ]
  sub <- sub %>% group_by(Patient, Region) %>%
    mutate(trial_id = row_number()) %>% ungroup()
  agg <- sub %>%
    group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy) %>%
    summarise(across(all_of(freq_cols), ~ mean(.x, na.rm = TRUE)),
              .groups = "drop") %>%
    select(-trial_id)
  agg <- as.data.frame(agg)
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
  cols <- freq_cols[freqs >= SLOW_GAMMA_COH[1] & freqs <= SLOW_GAMMA_COH[2]]
  agg$band_c <- rowMeans(agg[, cols, drop = FALSE], na.rm = TRUE)
  stim <- agg[agg$trial_type == "stim" & is.finite(agg$band_c), ]
  stim$Patient <- factor(stim$Patient)
  stim[, c("Patient", "Accuracy", "yes_or_no", "band_c")]
}

prep_pac_ec_prc <- function() {
  d <- read.csv(file.path(CSV_DIR, "combined_encoding_pac_all_mlmr_input.csv"),
                stringsAsFactors = FALSE, check.names = FALSE)
  d <- d[d$yes_or_no %in% c("yes", "no") &
         d$trial_type %in% c("nostim", "stim"), ]
  d$Patient[d$Patient == "BJH033"] <- "BJH032"
  d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
  pair <- d[d$Region == "EC_PRC", ]
  freq_cols <- grep("^diff_Freq_", names(pair), value = TRUE)
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
  cols <- freq_cols[freqs >= SLOW_GAMMA_PAC[1] & freqs <= SLOW_GAMMA_PAC[2]]
  pair$band_c <- rowMeans(pair[, cols, drop = FALSE], na.rm = TRUE)
  stim <- pair[pair$trial_type == "stim" & is.finite(pair$band_c), ]
  stim$Patient <- factor(stim$Patient)
  stim[, c("Patient", "Accuracy", "yes_or_no", "band_c")]
}


# ── Helpers ─────────────────────────────────────────────────────────────────

fit_glmm <- function(data) {
  data$Patient <- droplevels(factor(data$Patient))
  m <- tryCatch(
    glmer(Accuracy ~ band_c + (1 | Patient),
          data = data, family = binomial, control = ctrl),
    error = function(e) NULL,
    warning = function(w) {
      tryCatch(glmer(Accuracy ~ band_c + (1 | Patient),
                     data = data, family = binomial, control = ctrl),
               error = function(e) NULL)
    }
  )
  if (is.null(m)) return(list(or = NA_real_, z = NA_real_, p = NA_real_))
  s <- summary(m)$coefficients
  if (!"band_c" %in% rownames(s)) return(list(or = NA_real_, z = NA_real_, p = NA_real_))
  est <- s["band_c", "Estimate"]
  zv  <- s["band_c", "z value"]
  pv  <- s["band_c", "Pr(>|z|)"]
  list(or = exp(est), z = zv, p = pv)
}

# Within-subject shuffle of yes_or_no -> Accuracy
permute_within_subject <- function(data) {
  data %>% group_by(Patient) %>%
    mutate(Accuracy = sample(Accuracy)) %>% ungroup() %>% as.data.frame()
}

# For each subject, downsample the majority class to match minority class size.
balance_within_subject <- function(data) {
  data %>% group_by(Patient) %>%
    group_modify(~ {
      n_yes <- sum(.x$Accuracy == 1L); n_no <- sum(.x$Accuracy == 0L)
      k <- min(n_yes, n_no)
      if (k == 0) return(.x[0, ])
      yes <- .x[.x$Accuracy == 1L, ]
      no  <- .x[.x$Accuracy == 0L, ]
      bind_rows(yes[sample(nrow(yes), k), ],
                no [sample(nrow(no),  k), ])
    }) %>% ungroup() %>% as.data.frame()
}


# ── Run one effect ───────────────────────────────────────────────────────────

run_effect <- function(label, data) {
  cat(sprintf("\n========== %s ==========\n", label))
  cat(sprintf("Trials: %d (yes=%d, no=%d)  Subjects: %d\n",
              nrow(data), sum(data$Accuracy == 1L), sum(data$Accuracy == 0L),
              length(unique(data$Patient))))

  obs <- fit_glmm(data)
  cat(sprintf("Observed: OR = %.4g  z = %.3f  p = %.4f\n",
              obs$or, obs$z, obs$p))

  # ── Permutation ─────────────────────────────────────────────────────────
  perm_path <- file.path(OUT, sprintf("%s_permutation_null_z.csv", label))
  cat(sprintf("[1/2] Running %d within-subject permutations -> %s\n",
              N_PERM, perm_path))
  perm_z <- numeric(N_PERM)
  t0 <- Sys.time()
  for (i in seq_len(N_PERM)) {
    pd <- permute_within_subject(data)
    perm_z[i] <- fit_glmm(pd)$z
    if (i %% 50 == 0) {
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      eta <- elapsed / i * (N_PERM - i)
      cat(sprintf("    perm %d/%d  elapsed=%.0fs  ETA=%.0fs\n",
                  i, N_PERM, elapsed, eta))
      # Save partial progress
      write.csv(data.frame(iter = seq_len(i), z_null = perm_z[1:i],
                           observed_z = obs$z, observed_or = obs$or,
                           observed_p = obs$p),
                perm_path, row.names = FALSE)
    }
  }
  write.csv(data.frame(iter = seq_len(N_PERM), z_null = perm_z,
                       observed_z = obs$z, observed_or = obs$or,
                       observed_p = obs$p),
            perm_path, row.names = FALSE)
  valid <- !is.na(perm_z)
  p_two_sided <- mean(abs(perm_z[valid]) >= abs(obs$z), na.rm = TRUE)
  p_one_sided <- if (obs$z >= 0)
    mean(perm_z[valid] >= obs$z, na.rm = TRUE)
  else
    mean(perm_z[valid] <= obs$z, na.rm = TRUE)
  cat(sprintf("Permutation p (two-sided) = %.4f   (one-sided in obs direction) = %.4f\n",
              p_two_sided, p_one_sided))

  # ── Downsample ──────────────────────────────────────────────────────────
  ds_path <- file.path(OUT, sprintf("%s_downsample_estimates.csv", label))
  cat(sprintf("[2/2] Running %d balanced-subsample refits -> %s\n",
              N_DOWNSAMPLE, ds_path))
  ds_rows <- vector("list", N_DOWNSAMPLE)
  t0 <- Sys.time()
  for (i in seq_len(N_DOWNSAMPLE)) {
    bd <- balance_within_subject(data)
    fit <- fit_glmm(bd)
    ds_rows[[i]] <- data.frame(iter = i, n_trials = nrow(bd),
                               or = fit$or, z = fit$z, p = fit$p)
    if (i %% 20 == 0) {
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      eta <- elapsed / i * (N_DOWNSAMPLE - i)
      cat(sprintf("    ds %d/%d  elapsed=%.0fs  ETA=%.0fs\n",
                  i, N_DOWNSAMPLE, elapsed, eta))
      write.csv(do.call(rbind, ds_rows[1:i]), ds_path, row.names = FALSE)
    }
  }
  write.csv(do.call(rbind, ds_rows), ds_path, row.names = FALSE)
  ds <- do.call(rbind, ds_rows)
  cat(sprintf(paste0("Downsample summary: median OR = %.3f [%.3f, %.3f]; ",
                     "p<.05 in %d/%d iters (%.1f%%)\n"),
              median(ds$or, na.rm = TRUE),
              quantile(ds$or, 0.025, na.rm = TRUE),
              quantile(ds$or, 0.975, na.rm = TRUE),
              sum(ds$p < 0.05, na.rm = TRUE), nrow(ds),
              100 * mean(ds$p < 0.05, na.rm = TRUE)))
}


# ── Main ─────────────────────────────────────────────────────────────────────

run_effect("HPC_PRC_coherence", prep_coh_allhpc_prc())
run_effect("EC_PRC_pac",         prep_pac_ec_prc())

cat("\n===== DONE =====\n")
