##############################################################################
# Stability analyses for the two trending encoding LMM-cont panels:
#   EC slow gamma power (LRT p = 0.040, Wald p = 0.060, q_FDR = 0.144)
#   PRC slow gamma power (Wald p = 0.096, q_FDR = 0.144)
#
# Model: feature ~ mem_mod_z + (1 | Patient)
#   feature = trial-level slow-gamma power, baseline-corrected
#   mem_mod_z = z-scored avg_stim_dprime_diff  (one value per subject)
#
# Computes for each panel:
#   (1) Observed slope, SE, t, df (Satterthwaite), Wald p, LRT p
#   (2) LOSO refits (drop each subject, record slope and p)
#   (3) Bootstrap by subject (500 iters, with subject relabeling)
#   (4) Permutation: shuffle mem_mod_z across subjects (500 iters)
#   (5) Per-subject summary: mean feature, n_trials, mem_mod_z
#
# Output: OUTPUTS/encoding_memory_reports/
#         EC_PRC_slowgamma_power_lmm_cont_stability/
##############################################################################

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(broom.mixed); library(dplyr)
})

set.seed(42)

REPO <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_IN <- file.path(REPO, "OUTPUTS", "csvs")
OUT <- file.path(REPO, "OUTPUTS", "encoding_memory_reports",
                 "EC_PRC_slowgamma_power_lmm_cont_stability")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

SLOW_GAMMA <- c(30.27, 54.69)
MIN_PATIENTS <- 4
MIN_TRIALS   <- 30
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

N_BOOT <- 500
N_PERM <- 500

# ── Load and prep ──────────────────────────────────────────────────────────
resp <- read.csv(file.path(CSV_IN, "AMMEBLAES_responder_status.csv"),
                 stringsAsFactors = FALSE)
resp$Patient[resp$Patient == "BJH033"] <- "BJH032"
resp <- resp[!duplicated(resp$Patient), ]
resp$mem_mod_z <- as.numeric(scale(resp$avg_stim_dprime_diff))

d <- read.csv(file.path(CSV_IN,
                        "combined_encoding_power_all_mlmr_input.csv"),
              stringsAsFactors = FALSE, check.names = FALSE)
d <- d[d$yes_or_no %in% c("yes", "no") &
       d$trial_type %in% c("nostim", "stim"), ]
d$Patient[d$Patient == "BJH033"] <- "BJH032"
d <- merge(d, resp[, c("Patient", "mem_mod_z")], by = "Patient",
           all.x = TRUE)

freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
sg_cols <- freq_cols[freqs >= SLOW_GAMMA[1] & freqs <= SLOW_GAMMA[2]]
d$feature <- rowMeans(d[, sg_cols, drop = FALSE], na.rm = TRUE)

# ── Helpers ────────────────────────────────────────────────────────────────
fit_summary <- function(data) {
  m <- tryCatch(
    lmer(feature ~ mem_mod_z + (1 | Patient),
         data = data, REML = TRUE, control = ctrl),
    error = function(e) NULL
  )
  if (is.null(m)) return(list(slope = NA, se = NA, t = NA, df = NA,
                              p_wald = NA, p_lrt = NA))
  s <- summary(m)$coefficients
  if (!"mem_mod_z" %in% rownames(s)) {
    return(list(slope = NA, se = NA, t = NA, df = NA,
                p_wald = NA, p_lrt = NA))
  }
  # LRT via ML refits
  m_ml <- tryCatch(
    lmer(feature ~ mem_mod_z + (1 | Patient),
         data = data, REML = FALSE, control = ctrl),
    error = function(e) NULL)
  m0_ml <- tryCatch(
    lmer(feature ~ 1 + (1 | Patient),
         data = data, REML = FALSE, control = ctrl),
    error = function(e) NULL)
  p_lrt <- if (!is.null(m_ml) && !is.null(m0_ml)) {
    cmp <- tryCatch(anova(m0_ml, m_ml), error = function(e) NULL)
    if (!is.null(cmp)) as.numeric(cmp[["Pr(>Chisq)"]][2]) else NA
  } else NA
  list(
    slope = unname(s["mem_mod_z", "Estimate"]),
    se    = unname(s["mem_mod_z", "Std. Error"]),
    t     = unname(s["mem_mod_z", "t value"]),
    df    = unname(s["mem_mod_z", "df"]),
    p_wald = unname(s["mem_mod_z", "Pr(>|t|)"]),
    p_lrt = p_lrt
  )
}


# Bootstrap by subject: resample subjects with replacement, relabel them
# so each appears as a unique random-effect level.
boot_one <- function(data, subj_ids) {
  picked <- sample(subj_ids, length(subj_ids), replace = TRUE)
  pieces <- list()
  for (i in seq_along(picked)) {
    sub <- data[data$Patient == picked[i], ]
    if (nrow(sub) == 0) next
    sub$Patient <- paste0(picked[i], "_b", i)
    pieces[[length(pieces) + 1]] <- sub
  }
  bd <- do.call(rbind, pieces)
  bd$Patient <- factor(bd$Patient)
  bd
}

# Permutation: shuffle mem_mod_z across subjects, then broadcast back.
perm_one <- function(data, subj_ids) {
  # subj_ids is character vector of unique patient IDs (one per).
  mm <- unique(data[, c("Patient", "mem_mod_z")])
  mm <- mm[match(subj_ids, mm$Patient), ]
  perm <- sample(mm$mem_mod_z)
  mp <- data.frame(Patient = subj_ids, mem_mod_perm = perm,
                   stringsAsFactors = FALSE)
  out <- data
  out$mem_mod_z <- mp$mem_mod_perm[match(out$Patient, mp$Patient)]
  out
}


# ── Per-panel runner ───────────────────────────────────────────────────────
run_panel <- function(region) {
  cat(sprintf("\n========== %s slow gamma power ==========\n", region))
  panel <- d[d$Region == region & is.finite(d$feature) &
             !is.na(d$mem_mod_z), ]
  panel$Patient <- factor(panel$Patient)
  subj_ids <- as.character(unique(panel$Patient))
  n_subjects <- length(subj_ids)
  cat(sprintf("Trials: %d, Subjects: %d\n", nrow(panel), n_subjects))
  if (nrow(panel) < MIN_TRIALS || n_subjects < MIN_PATIENTS) {
    cat("  SKIP (insufficient data)\n"); return(invisible(NULL))
  }

  # ── Observed ───────────────────────────────────────────────────────────
  obs <- fit_summary(panel)
  cat(sprintf(paste0("Observed: slope=%+.4f, SE=%.4f, t=%.3f, df=%.1f, ",
                     "Wald p=%.4f, LRT p=%.4f\n"),
              obs$slope, obs$se, obs$t, obs$df, obs$p_wald, obs$p_lrt))
  write.csv(data.frame(
    region = region,
    n_trials = nrow(panel), n_subjects = n_subjects,
    slope = obs$slope, se = obs$se, t = obs$t, df = obs$df,
    ci_low = obs$slope - 1.96 * obs$se,
    ci_high = obs$slope + 1.96 * obs$se,
    p_wald = obs$p_wald, p_lrt = obs$p_lrt
  ), file.path(OUT, sprintf("%s_observed.csv", region)),
  row.names = FALSE)

  # ── Per-subject summary (for scatter plot) ─────────────────────────────
  per_subj <- panel %>%
    group_by(Patient, mem_mod_z) %>%
    summarise(n_trials = n(),
              feature_mean = mean(feature, na.rm = TRUE),
              feature_sem  = sd(feature, na.rm = TRUE) / sqrt(n()),
              .groups = "drop")
  write.csv(per_subj, file.path(OUT, sprintf("%s_per_subject.csv", region)),
            row.names = FALSE)

  # ── LOSO refits ────────────────────────────────────────────────────────
  cat("LOSO refits...\n")
  loso_rows <- list()
  loso_rows[[1]] <- data.frame(
    excluded = "<none>", n_trials = nrow(panel), n_subjects = n_subjects,
    slope = obs$slope, se = obs$se, t = obs$t, df = obs$df,
    p_wald = obs$p_wald, p_lrt = obs$p_lrt,
    stringsAsFactors = FALSE)
  for (s in subj_ids) {
    sub <- panel[panel$Patient != s, ]
    sub$Patient <- droplevels(factor(sub$Patient))
    if (nrow(sub) < MIN_TRIALS ||
        length(unique(sub$Patient)) < MIN_PATIENTS) {
      loso_rows[[length(loso_rows) + 1]] <- data.frame(
        excluded = s, n_trials = nrow(sub),
        n_subjects = length(unique(sub$Patient)),
        slope = NA, se = NA, t = NA, df = NA,
        p_wald = NA, p_lrt = NA, stringsAsFactors = FALSE)
      next
    }
    f <- fit_summary(sub)
    loso_rows[[length(loso_rows) + 1]] <- data.frame(
      excluded = s, n_trials = nrow(sub),
      n_subjects = length(unique(sub$Patient)),
      slope = f$slope, se = f$se, t = f$t, df = f$df,
      p_wald = f$p_wald, p_lrt = f$p_lrt, stringsAsFactors = FALSE)
  }
  loso_df <- do.call(rbind, loso_rows)
  write.csv(loso_df, file.path(OUT, sprintf("%s_loso.csv", region)),
            row.names = FALSE)

  # ── Bootstrap ──────────────────────────────────────────────────────────
  cat(sprintf("Bootstrap by subject (%d iters)...\n", N_BOOT))
  boot_path <- file.path(OUT, sprintf("%s_bootstrap.csv", region))
  boot_rows <- vector("list", N_BOOT)
  t0 <- Sys.time()
  for (i in seq_len(N_BOOT)) {
    bd <- boot_one(panel, subj_ids)
    f  <- fit_summary(bd)
    boot_rows[[i]] <- data.frame(iter = i, slope = f$slope, se = f$se,
                                 t = f$t, df = f$df, p_wald = f$p_wald,
                                 stringsAsFactors = FALSE)
    if (i %% 50 == 0) {
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      eta <- elapsed / i * (N_BOOT - i)
      cat(sprintf("    boot %d/%d  elapsed=%.0fs  ETA=%.0fs\n",
                  i, N_BOOT, elapsed, eta))
      write.csv(do.call(rbind, boot_rows[1:i]), boot_path,
                row.names = FALSE)
    }
  }
  write.csv(do.call(rbind, boot_rows), boot_path, row.names = FALSE)

  # ── Permutation ────────────────────────────────────────────────────────
  cat(sprintf("Permutation by subject (%d iters)...\n", N_PERM))
  perm_path <- file.path(OUT, sprintf("%s_permutation.csv", region))
  perm_rows <- vector("list", N_PERM)
  t0 <- Sys.time()
  for (i in seq_len(N_PERM)) {
    pd <- perm_one(panel, subj_ids)
    f  <- fit_summary(pd)
    perm_rows[[i]] <- data.frame(iter = i, slope = f$slope, t = f$t,
                                 p_wald = f$p_wald, p_lrt = f$p_lrt,
                                 stringsAsFactors = FALSE)
    if (i %% 50 == 0) {
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      eta <- elapsed / i * (N_PERM - i)
      cat(sprintf("    perm %d/%d  elapsed=%.0fs  ETA=%.0fs\n",
                  i, N_PERM, elapsed, eta))
      write.csv(do.call(rbind, perm_rows[1:i]), perm_path,
                row.names = FALSE)
    }
  }
  write.csv(do.call(rbind, perm_rows), perm_path, row.names = FALSE)
}


run_panel("EC")
run_panel("PRC")

cat("\n===== LMM-cont stability done -- outputs in", OUT, "=====\n")
