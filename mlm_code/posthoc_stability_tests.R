##############################################################################
# Stability testing for trending encoding GLMM panels (raw p < .10):
#   (1) Leave-one-subject-out (LOSO) refits
#   (2) Per-subject deltas of the focal feature
#   (3) Balanced-subsample (downsample) sensitivity
#
# Runs for all four analysis types. For main-effect models the focal term is
# `band_c`; for the interaction model it's `band_c:StimCondstim`. Subject
# deltas for main effects = mean(band_c | remembered) - mean(band_c | forgotten)
# within stim filter. Subject deltas for interactions = stim-vs-nostim difference
# in per-subject memory rate slopes.
#
# N_DOWNSAMPLE = 100 (matches prior sensitivity work, faster than 200).
#
# Outputs per trending panel:
#   stability_<measure>_<unit>_<band>_loso.csv
#   stability_<measure>_<unit>_<band>_deltas.csv
#   stability_<measure>_<unit>_<band>_downsample.csv
##############################################################################

source("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES/mlm_code/posthoc_common.R")

set.seed(42)

N_DOWNSAMPLE <- 100

is_interaction <- function(analysis) {
  identical(analysis, "GLMM_interactions_model_variations")
}

ri_formula <- function(analysis) {
  if (is_interaction(analysis))
    as.formula("Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)")
  else
    as.formula("Accuracy ~ band_c + (1 | Patient)")
}

focal_term_for <- function(analysis) {
  if (is_interaction(analysis)) "band_c:StimCondstim" else "band_c"
}

fit_or_centered <- function(formula, data) {
  m <- safe_glmer(formula, data)
  if (is.null(m)) {
    data$band_c <- data$band_c - mean(data$band_c, na.rm = TRUE)
    m <- safe_glmer(formula, data)
  }
  m
}

fit_summary <- function(formula, data, focal) {
  m <- fit_or_centered(formula, data)
  if (is.null(m)) return(list(estimate = NA, se = NA, z = NA, p = NA,
                              or = NA, ci_low = NA, ci_high = NA))
  s <- summary(m)$coefficients
  if (!focal %in% rownames(s)) return(list(estimate = NA, se = NA, z = NA, p = NA,
                                           or = NA, ci_low = NA, ci_high = NA))
  est <- s[focal, "Estimate"]; se <- s[focal, "Std. Error"]
  zv  <- s[focal, "z value"];  pv <- s[focal, "Pr(>|z|)"]
  ci  <- est + c(-1, 1) * 1.96 * se
  list(estimate = est, se = se, z = zv, p = pv,
       or = exp(est), ci_low = exp(ci[1]), ci_high = exp(ci[2]))
}

# Per-subject delta of band_c (mean across trials of one class minus the other).
# For main effects: yes - no within the contrast's StimCond filter.
# For interactions: (stim_yes - stim_no) - (nostim_yes - nostim_no)  per subject,
#                    i.e. the within-subject interaction signal.
compute_deltas <- function(d, analysis) {
  if (is_interaction(analysis)) {
    agg <- d %>%
      group_by(Patient, StimCond, yes_or_no) %>%
      summarise(band_c_mean = mean(band_c, na.rm = TRUE),
                n = n(), .groups = "drop")
    out <- agg %>%
      tidyr::pivot_wider(names_from = c(StimCond, yes_or_no),
                         values_from = c(band_c_mean, n),
                         values_fill = NA)
    # Components: band_c_mean_<cond>_<yesno>
    safe_get <- function(nm) if (nm %in% names(out)) out[[nm]] else rep(NA, nrow(out))
    out$stim_delta <-
      safe_get("band_c_mean_stim_yes")  - safe_get("band_c_mean_stim_no")
    out$nostim_delta <-
      safe_get("band_c_mean_nostim_yes") - safe_get("band_c_mean_nostim_no")
    out$interaction_delta <- out$stim_delta - out$nostim_delta
    out
  } else {
    agg <- d %>%
      group_by(Patient, yes_or_no) %>%
      summarise(band_c_mean = mean(band_c, na.rm = TRUE),
                n = n(), .groups = "drop") %>%
      tidyr::pivot_wider(names_from = yes_or_no,
                         values_from = c(band_c_mean, n))
    safe_get <- function(nm) if (nm %in% names(agg)) agg[[nm]] else rep(NA, nrow(agg))
    agg$delta <- safe_get("band_c_mean_yes") - safe_get("band_c_mean_no")
    agg
  }
}

# Balanced-subsample helper: within each Patient, downsample whichever class
# is larger to match the smaller class size. For interactions, balance within
# (Patient x StimCond). Returns the balanced data frame.
balance_within_subject <- function(d, analysis) {
  if (is_interaction(analysis)) {
    keys <- d %>% group_by(Patient, StimCond) %>% summarise(.groups = "drop")
    out <- list()
    for (i in seq_len(nrow(keys))) {
      sub <- d[d$Patient == keys$Patient[i] & d$StimCond == keys$StimCond[i], ]
      n_yes <- sum(sub$Accuracy == 1L); n_no <- sum(sub$Accuracy == 0L)
      k <- min(n_yes, n_no)
      if (k == 0) next
      yes <- sub[sub$Accuracy == 1L, ]; no <- sub[sub$Accuracy == 0L, ]
      out[[length(out) + 1]] <- rbind(
        yes[sample(nrow(yes), k), ],
        no[sample(nrow(no),  k), ]
      )
    }
    do.call(rbind, out)
  } else {
    out <- list()
    for (p in unique(d$Patient)) {
      sub <- d[d$Patient == p, ]
      n_yes <- sum(sub$Accuracy == 1L); n_no <- sum(sub$Accuracy == 0L)
      k <- min(n_yes, n_no)
      if (k == 0) next
      yes <- sub[sub$Accuracy == 1L, ]; no <- sub[sub$Accuracy == 0L, ]
      out[[length(out) + 1]] <- rbind(
        yes[sample(nrow(yes), k), ],
        no[sample(nrow(no),  k), ]
      )
    }
    do.call(rbind, out)
  }
}

# Run stability tests on a single panel for a single analysis.
run_panel <- function(measure, scope, unit, band, analysis, out_dir) {
  contrast <- ANALYSIS_CONTRAST[[analysis]]
  d <- panel_data(measure, scope, unit, band, contrast)
  n_pat <- length(unique(d$Patient))
  if (nrow(d) < MIN_TRIALS || n_pat < MIN_PATIENTS) {
    cat(sprintf("  SKIP %s/%s/%s (n=%d trials, %d pats)\n",
                measure, unit, band, nrow(d), n_pat))
    return(invisible(NULL))
  }
  focal <- focal_term_for(analysis)
  formula <- ri_formula(analysis)

  # Observed fit
  obs <- fit_summary(formula, d, focal)
  cat(sprintf("  obs: OR=%.3g, z=%.2f, p=%.4f\n",
              obs$or, obs$z, obs$p))

  # ── LOSO ─────────────────────────────────────────────────────────────
  loso_rows <- list()
  # First row: full sample
  loso_rows[[1]] <- data.frame(
    excluded = "<none>", n_trials = nrow(d), n_patients = n_pat,
    OR = obs$or, lo = obs$ci_low, hi = obs$ci_high,
    z = obs$z, p = obs$p, stringsAsFactors = FALSE)
  subjects <- as.character(unique(d$Patient))
  for (s in subjects) {
    sub <- d[d$Patient != s, ]
    sub$Patient <- droplevels(factor(sub$Patient))
    if (nrow(sub) < MIN_TRIALS || length(unique(sub$Patient)) < MIN_PATIENTS) {
      loso_rows[[length(loso_rows) + 1]] <- data.frame(
        excluded = s, n_trials = nrow(sub),
        n_patients = length(unique(sub$Patient)),
        OR = NA, lo = NA, hi = NA, z = NA, p = NA,
        stringsAsFactors = FALSE)
      next
    }
    f <- fit_summary(formula, sub, focal)
    loso_rows[[length(loso_rows) + 1]] <- data.frame(
      excluded = s, n_trials = nrow(sub),
      n_patients = length(unique(sub$Patient)),
      OR = f$or, lo = f$ci_low, hi = f$ci_high,
      z = f$z, p = f$p, stringsAsFactors = FALSE)
  }
  loso <- do.call(rbind, loso_rows)
  loso_path <- file.path(out_dir,
    sprintf("stability_%s_%s_%s_loso.csv", measure, unit, band))
  write.csv(loso, loso_path, row.names = FALSE)

  # ── Subject deltas ──────────────────────────────────────────────────
  deltas <- compute_deltas(d, analysis)
  deltas_path <- file.path(out_dir,
    sprintf("stability_%s_%s_%s_deltas.csv", measure, unit, band))
  write.csv(deltas, deltas_path, row.names = FALSE)

  # ── Balanced-subsample ──────────────────────────────────────────────
  ds_rows <- vector("list", N_DOWNSAMPLE)
  for (i in seq_len(N_DOWNSAMPLE)) {
    bd <- balance_within_subject(d, analysis)
    if (is.null(bd) || nrow(bd) < MIN_TRIALS) {
      ds_rows[[i]] <- data.frame(iter = i, n_trials = ifelse(is.null(bd), 0, nrow(bd)),
                                 OR = NA, z = NA, p = NA,
                                 stringsAsFactors = FALSE)
      next
    }
    bd$Patient <- droplevels(factor(bd$Patient))
    f <- fit_summary(formula, bd, focal)
    ds_rows[[i]] <- data.frame(iter = i, n_trials = nrow(bd),
                               OR = f$or, z = f$z, p = f$p,
                               stringsAsFactors = FALSE)
  }
  ds <- do.call(rbind, ds_rows)
  ds_path <- file.path(out_dir,
    sprintf("stability_%s_%s_%s_downsample.csv", measure, unit, band))
  write.csv(ds, ds_path, row.names = FALSE)

  cat(sprintf("    loso p range: %.3f-%.3f | DS median OR=%.3g, p<.05 in %d/%d\n",
              min(loso$p, na.rm = TRUE), max(loso$p, na.rm = TRUE),
              median(ds$OR, na.rm = TRUE),
              sum(ds$p < 0.05, na.rm = TRUE), nrow(ds)))
}

run_analysis <- function(analysis) {
  out_dir <- posthoc_out_dir(ANALYSIS_DIR[[analysis]])
  trending_csv <- file.path(out_dir, "trending_panels.csv")
  if (!file.exists(trending_csv)) {
    cat(sprintf("[%s] no trending_panels.csv; skipping\n", analysis)); return()
  }
  trending <- read.csv(trending_csv, stringsAsFactors = FALSE)
  cat(sprintf("\n========== %s: %d trending panels ==========\n",
              analysis, nrow(trending)))
  for (i in seq_len(nrow(trending))) {
    r <- trending[i, ]
    cat(sprintf("[%d/%d] %s | %s | %s | %s  (obs p=%.4f)\n",
                i, nrow(trending), r$measure, r$scope, r$unit, r$band, r$p_value))
    run_panel(r$measure, r$scope, r$unit, r$band, analysis, out_dir)
  }
}

.args <- commandArgs(trailingOnly = TRUE)
analyses <- if (length(.args) >= 1) .args[1] else names(ANALYSIS_DIR)
for (a in analyses) run_analysis(a)

cat("\n===== STABILITY TESTS DONE =====\n")
