##############################################################################
# Hypotheses 1a/1b/1c main-effect GLMM pipeline (imbalanced trials).
#
# Sibling of run_h1abc_full_retrieval.R but estimates the *main effect of the
# band on memory*, separately for three trial-subsets:
#
#   contrast in {all, nostim, stim}
#     all     -> all stim + nostim trials pooled
#     nostim  -> endogenous (no-stim) trials only
#     stim    -> stim trials only
#
# Per region/pair per band fits a binomial GLMM (one model per panel):
#
#   Accuracy ~ band_c + (1 | Patient)             (focal model)
#   Accuracy ~ 1      + (1 | Patient)             (null, for LR test)
#
# Focal term for FDR = band_c. band_c is the raw band value; if the model
# fails to converge it is grand-mean-centered as a numerical-stability
# fallback (logged to _glmm_centering_log.csv in OUT_BASE).
#
# Scopes covered: BLAMTL, HPCrhinal, HippSubBLA, HippSubRhinal (matches
# run_h1abc_full_retrieval.R). DG drop rule: a panel with < 4 unique patients
# or < 30 trials is dropped (and never enters the FDR family).
#
# Output: OUTPUTS/<phase>_memory_reports/stats/h1abc_maineffect_<phase>_mlm/
#         <measure>_<scope>_<contrast>/
#           <measure>_<unit>_<band>_coefs.csv
#           <measure>_<unit>_<band>_anova.csv
#           <measure>_<unit>_<band>_predictions.csv
#
# Usage:
#   Rscript run_h1abc_maineffect.R                              # all combos
#   Rscript run_h1abc_maineffect.R encoding power BLAMTL all    # one combo
#   Rscript run_h1abc_maineffect.R retrieval                    # all combos for retrieval
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
  if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
  if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")
  if (!require("dplyr"))       install.packages("dplyr",        repos = "https://cloud.r-project.org")
  library(lme4); library(lmerTest); library(broom.mixed); library(dplyr)
})

REPO_ROOT <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_IN <- file.path(REPO_ROOT, "OUTPUTS", "csvs")

MIN_PATIENTS <- 4
MIN_TRIALS   <- 30

THETA          <- c(4.88, 7.81)
SLOW_GAMMA     <- c(30.27, 54.69)
PAC_SLOW_GAMMA <- c(30, 50)

ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# ── Region scopes (mirror run_h1abc_full_retrieval.R) ────────────────────────
SCOPES <- list(
  BLAMTL = list(
    power_regions = c("BLA", "ALLHPC", "EC", "PRC"),
    coh_pairs     = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    pac_pairs     = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    needs_allhpc  = TRUE
  ),
  HPCrhinal = list(
    power_regions = c("ALLHPC", "EC", "PRC"),
    coh_pairs     = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
    pac_pairs     = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
    needs_allhpc  = TRUE
  ),
  HippSubBLA = list(
    power_regions = c("BLA", "CA", "DG", "HPC"),
    coh_pairs     = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    pac_pairs     = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    needs_allhpc  = FALSE
  ),
  HippSubRhinal = list(
    power_regions = c("CA", "DG", "HPC"),
    coh_pairs     = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"),
    pac_pairs     = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"),
    needs_allhpc  = FALSE
  )
)

HPC_SUB <- c("HPC", "CA", "DG")
ALLHPC_PAIR_SOURCES <- list(
  BLA_ALLHPC  = c("BLA_HPC", "BLA_CA", "BLA_DG"),
  ALLHPC_EC   = c("EC_HPC",  "CA_EC",  "DG_EC"),
  ALLHPC_PRC  = c("HPC_PRC", "CA_PRC", "DG_PRC")
)

CONTRASTS <- c("all", "nostim", "stim")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ensure_dir <- function(p) { dir.create(p, showWarnings = FALSE, recursive = TRUE); p }

save_csv <- function(df, path) {
  ensure_dir(dirname(path))
  write.csv(df, path, row.names = FALSE)
  cat(sprintf("    -> %s\n", basename(path)))
}

safe_glmer <- function(formula, data) {
  tryCatch(
    glmer(formula, data = data, family = binomial, control = ctrl_glmer),
    error = function(e) { cat("    GLMER FAILED:", conditionMessage(e), "\n"); NULL }
  )
}

load_measure_df <- function(measure, phase) {
  fname <- switch(measure,
                  power     = sprintf("combined_%s_power_all_mlmr_input.csv", phase),
                  coherence = sprintf("combined_%s_coherence_all_mlmr_input.csv", phase),
                  pac       = sprintf("combined_%s_pac_all_mlmr_input.csv", phase))
  d <- read.csv(file.path(CSV_IN, fname), stringsAsFactors = FALSE,
                check.names = FALSE)
  d <- d[d$yes_or_no %in% c("yes", "no") & d$trial_type %in% c("nostim", "stim"), ]
  d$Patient[d$Patient == "BJH033"] <- "BJH032"
  d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
  d$StimCond <- factor(ifelse(d$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))
  d
}

build_allhpc_power <- function(d) {
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  meta_cols <- intersect(c("Patient", "Region", "trial_type", "yes_or_no",
                           "Accuracy", "StimCond"), names(d))
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

build_allhpc_pair <- function(d, target_label, source_pairs) {
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  meta_cols <- intersect(c("Patient", "Region", "trial_type", "yes_or_no",
                           "Accuracy", "StimCond"), names(d))
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

prep_with_allhpc <- function(d, measure) {
  if (measure == "power") {
    extra <- build_allhpc_power(d)
    if (!is.null(extra)) d <- bind_rows(d, extra)
  } else {
    for (target in names(ALLHPC_PAIR_SOURCES)) {
      sources <- ALLHPC_PAIR_SOURCES[[target]]
      extra <- build_allhpc_pair(d, target, sources)
      if (!is.null(extra)) d <- bind_rows(d, extra)
    }
  }
  d
}

add_band_columns <- function(d, measure) {
  freq_cols <- sort(grep("^diff_Freq_", names(d), value = TRUE))
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
  if (measure == "pac") {
    sg_cols <- freq_cols[freqs >= PAC_SLOW_GAMMA[1] & freqs <= PAC_SLOW_GAMMA[2]]
    d$slow_gamma <- rowMeans(d[, sg_cols, drop = FALSE], na.rm = TRUE)
  } else {
    th_cols <- freq_cols[freqs >= THETA[1]      & freqs <= THETA[2]]
    sg_cols <- freq_cols[freqs >= SLOW_GAMMA[1] & freqs <= SLOW_GAMMA[2]]
    d$theta      <- rowMeans(d[, th_cols, drop = FALSE], na.rm = TRUE)
    d$slow_gamma <- rowMeans(d[, sg_cols, drop = FALSE], na.rm = TRUE)
  }
  d
}

apply_contrast_filter <- function(d, contrast) {
  if (contrast == "all")     return(d)
  if (contrast == "nostim")  return(d[d$StimCond == "nostim", ])
  if (contrast == "stim")    return(d[d$StimCond == "stim", ])
  stop("Unknown contrast: ", contrast)
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-panel GLMM (one model per region/pair per band per contrast)
#   Accuracy ~ band_c + (1 | Patient)
# Also save predictions on a band_c grid for plotting.
# ─────────────────────────────────────────────────────────────────────────────

save_anova_build <- function(fits, fmls, path) {
  rows <- list(); prev <- NULL
  for (nm in names(fits)) {
    m <- fits[[nm]]; if (is.null(m)) next
    npar <- attr(logLik(m), "df"); ll <- as.numeric(logLik(m))
    chisq <- NA_real_; df_v <- NA_real_; pv <- NA_real_
    if (!is.null(prev)) {
      cmp <- tryCatch(anova(prev, m, refit = FALSE), error = function(e) NULL)
      if (is.null(cmp)) cmp <- tryCatch(anova(prev, m), error = function(e) NULL)
      pick <- function(col) {
        v <- cmp[[col]]; if (is.null(v) || length(v) < 2) NA_real_ else as.numeric(v[2])
      }
      if (!is.null(cmp)) {
        chisq <- pick("Chisq"); df_v <- pick("Chi Df")
        if (is.na(df_v)) df_v <- pick("Df")
        pv    <- pick("Pr(>Chisq)")
      }
    }
    rows[[length(rows) + 1]] <- data.frame(
      Model = nm, Formula = fmls[[nm]], npar = npar,
      AIC = AIC(m), BIC = BIC(m), logLik = ll, deviance = -2 * ll,
      Chisq = chisq, Df = df_v, p.value = pv, stringsAsFactors = FALSE
    )
    prev <- m
  }
  if (length(rows) == 0) return(invisible(NULL))
  save_csv(do.call(rbind, rows), path)
}

save_predictions <- function(model, band_values, path) {
  beta <- fixef(model); V <- vcov(model)
  qs <- quantile(band_values, probs = c(0.02, 0.98), na.rm = TRUE)
  grid_band <- seq(qs[1], qs[2], length.out = 200)
  X <- cbind(`(Intercept)` = 1, band_c = grid_band)
  X <- X[, names(beta), drop = FALSE]
  eta    <- as.numeric(X %*% beta)
  se_eta <- sqrt(rowSums((X %*% V) * X))
  preds <- data.frame(
    band  = grid_band,
    p_hat = plogis(eta),
    p_lo  = plogis(eta - 1.96 * se_eta),
    p_hi  = plogis(eta + 1.96 * se_eta)
  )
  save_csv(preds, path)
}

run_glmm_unit <- function(sub, out_dir, modality, unit, band_label,
                          phase, contrast, log_path) {
  sub <- sub[is.finite(sub[[band_label]]), ]
  n_pat <- length(unique(sub$Patient))
  if (nrow(sub) < MIN_TRIALS || n_pat < MIN_PATIENTS) {
    cat(sprintf("    GLMM SKIP %s/%s/%s (n=%d trials, %d patients)\n",
                modality, unit, band_label, nrow(sub), n_pat))
    return(invisible(NULL))
  }
  sub$Patient <- factor(sub$Patient)
  feat <- sub[[band_label]]

  fmls <- list(
    m0     = "Accuracy ~ 1 + (1 | Patient)",
    m_full = "Accuracy ~ band_c + (1 | Patient)"
  )

  sub$band_c <- feat
  fits <- lapply(fmls, function(f) safe_glmer(as.formula(f), sub))
  centered_used <- FALSE
  if (any(sapply(fits, is.null))) {
    cat(sprintf("    GLMM convergence fallback to centered for %s/%s/%s\n",
                modality, unit, band_label))
    sub$band_c <- feat - mean(feat, na.rm = TRUE)
    fits <- lapply(fmls, function(f) safe_glmer(as.formula(f), sub))
    centered_used <- TRUE
  }

  coefs_path <- file.path(out_dir,
                          sprintf("%s_%s_%s_coefs.csv",
                                  modality, unit, band_label))
  anova_path <- file.path(out_dir,
                          sprintf("%s_%s_%s_anova.csv",
                                  modality, unit, band_label))
  preds_path <- file.path(out_dir,
                          sprintf("%s_%s_%s_predictions.csv",
                                  modality, unit, band_label))

  if (!is.null(fits$m_full)) {
    td <- tidy(fits$m_full, effects = "fixed", conf.int = TRUE,
               exponentiate = TRUE)
    save_csv(td, coefs_path)
    save_predictions(fits$m_full, sub$band_c, preds_path)
  }
  save_anova_build(fits, fmls, anova_path)

  log_row <- data.frame(
    phase = phase, contrast = contrast, modality = modality, unit = unit,
    band = band_label, centered = centered_used,
    n_trials = nrow(sub), n_patients = n_pat,
    stringsAsFactors = FALSE
  )
  write.table(log_row, log_path, sep = ",",
              append = file.exists(log_path),
              col.names = !file.exists(log_path), row.names = FALSE)
}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

run_combo <- function(phase, measure, scope_name, contrast) {
  scope <- SCOPES[[scope_name]]
  phase_folder <- if (phase == "retrieval") "retrieval_memory_reports" else "encoding_memory_reports"
  out_base <- file.path(REPO_ROOT, "OUTPUTS", phase_folder,
                        "stats", sprintf("h1abc_maineffect_%s_mlm", phase))
  out_dir <- ensure_dir(file.path(out_base,
                                  sprintf("%s_%s_%s",
                                          measure, scope_name, contrast)))
  log_path <- file.path(out_base, "_glmm_centering_log.csv")
  cat(sprintf("\n=== %s | %s | %s | %s ===\n",
              phase, measure, scope_name, contrast))

  d <- load_measure_df(measure, phase)
  if (scope$needs_allhpc) d <- prep_with_allhpc(d, measure)
  d <- add_band_columns(d, measure)
  d <- apply_contrast_filter(d, contrast)

  if (measure == "power") {
    units <- scope$power_regions
  } else if (measure == "coherence") {
    units <- scope$coh_pairs
  } else {
    units <- scope$pac_pairs
  }
  bands <- if (measure == "pac") c("slow_gamma") else c("theta", "slow_gamma")

  for (u in units) {
    sub <- d[d$Region == u, ]
    if (nrow(sub) == 0) {
      cat(sprintf("    NO ROWS for %s\n", u)); next
    }
    for (b in bands) run_glmm_unit(sub, out_dir, measure, u, b,
                                   phase, contrast, log_path)
  }
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

MEASURES <- c("power", "coherence", "pac")
SCOPE_NAMES <- names(SCOPES)
PHASES <- c("encoding", "retrieval")

.cmd_args <- commandArgs(trailingOnly = TRUE)

# Optional 1st arg: phase. Optional 2-4: measure scope contrast.
phases <- PHASES
if (length(.cmd_args) >= 1 && .cmd_args[1] %in% PHASES) {
  phases <- .cmd_args[1]
  .cmd_args <- .cmd_args[-1]
}

if (length(.cmd_args) >= 3) {
  measure  <- .cmd_args[1]
  scope    <- .cmd_args[2]
  contrast <- .cmd_args[3]
  for (p in phases) run_combo(p, measure, scope, contrast)
} else {
  for (p in phases) {
    for (m in MEASURES) {
      for (s in SCOPE_NAMES) {
        for (c in CONTRASTS) {
          run_combo(p, m, s, c)
        }
      }
    }
  }
}

cat("\n===== H1ABC MAIN-EFFECT PIPELINE COMPLETE =====\n")
