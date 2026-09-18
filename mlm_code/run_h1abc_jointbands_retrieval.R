##############################################################################
# Joint-bands GLMM test pipeline.
#
# Per (measure x scope), fit ONE GLMM per region/pair with theta AND slow_gamma
# entered together as predictors:
#
#   Accuracy ~ StimCond + theta_c + slow_gamma_c
#            + theta_c:StimCond + slow_gamma_c:StimCond + (1 | Patient)
#
# This is a robustness check against the per-band GLMMs in
# run_h1abc_full_retrieval.R: how does jointly entering theta + slow gamma
# change inference and FDR-BH correction for the StimCond x band
# interactions?
#
# Restricted to power + coherence (PAC has only slow_gamma).
#
# band predictors are RAW by default. On convergence failure we fall back to
# grand-mean centering both bands (and log it).
#
# Output:
#   OUTPUTS/<phase>_memory_reports/testing/stats/jointbands_glmm/
#       <measure>_<scope>/*.csv
#
# Usage:
#   Rscript run_h1abc_jointbands_retrieval.R                       # all retrieval combos
#   Rscript run_h1abc_jointbands_retrieval.R encoding              # all encoding combos
#   Rscript run_h1abc_jointbands_retrieval.R retrieval power BLAMTL
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

.cmd_args <- commandArgs(trailingOnly = TRUE)
PHASE <- "retrieval"
if (length(.cmd_args) >= 1 && .cmd_args[1] %in% c("retrieval", "encoding")) {
  PHASE <- .cmd_args[1]
  .cmd_args <- .cmd_args[-1]
}
phase_folder <- if (PHASE == "retrieval") "retrieval_memory_reports" else "encoding_memory_reports"
OUT_BASE <- file.path(REPO_ROOT, "OUTPUTS", phase_folder,
                      "testing", "stats", "jointbands_glmm")
dir.create(OUT_BASE, showWarnings = FALSE, recursive = TRUE)

MIN_PATIENTS <- 4
MIN_TRIALS   <- 30

THETA      <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)

ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# ── Region scopes (mirror of run_h1abc_full_retrieval.R) ─────────────────────
SCOPES <- list(
  BLAMTL = list(
    power_regions = c("BLA", "ALLHPC", "EC", "PRC"),
    coh_pairs     = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    needs_allhpc  = TRUE
  ),
  HPCrhinal = list(
    power_regions = c("ALLHPC", "EC", "PRC"),
    coh_pairs     = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
    needs_allhpc  = TRUE
  ),
  HippSubBLA = list(
    power_regions = c("BLA", "CA", "DG", "HPC"),
    coh_pairs     = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    needs_allhpc  = FALSE
  ),
  HippSubRhinal = list(
    # EC, PRC power and EC_PRC pairs belong to HPCrhinal only.
    # ALLHPC pairs are never used here (subregions only).
    power_regions = c("CA", "DG", "HPC"),
    coh_pairs     = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"),
    needs_allhpc  = FALSE
  )
)

HPC_SUB <- c("HPC", "CA", "DG")
ALLHPC_PAIR_SOURCES <- list(
  BLA_ALLHPC = c("BLA_HPC", "BLA_CA", "BLA_DG"),
  ALLHPC_EC  = c("EC_HPC",  "CA_EC",  "DG_EC"),
  ALLHPC_PRC = c("HPC_PRC", "CA_PRC", "DG_PRC")
)

# ── Responder data (StimCond is encoded in trial_type so resp not strictly
# needed, but kept for parity with the full pipeline) ────────────────────────
resp <- read.csv(file.path(CSV_IN, "AMMEBLAES_responder_status.csv"),
                 stringsAsFactors = FALSE)
names(resp)[names(resp) == "Responder.status"] <- "ResponderStatus"
resp$Patient[resp$Patient == "BJH033"] <- "BJH032"
resp <- resp[!duplicated(resp$Patient), ]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ensure_dir <- function(p) { dir.create(p, showWarnings = FALSE, recursive = TRUE); p }

save_csv <- function(df, path) {
  ensure_dir(dirname(path))
  write.csv(df, path, row.names = FALSE)
  cat(sprintf("    -> %s\n", basename(path)))
}

save_coefs <- function(model, path, exponentiate = FALSE) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = exponentiate)
  save_csv(td, path)
  td
}

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

safe_glmer <- function(formula, data) {
  tryCatch(
    glmer(formula, data = data, family = binomial, control = ctrl_glmer),
    error = function(e) { cat("    GLMER FAILED:", conditionMessage(e), "\n"); NULL }
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & ALLHPC aggregation (copied from run_h1abc_full_retrieval.R
# style; PAC excluded since it has only one band)
# ─────────────────────────────────────────────────────────────────────────────

load_measure_df <- function(measure) {
  fname <- switch(measure,
                  power     = sprintf("combined_%s_power_all_mlmr_input.csv", PHASE),
                  coherence = sprintf("combined_%s_coherence_all_mlmr_input.csv", PHASE))
  d <- read.csv(file.path(CSV_IN, fname), stringsAsFactors = FALSE,
                check.names = FALSE)
  d <- d[d$yes_or_no %in% c("yes", "no") & d$trial_type != "new", ]
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

add_band_columns <- function(d) {
  freq_cols <- sort(grep("^diff_Freq_", names(d), value = TRUE))
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
  th_cols <- freq_cols[freqs >= THETA[1]      & freqs <= THETA[2]]
  sg_cols <- freq_cols[freqs >= SLOW_GAMMA[1] & freqs <= SLOW_GAMMA[2]]
  d$theta      <- rowMeans(d[, th_cols, drop = FALSE], na.rm = TRUE)
  d$slow_gamma <- rowMeans(d[, sg_cols, drop = FALSE], na.rm = TRUE)
  d
}


# ─────────────────────────────────────────────────────────────────────────────
# Joint-bands GLMM (one model per region/pair with theta + slow_gamma)
# ─────────────────────────────────────────────────────────────────────────────

run_glmm_joint <- function(sub, out_dir, modality, unit) {
  sub <- sub[is.finite(sub$theta) & is.finite(sub$slow_gamma), ]
  n_pat <- length(unique(sub$Patient))
  if (nrow(sub) < MIN_TRIALS || n_pat < MIN_PATIENTS) {
    cat(sprintf("    GLMM SKIP %s/%s (n=%d trials, %d patients)\n",
                modality, unit, nrow(sub), n_pat))
    return(invisible(NULL))
  }
  sub$Patient <- factor(sub$Patient)

  fmls <- list(
    m0     = "Accuracy ~ 1 + (1 | Patient)",
    m1     = "Accuracy ~ theta_c + slow_gamma_c + (1 | Patient)",
    m2     = "Accuracy ~ theta_c + slow_gamma_c + StimCond + (1 | Patient)",
    m_full = paste(
      "Accuracy ~ theta_c + slow_gamma_c + StimCond",
      "+ theta_c:StimCond + slow_gamma_c:StimCond + (1 | Patient)"
    )
  )

  # First attempt: raw band values.
  sub$theta_c      <- sub$theta
  sub$slow_gamma_c <- sub$slow_gamma
  fits <- lapply(fmls, function(f) safe_glmer(as.formula(f), sub))
  centered_used <- FALSE

  if (any(sapply(fits, is.null))) {
    cat(sprintf("    GLMM convergence fallback to centered for %s/%s\n",
                modality, unit))
    sub$theta_c      <- sub$theta      - mean(sub$theta,      na.rm = TRUE)
    sub$slow_gamma_c <- sub$slow_gamma - mean(sub$slow_gamma, na.rm = TRUE)
    fits <- lapply(fmls, function(f) safe_glmer(as.formula(f), sub))
    centered_used <- TRUE
  }

  if (!is.null(fits$m_full)) {
    save_coefs(fits$m_full,
               file.path(out_dir,
                         sprintf("%s_%s_jointbands_coefs.csv",
                                 modality, unit)),
               exponentiate = TRUE)
  }
  save_anova_build(fits, fmls,
                   file.path(out_dir,
                             sprintf("%s_%s_jointbands_anova.csv",
                                     modality, unit)))

  log_path <- file.path(OUT_BASE, "_glmm_centering_log.csv")
  log_row <- data.frame(
    phase = PHASE, modality = modality, unit = unit,
    centered = centered_used,
    stringsAsFactors = FALSE
  )
  write.table(log_row, log_path, sep = ",", append = file.exists(log_path),
              col.names = !file.exists(log_path), row.names = FALSE)
}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

run_combo <- function(measure, scope_name) {
  scope <- SCOPES[[scope_name]]
  out_dir <- ensure_dir(file.path(OUT_BASE,
                                  sprintf("%s_%s", measure, scope_name)))
  cat(sprintf("\n=== %s | %s | jointbands GLMM ===\n", measure, scope_name))

  d <- load_measure_df(measure)
  if (scope$needs_allhpc) d <- prep_with_allhpc(d, measure)
  d <- add_band_columns(d)

  units <- if (measure == "power") scope$power_regions else scope$coh_pairs

  for (u in units) {
    sub <- d[d$Region == u, ]
    if (nrow(sub) == 0) {
      cat(sprintf("    NO ROWS for %s\n", u)); next
    }
    run_glmm_joint(sub, out_dir, measure, u)
  }
}


MEASURES <- c("power", "coherence")
SCOPE_NAMES <- names(SCOPES)

cat(sprintf("\n>>> Phase: %s   Output: %s\n", PHASE, OUT_BASE))

if (length(.cmd_args) >= 2) {
  run_combo(.cmd_args[1], .cmd_args[2])
} else {
  for (m in MEASURES) {
    for (s in SCOPE_NAMES) {
      run_combo(m, s)
    }
  }
}

cat(sprintf("\n===== JOINT-BANDS GLMM %s PIPELINE COMPLETE =====\n",
            toupper(PHASE)))
