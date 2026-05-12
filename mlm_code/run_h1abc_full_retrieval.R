##############################################################################
# Hypotheses 1a/1b/1c full retrieval pipeline (imbalanced trials).
#
# Builds 36 sets of CSVs covering:
#   measure  in {power, coherence, pac}
#   scope    in {BLAMTL, HPCrhinal, HippSubBLA, HippSubRhinal}
#       BLAMTL       = BLA + ALLHPC + EC + PRC                  (uses ALLHPC)
#       HPCrhinal    =        ALLHPC + EC + PRC                 (uses ALLHPC)
#       HippSubBLA   = BLA + CA + DG + HPC                      (no ALLHPC)
#       HippSubRhinal=       CA + DG + HPC + EC + PRC           (no ALLHPC)
#   modeltype in {LMMcont, LMMquad, GLMM}
#       LMMcont : feature ~ mem_mod_z + (1|Patient)             (per region/pair)
#       LMMquad : feature ~ QuadResponderGroup + (1|Patient)    (per region/pair)
#       GLMM    : Accuracy ~ band_c + StimCond + Region +
#                            Region:StimCond + (1|Patient)      (one model per band)
#
# band_c is the raw band value (no centering by default). On GLMM convergence
# failure the band is grand-mean-centered as a numerical-stability fallback;
# see _glmm_centering_log.csv in OUT_BASE for which panels used the fallback.
#
# DG drop rule: a panel/region with < 4 unique patients is dropped (and never
# enters the FDR family). MIN_PATIENTS = 4.
#
# Output: OUTPUTS/retrieval_memory_reports/stats/h1abc_full_retrieval_mlm/
#         <measure>_<scope>_<modeltype>/*.csv
#
# Usage:
#   Rscript run_h1abc_full_retrieval.R                # run everything
#   Rscript run_h1abc_full_retrieval.R power BLAMTL LMMcont
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

# Phase is the first arg if it matches "retrieval"/"encoding"; otherwise
# defaults to retrieval and all args are interpreted as combo selectors.
.cmd_args <- commandArgs(trailingOnly = TRUE)
PHASE <- "retrieval"
if (length(.cmd_args) >= 1 && .cmd_args[1] %in% c("retrieval", "encoding")) {
  PHASE <- .cmd_args[1]
  .cmd_args <- .cmd_args[-1]
}
phase_folder <- if (PHASE == "retrieval") "retrieval_memory_reports" else "encoding_memory_reports"
OUT_BASE <- file.path(REPO_ROOT, "OUTPUTS", phase_folder,
                      "stats", sprintf("h1abc_full_%s_mlm", PHASE))
dir.create(OUT_BASE, showWarnings = FALSE, recursive = TRUE)

MIN_PATIENTS <- 4   # drop rule per user spec (<4 = drop)
MIN_TRIALS   <- 30

THETA          <- c(4.88, 7.81)
SLOW_GAMMA     <- c(30.27, 54.69)
PAC_SLOW_GAMMA <- c(30, 50)

ctrl_lmer  <- lmerControl( optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# ── Region scopes & subregion sets ───────────────────────────────────────────
SCOPES <- list(
  BLAMTL = list(
    power_regions = c("BLA", "ALLHPC", "EC", "PRC"),
    coh_pairs     = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    pac_pairs     = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    glmm_ref_power = "BLA", glmm_ref_pair = "BLA_EC",
    needs_allhpc  = TRUE
  ),
  HPCrhinal = list(
    power_regions = c("ALLHPC", "EC", "PRC"),
    coh_pairs     = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
    pac_pairs     = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
    glmm_ref_power = "ALLHPC", glmm_ref_pair = "ALLHPC_EC",
    needs_allhpc  = TRUE
  ),
  HippSubBLA = list(
    power_regions = c("BLA", "CA", "DG", "HPC"),
    coh_pairs     = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    pac_pairs     = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    glmm_ref_power = "BLA", glmm_ref_pair = "BLA_HPC",
    needs_allhpc  = FALSE
  ),
  HippSubRhinal = list(
    # EC_PRC is owned by the HPCrhinal scope; excluded here to avoid
    # double-testing the same pair in two FDR families.
    power_regions = c("CA", "DG", "HPC", "EC", "PRC"),
    coh_pairs     = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"),
    pac_pairs     = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"),
    glmm_ref_power = "EC", glmm_ref_pair = "EC_HPC",
    needs_allhpc  = FALSE
  )
)

HPC_SUB <- c("HPC", "CA", "DG")

# Composition rules for ALLHPC pair labels (in the trial-aggregated dataset).
# Each entry maps a target ALLHPC-* pair label to the source pair labels in
# the raw input that should be averaged together. Direction is whatever the
# upstream PAC computation produced for that alphabetical pair name.
ALLHPC_PAIR_SOURCES <- list(
  BLA_ALLHPC  = c("BLA_HPC", "BLA_CA", "BLA_DG"),
  ALLHPC_EC   = c("EC_HPC",  "CA_EC",  "DG_EC"),
  ALLHPC_PRC  = c("HPC_PRC", "CA_PRC", "DG_PRC")
)

# ── Responder data ───────────────────────────────────────────────────────────
resp <- read.csv(file.path(CSV_IN, "AMMEBLAES_responder_status.csv"),
                 stringsAsFactors = FALSE)
names(resp)[names(resp) == "Responder.status"] <- "ResponderStatus"
resp$Patient[resp$Patient == "BJH033"] <- "BJH032"
resp <- resp[!duplicated(resp$Patient), ]
resp$QuadResponderGroup <- factor(
  resp$ResponderStatus,
  levels = c("Non-responders", "Anti-responders",
             "Moderate responders", "Strong responders")
)
levels(resp$QuadResponderGroup) <- c("NonResp", "AntiResp", "Moderate", "Strong")
resp$mem_mod_z <- as.numeric(scale(resp$avg_stim_dprime_diff))


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

safe_lmer <- function(formula, data, refit_ml = FALSE) {
  tryCatch(
    lmer(formula, data = data, REML = !refit_ml, control = ctrl_lmer),
    error = function(e) { cat("    LMER FAILED:", conditionMessage(e), "\n"); NULL }
  )
}

safe_glmer <- function(formula, data) {
  tryCatch(
    glmer(formula, data = data, family = binomial, control = ctrl_glmer),
    error = function(e) { cat("    GLMER FAILED:", conditionMessage(e), "\n"); NULL }
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & ALLHPC aggregation
# ─────────────────────────────────────────────────────────────────────────────

load_measure_df <- function(measure) {
  fname <- switch(measure,
                  power     = sprintf("combined_%s_power_all_mlmr_input.csv", PHASE),
                  coherence = sprintf("combined_%s_coherence_all_mlmr_input.csv", PHASE),
                  pac       = sprintf("combined_%s_pac_all_mlmr_input.csv", PHASE))
  d <- read.csv(file.path(CSV_IN, fname), stringsAsFactors = FALSE,
                check.names = FALSE)
  d <- d[d$yes_or_no %in% c("yes", "no") & d$trial_type != "new", ]
  d$Patient[d$Patient == "BJH033"] <- "BJH032"
  d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
  d$StimCond <- factor(ifelse(d$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))
  d <- merge(d, resp[, c("Patient", "QuadResponderGroup", "mem_mod_z")],
             by = "Patient", all.x = TRUE)
  d
}

# Build the ALLHPC region rows for a power dataframe.
# For each (Patient, trial_id) where trial_id is the row order within
# (Patient x Region), average the diff_Freq_* columns across HPC/CA/DG rows.
build_allhpc_power <- function(d) {
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  meta_cols <- c("Patient", "Region", "trial_type", "yes_or_no", "Accuracy",
                 "StimCond", "QuadResponderGroup", "mem_mod_z")
  meta_cols <- intersect(meta_cols, names(d))
  use_cols <- c(meta_cols, freq_cols)
  sub <- d[d$Region %in% HPC_SUB, use_cols]
  if (nrow(sub) == 0) return(NULL)

  sub <- sub %>%
    group_by(Patient, Region) %>%
    mutate(trial_id = row_number()) %>%
    ungroup()

  # Aggregate: for each (Patient, trial_id, trial_type, yes_or_no), average
  # numeric freq columns across whatever subregions are present for that
  # patient. trial_type and yes_or_no should be identical across regions for
  # the same trial_id, so they collapse cleanly.
  agg <- sub %>%
    group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy, StimCond,
             QuadResponderGroup, mem_mod_z) %>%
    summarise(across(all_of(freq_cols), ~ mean(.x, na.rm = TRUE)),
              .groups = "drop") %>%
    mutate(Region = "ALLHPC") %>%
    select(-trial_id)
  as.data.frame(agg)
}

# Build ALLHPC pair rows (coherence/PAC). Source pair list comes from
# ALLHPC_PAIR_SOURCES. Returns rows labeled with the new ALLHPC pair name.
build_allhpc_pair <- function(d, target_label, source_pairs) {
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  meta_cols <- c("Patient", "Region", "trial_type", "yes_or_no", "Accuracy",
                 "StimCond", "QuadResponderGroup", "mem_mod_z")
  meta_cols <- intersect(meta_cols, names(d))
  use_cols <- c(meta_cols, freq_cols)
  sub <- d[d$Region %in% source_pairs, use_cols]
  if (nrow(sub) == 0) return(NULL)

  sub <- sub %>%
    group_by(Patient, Region) %>%
    mutate(trial_id = row_number()) %>%
    ungroup()

  agg <- sub %>%
    group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy, StimCond,
             QuadResponderGroup, mem_mod_z) %>%
    summarise(across(all_of(freq_cols), ~ mean(.x, na.rm = TRUE)),
              .groups = "drop") %>%
    mutate(Region = target_label) %>%
    select(-trial_id)
  as.data.frame(agg)
}

# Append ALLHPC region/pair rows to the input dataframe.
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

# Add the band-aggregated feature column(s) to the dataframe.
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-unit LMM models (one per region/pair, no StimCond)
# ─────────────────────────────────────────────────────────────────────────────

run_lmm_unit <- function(sub, feature_col, modeltype, out_dir,
                         modality, unit, band_label) {
  sub <- sub[is.finite(sub[[feature_col]]) & is.finite(sub$mem_mod_z) &
             !is.na(sub$QuadResponderGroup), ]
  n_pat <- length(unique(sub$Patient))
  if (nrow(sub) < MIN_TRIALS || n_pat < MIN_PATIENTS) {
    cat(sprintf("    SKIP %s/%s/%s (n=%d trials, %d patients)\n",
                modality, unit, band_label, nrow(sub), n_pat))
    return(invisible(NULL))
  }
  sub$Patient <- factor(sub$Patient)
  sub$feature <- as.numeric(sub[[feature_col]])

  if (modeltype == "LMMcont") {
    fmls <- list(
      m0 = "feature ~ 1 + (1 | Patient)",
      m1 = "feature ~ mem_mod_z + (1 | Patient)"
    )
  } else if (modeltype == "LMMquad") {
    fmls <- list(
      m0 = "feature ~ 1 + (1 | Patient)",
      m1 = "feature ~ QuadResponderGroup + (1 | Patient)"
    )
  } else {
    stop("Unknown LMM modeltype: ", modeltype)
  }
  fits_ml <- lapply(fmls, function(f) safe_lmer(as.formula(f), sub, refit_ml = TRUE))
  full_reml <- safe_lmer(as.formula(fmls$m1), sub, refit_ml = FALSE)
  if (!is.null(full_reml)) {
    save_coefs(full_reml,
               file.path(out_dir,
                         sprintf("%s_%s_%s_coefs.csv", modality, unit, band_label)))
  }
  save_anova_build(fits_ml, fmls,
                   file.path(out_dir,
                             sprintf("%s_%s_%s_anova.csv", modality, unit, band_label)))
}


# ─────────────────────────────────────────────────────────────────────────────
# GLMM (one model per region per band)
#   Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)
# ─────────────────────────────────────────────────────────────────────────────

run_glmm_unit <- function(sub, out_dir, modality, unit, band_label) {
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
    m1     = "Accuracy ~ band_c + (1 | Patient)",
    m2     = "Accuracy ~ band_c + StimCond + (1 | Patient)",
    m_full = "Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)"
  )

  # First attempt: uncentered band. Input data is baseline-corrected, so 0
  # and direction are interpretable and we want to preserve that scale.
  sub$band_c <- feat
  fits <- lapply(fmls, function(f) safe_glmer(as.formula(f), sub))
  centered_used <- FALSE

  # Fallback: if any model failed to fit (returned NULL = a true convergence
  # error, not just a warning), retry with grand-mean centering for numerical
  # stability.
  if (any(sapply(fits, is.null))) {
    cat(sprintf("    GLMM convergence fallback to centered for %s/%s/%s\n",
                modality, unit, band_label))
    sub$band_c <- feat - mean(feat, na.rm = TRUE)
    fits <- lapply(fmls, function(f) safe_glmer(as.formula(f), sub))
    centered_used <- TRUE
  }

  if (!is.null(fits$m_full)) {
    save_coefs(fits$m_full,
               file.path(out_dir,
                         sprintf("%s_%s_%s_coefs.csv",
                                 modality, unit, band_label)),
               exponentiate = TRUE)
  }
  save_anova_build(fits, fmls,
                   file.path(out_dir,
                             sprintf("%s_%s_%s_anova.csv",
                                     modality, unit, band_label)))

  # Append per-panel centering metadata to a single log file in OUT_BASE.
  log_path <- file.path(OUT_BASE, "_glmm_centering_log.csv")
  log_row <- data.frame(
    phase = PHASE, modality = modality, unit = unit,
    band = band_label, centered = centered_used,
    stringsAsFactors = FALSE
  )
  write.table(log_row, log_path, sep = ",", append = file.exists(log_path),
              col.names = !file.exists(log_path), row.names = FALSE)
}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

run_combo <- function(measure, scope_name, modeltype) {
  scope <- SCOPES[[scope_name]]
  out_dir <- ensure_dir(file.path(OUT_BASE,
                                  sprintf("%s_%s_%s",
                                          measure, scope_name, modeltype)))
  cat(sprintf("\n=== %s | %s | %s ===\n", measure, scope_name, modeltype))

  d <- load_measure_df(measure)
  if (scope$needs_allhpc) d <- prep_with_allhpc(d, measure)
  d <- add_band_columns(d, measure)

  if (measure == "power") {
    units <- scope$power_regions
  } else if (measure == "coherence") {
    units <- scope$coh_pairs
  } else {
    units <- scope$pac_pairs
  }
  bands <- if (measure == "pac") c("slow_gamma") else c("theta", "slow_gamma")

  if (modeltype %in% c("LMMcont", "LMMquad")) {
    for (u in units) {
      sub <- d[d$Region == u, ]
      if (nrow(sub) == 0) {
        cat(sprintf("    NO ROWS for %s\n", u)); next
      }
      for (b in bands) run_lmm_unit(sub, b, modeltype, out_dir,
                                    measure, u, b)
    }
  } else if (modeltype == "GLMM") {
    for (u in units) {
      sub <- d[d$Region == u, ]
      if (nrow(sub) == 0) {
        cat(sprintf("    NO ROWS for %s\n", u)); next
      }
      for (b in bands) run_glmm_unit(sub, out_dir, measure, u, b)
    }
  } else {
    stop("Unknown modeltype: ", modeltype)
  }
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

MEASURES <- c("power", "coherence", "pac")
SCOPE_NAMES <- names(SCOPES)
MODELTYPES <- c("LMMcont", "LMMquad", "GLMM")

cat(sprintf("\n>>> Phase: %s   Output: %s\n", PHASE, OUT_BASE))

if (length(.cmd_args) >= 3) {
  run_combo(.cmd_args[1], .cmd_args[2], .cmd_args[3])
} else {
  for (m in MEASURES) {
    for (s in SCOPE_NAMES) {
      for (mt in MODELTYPES) {
        run_combo(m, s, mt)
      }
    }
  }
}

cat(sprintf("\n===== FULL H1ABC %s PIPELINE COMPLETE =====\n",
            toupper(PHASE)))
