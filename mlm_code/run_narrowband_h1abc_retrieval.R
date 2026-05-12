##############################################################################
# Narrow-band retrieval LMM/GLMM pipeline.
#
# Mirrors run_h1abc_full_retrieval.R but replaces the fixed THETA/SLOW_GAMMA
# freq ranges with PER-REGION narrow bands derived from the significant
# permutation clusters (see permutation/derive_narrowbands_for_retrieval_lmm.py).
#
# Models fit per (measure x scope x modeltype) combo:
#   LMMcont : feature ~ mem_mod_z + (1|Patient)
#   LMMquad : feature ~ QuadResponderGroup + (1|Patient)
#   GLMM    : Accuracy ~ band_c + StimCond + band_c:StimCond + (1|Patient)
# where 'feature' / 'band_c' is the per-region narrow-band-averaged value.
#
# Only fits regions/pairs that have a narrow-band entry in narrow_bands.csv.
#
# Output: OUTPUTS/retrieval memory reports permutation test based/stats/
#         narrowband_h1abc_retrieval_mlm/<measure>_<scope>_<modeltype>/...
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
  if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
  if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")
  if (!require("dplyr"))       install.packages("dplyr",        repos = "https://cloud.r-project.org")
  library(lme4); library(lmerTest); library(broom.mixed); library(dplyr)
})

REPO_ROOT <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_IN    <- file.path(REPO_ROOT, "OUTPUTS", "csvs")
NB_CSV    <- file.path(REPO_ROOT, "OUTPUTS",
                       "retrieval memory reports permutation test based",
                       "narrow_bands.csv")
OUT_BASE  <- file.path(REPO_ROOT, "OUTPUTS",
                       "retrieval memory reports permutation test based",
                       "stats", "narrowband_h1abc_retrieval_mlm")
dir.create(OUT_BASE, showWarnings = FALSE, recursive = TRUE)

PHASE <- "retrieval"
MIN_PATIENTS <- 4
MIN_TRIALS   <- 30

ctrl_lmer  <- lmerControl( optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# Region scopes (mirror of run_h1abc_full_retrieval.R), with EC_PRC added
# to HippSubRhinal so the only sig cluster outside the original scopes
# still gets reported.
SCOPES <- list(
  BLAMTL = list(
    power_regions = c("BLA", "ALLHPC", "EC", "PRC"),
    coh_pairs     = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    pac_pairs     = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    needs_allhpc  = TRUE
  ),
  HPCrhinal = list(
    power_regions = c("ALLHPC", "EC", "PRC"),
    coh_pairs     = c("ALLHPC_EC", "ALLHPC_PRC"),
    pac_pairs     = c("ALLHPC_EC", "ALLHPC_PRC"),
    needs_allhpc  = TRUE
  ),
  HippSubBLA = list(
    power_regions = c("BLA", "CA", "DG", "HPC"),
    coh_pairs     = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    pac_pairs     = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    needs_allhpc  = FALSE
  ),
  HippSubRhinal = list(
    power_regions = c("CA", "DG", "HPC", "EC", "PRC"),
    coh_pairs     = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC",
                      "HPC_PRC", "EC_PRC"),  # EC_PRC added for narrow-band run
    pac_pairs     = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC",
                      "HPC_PRC", "EC_PRC"),
    needs_allhpc  = FALSE
  )
)

HPC_SUB <- c("HPC", "CA", "DG")
ALLHPC_PAIR_SOURCES <- list(
  BLA_ALLHPC = c("BLA_HPC", "BLA_CA", "BLA_DG"),
  ALLHPC_EC  = c("EC_HPC",  "CA_EC",  "DG_EC"),
  ALLHPC_PRC = c("HPC_PRC", "CA_PRC", "DG_PRC")
)

# Responder data (same as h1abc)
resp <- read.csv(file.path(CSV_IN, "AMMEBLAES_responder_status.csv"),
                 stringsAsFactors = FALSE)
names(resp)[names(resp) == "Responder.status"] <- "ResponderStatus"
resp$Patient[resp$Patient == "BJH033"] <- "BJH032"
resp <- resp[!duplicated(resp$Patient), ]
resp$QuadResponderGroup <- factor(
  resp$ResponderStatus,
  levels = c("Non-responders", "Anti-responders",
             "Moderate responders", "Strong responders"))
levels(resp$QuadResponderGroup) <- c("NonResp", "AntiResp", "Moderate", "Strong")
resp$mem_mod_z <- as.numeric(scale(resp$avg_stim_dprime_diff))

# Narrow band table (long form: modality, region, family, narrow_lo_hz, narrow_hi_hz)
nb <- read.csv(NB_CSV, stringsAsFactors = FALSE)

normalize_region_label <- function(region) {
  if (is.na(region)) return(NA_character_)
  parts <- strsplit(trimws(region), "_", fixed = TRUE)[[1]]
  parts <- trimws(parts); parts <- parts[nzchar(parts)]
  if (length(parts) == 0) return(NA_character_)
  parts[parts == "ER"] <- "EC"
  if (length(parts) == 2) parts <- sort(parts)
  paste(parts, collapse = "_")
}

ensure_dir <- function(p) { dir.create(p, showWarnings = FALSE, recursive = TRUE); p }
save_csv <- function(df, path) {
  ensure_dir(dirname(path)); write.csv(df, path, row.names = FALSE)
  cat(sprintf("    -> %s\n", basename(path)))
}
save_coefs <- function(model, path, exponentiate = FALSE) {
  td <- tryCatch(
    tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = exponentiate),
    error = function(e) NULL
  )
  if (is.null(td)) return(invisible(NULL))
  save_csv(td, path); td
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
        pv <- pick("Pr(>Chisq)")
      }
    }
    rows[[length(rows) + 1]] <- data.frame(
      Model = nm, Formula = fmls[[nm]], npar = npar,
      AIC = AIC(m), BIC = BIC(m), logLik = ll, deviance = -2 * ll,
      Chisq = chisq, Df = df_v, p.value = pv, stringsAsFactors = FALSE)
    prev <- m
  }
  if (length(rows) == 0) return(invisible(NULL))
  save_csv(do.call(rbind, rows), path)
}
safe_lmer <- function(formula, data, refit_ml = FALSE) {
  tryCatch(lmer(formula, data = data, REML = !refit_ml, control = ctrl_lmer),
           error = function(e) { cat("    LMER FAILED:", conditionMessage(e), "\n"); NULL })
}
safe_glmer <- function(formula, data) {
  tryCatch(glmer(formula, data = data, family = binomial, control = ctrl_glmer),
           error = function(e) { cat("    GLMER FAILED:", conditionMessage(e), "\n"); NULL })
}

load_measure_df <- function(measure) {
  fname <- switch(measure,
                  power     = "combined_retrieval_power_all_mlmr_input.csv",
                  coherence = "combined_retrieval_coherence_all_mlmr_input.csv",
                  pac       = "combined_retrieval_pac_all_mlmr_input.csv")
  d <- read.csv(file.path(CSV_IN, fname), stringsAsFactors = FALSE,
                check.names = FALSE)
  d$Region <- vapply(d$Region, normalize_region_label, character(1), USE.NAMES = FALSE)
  d <- d[d$yes_or_no %in% c("yes", "no") & d$trial_type != "new", ]
  d$Patient[d$Patient == "BJH033"] <- "BJH032"
  d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
  d$StimCond <- factor(ifelse(d$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))
  d <- merge(d, resp[, c("Patient", "QuadResponderGroup", "mem_mod_z")],
             by = "Patient", all.x = TRUE)
  d
}

build_allhpc_power <- function(d) {
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  meta_cols <- intersect(c("Patient", "Region", "trial_type", "yes_or_no",
                           "Accuracy", "StimCond", "QuadResponderGroup",
                           "mem_mod_z"), names(d))
  sub <- d[d$Region %in% HPC_SUB, c(meta_cols, freq_cols)]
  if (nrow(sub) == 0) return(NULL)
  sub <- sub %>% group_by(Patient, Region) %>%
    mutate(trial_id = row_number()) %>% ungroup()
  agg <- sub %>%
    group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy, StimCond,
             QuadResponderGroup, mem_mod_z) %>%
    summarise(across(all_of(freq_cols), ~ mean(.x, na.rm = TRUE)),
              .groups = "drop") %>%
    mutate(Region = "ALLHPC") %>% select(-trial_id)
  as.data.frame(agg)
}

build_allhpc_pair <- function(d, target_label, source_pairs) {
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  meta_cols <- intersect(c("Patient", "Region", "trial_type", "yes_or_no",
                           "Accuracy", "StimCond", "QuadResponderGroup",
                           "mem_mod_z"), names(d))
  sub <- d[d$Region %in% source_pairs, c(meta_cols, freq_cols)]
  if (nrow(sub) == 0) return(NULL)
  sub <- sub %>% group_by(Patient, Region) %>%
    mutate(trial_id = row_number()) %>% ungroup()
  agg <- sub %>%
    group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy, StimCond,
             QuadResponderGroup, mem_mod_z) %>%
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

# Add a band column using a per-region narrow band freq range. Returns d
# with new column `feature` populated for rows in this region; NA elsewhere.
add_narrow_band_column <- function(d, region, lo, hi) {
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
  in_band <- freq_cols[freqs >= lo & freqs <= hi]
  if (length(in_band) == 0) {
    cat(sprintf("    no freq cols in %.2f-%.2f Hz; skipping\n", lo, hi))
    d$feature <- NA_real_
    return(d)
  }
  d$feature <- NA_real_
  mask <- d$Region == region
  d$feature[mask] <- rowMeans(d[mask, in_band, drop = FALSE], na.rm = TRUE)
  d
}

run_lmm_unit <- function(sub, modeltype, out_dir, modality, unit, band_label) {
  sub <- sub[is.finite(sub$feature), ]
  if (modeltype == "LMMcont") {
    sub <- sub[is.finite(sub$mem_mod_z), ]
  } else if (modeltype == "LMMquad") {
    sub <- sub[!is.na(sub$QuadResponderGroup), ]
  }
  n_pat <- length(unique(sub$Patient))
  if (nrow(sub) < MIN_TRIALS || n_pat < MIN_PATIENTS) {
    cat(sprintf("    SKIP %s/%s/%s (n=%d trials, %d patients)\n",
                modality, unit, band_label, nrow(sub), n_pat))
    return(invisible(NULL))
  }
  sub$Patient <- factor(sub$Patient)
  fmls <- if (modeltype == "LMMcont") {
    list(m0 = "feature ~ 1 + (1 | Patient)",
         m1 = "feature ~ mem_mod_z + (1 | Patient)")
  } else {
    list(m0 = "feature ~ 1 + (1 | Patient)",
         m1 = "feature ~ QuadResponderGroup + (1 | Patient)")
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

run_glmm_unit <- function(sub, out_dir, modality, unit, band_label) {
  sub <- sub[is.finite(sub$feature), ]
  n_pat <- length(unique(sub$Patient))
  if (nrow(sub) < MIN_TRIALS || n_pat < MIN_PATIENTS) {
    cat(sprintf("    GLMM SKIP %s/%s/%s (n=%d trials, %d patients)\n",
                modality, unit, band_label, nrow(sub), n_pat))
    return(invisible(NULL))
  }
  sub$Patient <- factor(sub$Patient)
  fmls <- list(
    m0     = "Accuracy ~ 1 + (1 | Patient)",
    m1     = "Accuracy ~ band_c + (1 | Patient)",
    m2     = "Accuracy ~ band_c + StimCond + (1 | Patient)",
    m_full = "Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)"
  )
  sub$band_c <- sub$feature
  fits <- lapply(fmls, function(f) safe_glmer(as.formula(f), sub))
  if (any(sapply(fits, is.null))) {
    sub$band_c <- sub$feature - mean(sub$feature, na.rm = TRUE)
    fits <- lapply(fmls, function(f) safe_glmer(as.formula(f), sub))
  }
  if (!is.null(fits$m_full)) {
    save_coefs(fits$m_full,
               file.path(out_dir,
                         sprintf("%s_%s_%s_coefs.csv", modality, unit, band_label)),
               exponentiate = TRUE)
  }
  save_anova_build(fits, fmls,
                   file.path(out_dir,
                             sprintf("%s_%s_%s_anova.csv", modality, unit, band_label)))
}

run_combo <- function(measure, scope, modeltype) {
  sc <- SCOPES[[scope]]
  units <- if (measure == "power") sc$power_regions
           else if (measure == "coherence") sc$coh_pairs
           else sc$pac_pairs
  if (sc$needs_allhpc && !any(grepl("ALLHPC", units))) return(invisible())

  out_dir <- file.path(OUT_BASE, sprintf("%s_%s_%s", measure, scope, modeltype))
  ensure_dir(out_dir)
  cat(sprintf("\n=== %s | %s | %s ===\n", measure, scope, modeltype))

  # Per-modality narrow bands (median across sig clusters, applied to ALL
  # regions in this scope -- not just regions with their own sig cluster).
  nb_mod <- nb[nb$modality == measure, ]
  if (nrow(nb_mod) == 0) {
    cat("    no narrow bands defined for this modality; skipping\n")
    return(invisible())
  }

  # Persist what bands are being used (and which regions they get applied to).
  expanded <- do.call(rbind, lapply(units, function(u) {
    df <- nb_mod
    df$region <- u
    df
  }))
  save_csv(expanded, file.path(out_dir, "_narrow_bands_used.csv"))

  d <- load_measure_df(measure)
  d <- prep_with_allhpc(d, measure)

  for (region in units) {
    for (i in seq_len(nrow(nb_mod))) {
      family <- nb_mod$family[i]
      lo <- nb_mod$narrow_lo_hz[i]
      hi <- nb_mod$narrow_hi_hz[i]
      band_label <- sprintf("narrow_%s", family)
      cat(sprintf("  -> %s | %s [%.2f-%.2f Hz]\n",
                  region, band_label, lo, hi))

      d2 <- add_narrow_band_column(d, region, lo, hi)
      sub <- d2[d2$Region == region & is.finite(d2$feature), ]
      if (nrow(sub) == 0) {
        cat("    no rows in region after band column add; skipping\n")
        next
      }

      if (modeltype %in% c("LMMcont", "LMMquad")) {
        run_lmm_unit(sub, modeltype, out_dir, measure, region, band_label)
      } else if (modeltype == "GLMM") {
        run_glmm_unit(sub, out_dir, measure, region, band_label)
      }
    }
  }
}


# Main: iterate all combos.
.cmd_args <- commandArgs(trailingOnly = TRUE)
COMBOS <- if (length(.cmd_args) >= 3) {
  list(list(measure = .cmd_args[1], scope = .cmd_args[2], modeltype = .cmd_args[3]))
} else {
  out <- list()
  for (mz in c("LMMcont", "LMMquad", "GLMM")) {
    for (sc in names(SCOPES)) {
      for (m in c("power", "coherence", "pac")) {
        out[[length(out) + 1]] <- list(measure = m, scope = sc, modeltype = mz)
      }
    }
  }
  out
}

for (combo in COMBOS) {
  run_combo(combo$measure, combo$scope, combo$modeltype)
}
cat("\n===== ALL DONE =====\n")
