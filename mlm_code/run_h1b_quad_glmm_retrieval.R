##############################################################################
# Hypothesis 1b - QUAD RESPONDER CATEGORY - GLMM (3-way)
#   Same 3-way GLMM design as run_h1b_glmm_retrieval.R, but with the original
#   four responder labels: Anti / Non (ref) / Moderate / Strong.
#
# Per region (power) or pair (coherence/PAC), per band:
#   Accuracy ~ StimCond * feature_c * QuadResponderGroup + (1 | Patient)
#
# Continuous moderator (z-scored avg_stim_dprime_diff) also fit per panel.
# Bands: theta, slow_gamma. PAC: SG PAC only.
# BJH032 == BJH033 -> folded together (54 subjects).
#
# Usage: Rscript run_h1b_quad_glmm_retrieval.R --imbalanced
##############################################################################

suppressPackageStartupMessages({
  if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
  if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
  if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")
  library(lme4); library(lmerTest); library(broom.mixed)
})

args <- commandArgs(trailingOnly = TRUE)
balanced <- "--balanced" %in% args
MIN_TRIALS <- 10
MIN_PATIENTS <- 5

script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    a <- commandArgs(trailingOnly = FALSE)
    f <- grep("--file=", a, value = TRUE)
    if (length(f) > 0) dirname(normalizePath(sub("--file=", "", f))) else getwd()
  }
)

label <- if (balanced) "balanced" else "imbalanced"
out_dir <- file.path(script_dir, "outputs",
                    paste0("h1b_quad_glmm_", label, "_retrieval_mlm"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)
PAC_SLOW_GAMMA <- c(30, 50)

BANDS <- list(
  theta      = list(range = THETA,      label = "Theta"),
  slow_gamma = list(range = SLOW_GAMMA, label = "Slow Gamma")
)

POWER_REGIONS <- c("BLA", "HPC", "EC", "PRC")
COH_PAIRS <- c("BLA_HPC", "BLA_EC", "BLA_PRC", "EC_HPC", "EC_PRC", "HPC_PRC")
PAC_PAIRS <- c("BLA_HPC", "BLA_EC", "BLA_PRC")

resp <- read.csv(file.path(script_dir, "outputs", "csvs",
                           "AMMEBLAES_responder_status.csv"),
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

cat(sprintf("Quad responder coverage (n=%d): %s\n", nrow(resp),
            paste(sprintf("%s=%d", levels(resp$QuadResponderGroup),
                          tabulate(resp$QuadResponderGroup)),
                  collapse=", ")))

save_coefs <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  cat(sprintf("  -> Saved %s\n", filename))
  td
}

save_model_build <- function(models, formulas, filename) {
  rows <- list(); prev <- NULL
  for (nm in names(models)) {
    m <- models[[nm]]; if (is.null(m)) next
    npar <- attr(logLik(m), "df"); ll <- as.numeric(logLik(m))
    chisq <- NA_real_; df_v <- NA_real_; pv <- NA_real_
    if (!is.null(prev)) {
      cmp <- tryCatch(anova(prev, m, refit = FALSE), error = function(e) NULL)
      if (is.null(cmp)) cmp <- tryCatch(anova(prev, m), error = function(e) NULL)
      pick <- function(col) { v <- cmp[[col]]; if (is.null(v) || length(v) < 2) NA_real_ else as.numeric(v[2]) }
      if (!is.null(cmp)) { chisq <- pick("Chisq"); df_v <- pick("Chi Df")
        if (is.na(df_v)) df_v <- pick("Df"); pv <- pick("Pr(>Chisq)") }
    }
    rows[[length(rows)+1]] <- data.frame(Model=nm, Formula=formulas[[nm]],
      npar=npar, AIC=AIC(m), BIC=BIC(m), logLik=ll, deviance=-2*ll,
      Chisq=chisq, Df=df_v, p.value=pv, stringsAsFactors=FALSE)
    prev <- m
  }
  if (length(rows) == 0) return(invisible(NULL))
  write.csv(do.call(rbind, rows), file.path(out_dir, filename), row.names = FALSE)
  cat(sprintf("  -> Saved %s\n", filename))
}

safe_glmer <- function(formula, data) tryCatch(
  glmer(formula, data = data, family = binomial, control = ctrl),
  error = function(e) { cat("    FAILED:", conditionMessage(e), "\n"); NULL })

filter_balanced <- function(df, lab) {
  if (!balanced) {
    cat(sprintf("\n  [%s] No filter (imbalanced): %d patients\n",
                lab, length(unique(df$Patient))))
    return(df)
  }
  patients <- unique(df$Patient)
  n_regions <- sapply(patients, function(p) length(unique(df$Region[df$Patient==p])))
  cnt2 <- data.frame(Patient=patients,
                     n_rem=sapply(patients, function(p) sum(df$Patient==p & df$yes_or_no=="yes")),
                     n_forg=sapply(patients, function(p) sum(df$Patient==p & df$yes_or_no=="no")),
                     n_regions=n_regions)
  cnt2$rem_per_reg <- round(cnt2$n_rem / cnt2$n_regions)
  cnt2$forg_per_reg <- round(cnt2$n_forg / cnt2$n_regions)
  inc <- cnt2$Patient[cnt2$rem_per_reg >= MIN_TRIALS & cnt2$forg_per_reg >= MIN_TRIALS]
  cat(sprintf("\n  [%s] Balanced filter: %d / %d patients\n", lab, length(inc), nrow(cnt2)))
  df[df$Patient %in% inc, ]
}

prepare_trial_df <- function(csv_path, lab) {
  d <- read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)
  d <- d[d$yes_or_no %in% c("yes","no") & d$trial_type != "new", ]
  d <- filter_balanced(d, lab)
  d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
  d$StimCond <- factor(ifelse(d$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))
  d <- merge(d, resp[, c("Patient", "QuadResponderGroup", "mem_mod_z")],
             by = "Patient", all.x = TRUE)
  d
}

add_band_columns <- function(d) {
  freq_cols <- sort(grep("^diff_Freq_", names(d), value = TRUE))
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
  for (b in names(BANDS)) {
    r <- BANDS[[b]]$range
    cols <- freq_cols[freqs >= r[1] & freqs <= r[2]]
    d[[b]] <- rowMeans(d[, cols, drop=FALSE], na.rm=TRUE)
  }
  d
}

run_unit <- function(sub, feature_col, modality, unit, band_label) {
  sub <- sub[is.finite(sub[[feature_col]]) & is.finite(sub$mem_mod_z) &
             !is.na(sub$QuadResponderGroup), ]
  if (nrow(sub) < 30 || length(unique(sub$Patient)) < MIN_PATIENTS) {
    cat(sprintf("  Skipping %s %s %s: %d trials, %d patients\n",
                modality, unit, band_label, nrow(sub),
                length(unique(sub$Patient))))
    return(invisible(NULL))
  }
  sub$Patient <- factor(sub$Patient)
  sub$feature_c <- as.numeric(sub[[feature_col]] - mean(sub[[feature_col]], na.rm=TRUE))

  fmls_cont <- list(
    m0     = "Accuracy ~ 1 + (1 | Patient)",
    m1     = "Accuracy ~ StimCond + feature_c + mem_mod_z + (1 | Patient)",
    m2     = "Accuracy ~ StimCond * feature_c + mem_mod_z + StimCond:mem_mod_z + feature_c:mem_mod_z + (1 | Patient)",
    m_full = "Accuracy ~ StimCond * feature_c * mem_mod_z + (1 | Patient)"
  )
  fits <- list()
  for (nm in names(fmls_cont)) fits[[nm]] <- safe_glmer(as.formula(fmls_cont[[nm]]), sub)
  if (!is.null(fits$m_full)) {
    save_coefs(fits$m_full, sprintf("%s_%s_%s_modContinuous_coefs.csv",
                                     modality, unit, band_label))
    save_model_build(fits, fmls_cont, sprintf("%s_%s_%s_modContinuous_anova.csv",
                                              modality, unit, band_label))
  }

  fmls_cat <- list(
    m0     = "Accuracy ~ 1 + (1 | Patient)",
    m1     = "Accuracy ~ StimCond + feature_c + QuadResponderGroup + (1 | Patient)",
    m2     = "Accuracy ~ StimCond * feature_c + QuadResponderGroup + StimCond:QuadResponderGroup + feature_c:QuadResponderGroup + (1 | Patient)",
    m_full = "Accuracy ~ StimCond * feature_c * QuadResponderGroup + (1 | Patient)"
  )
  fits2 <- list()
  for (nm in names(fmls_cat)) fits2[[nm]] <- safe_glmer(as.formula(fmls_cat[[nm]]), sub)
  if (!is.null(fits2$m_full)) {
    save_coefs(fits2$m_full, sprintf("%s_%s_%s_modQuadCategorical_coefs.csv",
                                      modality, unit, band_label))
    save_model_build(fits2, fmls_cat, sprintf("%s_%s_%s_modQuadCategorical_anova.csv",
                                              modality, unit, band_label))
  }
}

cat("\n", paste(rep("=",70),collapse=""), "\nH1B QUAD GLMM POWER -", toupper(label), "\n",
    paste(rep("=",70),collapse=""), "\n", sep="")
pw <- prepare_trial_df(file.path(script_dir, "outputs", "csvs",
                                 "combined_retrieval_power_all_mlmr_input.csv"), "Power")
pw <- add_band_columns(pw)
for (reg in POWER_REGIONS) {
  sub <- pw[pw$Region == reg, ]
  cat(sprintf("\n--- Power %s: %d trials, %d patients ---\n",
              reg, nrow(sub), length(unique(sub$Patient))))
  for (b in names(BANDS)) run_unit(sub, b, "power", reg, b)
}

cat("\n", paste(rep("=",70),collapse=""), "\nH1B QUAD GLMM COHERENCE -", toupper(label), "\n",
    paste(rep("=",70),collapse=""), "\n", sep="")
coh <- prepare_trial_df(file.path(script_dir, "outputs", "csvs",
                                  "combined_retrieval_coherence_all_mlmr_input.csv"), "Coherence")
coh <- add_band_columns(coh)
for (pair in COH_PAIRS) {
  sub <- coh[coh$Region == pair, ]
  cat(sprintf("\n--- Coherence %s: %d trials, %d patients ---\n",
              pair, nrow(sub), length(unique(sub$Patient))))
  for (b in names(BANDS)) run_unit(sub, b, "coherence", pair, b)
}

cat("\n", paste(rep("=",70),collapse=""), "\nH1B QUAD GLMM PAC (SG) -", toupper(label), "\n",
    paste(rep("=",70),collapse=""), "\n", sep="")
pac <- prepare_trial_df(file.path(script_dir, "outputs", "csvs",
                                  "combined_retrieval_pac_all_mlmr_input.csv"), "PAC")
freq_cols <- sort(grep("^diff_Freq_", names(pac), value = TRUE))
freqs_p <- as.numeric(sub("diff_Freq_", "", freq_cols))
sg_cols <- freq_cols[freqs_p >= PAC_SLOW_GAMMA[1] & freqs_p <= PAC_SLOW_GAMMA[2]]
pac$slow_gamma_pac <- rowMeans(pac[, sg_cols, drop=FALSE], na.rm=TRUE)
for (pair in PAC_PAIRS) {
  sub <- pac[pac$Region == pair, ]
  cat(sprintf("\n--- PAC %s: %d trials, %d patients ---\n",
              pair, nrow(sub), length(unique(sub$Patient))))
  run_unit(sub, "slow_gamma_pac", "pac", pair, "slow_gamma")
}

cat(sprintf("\n\n===== H1B QUAD GLMM RETRIEVAL (%s) COMPLETE =====\n", toupper(label)))
