##############################################################################
# Hypothesis 1c Retrieval MLM:
#   Coherence and PAC ONLY, restricted to NON-BLA region pairs:
#   CA, DG, HPC, EC, PRC -> 10 pairs (CA-DG, CA-EC, CA-HPC, CA-PRC,
#   DG-EC, DG-HPC, DG-PRC, EC-HPC, EC-PRC, HPC-PRC).
#
# Mirrors the v2 report's model structure:
#   - Across-all-pairs Pair x StimCond per band/PAC type, with predictor
#     entry order: band_c (or pac_z) -> StimCond -> Region -> Region:StimCond.
#   - Per-pair multi-band model (theta + slow gamma) for coherence.
#   - Per-pair two-PAC-type model (SG PAC + HFA PAC) for PAC.
#
# Per-pair tests require >= MIN_PATIENTS (5) patients.
#
# Usage: Rscript run_h1c_retrieval.R --imbalanced
#        Rscript run_h1c_retrieval.R --balanced
##############################################################################

if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")

library(lme4)
library(lmerTest)
library(broom.mixed)

# ── Parse args ──
args <- commandArgs(trailingOnly = TRUE)
balanced <- "--balanced" %in% args
MIN_TRIALS <- 10
MIN_PATIENTS <- 5

script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    a <- commandArgs(trailingOnly = FALSE)
    f <- grep("--file=", a, value = TRUE)
    if (length(f) > 0) dirname(normalizePath(sub("--file=", "", f)))
    else getwd()
  }
)

label <- if (balanced) "balanced" else "imbalanced"
out_dir <- file.path(script_dir, "outputs", paste0("h1c_", label, "_retrieval_mlm"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)
PAC_SLOW_GAMMA <- c(30, 50)
PAC_HFA <- c(70, 100)

BANDS <- list(
  theta      = list(range = THETA,      label = "Theta"),
  slow_gamma = list(range = SLOW_GAMMA, label = "Slow Gamma")
)

# Non-BLA pairs (CA, DG, HPC, EC, PRC).
NONBLA_PAIRS <- c(
  "CA_DG", "CA_EC", "CA_HPC", "CA_PRC",
  "DG_EC", "DG_HPC", "DG_PRC",
  "EC_HPC", "EC_PRC",
  "HPC_PRC"
)

save_coefs <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  cat(sprintf("  -> Saved %s\n", filename))
  return(td)
}

save_model_build <- function(models, formulas, filename) {
  ord <- names(models)
  rows <- list()
  prev <- NULL
  for (nm in ord) {
    m <- models[[nm]]
    if (is.null(m)) next
    npar <- attr(logLik(m), "df")
    ll <- as.numeric(logLik(m))
    aic_v <- AIC(m); bic_v <- BIC(m)
    chisq <- NA_real_; df_v <- NA_real_; pv <- NA_real_
    if (!is.null(prev)) {
      cmp <- tryCatch(anova(prev, m, refit = FALSE), error = function(e) NULL)
      if (is.null(cmp)) cmp <- tryCatch(anova(prev, m), error = function(e) NULL)
      pick <- function(col) {
        v <- cmp[[col]]
        if (is.null(v) || length(v) < 2) return(NA_real_)
        as.numeric(v[2])
      }
      if (!is.null(cmp)) {
        chisq <- pick("Chisq")
        df_v  <- pick("Chi Df")
        if (is.na(df_v)) df_v <- pick("Df")
        pv    <- pick("Pr(>Chisq)")
      }
    }
    rows[[length(rows) + 1]] <- data.frame(
      Model = nm, Formula = formulas[[nm]], npar = npar,
      AIC = aic_v, BIC = bic_v, logLik = ll, deviance = -2 * ll,
      Chisq = chisq, Df = df_v, p.value = pv,
      stringsAsFactors = FALSE
    )
    prev <- m
  }
  if (length(rows) == 0) return(invisible(NULL))
  out <- do.call(rbind, rows)
  write.csv(out, file.path(out_dir, filename), row.names = FALSE)
  cat(sprintf("  -> Saved %s\n", filename))
}

safe_glmer <- function(formula, data) {
  tryCatch(
    glmer(formula, data = data, family = binomial, control = ctrl),
    error = function(e) { cat("    FAILED:", conditionMessage(e), "\n"); NULL }
  )
}

filter_balanced <- function(df, measure_label) {
  if (!balanced) {
    cat(sprintf("\n  [%s] No filter (imbalanced): %d patients\n",
                measure_label, length(unique(df$Patient))))
    return(df)
  }
  patients <- unique(df$Patient)
  counts <- data.frame(Patient = patients, stringsAsFactors = FALSE)
  counts$n_regions <- sapply(patients, function(p) length(unique(df$Region[df$Patient == p])))
  counts$n_rem <- sapply(patients, function(p) sum(df$Patient == p & df$yes_or_no == "yes"))
  counts$n_forg <- sapply(patients, function(p) sum(df$Patient == p & df$yes_or_no == "no"))
  counts$n_rem_unique <- round(counts$n_rem / counts$n_regions)
  counts$n_forg_unique <- round(counts$n_forg / counts$n_regions)
  included <- counts$Patient[counts$n_rem_unique >= MIN_TRIALS & counts$n_forg_unique >= MIN_TRIALS]
  cat(sprintf("\n  [%s] Balanced filter (>=%d per condition): %d / %d patients kept\n",
              measure_label, MIN_TRIALS, length(included), nrow(counts)))
  df[df$Patient %in% included, ]
}

##############################################################################
# COHERENCE
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("H1C COHERENCE (non-BLA pairs) - %s\n", toupper(label)))
cat(paste(rep("=", 70), collapse=""), "\n")

coh_csv <- file.path(script_dir, "outputs", "csvs",
                    "combined_retrieval_coherence_all_mlmr_input.csv")
coh <- read.csv(coh_csv, stringsAsFactors = FALSE, check.names = FALSE)
coh <- coh[coh$yes_or_no %in% c("yes", "no") & coh$trial_type != "new", ]
coh <- filter_balanced(coh, "Coherence")

freq_cols_c <- sort(grep("^diff_Freq_", names(coh), value = TRUE))
freqs_c <- as.numeric(sub("diff_Freq_", "", freq_cols_c))

for (b in names(BANDS)) {
  r <- BANDS[[b]]$range
  coh[[b]] <- rowMeans(coh[, freq_cols_c[freqs_c >= r[1] & freqs_c <= r[2]],
                            drop = FALSE], na.rm = TRUE)
}

coh$Accuracy <- ifelse(coh$yes_or_no == "yes", 1, 0)
coh$StimCond <- factor(ifelse(coh$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

coh <- coh[coh$Region %in% NONBLA_PAIRS, ]
cat(sprintf("\n  Non-BLA pairs available: %s\n",
            paste(sort(unique(coh$Region)), collapse=", ")))

# ── Across-all-pairs Pair * StimCond per band ──
cat("\n--- H1C: Coherence Pair x StimCond (non-BLA pairs, per band) ---\n")
{
  sub_all <- coh
  sub_all$Region <- factor(sub_all$Region, levels = sort(unique(sub_all$Region)))
  sub_all$Patient <- factor(sub_all$Patient)
  cat(sprintf("\n  Non-BLA pairs: %d trials, %d patients\n",
              nrow(sub_all), nlevels(sub_all$Patient)))
  for (b in names(BANDS)) {
    sub_all$band_c <- sub_all[[b]] - mean(sub_all[[b]], na.rm = TRUE)
    cat(sprintf("    Non-BLA coherence %s Pair*StimCond: ", BANDS[[b]]$label))
    fmls <- list(
      m0     = "Accuracy ~ 1 + (1 | Patient)",
      m1     = "Accuracy ~ band_c + (1 | Patient)",
      m2     = "Accuracy ~ band_c + StimCond + (1 | Patient)",
      m3     = "Accuracy ~ band_c + StimCond + Region + (1 | Patient)",
      m_full = "Accuracy ~ band_c + StimCond + Region + Region:StimCond + (1 | Patient)"
    )
    fits <- list()
    for (nm in names(fmls)) fits[[nm]] <- safe_glmer(as.formula(fmls[[nm]]), sub_all)
    if (!is.null(fits$m_full)) {
      save_coefs(fits$m_full, sprintf("coherence_allpairs_pairbystim_%s_coefs.csv", b))
      save_model_build(fits, fmls, sprintf("coherence_allpairs_pairbystim_%s_anova.csv", b))
    }
  }
}

# ── Per-pair two-band coherence ──
cat("\n--- H1C: Per-Pair Two-Band Coherence Models ---\n")
for (pair in NONBLA_PAIRS) {
  sub <- coh[coh$Region == pair, ]
  sub$Patient <- factor(sub$Patient)
  if (nrow(sub) < 30 || nlevels(sub$Patient) < MIN_PATIENTS) {
    cat(sprintf("\n  Skipping %s: %d trials, %d patients (< %d)\n",
                pair, nrow(sub), nlevels(sub$Patient), MIN_PATIENTS))
    next
  }
  sub$theta_c <- sub$theta - mean(sub$theta, na.rm = TRUE)
  sub$slow_gamma_c <- sub$slow_gamma - mean(sub$slow_gamma, na.rm = TRUE)
  cat(sprintf("\n  Pair %s (two-band): %d trials, %d patients\n",
              pair, nrow(sub), nlevels(sub$Patient)))
  fmls <- list(
    m0     = "Accuracy ~ 1 + (1 | Patient)",
    m1     = "Accuracy ~ StimCond + (1 | Patient)",
    m2     = "Accuracy ~ StimCond + theta_c + (1 | Patient)",
    m3     = "Accuracy ~ StimCond + theta_c + slow_gamma_c + (1 | Patient)",
    m_full = "Accuracy ~ StimCond + theta_c + slow_gamma_c + theta_c:StimCond + slow_gamma_c:StimCond + (1 | Patient)"
  )
  fits <- list()
  for (nm in names(fmls)) fits[[nm]] <- safe_glmer(as.formula(fmls[[nm]]), sub)
  if (!is.null(fits$m_full)) {
    save_coefs(fits$m_full, sprintf("coherence_twoband_%s_coefs.csv", pair))
    save_model_build(fits, fmls, sprintf("coherence_twoband_%s_anova.csv", pair))
  }
}

##############################################################################
# PAC
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("H1C PAC (non-BLA pairs) - %s\n", toupper(label)))
cat(paste(rep("=", 70), collapse=""), "\n")

pac_csv <- file.path(script_dir, "outputs", "csvs",
                    "combined_retrieval_pac_all_mlmr_input.csv")
pac <- read.csv(pac_csv, stringsAsFactors = FALSE, check.names = FALSE)
pac <- pac[pac$yes_or_no %in% c("yes", "no") & pac$trial_type != "new", ]
pac <- filter_balanced(pac, "PAC")

freq_cols_p <- sort(grep("^diff_Freq_", names(pac), value = TRUE))
freqs_p <- as.numeric(sub("diff_Freq_", "", freq_cols_p))

sg_cols <- freq_cols_p[freqs_p >= PAC_SLOW_GAMMA[1] & freqs_p <= PAC_SLOW_GAMMA[2]]
hfa_cols <- freq_cols_p[freqs_p >= PAC_HFA[1] & freqs_p <= PAC_HFA[2]]

pac$slow_gamma_pac <- rowMeans(pac[, sg_cols, drop = FALSE], na.rm = TRUE)
pac$hfa_pac <- rowMeans(pac[, hfa_cols, drop = FALSE], na.rm = TRUE)

pac$Accuracy <- ifelse(pac$yes_or_no == "yes", 1, 0)
pac$StimCond <- factor(ifelse(pac$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

pac <- pac[pac$Region %in% NONBLA_PAIRS, ]
cat(sprintf("\n  Non-BLA PAC pairs available: %s\n",
            paste(sort(unique(pac$Region)), collapse=", ")))

# ── Across-all-pairs Pair * StimCond per PAC type ──
cat("\n--- H1C: PAC Pair x StimCond (non-BLA pairs, per PAC type) ---\n")
{
  sub_all <- pac
  sub_all$Region <- factor(sub_all$Region, levels = sort(unique(sub_all$Region)))
  sub_all$Patient <- factor(sub_all$Patient)
  cat(sprintf("\n  Non-BLA PAC pairs: %d trials, %d patients\n",
              nrow(sub_all), nlevels(sub_all$Patient)))
  for (pac_type in c("slow_gamma", "hfa")) {
    pac_col <- paste0(pac_type, "_pac")
    sub_all$pac_z <- as.numeric(scale(sub_all[[pac_col]]))
    cat(sprintf("    Non-BLA PAC %s Pair*StimCond: ", pac_type))
    fmls <- list(
      m0     = "Accuracy ~ 1 + (1 | Patient)",
      m1     = "Accuracy ~ pac_z + (1 | Patient)",
      m2     = "Accuracy ~ pac_z + StimCond + (1 | Patient)",
      m3     = "Accuracy ~ pac_z + StimCond + Region + (1 | Patient)",
      m_full = "Accuracy ~ pac_z + StimCond + Region + Region:StimCond + (1 | Patient)"
    )
    fits <- list()
    for (nm in names(fmls)) fits[[nm]] <- safe_glmer(as.formula(fmls[[nm]]), sub_all)
    if (!is.null(fits$m_full)) {
      save_coefs(fits$m_full, sprintf("pac_allpairs_pairbystim_%s_coefs.csv", pac_type))
      save_model_build(fits, fmls, sprintf("pac_allpairs_pairbystim_%s_anova.csv", pac_type))
    }
  }
}

# ── Per-pair two-PAC-type ──
cat("\n--- H1C: Per-Pair Two-PAC-Type Models ---\n")
for (pair in NONBLA_PAIRS) {
  sub <- pac[pac$Region == pair, ]
  sub$Patient <- factor(sub$Patient)
  if (nrow(sub) < 30 || nlevels(sub$Patient) < MIN_PATIENTS) {
    cat(sprintf("\n  Skipping %s: %d trials, %d patients (< %d)\n",
                pair, nrow(sub), nlevels(sub$Patient), MIN_PATIENTS))
    next
  }
  sg_sd <- sd(sub$slow_gamma_pac, na.rm = TRUE)
  hfa_sd <- sd(sub$hfa_pac, na.rm = TRUE)
  if (sg_sd < 1e-10 || hfa_sd < 1e-10) {
    cat(sprintf("\n  Skipping pair %s: near-zero SD\n", pair))
    next
  }
  sub$sg_pac_z <- as.numeric(scale(sub$slow_gamma_pac))
  sub$hfa_pac_z <- as.numeric(scale(sub$hfa_pac))
  cat(sprintf("\n  Pair %s (two-PAC-type): %d trials, %d patients\n",
              pair, nrow(sub), nlevels(sub$Patient)))
  fmls <- list(
    m0     = "Accuracy ~ 1 + (1 | Patient)",
    m1     = "Accuracy ~ StimCond + (1 | Patient)",
    m2     = "Accuracy ~ StimCond + sg_pac_z + (1 | Patient)",
    m3     = "Accuracy ~ StimCond + sg_pac_z + hfa_pac_z + (1 | Patient)",
    m_full = "Accuracy ~ StimCond + sg_pac_z + hfa_pac_z + sg_pac_z:StimCond + hfa_pac_z:StimCond + (1 | Patient)"
  )
  fits <- list()
  for (nm in names(fmls)) fits[[nm]] <- safe_glmer(as.formula(fmls[[nm]]), sub)
  if (!is.null(fits$m_full)) {
    save_coefs(fits$m_full, sprintf("pac_twotype_%s_coefs.csv", pair))
    save_model_build(fits, fmls, sprintf("pac_twotype_%s_anova.csv", pair))
  }
}

cat(sprintf("\n\n===== H1C RETRIEVAL MLM (%s) COMPLETE =====\n", toupper(label)))
