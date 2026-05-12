##############################################################################
# Retrieval MLM v2: Per-Region + Across-Region Power/Coherence + PAC
#
# Usage: Rscript run_v2_retrieval.R --balanced
#        Rscript run_v2_retrieval.R --imbalanced
#
# Retrieval bands: theta and slow gamma only (for power/coherence)
# PAC: SG and HFA (separate models)
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
MIN_PATIENTS <- 5  # require at least 5 patients per per-region/per-pair test

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
out_dir <- file.path(script_dir, "outputs", paste0(label, "_retrieval_mlm"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)
HFA <- c(70.31, 99.61)
PAC_SLOW_GAMMA <- c(30, 50)
PAC_HFA <- c(70, 100)

# Retrieval: only theta and slow gamma for power/coherence
BANDS <- list(
  theta = list(range = THETA, label = "Theta"),
  slow_gamma = list(range = SLOW_GAMMA, label = "Slow Gamma")
)

save_coefs <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  cat(sprintf("  -> Saved %s\n", filename))
  return(td)
}

# Build a sequential model-comparison table from a list of fitted lme4 models.
# `models` is a named list (names become Model labels). Saves rows with
# Model, Formula, npar, AIC, BIC, logLik, deviance, Chisq, Df, p.value.
save_model_build <- function(models, formulas, filename) {
  ord <- names(models)
  rows <- list()
  prev <- NULL
  for (i in seq_along(ord)) {
    nm <- ord[i]
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
      Model    = nm,
      Formula  = formulas[[nm]],
      npar     = npar,
      AIC      = aic_v,
      BIC      = bic_v,
      logLik   = ll,
      deviance = -2 * ll,
      Chisq    = chisq,
      Df       = df_v,
      p.value  = pv,
      stringsAsFactors = FALSE
    )
    prev <- m
  }
  if (length(rows) == 0) return(invisible(NULL))
  out <- do.call(rbind, rows)
  write.csv(out, file.path(out_dir, filename), row.names = FALSE)
  cat(sprintf("  -> Saved %s\n", filename))
}

# Helper: fit a glmer with a given formula, returning NULL on failure.
safe_glmer <- function(formula, data) {
  tryCatch(
    glmer(formula, data = data, family = binomial, control = ctrl),
    error = function(e) { cat("    FAILED:", conditionMessage(e), "\n"); NULL }
  )
}

print_interaction <- function(td, pattern) {
  row <- td[grepl(pattern, td$term), ]
  if (nrow(row) > 0) {
    cat(sprintf("    %s: OR=%.4f [%.4f, %.4f], p=%.4f\n",
                row$term[1], row$estimate[1], row$conf.low[1], row$conf.high[1],
                row$p.value[1]))
  }
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
  excluded <- counts[!(counts$Patient %in% included), ]

  cat(sprintf("\n  [%s] Balanced filter (>=%d per condition):\n", measure_label, MIN_TRIALS))
  cat(sprintf("    Included: %d / %d patients\n", length(included), nrow(counts)))
  if (nrow(excluded) > 0) {
    for (i in seq_len(nrow(excluded))) {
      cat(sprintf("    Excluded: %s (rem=%d, forg=%d)\n",
                  excluded$Patient[i], excluded$n_rem_unique[i], excluded$n_forg_unique[i]))
    }
  }
  df[df$Patient %in% included, ]
}

##############################################################################
# POWER
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("RETRIEVAL POWER - %s\n", toupper(label)))
cat(paste(rep("=", 70), collapse=""), "\n")

pw_csv <- file.path(script_dir, "outputs", "csvs", "combined_retrieval_power_all_mlmr_input.csv")
pw <- read.csv(pw_csv, stringsAsFactors = FALSE, check.names = FALSE)
pw <- pw[pw$yes_or_no %in% c("yes", "no") & pw$trial_type != "new", ]
pw <- filter_balanced(pw, "Power")

freq_cols <- sort(grep("^diff_Freq_", names(pw), value = TRUE))
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))

for (b in names(BANDS)) {
  r <- BANDS[[b]]$range
  pw[[b]] <- rowMeans(pw[, freq_cols[freqs >= r[1] & freqs <= r[2]], drop=FALSE], na.rm=TRUE)
}

pw$Accuracy <- ifelse(pw$yes_or_no == "yes", 1, 0)
pw$StimCond <- factor(ifelse(pw$trial_type == "nostim", "nostim", "stim"),
                      levels = c("nostim", "stim"))

mtl_regions <- c("BLA", "HPC", "EC", "PRC")
hpc_regions <- c("BLA", "CA", "DG")
all_regions <- unique(c(mtl_regions, hpc_regions))

# ── Per-region power models ──
cat("\n--- Per-Region Power Models ---\n")
for (reg in all_regions) {
  sub <- pw[pw$Region == reg, ]
  sub$Patient <- factor(sub$Patient)
  if (nrow(sub) < 30 || nlevels(sub$Patient) < 3) next

  cat(sprintf("\n  Region %s: %d trials, %d patients\n", reg, nrow(sub), nlevels(sub$Patient)))

  for (b in names(BANDS)) {
    sub$band_c <- sub[[b]] - mean(sub[[b]], na.rm=TRUE)
    cat(sprintf("    %s %s: ", reg, BANDS[[b]]$label))

    m <- tryCatch({
      glmer(Accuracy ~ band_c * StimCond + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("FAILED\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("power_region_%s_%s_coefs.csv", reg, b))
      print_interaction(td, "band_c:StimCond")
    }
  }
}

# ── NEW: Across-all-regions Region * StimCond per band ──
# Predictor entry order: band_c -> StimCond -> Region -> Region:StimCond.
# All 6 regions (BLA, CA, DG, HPC, EC, PRC) included; BLA is the reference.
cat("\n--- NEW: Power Region x StimCond (all regions, per band) ---\n")
{
  sub_all <- pw[pw$Region %in% all_regions, ]
  sub_all$Region <- factor(sub_all$Region, levels = sort(unique(sub_all$Region)))
  sub_all$Patient <- factor(sub_all$Patient)
  cat(sprintf("\n  All-regions: %d trials, %d patients, regions: %s\n",
              nrow(sub_all), nlevels(sub_all$Patient),
              paste(levels(sub_all$Region), collapse=", ")))
  for (b in names(BANDS)) {
    sub_all$band_c <- sub_all[[b]] - mean(sub_all[[b]], na.rm=TRUE)
    cat(sprintf("    All-regions %s Region*StimCond: ", BANDS[[b]]$label))
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
      td <- save_coefs(fits$m_full, sprintf("power_allregions_regbystim_%s_coefs.csv", b))
      print_interaction(td, "Region.*:StimCondstim")
      save_model_build(fits, fmls, sprintf("power_allregions_regbystim_%s_anova.csv", b))
    }
  }
}

# ── NEW: Per-region two-band power model ──
# Predictor entry order: StimCond -> theta_c -> slow_gamma_c -> band:StimCond ints.
cat("\n--- NEW: Per-Region Two-Band Power Models (>=5 patients) ---\n")
for (reg in all_regions) {
  sub <- pw[pw$Region == reg, ]
  sub$Patient <- factor(sub$Patient)
  if (nrow(sub) < 30 || nlevels(sub$Patient) < MIN_PATIENTS) {
    cat(sprintf("\n  Skipping %s: %d trials, %d patients (< %d)\n",
                reg, nrow(sub), nlevels(sub$Patient), MIN_PATIENTS))
    next
  }
  sub$theta_c <- sub$theta - mean(sub$theta, na.rm=TRUE)
  sub$slow_gamma_c <- sub$slow_gamma - mean(sub$slow_gamma, na.rm=TRUE)
  cat(sprintf("\n  Region %s (two-band): %d trials, %d patients\n",
              reg, nrow(sub), nlevels(sub$Patient)))
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
    save_coefs(fits$m_full, sprintf("power_twoband_%s_coefs.csv", reg))
    save_model_build(fits, fmls, sprintf("power_twoband_%s_anova.csv", reg))
  }
}

# ── Across-region power models ──
cat("\n--- Across-Region Power Models ---\n")
for (region_set_name in c("MTL", "HPC_subfields")) {
  allowed <- if (region_set_name == "MTL") mtl_regions else hpc_regions
  sub <- pw[pw$Region %in% allowed, ]
  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n  %s: %d trials, %d patients, regions: %s\n",
              region_set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))

  for (b in names(BANDS)) {
    sub$band_c <- sub[[b]] - mean(sub[[b]], na.rm=TRUE)
    cat(sprintf("    %s %s: ", region_set_name, BANDS[[b]]$label))

    m <- tryCatch({
      glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("FAILED\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("power_across_%s_%s_coefs.csv", region_set_name, b))
      print_interaction(td, "StimCondstim:band_c")
    }
  }
}

##############################################################################
# COHERENCE
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("RETRIEVAL COHERENCE - %s\n", toupper(label)))
cat(paste(rep("=", 70), collapse=""), "\n")

coh_csv <- file.path(script_dir, "outputs", "csvs", "combined_retrieval_coherence_all_mlmr_input.csv")
coh <- read.csv(coh_csv, stringsAsFactors = FALSE, check.names = FALSE)
coh <- coh[coh$yes_or_no %in% c("yes", "no") & coh$trial_type != "new", ]
coh <- filter_balanced(coh, "Coherence")

freq_cols_c <- sort(grep("^diff_Freq_", names(coh), value = TRUE))
freqs_c <- as.numeric(sub("diff_Freq_", "", freq_cols_c))

for (b in names(BANDS)) {
  r <- BANDS[[b]]$range
  coh[[b]] <- rowMeans(coh[, freq_cols_c[freqs_c >= r[1] & freqs_c <= r[2]], drop=FALSE], na.rm=TRUE)
}

coh$Accuracy <- ifelse(coh$yes_or_no == "yes", 1, 0)
coh$StimCond <- factor(ifelse(coh$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

coh_mtl_pairs_fn <- function(df) {
  allowed <- c("BLA", "HPC", "EC", "PRC")
  df[sapply(strsplit(df$Region, "_", fixed=TRUE), function(p) all(p %in% allowed)), ]
}
coh_hpc_pairs_fn <- function(df) {
  allowed <- c("BLA", "CA", "DG")
  df[sapply(strsplit(df$Region, "_", fixed=TRUE), function(p) all(p %in% allowed)), ]
}

# ── Per-region-pair coherence models ──
cat("\n--- Per-Region-Pair Coherence Models ---\n")
mtl_coh <- coh_mtl_pairs_fn(coh)
hpc_coh <- coh_hpc_pairs_fn(coh)
valid_pairs <- sort(unique(c(mtl_coh$Region, hpc_coh$Region)))

for (reg in valid_pairs) {
  sub <- coh[coh$Region == reg, ]
  sub$Patient <- factor(sub$Patient)
  if (nrow(sub) < 30 || nlevels(sub$Patient) < 3) {
    cat(sprintf("\n  Skipping %s: %d trials, %d patients\n", reg, nrow(sub), nlevels(sub$Patient)))
    next
  }

  cat(sprintf("\n  Pair %s: %d trials, %d patients\n", reg, nrow(sub), nlevels(sub$Patient)))

  for (b in names(BANDS)) {
    sub$band_c <- sub[[b]] - mean(sub[[b]], na.rm=TRUE)
    cat(sprintf("    %s %s: ", reg, BANDS[[b]]$label))

    m <- tryCatch({
      glmer(Accuracy ~ band_c * StimCond + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("FAILED\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("coherence_region_%s_%s_coefs.csv", reg, b))
      print_interaction(td, "band_c:StimCond")
    }
  }
}

# ── NEW: Across-all-pairs Pair * StimCond per band (coherence) ──
# Restricted to BLA-X pairs only (BLA-CA, BLA-DG, BLA-EC, BLA-HPC, BLA-PRC).
# Predictor entry order: band_c -> StimCond -> Region(=pair) -> Region:StimCond.
BLA_PAIRS_COH <- c("BLA_CA", "BLA_DG", "BLA_EC", "BLA_HPC", "BLA_PRC")
cat("\n--- NEW: Coherence Pair x StimCond (BLA-X pairs, per band) ---\n")
{
  sub_all <- coh[coh$Region %in% BLA_PAIRS_COH, ]
  sub_all$Region <- factor(sub_all$Region, levels = sort(unique(sub_all$Region)))
  sub_all$Patient <- factor(sub_all$Patient)
  cat(sprintf("\n  BLA-X pairs: %d trials, %d patients, pairs: %s\n",
              nrow(sub_all), nlevels(sub_all$Patient),
              paste(levels(sub_all$Region), collapse=", ")))
  for (b in names(BANDS)) {
    sub_all$band_c <- sub_all[[b]] - mean(sub_all[[b]], na.rm=TRUE)
    cat(sprintf("    BLA-X coherence %s Pair*StimCond: ", BANDS[[b]]$label))
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
      td <- save_coefs(fits$m_full, sprintf("coherence_allpairs_pairbystim_%s_coefs.csv", b))
      print_interaction(td, "Region.*:StimCondstim")
      save_model_build(fits, fmls, sprintf("coherence_allpairs_pairbystim_%s_anova.csv", b))
    }
  }
}

# ── NEW: Per-pair two-band coherence model (BLA-X pairs, >=5 patients) ──
cat("\n--- NEW: Per-Pair Two-Band Coherence Models (BLA-X) ---\n")
for (pair in BLA_PAIRS_COH) {
  sub <- coh[coh$Region == pair, ]
  sub$Patient <- factor(sub$Patient)
  if (nrow(sub) < 30 || nlevels(sub$Patient) < MIN_PATIENTS) {
    cat(sprintf("\n  Skipping %s: %d trials, %d patients (< %d)\n",
                pair, nrow(sub), nlevels(sub$Patient), MIN_PATIENTS))
    next
  }
  sub$theta_c <- sub$theta - mean(sub$theta, na.rm=TRUE)
  sub$slow_gamma_c <- sub$slow_gamma - mean(sub$slow_gamma, na.rm=TRUE)
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

# ── Across-region coherence models ──
cat("\n--- Across-Region Coherence Models ---\n")
for (region_set_name in c("MTL", "HPC_subfields")) {
  sub <- if (region_set_name == "MTL") coh_mtl_pairs_fn(coh) else coh_hpc_pairs_fn(coh)
  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n  %s: %d trials, %d patients, pairs: %s\n",
              region_set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))

  for (b in names(BANDS)) {
    sub$band_c <- sub[[b]] - mean(sub[[b]], na.rm=TRUE)
    cat(sprintf("    %s %s: ", region_set_name, BANDS[[b]]$label))

    m <- tryCatch({
      glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("FAILED\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("coherence_across_%s_%s_coefs.csv", region_set_name, b))
      print_interaction(td, "StimCondstim:band_c")
    }
  }
}

##############################################################################
# PAC (Z-scored, separate SG and HFA models)
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("RETRIEVAL PAC - %s\n", toupper(label)))
cat(paste(rep("=", 70), collapse=""), "\n")

pac_csv <- file.path(script_dir, "outputs", "csvs", "combined_retrieval_pac_all_mlmr_input.csv")
pac <- read.csv(pac_csv, stringsAsFactors = FALSE, check.names = FALSE)
pac <- pac[pac$yes_or_no %in% c("yes", "no") & pac$trial_type != "new", ]
pac <- filter_balanced(pac, "PAC")

freq_cols_p <- sort(grep("^diff_Freq_", names(pac), value = TRUE))
freqs_p <- as.numeric(sub("diff_Freq_", "", freq_cols_p))

sg_cols <- freq_cols_p[freqs_p >= PAC_SLOW_GAMMA[1] & freqs_p <= PAC_SLOW_GAMMA[2]]
hfa_cols <- freq_cols_p[freqs_p >= PAC_HFA[1] & freqs_p <= PAC_HFA[2]]

pac$slow_gamma_pac <- rowMeans(pac[, sg_cols, drop=FALSE], na.rm=TRUE)
pac$hfa_pac <- rowMeans(pac[, hfa_cols, drop=FALSE], na.rm=TRUE)

pac$Accuracy <- ifelse(pac$yes_or_no == "yes", 1, 0)
pac$StimCond <- factor(ifelse(pac$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

pac_sets <- list(
  MTL = c("BLA_HPC", "BLA_EC", "BLA_PRC"),
  HPC_subfields = c("BLA_HPC", "BLA_CA", "BLA_DG")
)

for (set_name in names(pac_sets)) {
  pairs <- pac_sets[[set_name]]
  sub <- pac[pac$Region %in% pairs, ]
  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n--- PAC %s: %d trials, %d patients, pairs: %s ---\n",
              set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))

  for (pac_type in c("slow_gamma", "hfa")) {
    pac_col <- paste0(pac_type, "_pac")
    pac_label <- if (pac_type == "slow_gamma") "SG" else "HFA"

    pac_sd <- sd(sub[[pac_col]], na.rm = TRUE)
    if (pac_sd < 1e-10) {
      cat(sprintf("  Skipping %s %s PAC: near-zero SD\n", set_name, pac_label))
      next
    }
    # Try raw PAC first; fall back to z-scored only on convergence error.
    sub$pac_z <- sub[[pac_col]]

    cat(sprintf("  %s %s PAC (SD=%.6f): ", set_name, pac_label, pac_sd))

    m <- tryCatch({
      glmer(Accuracy ~ pac_z * StimCond + Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) NULL)
    if (is.null(m)) {
      cat("    Convergence fallback: z-scored PAC. ")
      sub$pac_z <- as.numeric(scale(sub[[pac_col]]))
      m <- tryCatch({
        glmer(Accuracy ~ pac_z * StimCond + Region + (1 | Patient),
              data = sub, family = binomial, control = ctrl)
      }, error = function(e) { cat("FAILED\n"); NULL })
    }

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("pac_%s_%s_coefs.csv", set_name, pac_type))
      print_interaction(td, "pac_z:StimCond")
      cat("\n")
      print(summary(m))
    }
  }
}

# ── NEW: Across-all-pairs Pair * StimCond per PAC type ──
# Restricted to BLA-X pairs (already the case in pac_sets); reorder predictors
# pac_z -> StimCond -> Region -> Region:StimCond. Drop pairs with <5 patients.
BLA_PAIRS_PAC <- c("BLA_CA", "BLA_DG", "BLA_EC", "BLA_HPC", "BLA_PRC")
cat("\n--- NEW: PAC Pair x StimCond (BLA-X pairs, per PAC type) ---\n")
{
  sub_all <- pac[pac$Region %in% BLA_PAIRS_PAC, ]
  sub_all$Region <- factor(sub_all$Region, levels = sort(unique(sub_all$Region)))
  sub_all$Patient <- factor(sub_all$Patient)
  cat(sprintf("\n  BLA-X PAC pairs: %d trials, %d patients, pairs: %s\n",
              nrow(sub_all), nlevels(sub_all$Patient),
              paste(levels(sub_all$Region), collapse=", ")))
  for (pac_type in c("slow_gamma", "hfa")) {
    pac_col <- paste0(pac_type, "_pac")
    # Try raw PAC first; fall back to z-scored only on convergence error.
    sub_all$pac_z <- sub_all[[pac_col]]
    cat(sprintf("    BLA-X PAC %s Pair*StimCond: ", pac_type))
    fmls <- list(
      m0     = "Accuracy ~ 1 + (1 | Patient)",
      m1     = "Accuracy ~ pac_z + (1 | Patient)",
      m2     = "Accuracy ~ pac_z + StimCond + (1 | Patient)",
      m3     = "Accuracy ~ pac_z + StimCond + Region + (1 | Patient)",
      m_full = "Accuracy ~ pac_z + StimCond + Region + Region:StimCond + (1 | Patient)"
    )
    fits <- list()
    for (nm in names(fmls)) fits[[nm]] <- safe_glmer(as.formula(fmls[[nm]]), sub_all)
    if (any(sapply(fits, is.null))) {
      cat("    Convergence fallback: z-scored PAC.\n")
      sub_all$pac_z <- as.numeric(scale(sub_all[[pac_col]]))
      for (nm in names(fmls)) fits[[nm]] <- safe_glmer(as.formula(fmls[[nm]]), sub_all)
    }
    if (!is.null(fits$m_full)) {
      td <- save_coefs(fits$m_full, sprintf("pac_allpairs_pairbystim_%s_coefs.csv", pac_type))
      print_interaction(td, "Region.*:StimCondstim")
      save_model_build(fits, fmls, sprintf("pac_allpairs_pairbystim_%s_anova.csv", pac_type))
    }
  }
}

# ── NEW: Per-pair two-PAC-type model (BLA-X pairs, >=5 patients) ──
cat("\n--- NEW: Per-Pair Two-PAC-Type Models (BLA-X) ---\n")
for (pair in BLA_PAIRS_PAC) {
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
  # Try raw PAC first; fall back to z-scored only on convergence error.
  sub$sg_pac_z <- sub$slow_gamma_pac
  sub$hfa_pac_z <- sub$hfa_pac
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
  if (any(sapply(fits, is.null))) {
    cat("    Convergence fallback: z-scored PAC.\n")
    sub$sg_pac_z <- as.numeric(scale(sub$slow_gamma_pac))
    sub$hfa_pac_z <- as.numeric(scale(sub$hfa_pac))
    for (nm in names(fmls)) fits[[nm]] <- safe_glmer(as.formula(fmls[[nm]]), sub)
  }
  if (!is.null(fits$m_full)) {
    save_coefs(fits$m_full, sprintf("pac_twotype_%s_coefs.csv", pair))
    save_model_build(fits, fmls, sprintf("pac_twotype_%s_anova.csv", pair))
  }
}

cat(sprintf("\n\n===== RETRIEVAL MLM V2 (%s) COMPLETE =====\n", toupper(label)))
