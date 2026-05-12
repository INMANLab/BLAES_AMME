##############################################################################
# Balanced-Trials Encoding MLM: Power, Coherence, PAC × Stim Interactions
#
# Excludes subjects with <10 trials in either remembered or forgotten
# condition. Runs trial-level GLMMs (binomial) for encoding only.
#
# Models:
#   Power/Coherence: Accuracy ~ band_c * StimCond + Region + band_c:Region + (1|Patient)
#   PAC (per region): Accuracy ~ (slow_gamma_pac_c + hfa_pac_c) * StimCond + (1|Patient)
##############################################################################

if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")

library(lme4)
library(lmerTest)
library(broom.mixed)

script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg)))
    else getwd()
  }
)

MIN_TRIALS <- 10
out_dir <- file.path(script_dir, "outputs", "balanced_retrieval_mlm")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# Band definitions
THETA <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)
HFA <- c(70.31, 99.61)

PAC_SLOW_GAMMA <- c(30, 50)
PAC_HFA <- c(70, 100)

save_coefs <- function(model, filename, logistic = TRUE) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = logistic)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  return(td)
}

##############################################################################
# Helper: balanced-trials filter (base R, no dplyr)
##############################################################################
filter_balanced <- function(df, measure_label) {
  patients <- unique(df$Patient)
  counts <- data.frame(Patient = patients, stringsAsFactors = FALSE)
  counts$n_rem <- sapply(patients, function(p) sum(df$Patient == p & df$yes_or_no == "yes"))
  counts$n_forg <- sapply(patients, function(p) sum(df$Patient == p & df$yes_or_no == "no"))
  counts$n_regions <- sapply(patients, function(p) length(unique(df$Region[df$Patient == p])))

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

filter_balanced_pac <- function(df, measure_label) {
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
cat("RETRIEVAL POWER - BALANCED TRIALS (RETRIEVAL)\n")
cat(paste(rep("=", 70), collapse=""), "\n")

pw_csv <- file.path(script_dir, "outputs", "csvs", "combined_retrieval_power_all_mlmr_input.csv")
pw <- read.csv(pw_csv, stringsAsFactors = FALSE, check.names = FALSE)
pw <- pw[pw$yes_or_no %in% c("yes", "no") & pw$trial_type != "new", ]

pw_bal <- filter_balanced(pw, "Power")

# Compute band means
freq_cols <- sort(grep("^diff_Freq_", names(pw_bal), value = TRUE))
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))

pw_bal$theta <- rowMeans(pw_bal[, freq_cols[freqs >= THETA[1] & freqs <= THETA[2]], drop=FALSE], na.rm=TRUE)
pw_bal$slow_gamma <- rowMeans(pw_bal[, freq_cols[freqs >= SLOW_GAMMA[1] & freqs <= SLOW_GAMMA[2]], drop=FALSE], na.rm=TRUE)
pw_bal$fast_gamma <- rowMeans(pw_bal[, freq_cols[freqs >= HFA[1] & freqs <= HFA[2]], drop=FALSE], na.rm=TRUE)

pw_bal$Accuracy <- ifelse(pw_bal$yes_or_no == "yes", 1, 0)
pw_bal$StimCond <- factor(ifelse(pw_bal$trial_type == "nostim", "nostim", "stim"),
                          levels = c("nostim", "stim"))

# Region subsets
mtl_regions <- c("BLA", "HPC", "EC", "PRC")
hpc_regions <- c("BLA", "CA", "DG")

for (region_set_name in c("MTL", "HPC_subfields")) {
  if (region_set_name == "MTL") {
    allowed <- mtl_regions
  } else {
    allowed <- hpc_regions
  }

  sub <- pw_bal[pw_bal$Region %in% allowed, ]
  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n--- Power %s: %d trials, %d patients, regions: %s ---\n",
              region_set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))
  cat(sprintf("  Memory rate: %.1f%%, Stim rate: %.1f%%\n",
              mean(sub$Accuracy)*100, mean(sub$StimCond == "stim")*100))

  for (band in c("theta", "slow_gamma", "fast_gamma")) {
    band_label <- switch(band, theta="Theta", slow_gamma="Slow Gamma", fast_gamma="HFA")
    # Try raw band first; fall back to grand-mean-centered only on convergence error.
    sub$band_c <- sub[[band]]

    cat(sprintf("\n  MODEL: Power %s %s x StimCond\n", region_set_name, band_label))

    m <- tryCatch({
      glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) NULL)
    if (is.null(m)) {
      cat("    Convergence fallback: grand-mean-centered band\n")
      sub$band_c <- sub[[band]] - mean(sub[[band]], na.rm=TRUE)
      m <- tryCatch({
        glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
              data = sub, family = binomial, control = ctrl)
      }, error = function(e) { cat("    FAILED:", conditionMessage(e), "\n"); NULL })
    }

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("power_%s_%s_coefs.csv", region_set_name, band))
      # Print key interaction
      stim_int <- td[grepl("StimCond", td$term) & grepl("band_c", td$term), ]
      if (nrow(stim_int) > 0) {
        cat(sprintf("    band_c x StimCond: OR=%.4f [%.4f, %.4f], p=%.4f\n",
                    stim_int$estimate[1], stim_int$conf.low[1], stim_int$conf.high[1],
                    stim_int$p.value[1]))
      }
      print(summary(m))
    }
  }
}

##############################################################################
# COHERENCE
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("RETRIEVAL COHERENCE - BALANCED TRIALS (RETRIEVAL)\n")
cat(paste(rep("=", 70), collapse=""), "\n")

coh_csv <- file.path(script_dir, "outputs", "csvs", "combined_retrieval_coherence_all_mlmr_input.csv")
coh <- read.csv(coh_csv, stringsAsFactors = FALSE, check.names = FALSE)
coh <- coh[coh$yes_or_no %in% c("yes", "no") & coh$trial_type != "new", ]

coh_bal <- filter_balanced(coh, "Coherence")

freq_cols_c <- sort(grep("^diff_Freq_", names(coh_bal), value = TRUE))
freqs_c <- as.numeric(sub("diff_Freq_", "", freq_cols_c))

coh_bal$theta <- rowMeans(coh_bal[, freq_cols_c[freqs_c >= THETA[1] & freqs_c <= THETA[2]], drop=FALSE], na.rm=TRUE)
coh_bal$slow_gamma <- rowMeans(coh_bal[, freq_cols_c[freqs_c >= SLOW_GAMMA[1] & freqs_c <= SLOW_GAMMA[2]], drop=FALSE], na.rm=TRUE)
coh_bal$fast_gamma <- rowMeans(coh_bal[, freq_cols_c[freqs_c >= HFA[1] & freqs_c <= HFA[2]], drop=FALSE], na.rm=TRUE)

coh_bal$Accuracy <- ifelse(coh_bal$yes_or_no == "yes", 1, 0)
coh_bal$StimCond <- factor(ifelse(coh_bal$trial_type == "nostim", "nostim", "stim"),
                           levels = c("nostim", "stim"))

# Coherence region filters
coh_mtl_pairs <- function(df) {
  allowed <- c("BLA", "HPC", "EC", "PRC")
  df[sapply(strsplit(df$Region, "_", fixed=TRUE), function(p) all(p %in% allowed)), ]
}

coh_hpc_pairs <- function(df) {
  allowed <- c("BLA", "CA", "DG")
  df[sapply(strsplit(df$Region, "_", fixed=TRUE), function(p) all(p %in% allowed)), ]
}

for (region_set_name in c("MTL", "HPC_subfields")) {
  if (region_set_name == "MTL") {
    sub <- coh_mtl_pairs(coh_bal)
  } else {
    sub <- coh_hpc_pairs(coh_bal)
  }

  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n--- Coherence %s: %d trials, %d patients, regions: %s ---\n",
              region_set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))

  for (band in c("theta", "slow_gamma", "fast_gamma")) {
    band_label <- switch(band, theta="Theta", slow_gamma="Slow Gamma", fast_gamma="HFA")
    # Try raw band first; fall back to grand-mean-centered only on convergence error.
    sub$band_c <- sub[[band]]

    cat(sprintf("\n  MODEL: Coherence %s %s x StimCond\n", region_set_name, band_label))

    m <- tryCatch({
      glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) NULL)
    if (is.null(m)) {
      cat("    Convergence fallback: grand-mean-centered band\n")
      sub$band_c <- sub[[band]] - mean(sub[[band]], na.rm=TRUE)
      m <- tryCatch({
        glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
              data = sub, family = binomial, control = ctrl)
      }, error = function(e) { cat("    FAILED:", conditionMessage(e), "\n"); NULL })
    }

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("coherence_%s_%s_coefs.csv", region_set_name, band))
      stim_int <- td[grepl("StimCond", td$term) & grepl("band_c", td$term), ]
      if (nrow(stim_int) > 0) {
        cat(sprintf("    band_c x StimCond: OR=%.4f [%.4f, %.4f], p=%.4f\n",
                    stim_int$estimate[1], stim_int$conf.low[1], stim_int$conf.high[1],
                    stim_int$p.value[1]))
      }
      print(summary(m))
    }
  }
}

##############################################################################
# PAC (per region pair)
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("RETRIEVAL PAC - BALANCED TRIALS (RETRIEVAL)\n")
cat(paste(rep("=", 70), collapse=""), "\n")

pac_csv <- file.path(script_dir, "outputs", "csvs", "combined_retrieval_pac_all_mlmr_input.csv")
pac <- read.csv(pac_csv, stringsAsFactors = FALSE, check.names = FALSE)
pac <- pac[pac$yes_or_no %in% c("yes", "no") & pac$trial_type != "new", ]

pac_bal <- filter_balanced_pac(pac, "PAC")

freq_cols_p <- sort(grep("^diff_Freq_", names(pac_bal), value = TRUE))
freqs_p <- as.numeric(sub("diff_Freq_", "", freq_cols_p))

sg_cols <- freq_cols_p[freqs_p >= PAC_SLOW_GAMMA[1] & freqs_p <= PAC_SLOW_GAMMA[2]]
hfa_cols <- freq_cols_p[freqs_p >= PAC_HFA[1] & freqs_p <= PAC_HFA[2]]

pac_bal$slow_gamma_pac <- rowMeans(pac_bal[, sg_cols, drop=FALSE], na.rm=TRUE)
pac_bal$hfa_pac <- rowMeans(pac_bal[, hfa_cols, drop=FALSE], na.rm=TRUE)

# Default: use raw PAC values (no centering). Per-region fallback below
# replaces these with grand-mean-centered values only on convergence error.
pac_bal$slow_gamma_pac_c <- pac_bal$slow_gamma_pac
pac_bal$hfa_pac_c <- pac_bal$hfa_pac

pac_bal$Accuracy <- ifelse(pac_bal$yes_or_no == "yes", 1, 0)
pac_bal$StimCond <- factor(ifelse(pac_bal$trial_type == "nostim", "nostim", "stim"),
                           levels = c("nostim", "stim"))

# Run per region pair
pac_regions <- sort(unique(pac_bal$Region))
cat(sprintf("\n  PAC region pairs: %s\n", paste(pac_regions, collapse=", ")))

for (reg in pac_regions) {
  sub <- pac_bal[pac_bal$Region == reg, ]
  sub$Patient <- factor(sub$Patient)

  if (nrow(sub) < 30 || nlevels(sub$Patient) < 3) {
    cat(sprintf("\n  Skipping %s: %d trials, %d patients\n", reg, nrow(sub), nlevels(sub$Patient)))
    next
  }

  # Try raw PAC values first; fall back to within-region grand-mean-centered
  # only on convergence error.
  sub$slow_gamma_pac_c <- sub$slow_gamma_pac
  sub$hfa_pac_c <- sub$hfa_pac

  cat(sprintf("\n--- PAC %s: %d trials, %d patients ---\n",
              reg, nrow(sub), nlevels(sub$Patient)))
  cat(sprintf("  Memory rate: %.1f%%, Stim rate: %.1f%%\n",
              mean(sub$Accuracy)*100, mean(sub$StimCond == "stim")*100))

  cat(sprintf("  MODEL: PAC %s - (SG + HFA) x StimCond\n", reg))

  m <- tryCatch({
    glmer(Accuracy ~ (slow_gamma_pac_c + hfa_pac_c) * StimCond + (1 | Patient),
          data = sub, family = binomial, control = ctrl)
  }, error = function(e) NULL)
  if (is.null(m)) {
    cat("    Convergence fallback: within-region grand-mean-centered PAC\n")
    sub$slow_gamma_pac_c <- sub$slow_gamma_pac - mean(sub$slow_gamma_pac, na.rm=TRUE)
    sub$hfa_pac_c <- sub$hfa_pac - mean(sub$hfa_pac, na.rm=TRUE)
    m <- tryCatch({
      glmer(Accuracy ~ (slow_gamma_pac_c + hfa_pac_c) * StimCond + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    FAILED:", conditionMessage(e), "\n"); NULL })
  }

  if (!is.null(m)) {
    td <- save_coefs(m, sprintf("pac_%s_coefs.csv", gsub("/", "_", reg)))
    # Print key interactions
    for (term_pat in c("slow_gamma_pac_c:StimCond", "hfa_pac_c:StimCond")) {
      row <- td[grepl(term_pat, td$term, fixed=TRUE), ]
      if (nrow(row) > 0) {
        cat(sprintf("    %s: OR=%.4f [%.4f, %.4f], p=%.4f\n",
                    row$term[1], row$estimate[1], row$conf.low[1], row$conf.high[1],
                    row$p.value[1]))
      }
    }
    print(summary(m))
  }
}

cat("\n\n===== BALANCED ENCODING MLM COMPLETE =====\n")
