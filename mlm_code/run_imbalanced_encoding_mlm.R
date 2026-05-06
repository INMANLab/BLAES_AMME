##############################################################################
# All-Subjects Encoding MLM: Power, Coherence × Stim Interactions
#
# NO balanced-trials filter — includes ALL patients regardless of
# memory trial counts.
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

out_dir <- file.path(script_dir, "outputs", "imbalanced_encoding_mlm")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)
HFA <- c(70.31, 99.61)

save_coefs <- function(model, filename, logistic = TRUE) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = logistic)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  return(td)
}

##############################################################################
# POWER
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("ENCODING POWER - ALL SUBJECTS (IMBALANCED)\n")
cat(paste(rep("=", 70), collapse=""), "\n")

pw_csv <- file.path(script_dir, "outputs", "csvs", "combined_encoding_power_all_mlmr_input.csv")
pw <- read.csv(pw_csv, stringsAsFactors = FALSE, check.names = FALSE)
pw <- pw[pw$yes_or_no %in% c("yes", "no") & pw$trial_type != "new", ]

cat(sprintf("\n  All subjects: %d patients\n", length(unique(pw$Patient))))

freq_cols <- sort(grep("^diff_Freq_", names(pw), value = TRUE))
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))

pw$theta <- rowMeans(pw[, freq_cols[freqs >= THETA[1] & freqs <= THETA[2]], drop=FALSE], na.rm=TRUE)
pw$slow_gamma <- rowMeans(pw[, freq_cols[freqs >= SLOW_GAMMA[1] & freqs <= SLOW_GAMMA[2]], drop=FALSE], na.rm=TRUE)
pw$fast_gamma <- rowMeans(pw[, freq_cols[freqs >= HFA[1] & freqs <= HFA[2]], drop=FALSE], na.rm=TRUE)

pw$Accuracy <- ifelse(pw$yes_or_no == "yes", 1, 0)
pw$StimCond <- factor(ifelse(pw$trial_type == "nostim", "nostim", "stim"),
                      levels = c("nostim", "stim"))

mtl_regions <- c("BLA", "HPC", "EC", "PRC")
hpc_regions <- c("BLA", "CA", "DG")

for (region_set_name in c("MTL", "HPC_subfields")) {
  allowed <- if (region_set_name == "MTL") mtl_regions else hpc_regions
  sub <- pw[pw$Region %in% allowed, ]
  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n--- Power %s: %d trials, %d patients, regions: %s ---\n",
              region_set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))
  cat(sprintf("  Memory rate: %.1f%%, Stim rate: %.1f%%\n",
              mean(sub$Accuracy)*100, mean(sub$StimCond == "stim")*100))

  for (band in c("theta", "slow_gamma", "fast_gamma")) {
    band_label <- switch(band, theta="Theta", slow_gamma="Slow Gamma", fast_gamma="HFA")
    sub$band_c <- sub[[band]] - mean(sub[[band]], na.rm=TRUE)

    cat(sprintf("\n  MODEL: Power %s %s x StimCond\n", region_set_name, band_label))

    m <- tryCatch({
      glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    FAILED:", conditionMessage(e), "\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("power_%s_%s_coefs.csv", region_set_name, band))
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
cat("ENCODING COHERENCE - ALL SUBJECTS (IMBALANCED)\n")
cat(paste(rep("=", 70), collapse=""), "\n")

coh_csv <- file.path(script_dir, "outputs", "csvs", "combined_encoding_coherence_all_mlmr_input.csv")
coh <- read.csv(coh_csv, stringsAsFactors = FALSE, check.names = FALSE)
coh <- coh[coh$yes_or_no %in% c("yes", "no") & coh$trial_type != "new", ]

cat(sprintf("\n  All subjects: %d patients\n", length(unique(coh$Patient))))

freq_cols_c <- sort(grep("^diff_Freq_", names(coh), value = TRUE))
freqs_c <- as.numeric(sub("diff_Freq_", "", freq_cols_c))

coh$theta <- rowMeans(coh[, freq_cols_c[freqs_c >= THETA[1] & freqs_c <= THETA[2]], drop=FALSE], na.rm=TRUE)
coh$slow_gamma <- rowMeans(coh[, freq_cols_c[freqs_c >= SLOW_GAMMA[1] & freqs_c <= SLOW_GAMMA[2]], drop=FALSE], na.rm=TRUE)
coh$fast_gamma <- rowMeans(coh[, freq_cols_c[freqs_c >= HFA[1] & freqs_c <= HFA[2]], drop=FALSE], na.rm=TRUE)

coh$Accuracy <- ifelse(coh$yes_or_no == "yes", 1, 0)
coh$StimCond <- factor(ifelse(coh$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

coh_mtl_pairs <- function(df) {
  allowed <- c("BLA", "HPC", "EC", "PRC")
  df[sapply(strsplit(df$Region, "_", fixed=TRUE), function(p) all(p %in% allowed)), ]
}

coh_hpc_pairs <- function(df) {
  allowed <- c("BLA", "CA", "DG")
  df[sapply(strsplit(df$Region, "_", fixed=TRUE), function(p) all(p %in% allowed)), ]
}

for (region_set_name in c("MTL", "HPC_subfields")) {
  sub <- if (region_set_name == "MTL") coh_mtl_pairs(coh) else coh_hpc_pairs(coh)
  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n--- Coherence %s: %d trials, %d patients, regions: %s ---\n",
              region_set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))

  for (band in c("theta", "slow_gamma", "fast_gamma")) {
    band_label <- switch(band, theta="Theta", slow_gamma="Slow Gamma", fast_gamma="HFA")
    sub$band_c <- sub[[band]] - mean(sub[[band]], na.rm=TRUE)

    cat(sprintf("\n  MODEL: Coherence %s %s x StimCond\n", region_set_name, band_label))

    m <- tryCatch({
      glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    FAILED:", conditionMessage(e), "\n"); NULL })

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

cat("\n\n===== IMBALANCED ENCODING POWER/COHERENCE MLM COMPLETE =====\n")
