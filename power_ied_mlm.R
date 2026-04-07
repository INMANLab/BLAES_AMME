##############################################################################
#  Power x IED Timing Windows MLM - Encoding Only
#
#  Per-region power (BLA, HPC, CA, DG) on IED trials only.
#  Tests whether trial-level power predicts subsequent memory,
#  controlling for IED timing windows and stimulation.
#
#  Models per region and band (theta, slow gamma, HFA):
#    Part 1: All power trials (no IED data needed)
#      A1: memory ~ pow_band_z + stim + (1|patient_id)
#    Part 2: IED+power merged trials
#      T1: memory ~ pow_band_z + timing windows + (1|patient_id)
#      TI1: memory ~ pow_band_z * timing windows + (1|patient_id)
#    Part 3: Power + IED Spread
#      SP1: memory ~ pow_band_z + spread + (1|patient_id)
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
out_dir <- file.path(script_dir, "outputs", "ied_timing_memory")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

save_tidy <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  return(td)
}

bands <- c("theta", "slow_gamma", "hfa")
band_labels <- c("Theta (4-8 Hz)", "Slow Gamma (30-55 Hz)", "HFA (70-100 Hz)")
regions <- c("BLA", "HPC", "CA", "DG")

##############################################################################
# PART 1: ALL POWER TRIALS -> MEMORY
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 1: POWER -> MEMORY (ALL TRIALS)\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

pw_all <- read.csv(file.path(out_dir, "power_all.csv"), stringsAsFactors = FALSE)

for (reg in regions) {
  sub <- pw_all[pw_all$region == reg, ]
  sub$patient_id <- factor(sub$patient_id)

  if (nrow(sub) < 20 || nlevels(sub$patient_id) < 3) {
    cat(sprintf("  Skipping %s: too few data\n", reg))
    next
  }

  cat(sprintf("\n--- %s: %d trials, %d patients ---\n", reg, nrow(sub), nlevels(sub$patient_id)))

  for (b in bands) {
    col <- paste0("pow_", b)
    sub[[paste0(col, "_z")]] <- as.numeric(scale(sub[[col]]))
  }

  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")
    cat(sprintf("\n  MODEL A_%s_%s: %s (All Trials)\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0("memory ~ ", zcol, " + stim + (1 | patient_id)")),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      print(summary(m))
      save_tidy(m, sprintf("pow_A_%s_%s_odds_ratios.csv", reg, b))
    }
  }
}

##############################################################################
# PART 2: POWER x IED TIMING WINDOWS -> MEMORY (IED TRIALS ONLY)
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 2: POWER x IED TIMING WINDOWS (IED TRIALS ONLY)\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

ied_pw <- read.csv(file.path(out_dir, "power_ied_merged.csv"), stringsAsFactors = FALSE)

for (reg in regions) {
  sub <- ied_pw[ied_pw$region == reg, ]
  # Filter patients with >= 5 trials
  pc <- table(sub$patient_id)
  kp <- names(pc)[pc >= 5]
  sub <- sub[sub$patient_id %in% kp, ]
  sub$patient_id <- factor(sub$patient_id)

  if (nrow(sub) < 20 || nlevels(sub$patient_id) < 3) {
    cat(sprintf("  Skipping %s: %d trials, %d patients (insufficient)\n",
                reg, nrow(sub), nlevels(sub$patient_id)))
    next
  }

  cat(sprintf("\n--- %s: %d IED trials, %d patients (>=5 each) ---\n",
              reg, nrow(sub), nlevels(sub$patient_id)))
  cat(sprintf("  Memory rate: %.1f%%\n", mean(sub$memory) * 100))

  cat("  Timing window distribution:\n")
  for (v in c("ied_before_image", "ied_during_image",
              "ied_after_image", "ied_during_stim")) {
    n1 <- sum(sub[[v]])
    cat(sprintf("    %s: %d/%d (%.1f%%)\n", v, n1, nrow(sub),
                n1 / nrow(sub) * 100))
  }

  for (b in bands) {
    col <- paste0("pow_", b)
    sub[[paste0(col, "_z")]] <- as.numeric(scale(sub[[col]]))
  }

  # Part 2a: Main effects with timing windows
  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")

    cat(sprintf("\n  MODEL T_%s_%s: %s + Timing Windows\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " + ied_before_image + ied_during_image + ",
        "ied_after_image + ied_during_stim + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      print(summary(m))
      save_tidy(m, sprintf("pow_T_%s_%s_odds_ratios.csv", reg, b))
    }
  }

  # Part 2b: Interaction models
  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")

    cat(sprintf("\n  MODEL TI_%s_%s: %s x Timing Interactions\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " * ied_before_image + ", zcol, " * ied_during_image + ",
        zcol, " * ied_after_image + ", zcol, " * ied_during_stim + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      print(summary(m))
      save_tidy(m, sprintf("pow_TI_%s_%s_odds_ratios.csv", reg, b))
    }
  }

  # Part 2c: Spread models
  sub$n_regions_c <- as.numeric(scale(sub$n_regions, scale = FALSE))
  sub$n_channels_c <- as.numeric(scale(sub$n_channels, scale = FALSE))

  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("pow_", b, "_z")

    cat(sprintf("\n  MODEL SP_%s_%s: %s + Spread\n", reg, b, bl))
    m <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " + n_regions_c + n_channels_c + (1 | patient_id)"
      )), data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("    Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) {
      print(summary(m))
      save_tidy(m, sprintf("pow_SP_%s_%s_odds_ratios.csv", reg, b))
    }
  }
}

cat("\n\n===== ALL DONE =====\n")
