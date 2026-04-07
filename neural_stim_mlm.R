##############################################################################
#  Neural and Stim Effects MLM - Encoding Only
#
#  COHERENCE:
#  Part 1: BLA-HPC Coherence -> Memory (all coherence trials)
#    Models C1-C3: memory ~ coh_band + stim + (1|patient_id)
#  Part 2: Coherence x IED Timing Windows -> Memory (IED+coh trials)
#    Models T1-T3: + timing windows; TI1-TI3: + interactions
#  Part 3: During-Stim IED Trials (stim only, coherence -> memory)
#    Models DS1-DS3: memory ~ coh_band + (1|patient_id)
#  Part 4: With IED Spread
#    Models SP1-SP3: + spread; SPT1-SPT3: + spread + timing
#
#  PAC:
#  Part 5: BLA-HPC PAC -> Memory (all PAC trials)
#    Models P1-P2: memory ~ pac_band + stim + (1|patient_id)
#  Part 6: PAC x IED Timing Windows -> Memory
#    Models PT1-PT2: + timing windows; PTI1-PTI2: + interactions
#  Part 7: PAC + IED Spread
#    Models PSP1-PSP2: + spread; PSPT1-PSPT2: + spread + timing
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

# Region pair from command line (default: BLA_HPC)
cli_args <- commandArgs(trailingOnly = TRUE)
REGION_PAIR <- if (length(cli_args) >= 1) cli_args[1] else "BLA_HPC"
# File prefix for non-default pairs
FILE_PREFIX <- if (REGION_PAIR == "BLA_HPC") "ns_model" else paste0("ns_", REGION_PAIR, "_model")

cat(sprintf("\n>>> Running models for region pair: %s\n", REGION_PAIR))
cat(sprintf(">>> File prefix: %s\n\n", FILE_PREFIX))

save_tidy <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  return(td)
}

bands <- c("theta", "slow_gamma", "hfa")
band_labels <- c("Theta (4-8 Hz)", "Slow Gamma (30-55 Hz)", "HFA (70-100 Hz)")
pac_bands <- c("slow_gamma", "hfa")
pac_band_labels <- c("Slow Gamma (30-55 Hz)", "HFA (70-100 Hz)")

##############################################################################
# PART 1: COHERENCE -> MEMORY (ALL TRIALS)
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("PART 1: %s COHERENCE -> MEMORY (ALL TRIALS)\n", REGION_PAIR))
cat(paste(rep("=", 70), collapse=""), "\n\n")

coh_all <- read.csv(file.path(out_dir, "neural_stim_coh_all.csv"),
                     stringsAsFactors = FALSE)
coh_bla <- coh_all[coh_all$region_pair == REGION_PAIR, ]
coh_bla$patient_id <- factor(coh_bla$patient_id)

cat(sprintf("All %s trials: %d trials, %d patients\n", REGION_PAIR,
            nrow(coh_bla), nlevels(coh_bla$patient_id)))
cat(sprintf("Memory rate: %.1f%% remembered\n", mean(coh_bla$memory) * 100))

for (b in bands) {
  col <- paste0("coh_", b)
  coh_bla[[paste0(col, "_z")]] <- as.numeric(scale(coh_bla[[col]]))
}

for (i in seq_along(bands)) {
  b <- bands[i]
  bl <- band_labels[i]
  zcol <- paste0("coh_", b, "_z")
  cat(sprintf("\n========== MODEL C%d: %s (All Trials) ==========\n", i, bl))
  m <- glmer(
    as.formula(paste0("memory ~ ", zcol, " + stim + (1 | patient_id)")),
    data = coh_bla, family = binomial, control = ctrl
  )
  print(summary(m))
  save_tidy(m, sprintf("%sC%d_%s_odds_ratios.csv", FILE_PREFIX, i, b))
}

##############################################################################
# PART 2: COHERENCE x IED TIMING WINDOWS -> MEMORY
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 2: COHERENCE x IED TIMING WINDOWS -> MEMORY\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

ied_coh <- read.csv(file.path(out_dir, "neural_stim_coh_ied_merged.csv"),
                     stringsAsFactors = FALSE)
ied_bla <- ied_coh[ied_coh$region_pair == REGION_PAIR, ]
pat_counts <- table(ied_bla$patient_id)
keep_pats <- names(pat_counts)[pat_counts >= 5]
ied_bla <- ied_bla[ied_bla$patient_id %in% keep_pats, ]
ied_bla$patient_id <- factor(ied_bla$patient_id)

cat(sprintf("IED+Coherence %s: %d trials, %d patients (>=5 trials each)\n", REGION_PAIR,
            nrow(ied_bla), nlevels(ied_bla$patient_id)))

cat("\n--- IED Timing Window Distributions ---\n")
for (v in c("ied_before_image", "ied_during_image",
            "ied_after_image", "ied_during_stim")) {
  n1 <- sum(ied_bla[[v]])
  cat(sprintf("  %s: %d/%d (%.1f%%)\n", v, n1, nrow(ied_bla),
              n1 / nrow(ied_bla) * 100))
}

for (b in bands) {
  col <- paste0("coh_", b)
  ied_bla[[paste0(col, "_z")]] <- as.numeric(scale(ied_bla[[col]]))
}

# Part 2a: Main effects
for (i in seq_along(bands)) {
  b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("coh_", b, "_z")
  cat(sprintf("\n========== MODEL T%d: %s + Timing Windows ==========\n", i, bl))
  m <- tryCatch({
    glmer(as.formula(paste0(
      "memory ~ ", zcol, " + ied_before_image + ied_during_image + ",
      "ied_after_image + ied_during_stim + (1 | patient_id)"
    )), data = ied_bla, family = binomial, control = ctrl)
  }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
  if (!is.null(m)) { print(summary(m)); save_tidy(m, sprintf("%sT%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }
}

# Part 2b: Interaction models
for (i in seq_along(bands)) {
  b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("coh_", b, "_z")
  cat(sprintf("\n========== MODEL TI%d: %s x Timing Interactions ==========\n", i, bl))
  m <- tryCatch({
    glmer(as.formula(paste0(
      "memory ~ ", zcol, " * ied_before_image + ", zcol, " * ied_during_image + ",
      zcol, " * ied_after_image + ", zcol, " * ied_during_stim + (1 | patient_id)"
    )), data = ied_bla, family = binomial, control = ctrl)
  }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
  if (!is.null(m)) { print(summary(m)); save_tidy(m, sprintf("%sTI%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }
}

##############################################################################
# PART 3: DURING-STIM IED TRIALS (stim trials only, coherence -> memory)
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 3: DURING-STIM IED TRIALS (STIM ONLY)\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

# Keep only stim trials with during-stim IEDs
ds_bla <- ied_bla[ied_bla$ied_during_stim == 1 & ied_bla$stim == 1, ]
ds_bla$patient_id <- droplevels(ds_bla$patient_id)

cat(sprintf("During-stim IED trials (stim only): %d trials, %d patients\n",
            nrow(ds_bla), nlevels(ds_bla$patient_id)))
cat(sprintf("  Remembered: %d (%.1f%%)\n",
            sum(ds_bla$memory), mean(ds_bla$memory) * 100))

if (nrow(ds_bla) >= 20 && nlevels(ds_bla$patient_id) >= 3) {
  for (b in bands) {
    col <- paste0("coh_", b)
    ds_bla[[paste0(col, "_z")]] <- as.numeric(scale(ds_bla[[col]]))
  }
  for (i in seq_along(bands)) {
    b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("coh_", b, "_z")
    cat(sprintf("\n========== MODEL DS%d: %s (During-Stim, Stim Only) ==========\n", i, bl))
    m <- tryCatch({
      glmer(as.formula(paste0("memory ~ ", zcol, " + (1 | patient_id)")),
            data = ds_bla, family = binomial, control = ctrl)
    }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) { print(summary(m)); save_tidy(m, sprintf("%sDS%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }
  }
} else {
  cat("  Too few during-stim IED trials for modeling.\n")
}

##############################################################################
# PART 4: COHERENCE + IED SPREAD -> MEMORY
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 4: COHERENCE + IED SPREAD -> MEMORY\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

ied_bla$n_regions_c <- as.numeric(scale(ied_bla$n_regions, scale = FALSE))
ied_bla$n_channels_c <- as.numeric(scale(ied_bla$n_channels, scale = FALSE))

for (i in seq_along(bands)) {
  b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("coh_", b, "_z")
  cat(sprintf("\n========== MODEL SP%d: %s + Spread ==========\n", i, bl))
  m <- tryCatch({
    glmer(as.formula(paste0(
      "memory ~ ", zcol, " + n_regions_c + n_channels_c + (1 | patient_id)"
    )), data = ied_bla, family = binomial, control = ctrl)
  }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
  if (!is.null(m)) { print(summary(m)); save_tidy(m, sprintf("%sSP%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }
}

for (i in seq_along(bands)) {
  b <- bands[i]; bl <- band_labels[i]; zcol <- paste0("coh_", b, "_z")
  cat(sprintf("\n========== MODEL SPT%d: %s + Spread + Timing ==========\n", i, bl))
  m <- tryCatch({
    glmer(as.formula(paste0(
      "memory ~ ", zcol, " + n_regions_c + n_channels_c + ",
      "ied_before_image + ied_during_image + ied_after_image + ",
      "ied_during_stim + (1 | patient_id)"
    )), data = ied_bla, family = binomial, control = ctrl)
  }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
  if (!is.null(m)) { print(summary(m)); save_tidy(m, sprintf("%sSPT%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }
}

##############################################################################
# PART 5: PAC -> MEMORY (ALL TRIALS)
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("PART 5: %s PAC -> MEMORY (ALL TRIALS)\n", REGION_PAIR))
cat(paste(rep("=", 70), collapse=""), "\n\n")

pac_file <- file.path(out_dir, "neural_stim_pac_all.csv")
if (file.exists(pac_file)) {
  pac_all <- read.csv(pac_file, stringsAsFactors = FALSE)
  pac_bla <- pac_all[pac_all$region_pair == REGION_PAIR, ]
  pac_bla$patient_id <- factor(pac_bla$patient_id)

  cat(sprintf("All %s PAC trials: %d trials, %d patients\n", REGION_PAIR,
              nrow(pac_bla), nlevels(pac_bla$patient_id)))

  for (b in pac_bands) {
    col <- paste0("pac_", b)
    if (col %in% colnames(pac_bla)) {
      pac_bla[[paste0(col, "_z")]] <- as.numeric(scale(pac_bla[[col]]))
    }
  }

  for (i in seq_along(pac_bands)) {
    b <- pac_bands[i]; bl <- pac_band_labels[i]
    zcol <- paste0("pac_", b, "_z")
    if (!(zcol %in% colnames(pac_bla))) next
    cat(sprintf("\n========== MODEL P%d: PAC %s (All Trials) ==========\n", i, bl))
    m <- tryCatch({
      glmer(as.formula(paste0("memory ~ ", zcol, " + stim + (1 | patient_id)")),
            data = pac_bla, family = binomial, control = ctrl)
    }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) { print(summary(m)); save_tidy(m, sprintf("%sP%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }
  }
} else {
  cat("  PAC all-trials file not found, skipping.\n")
}

##############################################################################
# PART 6: PAC x IED TIMING WINDOWS -> MEMORY
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 6: PAC x IED TIMING WINDOWS -> MEMORY\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

pac_ied_file <- file.path(out_dir, "neural_stim_pac_ied_merged.csv")
if (file.exists(pac_ied_file)) {
  ied_pac <- read.csv(pac_ied_file, stringsAsFactors = FALSE)
  ied_pac_bla <- ied_pac[ied_pac$region_pair == REGION_PAIR, ]
  # Filter patients with >= 5 trials
  pc <- table(ied_pac_bla$patient_id)
  kp <- names(pc)[pc >= 5]
  ied_pac_bla <- ied_pac_bla[ied_pac_bla$patient_id %in% kp, ]
  ied_pac_bla$patient_id <- factor(ied_pac_bla$patient_id)

  cat(sprintf("IED+PAC %s: %d trials, %d patients (>=5 trials each)\n", REGION_PAIR,
              nrow(ied_pac_bla), nlevels(ied_pac_bla$patient_id)))

  if (nrow(ied_pac_bla) >= 20 && nlevels(ied_pac_bla$patient_id) >= 3) {
    for (b in pac_bands) {
      col <- paste0("pac_", b)
      if (col %in% colnames(ied_pac_bla)) {
        ied_pac_bla[[paste0(col, "_z")]] <- as.numeric(scale(ied_pac_bla[[col]]))
      }
    }

    # Part 6a: Main effects
    for (i in seq_along(pac_bands)) {
      b <- pac_bands[i]; bl <- pac_band_labels[i]
      zcol <- paste0("pac_", b, "_z")
      if (!(zcol %in% colnames(ied_pac_bla))) next
      cat(sprintf("\n========== MODEL PT%d: PAC %s + Timing Windows ==========\n", i, bl))
      m <- tryCatch({
        glmer(as.formula(paste0(
          "memory ~ ", zcol, " + ied_before_image + ied_during_image + ",
          "ied_after_image + ied_during_stim + (1 | patient_id)"
        )), data = ied_pac_bla, family = binomial, control = ctrl)
      }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
      if (!is.null(m)) { print(summary(m)); save_tidy(m, sprintf("%sPT%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }
    }

    # Part 6b: Interaction models
    for (i in seq_along(pac_bands)) {
      b <- pac_bands[i]; bl <- pac_band_labels[i]
      zcol <- paste0("pac_", b, "_z")
      if (!(zcol %in% colnames(ied_pac_bla))) next
      cat(sprintf("\n========== MODEL PTI%d: PAC %s x Timing Interactions ==========\n", i, bl))
      m <- tryCatch({
        glmer(as.formula(paste0(
          "memory ~ ", zcol, " * ied_before_image + ", zcol, " * ied_during_image + ",
          zcol, " * ied_after_image + ", zcol, " * ied_during_stim + (1 | patient_id)"
        )), data = ied_pac_bla, family = binomial, control = ctrl)
      }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
      if (!is.null(m)) { print(summary(m)); save_tidy(m, sprintf("%sPTI%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }
    }
  } else {
    cat("  Too few IED+PAC trials for timing window models.\n")
  }
} else {
  cat("  PAC IED-merged file not found, skipping.\n")
}

##############################################################################
# PART 7: PAC + IED SPREAD -> MEMORY
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 7: PAC + IED SPREAD -> MEMORY\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

if (exists("ied_pac_bla") && nrow(ied_pac_bla) >= 20 && nlevels(ied_pac_bla$patient_id) >= 3) {
  ied_pac_bla$n_regions_c <- as.numeric(scale(ied_pac_bla$n_regions, scale = FALSE))
  ied_pac_bla$n_channels_c <- as.numeric(scale(ied_pac_bla$n_channels, scale = FALSE))

  for (i in seq_along(pac_bands)) {
    b <- pac_bands[i]; bl <- pac_band_labels[i]
    zcol <- paste0("pac_", b, "_z")
    if (!(zcol %in% colnames(ied_pac_bla))) next

    cat(sprintf("\n========== MODEL PSP%d: PAC %s + Spread ==========\n", i, bl))
    m <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " + n_regions_c + n_channels_c + (1 | patient_id)"
      )), data = ied_pac_bla, family = binomial, control = ctrl)
    }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m)) { print(summary(m)); save_tidy(m, sprintf("%sPSP%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }

    cat(sprintf("\n========== MODEL PSPT%d: PAC %s + Spread + Timing ==========\n", i, bl))
    m2 <- tryCatch({
      glmer(as.formula(paste0(
        "memory ~ ", zcol, " + n_regions_c + n_channels_c + ",
        "ied_before_image + ied_during_image + ied_after_image + ",
        "ied_during_stim + (1 | patient_id)"
      )), data = ied_pac_bla, family = binomial, control = ctrl)
    }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(m2)) { print(summary(m2)); save_tidy(m2, sprintf("%sPSPT%d_%s_odds_ratios.csv", FILE_PREFIX, i, b)) }
  }
} else {
  cat("  Too few IED+PAC trials for spread models.\n")
}

##############################################################################
# Save summaries
##############################################################################
sink(file.path(out_dir, "neural_stim_all_summaries.txt"))
cat("Neural and Stim Effects MLM - Encoding Only\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")
cat("See ns_model*_odds_ratios.csv files for detailed results.\n")
sink()

cat("\n\n===== ALL DONE =====\n")
