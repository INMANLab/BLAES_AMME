##############################################################################
# Subject-resampling-with-replacement bootstrap for the EC-PRC PAC slow-gamma
# stim-only main-effect GLMM:
#   Accuracy ~ band_c + (1 | Patient), family = binomial
#
# Mirrors the bootstrap done for the PRC LMM-cont report so we can produce
# a parallel 2x2 stability figure for the FDR-significant stim-only finding.
#
# Outputs (to stim_effect/post_hoc_testing/):
#   EC_PRC_pac_stim_bootstrap_slopes.csv     -- one row per bootstrap iter
#   EC_PRC_pac_stim_band_sd.csv              -- SD of slow-gamma PAC values
#                                               (for OR-per-SD conversion)
##############################################################################

suppressPackageStartupMessages({
  library(lme4); library(dplyr)
})

set.seed(42)

REPO <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_IN <- file.path(REPO, "OUTPUTS", "csvs")
OUT <- file.path(REPO, "OUTPUTS", "encoding_memory_reports",
                 "stim_effect", "post_hoc_testing")

PAC_SG <- c(30, 50)
PAIR <- "EC_PRC"
N_BOOT <- 500

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# ── Data prep ──────────────────────────────────────────────────────────────
d <- read.csv(file.path(CSV_IN, "combined_encoding_pac_all_mlmr_input.csv"),
              stringsAsFactors = FALSE, check.names = FALSE)
d <- d[d$yes_or_no %in% c("yes", "no") &
       d$trial_type %in% c("nostim", "stim"), ]
d$Patient[d$Patient == "BJH033"] <- "BJH032"
d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
pair <- d[d$Region == PAIR & d$trial_type == "stim", ]
freq_cols <- grep("^diff_Freq_", names(pair), value = TRUE)
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
sg_cols <- freq_cols[freqs >= PAC_SG[1] & freqs <= PAC_SG[2]]
pair$band_c <- rowMeans(pair[, sg_cols, drop = FALSE], na.rm = TRUE)
pair <- pair[is.finite(pair$band_c), ]
pair$Patient <- factor(pair$Patient)

subjects <- as.character(unique(pair$Patient))
n_subj <- length(subjects)
cat(sprintf("EC-PRC PAC stim-only: %d trials, %d subjects\n",
            nrow(pair), n_subj))

# SD of the band_c predictor across all included trials -- used to convert
# raw beta to "OR per 1 SD increase" downstream.
sd_band <- sd(pair$band_c, na.rm = TRUE)
write.csv(data.frame(sd_band_c = sd_band), file.path(OUT, "EC_PRC_pac_stim_band_sd.csv"),
          row.names = FALSE)
cat(sprintf("SD of slow-gamma PAC values: %.5f\n", sd_band))

# ── Bootstrap helpers ──────────────────────────────────────────────────────
fit_one <- function(data) {
  data$Patient <- droplevels(factor(data$Patient))
  m <- tryCatch(
    suppressWarnings(glmer(Accuracy ~ band_c + (1 | Patient),
                            data = data, family = binomial, control = ctrl)),
    error = function(e) NULL)
  if (is.null(m)) return(list(beta = NA, se = NA, z = NA, p = NA))
  s <- summary(m)$coefficients
  if (!"band_c" %in% rownames(s))
    return(list(beta = NA, se = NA, z = NA, p = NA))
  list(beta = unname(s["band_c", "Estimate"]),
       se   = unname(s["band_c", "Std. Error"]),
       z    = unname(s["band_c", "z value"]),
       p    = unname(s["band_c", "Pr(>|z|)"]))
}

boot_sample <- function(data, subj_ids) {
  picked <- sample(subj_ids, length(subj_ids), replace = TRUE)
  pieces <- list()
  for (i in seq_along(picked)) {
    sub <- data[data$Patient == picked[i], ]
    if (nrow(sub) == 0) next
    sub$Patient <- paste0(picked[i], "_b", i)
    pieces[[length(pieces) + 1]] <- sub
  }
  bd <- do.call(rbind, pieces)
  bd$Patient <- factor(bd$Patient)
  bd
}

# ── Run bootstrap ─────────────────────────────────────────────────────────
boot_path <- file.path(OUT, "EC_PRC_pac_stim_bootstrap_slopes.csv")
rows <- vector("list", N_BOOT)
t0 <- Sys.time()
for (i in seq_len(N_BOOT)) {
  bd <- boot_sample(pair, subjects)
  f  <- fit_one(bd)
  rows[[i]] <- data.frame(
    iter = i, n_trials = nrow(bd),
    beta = f$beta, se = f$se, z = f$z, p = f$p,
    stringsAsFactors = FALSE)
  if (i %% 25 == 0) {
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    eta <- elapsed / i * (N_BOOT - i)
    cat(sprintf("  boot %d/%d  elapsed=%.0fs  ETA=%.0fs\n",
                i, N_BOOT, elapsed, eta))
    write.csv(do.call(rbind, rows[1:i]), boot_path, row.names = FALSE)
  }
}
write.csv(do.call(rbind, rows), boot_path, row.names = FALSE)
cat("Saved bootstrap CSV ->", boot_path, "\n")
