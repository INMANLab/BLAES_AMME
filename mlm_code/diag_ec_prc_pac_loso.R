##############################################################################
# Leave-one-subject-out (LOSO) + subject-delta sensitivity for the EC-PRC
# stim-only slow-gamma PAC GLMM main effect
#   (Accuracy ~ slow_gamma + (1 | Patient), encoding, EC_PRC pair).
#
# PAC slow-gamma band = 30-50 Hz (matches run_h1abc_maineffect.R).
# Output:
#   OUTPUTS/encoding_memory_reports/stim_effect/
#     EC_PRC_stim_slow_gamma_pac_loso.csv
#     EC_PRC_stim_slow_gamma_pac_subject_table.csv
##############################################################################

suppressPackageStartupMessages({
  library(lme4); library(broom.mixed); library(dplyr)
})

REPO <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV  <- file.path(REPO, "OUTPUTS", "csvs",
                  "combined_encoding_pac_all_mlmr_input.csv")
OUT  <- file.path(REPO, "OUTPUTS", "encoding_memory_reports", "stim_effect")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

LOSO_CSV  <- file.path(OUT, "EC_PRC_stim_slow_gamma_pac_loso.csv")
SUBJ_CSV  <- file.path(OUT, "EC_PRC_stim_slow_gamma_pac_subject_table.csv")

PAC_SG <- c(30, 50)
PAIR <- "EC_PRC"
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

d <- read.csv(CSV, stringsAsFactors = FALSE, check.names = FALSE)
d <- d[d$yes_or_no %in% c("yes", "no") &
       d$trial_type %in% c("nostim", "stim"), ]
d$Patient[d$Patient == "BJH033"] <- "BJH032"
d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)

freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
sg_cols <- freq_cols[freqs >= PAC_SG[1] & freqs <= PAC_SG[2]]

pair <- d[d$Region == PAIR, ]
pair$slow_gamma <- rowMeans(pair[, sg_cols, drop = FALSE], na.rm = TRUE)
stim <- pair[pair$trial_type == "stim" & is.finite(pair$slow_gamma), ]
stim$Patient <- factor(stim$Patient)

cat(sprintf("EC_PRC stim PAC: %d trials, %d subjects\n",
            nrow(stim), nlevels(stim$Patient)))

# Per-subject delta table (mean slow_gamma | remembered) - (mean slow_gamma | forgotten).
per_subj <- stim %>%
  group_by(Patient, yes_or_no) %>%
  summarise(slow_gamma_mean = mean(slow_gamma), n = n(), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = yes_or_no,
                     values_from = c(slow_gamma_mean, n))
names(per_subj) <- sub("^slow_gamma_mean_", "", names(per_subj))
names(per_subj) <- sub("^n_", "n_", names(per_subj))
per_subj$delta <- per_subj$yes - per_subj$no
per_subj <- per_subj[order(per_subj$delta), ]
write.csv(per_subj, SUBJ_CSV, row.names = FALSE)
cat("Saved subject table:", SUBJ_CSV, "\n")
print(per_subj)

# LOSO refits
fit_one <- function(data, label, n_excluded) {
  data$Patient <- droplevels(factor(data$Patient))
  m <- tryCatch(
    glmer(Accuracy ~ slow_gamma + (1 | Patient),
          data = data, family = binomial, control = ctrl),
    error = function(e) { cat("FAILED:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(m)) {
    return(data.frame(excluded = label, n_trials = nrow(data),
                      n_subjects = nlevels(data$Patient),
                      n_trials_excluded = n_excluded,
                      OR = NA, lo = NA, hi = NA, z = NA, p = NA))
  }
  td <- tidy(m, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  r <- td[td$term == "slow_gamma", ]
  data.frame(
    excluded = label, n_trials = nrow(data),
    n_subjects = nlevels(data$Patient),
    n_trials_excluded = n_excluded,
    OR = r$estimate, lo = r$conf.low, hi = r$conf.high,
    z = r$statistic, p = r$p.value
  )
}

rows <- list()
rows[[length(rows) + 1]] <- fit_one(stim, "<none>", 0)

subjects <- levels(stim$Patient)
for (s in subjects) {
  data_s <- stim[stim$Patient != s, ]
  n_excluded <- nrow(stim) - nrow(data_s)
  rows[[length(rows) + 1]] <- fit_one(data_s, s, n_excluded)
  cat(sprintf("  -%s: n_excluded=%d  OR=%.2f  p=%.3f\n",
              s, n_excluded,
              tail(rows, 1)[[1]]$OR, tail(rows, 1)[[1]]$p))
}

out <- do.call(rbind, rows)
write.csv(out, LOSO_CSV, row.names = FALSE)
cat("\nSaved LOSO:", LOSO_CSV, "\n")
print(out, row.names = FALSE)
