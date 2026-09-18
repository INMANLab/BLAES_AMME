##############################################################################
# Leave-one-subject-out (LOSO) sensitivity for the HPC-PRC stim-only slow-gamma
# coherence GLMM main effect (Accuracy ~ band_c + (1 | Patient)).
#
# For each subject, refit excluding that subject and record the band_c OR,
# 95% CI, z, and p. Output:
#   OUTPUTS/encoding_memory_reports/stim_effect/
#     HPC_PRC_stim_slow_gamma_loso.csv
##############################################################################

suppressPackageStartupMessages({
  library(lme4); library(broom.mixed); library(dplyr)
})

REPO <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV  <- file.path(REPO, "OUTPUTS", "csvs",
                  "combined_encoding_coherence_all_mlmr_input.csv")
OUT  <- file.path(REPO, "OUTPUTS", "encoding_memory_reports", "stim_effect",
                  "HPC_PRC_stim_slow_gamma_loso.csv")

SG <- c(30.27, 54.69)
SOURCES <- c("HPC_PRC", "CA_PRC", "DG_PRC")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

d <- read.csv(CSV, stringsAsFactors = FALSE, check.names = FALSE)
d <- d[d$yes_or_no %in% c("yes", "no") &
       d$trial_type %in% c("nostim", "stim"), ]
d$Patient[d$Patient == "BJH033"] <- "BJH032"
d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)

freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
sub <- d[d$Region %in% SOURCES, ]
sub <- sub %>% group_by(Patient, Region) %>%
  mutate(trial_id = row_number()) %>% ungroup()
agg <- sub %>%
  group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy) %>%
  summarise(across(all_of(freq_cols), ~ mean(.x, na.rm = TRUE)),
            .groups = "drop") %>%
  select(-trial_id)
agg <- as.data.frame(agg)

freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
sg_cols <- freq_cols[freqs >= SG[1] & freqs <= SG[2]]
agg$slow_gamma <- rowMeans(agg[, sg_cols, drop = FALSE], na.rm = TRUE)

stim <- agg[agg$trial_type == "stim" & is.finite(agg$slow_gamma), ]
stim$Patient <- factor(stim$Patient)
cat(sprintf("Full data: %d trials across %d subjects\n",
            nrow(stim), nlevels(stim$Patient)))

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
write.csv(out, OUT, row.names = FALSE)
cat("\nSaved:", OUT, "\n")
print(out, row.names = FALSE)
