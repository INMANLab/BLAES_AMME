##############################################################################
#  Hypothesis 3a — Mixed-Effects Logistic Regression
#  Model 1: Encoding IED timing windows → memory
#  Model 2: Stim * retrieval IED frequency → memory
##############################################################################

# ── Packages ─────────────────────────────────────────────────────────────────
if (!require("lme4"))      install.packages("lme4",      repos = "https://cloud.r-project.org")
if (!require("lmerTest"))  install.packages("lmerTest",   repos = "https://cloud.r-project.org")
if (!require("ggplot2"))   install.packages("ggplot2",    repos = "https://cloud.r-project.org")
if (!require("sjPlot"))    install.packages("sjPlot",     repos = "https://cloud.r-project.org")
if (!require("effects"))   install.packages("effects",    repos = "https://cloud.r-project.org")
if (!require("broom.mixed")) install.packages("broom.mixed", repos = "https://cloud.r-project.org")

library(lme4)
library(lmerTest)
library(ggplot2)
library(sjPlot)
library(effects)
library(broom.mixed)

# ── Paths (edit if needed) ───────────────────────────────────────────────────
# Detect script directory (works from Rscript and RStudio)
script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    # When run via Rscript, use the command-line arg
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(normalizePath(sub("--file=", "", file_arg)))
    } else {
      getwd()
    }
  }
)
out_dir <- file.path(script_dir, "outputs", "ied_timing_memory")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data_file <- file.path(out_dir, "mlm_encoding_trials.csv")

# ── Load data ────────────────────────────────────────────────────────────────
d <- read.csv(data_file, stringsAsFactors = FALSE)
d$patient_id <- factor(d$patient_id)
d$memory     <- as.integer(d$memory)
d$stim       <- as.integer(d$stim)

cat("=== Data summary ===\n")
cat(sprintf("Trials: %d | Patients: %d | Remembered: %d | Forgotten: %d\n",
            nrow(d), nlevels(d$patient_id), sum(d$memory), sum(1 - d$memory)))
cat(sprintf("Patients with retrieval IED freq: %d\n",
            length(unique(d$patient_id[!is.na(d$retrieval_ied_freq)]))))

##############################################################################
# MODEL 1a: Encoding IED per window → memory  (random intercept)
##############################################################################
cat("\n\n========== MODEL 1a: Encoding IED Windows (random intercept) ==========\n")

m1a <- glmer(
  memory ~ ied_before_image + ied_during_image + ied_after_image + ied_during_stim +
    (1 | patient_id),
  data   = d,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)

cat("\n--- Model 1a summary ---\n")
print(summary(m1a))

# Odds ratios
or_1a <- exp(fixef(m1a))
ci_1a <- exp(confint(m1a, method = "Wald"))
cat("\n--- Model 1a: Odds Ratios ---\n")
print(round(or_1a, 3))
cat("\n--- Model 1a: 95% CI ---\n")
print(round(ci_1a, 3))

# Save tidy output
tidy_1a <- tidy(m1a, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
write.csv(tidy_1a, file.path(out_dir, "mlm_model1a_odds_ratios.csv"), row.names = FALSE)

##############################################################################
# MODEL 1b: Add stim and stim x window interactions  (random intercept)
##############################################################################
cat("\n\n========== MODEL 1b: IED Windows + Stim + Interactions ==========\n")

m1b <- glmer(
  memory ~ (ied_before_image + ied_during_image + ied_after_image + ied_during_stim) * stim +
    (1 | patient_id),
  data   = d,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)

cat("\n--- Model 1b summary ---\n")
print(summary(m1b))

tidy_1b <- tidy(m1b, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
write.csv(tidy_1b, file.path(out_dir, "mlm_model1b_odds_ratios.csv"), row.names = FALSE)

# Compare models
cat("\n--- Model comparison: 1a vs 1b ---\n")
print(anova(m1a, m1b))

##############################################################################
# MODEL 1c: Random slope for ied_after_image (if convergence allows)
##############################################################################
cat("\n\n========== MODEL 1c: Random slope for After Image IED ==========\n")

m1c <- tryCatch({
  glmer(
    memory ~ ied_before_image + ied_during_image + ied_after_image + ied_during_stim +
      (1 + ied_after_image | patient_id),
    data   = d,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
  )
}, warning = function(w) {
  cat("Warning in Model 1c:", conditionMessage(w), "\n")
  return(NULL)
}, error = function(e) {
  cat("Model 1c failed to converge:", conditionMessage(e), "\n")
  return(NULL)
})

if (!is.null(m1c)) {
  cat("\n--- Model 1c summary ---\n")
  print(summary(m1c))

  tidy_1c <- tidy(m1c, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(tidy_1c, file.path(out_dir, "mlm_model1c_odds_ratios.csv"), row.names = FALSE)

  cat("\n--- Model comparison: 1a vs 1c ---\n")
  print(anova(m1a, m1c))
} else {
  cat("Model 1c did not converge. Skipping.\n")
}

##############################################################################
# MODEL 1a Forest Plot
##############################################################################
cat("\n--- Generating forest plot for Model 1a ---\n")
fp_data <- tidy_1a[tidy_1a$term != "(Intercept)", ]
fp_data$term <- c("Before Image", "During Image", "After Image", "During Stim")
fp_data$term <- factor(fp_data$term, levels = rev(fp_data$term))

p_forest <- ggplot(fp_data, aes(x = estimate, y = term)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
  geom_text(aes(label = sprintf("OR = %.2f\np = %s", estimate,
                                ifelse(p.value < .001, "< .001",
                                       sprintf("%.3f", p.value)))),
            hjust = -0.15, size = 3.2) +
  scale_x_continuous(limits = c(0.2, 2.5)) +
  labs(x = "Odds Ratio (95% CI)", y = "",
       title = "Model 1a: Encoding IED Timing Windows",
       subtitle = "Logistic GLMM with random intercept for patient") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(out_dir, "mlm_model1a_forest.png"), p_forest,
       width = 7, height = 4, dpi = 300)
cat("Saved forest plot.\n")


##############################################################################
# MODEL 2: Stim * Retrieval IED Frequency → memory
##############################################################################
cat("\n\n========== MODEL 2: Stim * Retrieval IED Frequency ==========\n")

# Subset to patients with retrieval data
d2 <- d[!is.na(d$retrieval_ied_freq), ]
d2$patient_id <- droplevels(d2$patient_id)

cat(sprintf("Model 2 data: %d trials, %d patients\n", nrow(d2), nlevels(d2$patient_id)))

# Center retrieval IED frequency (grand-mean centering)
d2$ret_ied_freq_c <- d2$retrieval_ied_freq - mean(d2$retrieval_ied_freq)
# Also create standardized version (z-score)
d2$ret_ied_freq_z <- scale(d2$retrieval_ied_freq)[, 1]

cat(sprintf("Retrieval IED freq: M = %.3f, SD = %.3f, range = [%.3f, %.3f]\n",
            mean(d2$retrieval_ied_freq), sd(d2$retrieval_ied_freq),
            min(d2$retrieval_ied_freq), max(d2$retrieval_ied_freq)))

# Model 2a: centered version
m2a <- glmer(
  memory ~ stim * ret_ied_freq_c + (1 | patient_id),
  data   = d2,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)

cat("\n--- Model 2a summary (centered freq) ---\n")
print(summary(m2a))

or_2a <- exp(fixef(m2a))
ci_2a <- exp(confint(m2a, method = "Wald"))
cat("\n--- Model 2a: Odds Ratios ---\n")
print(round(or_2a, 3))
cat("\n--- Model 2a: 95% CI ---\n")
print(round(ci_2a, 3))

tidy_2a <- tidy(m2a, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
write.csv(tidy_2a, file.path(out_dir, "mlm_model2a_odds_ratios.csv"), row.names = FALSE)

# Model 2b: standardized version (for comparison)
m2b <- glmer(
  memory ~ stim * ret_ied_freq_z + (1 | patient_id),
  data   = d2,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)

cat("\n--- Model 2b summary (standardized freq) ---\n")
print(summary(m2b))

tidy_2b <- tidy(m2b, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
write.csv(tidy_2b, file.path(out_dir, "mlm_model2b_odds_ratios.csv"), row.names = FALSE)


##############################################################################
# MODEL 2 Interaction Plot
##############################################################################
cat("\n--- Generating interaction plot for Model 2a ---\n")

# Create prediction grid
freq_vals <- quantile(d2$retrieval_ied_freq, probs = c(0.25, 0.50, 0.75))
freq_c_vals <- freq_vals - mean(d2$retrieval_ied_freq)

newdata <- expand.grid(
  stim = c(0, 1),
  ret_ied_freq_c = freq_c_vals
)
newdata$retrieval_ied_freq <- newdata$ret_ied_freq_c + mean(d2$retrieval_ied_freq)
newdata$freq_label <- factor(
  rep(c("Low (Q1)", "Medium (Q2)", "High (Q3)"), each = 2),
  levels = c("Low (Q1)", "Medium (Q2)", "High (Q3)")
)
newdata$stim_label <- ifelse(newdata$stim == 1, "Stimulation", "No Stimulation")

# Predicted probabilities (population-level, re.form = NA)
newdata$pred <- predict(m2a, newdata = newdata, type = "response", re.form = NA)

# Bootstrap CIs via parametric bootstrap (quick approach)
# Use SE from the linear predictor
lp <- predict(m2a, newdata = newdata, type = "link", re.form = NA, se.fit = TRUE)
# se.fit not available for glmer, use manual approach
mm <- model.matrix(~ stim * ret_ied_freq_c, data = newdata)
pvar <- diag(mm %*% tcrossprod(vcov(m2a), mm))
newdata$pred_se <- sqrt(pvar)
newdata$pred_lo <- plogis(qlogis(newdata$pred) - 1.96 * newdata$pred_se)
newdata$pred_hi <- plogis(qlogis(newdata$pred) + 1.96 * newdata$pred_se)

p_interact <- ggplot(newdata, aes(x = stim_label, y = pred * 100,
                                   color = freq_label, group = freq_label)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = pred_lo * 100, ymax = pred_hi * 100), width = 0.1) +
  scale_color_manual(values = c("#2166ac", "#636363", "#b2182b"),
                     name = "Retrieval IED\nFrequency") +
  labs(x = "", y = "Predicted % Remembered",
       title = "Model 2: Stim x Retrieval IED Frequency Interaction",
       subtitle = "Population-level predictions from GLMM") +
  ylim(40, 100) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "right")

ggsave(file.path(out_dir, "mlm_model2_interaction.png"), p_interact,
       width = 7, height = 5, dpi = 300)
cat("Saved interaction plot.\n")

##############################################################################
# Save all model summaries to text file for the PDF builder
##############################################################################
sink(file.path(out_dir, "mlm_all_summaries.txt"))

cat("========== MODEL 1a: Encoding IED Windows (random intercept) ==========\n\n")
print(summary(m1a))
cat("\nOdds Ratios:\n")
print(round(or_1a, 3))
cat("\n95% CIs:\n")
print(round(ci_1a, 3))

cat("\n\n========== MODEL 1b: IED Windows + Stim + Interactions ==========\n\n")
print(summary(m1b))
cat("\nModel comparison 1a vs 1b:\n")
print(anova(m1a, m1b))

if (!is.null(m1c)) {
  cat("\n\n========== MODEL 1c: Random slope for After Image ==========\n\n")
  print(summary(m1c))
  cat("\nModel comparison 1a vs 1c:\n")
  print(anova(m1a, m1c))
}

cat("\n\n========== MODEL 2a: Stim * Retrieval IED Freq (centered) ==========\n\n")
print(summary(m2a))
cat("\nOdds Ratios:\n")
print(round(or_2a, 3))
cat("\n95% CIs:\n")
print(round(ci_2a, 3))

cat("\n\n========== MODEL 2b: Stim * Retrieval IED Freq (standardized) ==========\n\n")
print(summary(m2b))

sink()
cat("\nSaved all summaries to mlm_all_summaries.txt\n")

cat("\n\n===== ALL DONE =====\n")
