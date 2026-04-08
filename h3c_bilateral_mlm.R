##############################################################################
#  Hypothesis 3c — Bilateral IED Laterality and Memory
#  Encoding: Models A (bilateral IED x stim), B (L/R/Bilateral x stim)
#  Retrieval: Models C (bilateral IED), D (L/R/Bilateral)
##############################################################################

if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
if (!require("ggplot2"))     install.packages("ggplot2",      repos = "https://cloud.r-project.org")
if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")

library(lme4)
library(lmerTest)
library(ggplot2)
library(broom.mixed)

# ── Paths ────────────────────────────────────────────────────────────────────
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

##############################################################################
# ═══════════════════════════ ENCODING PHASE ════════════════════════════════
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("ENCODING PHASE\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

enc_bi <- read.csv(file.path(out_dir, "h3c_encoding_bilateral.csv"), stringsAsFactors = FALSE)
enc_all <- read.csv(file.path(out_dir, "h3c_encoding_all.csv"), stringsAsFactors = FALSE)

enc_bi$patient_id <- factor(enc_bi$patient_id)
enc_all$patient_id <- factor(enc_all$patient_id)

cat(sprintf("Bilateral subset: %d trials, %d patients\n", nrow(enc_bi), nlevels(enc_bi$patient_id)))
cat(sprintf("All patients: %d trials, %d patients\n", nrow(enc_all), nlevels(enc_all$patient_id)))

# Hemisphere distribution
cat("\n--- Encoding: Trial hemisphere distribution (bilateral patients) ---\n")
print(table(enc_bi$trial_hemisphere))
cat("\n--- Encoding: Memory by hemisphere ---\n")
print(tapply(enc_bi$memory, enc_bi$trial_hemisphere, function(x) {
  sprintf("n=%d, rem=%d (%.1f%%)", length(x), sum(x), mean(x)*100)
}))

##############################################################################
# MODEL A: Bilateral IED x Stim (bilateral patients only)
##############################################################################
cat("\n\n========== MODEL A: Bilateral IED x Stim (Encoding) ==========\n")

mA <- glmer(
  memory ~ bilateral_ied * stim + (1 | patient_id),
  data = enc_bi, family = binomial, control = ctrl
)
cat("\n--- Model A summary ---\n")
print(summary(mA))
tdA <- save_tidy(mA, "h3c_modelA_odds_ratios.csv")

# Model A without interaction for comparison
mA0 <- glmer(
  memory ~ bilateral_ied + stim + (1 | patient_id),
  data = enc_bi, family = binomial, control = ctrl
)
cat("\n--- Model A: LRT for interaction ---\n")
print(anova(mA0, mA))

##############################################################################
# MODEL B: Full Laterality (L/R/Bilateral) x Stim (all patients)
##############################################################################
cat("\n\n========== MODEL B: Laterality x Stim (Encoding, All Patients) ==========\n")

# Create factor: L as reference (all trials have IEDs; no "None" category)
enc_all$hemi_cat <- factor(enc_all$trial_hemisphere,
                            levels = c("L", "R", "Bilateral"))

# Model with laterality and stim
mB <- glmer(
  memory ~ hemi_cat * stim + (1 | patient_id),
  data = enc_all, family = binomial, control = ctrl
)
cat("\n--- Model B summary ---\n")
print(summary(mB))
tdB <- save_tidy(mB, "h3c_modelB_odds_ratios.csv")

# Model without interactions
mB0 <- glmer(
  memory ~ hemi_cat + stim + (1 | patient_id),
  data = enc_all, family = binomial, control = ctrl
)
cat("\n--- Model B: LRT for interactions ---\n")
print(anova(mB0, mB))

# Model B2: L vs R contrast (exclude None and Bilateral for direct comparison)
enc_lr <- enc_all[enc_all$trial_hemisphere %in% c("L", "R"), ]
enc_lr$patient_id <- droplevels(factor(enc_lr$patient_id))
enc_lr$hemi_R <- ifelse(enc_lr$trial_hemisphere == "R", 1, 0)

cat("\n\n========== MODEL B2: L vs R Direct Comparison (Encoding) ==========\n")
cat(sprintf("L vs R subset: %d trials, %d patients\n", nrow(enc_lr), nlevels(enc_lr$patient_id)))

mB2 <- glmer(
  memory ~ hemi_R * stim + (1 | patient_id),
  data = enc_lr, family = binomial, control = ctrl
)
cat("\n--- Model B2 summary ---\n")
print(summary(mB2))
tdB2 <- save_tidy(mB2, "h3c_modelB2_odds_ratios.csv")

##############################################################################
# ENCODING FOREST PLOT
##############################################################################
# Model B main effects (no interactions, cleaner)
# Note: reference is "None" (no IED), so hemi_catL is the first non-intercept term
td_plot <- tidy(mB0, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
td_plot <- td_plot[td_plot$term != "(Intercept)", ]
labels_map <- c("hemi_catR" = "Right IED (vs Left)",
                "hemi_catBilateral" = "Bilateral IED (vs Left)",
                "stim" = "Stimulation")
td_plot$label <- labels_map[td_plot$term]
td_plot$label <- factor(td_plot$label, levels = rev(unname(labels_map)))

p_enc <- ggplot(td_plot, aes(x = estimate, y = label)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(size = 3) +
  geom_errorbar(aes(xmin = conf.low, xmax = conf.high), width = 0.2,
                orientation = "y") +
  geom_text(aes(label = sprintf("OR=%.2f, p=%s", estimate,
                                ifelse(p.value < .001, "<.001",
                                       sprintf("%.3f", p.value)))),
            hjust = -0.1, size = 3) +
  scale_x_continuous(limits = c(0.1, 3.5)) +
  labs(x = "Odds Ratio (95% CI)", y = "",
       title = "Encoding: IED Laterality and Memory",
       subtitle = "Reference = No IED; GLMM with random intercept") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(out_dir, "h3c_encoding_forest.png"), p_enc,
       width = 8, height = 4.5, dpi = 300)

##############################################################################
# ═══════════════════════════ RETRIEVAL PHASE ═══════════════════════════════
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("RETRIEVAL PHASE\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

ret_bi <- read.csv(file.path(out_dir, "h3c_retrieval_bilateral.csv"), stringsAsFactors = FALSE)
ret_all <- read.csv(file.path(out_dir, "h3c_retrieval_all.csv"), stringsAsFactors = FALSE)

ret_bi$patient_id <- factor(ret_bi$patient_id)
ret_all$patient_id <- factor(ret_all$patient_id)

cat(sprintf("Bilateral subset: %d trials, %d patients\n", nrow(ret_bi), nlevels(ret_bi$patient_id)))
cat(sprintf("All patients: %d trials, %d patients\n", nrow(ret_all), nlevels(ret_all$patient_id)))

cat("\n--- Retrieval: Trial hemisphere distribution (bilateral patients) ---\n")
print(table(ret_bi$trial_hemisphere))
cat("\n--- Retrieval: Memory by hemisphere ---\n")
print(tapply(ret_bi$memory, ret_bi$trial_hemisphere, function(x) {
  sprintf("n=%d, rem=%d (%.1f%%)", length(x), sum(x), mean(x)*100)
}))

##############################################################################
# MODEL C: Bilateral IED (retrieval, bilateral patients)
##############################################################################
cat("\n\n========== MODEL C: Bilateral IED (Retrieval) ==========\n")

# Note: retrieval has no stim condition per trial, but we can test bilateral IED effect
mC <- tryCatch({
  glmer(
    memory ~ bilateral_ied + (1 | patient_id),
    data = ret_bi, family = binomial, control = ctrl
  )
}, error = function(e) {
  cat("Model C failed:", conditionMessage(e), "\n")
  NULL
})

if (!is.null(mC)) {
  cat("\n--- Model C summary ---\n")
  print(summary(mC))
  tdC <- save_tidy(mC, "h3c_modelC_odds_ratios.csv")
}

##############################################################################
# MODEL D: Full Laterality (retrieval, all patients)
##############################################################################
cat("\n\n========== MODEL D: Laterality (Retrieval, All Patients) ==========\n")

ret_all$hemi_cat <- factor(ret_all$trial_hemisphere,
                            levels = c("L", "R", "Bilateral"))

mD <- tryCatch({
  glmer(
    memory ~ hemi_cat + (1 | patient_id),
    data = ret_all, family = binomial, control = ctrl
  )
}, error = function(e) {
  cat("Model D failed:", conditionMessage(e), "\n")
  NULL
})

if (!is.null(mD)) {
  cat("\n--- Model D summary ---\n")
  print(summary(mD))
  tdD <- save_tidy(mD, "h3c_modelD_odds_ratios.csv")
}

# Model D2: L vs R (retrieval)
ret_lr <- ret_all[ret_all$trial_hemisphere %in% c("L", "R"), ]
ret_lr$patient_id <- droplevels(factor(ret_lr$patient_id))
ret_lr$hemi_R <- ifelse(ret_lr$trial_hemisphere == "R", 1, 0)

if (nrow(ret_lr) > 10 && nlevels(ret_lr$patient_id) > 2) {
  cat("\n\n========== MODEL D2: L vs R Direct Comparison (Retrieval) ==========\n")
  cat(sprintf("L vs R subset: %d trials, %d patients\n", nrow(ret_lr), nlevels(ret_lr$patient_id)))

  mD2 <- tryCatch({
    glmer(
      memory ~ hemi_R + (1 | patient_id),
      data = ret_lr, family = binomial, control = ctrl
    )
  }, error = function(e) {
    cat("Model D2 failed:", conditionMessage(e), "\n")
    NULL
  })

  if (!is.null(mD2)) {
    cat("\n--- Model D2 summary ---\n")
    print(summary(mD2))
    tdD2 <- save_tidy(mD2, "h3c_modelD2_odds_ratios.csv")
  }
}

# Retrieval forest plot (if Model D converged)
if (!is.null(mD)) {
  td_ret <- tidy(mD, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  td_ret <- td_ret[td_ret$term != "(Intercept)", ]
  ret_labels <- c("hemi_catR" = "Right IED (vs Left)",
                  "hemi_catBilateral" = "Bilateral IED (vs Left)")
  td_ret$label <- ret_labels[td_ret$term]
  td_ret$label <- factor(td_ret$label, levels = rev(unname(ret_labels)))

  p_ret <- ggplot(td_ret, aes(x = estimate, y = label)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    geom_point(size = 3) +
    geom_errorbar(aes(xmin = conf.low, xmax = conf.high), width = 0.2,
                  orientation = "y") +
    geom_text(aes(label = sprintf("OR=%.2f, p=%s", estimate,
                                  ifelse(p.value < .001, "<.001",
                                         sprintf("%.3f", p.value)))),
              hjust = -0.1, size = 3) +
    scale_x_continuous(limits = c(0.05, 5)) +
    labs(x = "Odds Ratio (95% CI)", y = "",
         title = "Retrieval: IED Laterality and Recall",
         subtitle = "Reference = No IED; GLMM with random intercept") +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(out_dir, "h3c_retrieval_forest.png"), p_ret,
         width = 8, height = 4, dpi = 300)
}

##############################################################################
# Save all summaries
##############################################################################
sink(file.path(out_dir, "h3c_all_summaries.txt"))

cat("========== MODEL A: Bilateral IED x Stim (Encoding) ==========\n\n")
print(summary(mA))
cat("\nLRT for interaction:\n")
print(anova(mA0, mA))

cat("\n\n========== MODEL B: Laterality x Stim (Encoding) ==========\n\n")
print(summary(mB))
cat("\nLRT for interactions:\n")
print(anova(mB0, mB))

cat("\n\n========== MODEL B2: L vs R (Encoding) ==========\n\n")
print(summary(mB2))

if (!is.null(mC)) {
  cat("\n\n========== MODEL C: Bilateral IED (Retrieval) ==========\n\n")
  print(summary(mC))
}

if (!is.null(mD)) {
  cat("\n\n========== MODEL D: Laterality (Retrieval) ==========\n\n")
  print(summary(mD))
}

sink()

cat("\n\n===== ALL DONE =====\n")
