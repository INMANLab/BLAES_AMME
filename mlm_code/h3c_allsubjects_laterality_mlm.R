##############################################################################
#  Hypothesis 3c (All Subjects) — IED Laterality and Memory
#  All patients included; laterality = L (ref) / Bilateral / R
#  Encoding:
#    Model 1 — Main effects: memory ~ hemi_cat + stim + (1|patient_id)
#    Model 2 — Interactions: memory ~ hemi_cat * stim + (1|patient_id)
#    Model 3 — Bilateral vs Unilateral: memory ~ bilateral_ied + stim + (1|patient_id)
#  Retrieval:
#    Model 4 — Main effects: memory ~ hemi_cat + (1|patient_id)
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
cat("ENCODING PHASE — ALL SUBJECTS\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

enc <- read.csv(file.path(out_dir, "h3c_encoding_all.csv"), stringsAsFactors = FALSE)
enc$patient_id <- factor(enc$patient_id)

cat(sprintf("All patients: %d trials, %d patients\n", nrow(enc), nlevels(enc$patient_id)))
cat("\n--- Encoding: Trial hemisphere distribution ---\n")
print(table(enc$trial_hemisphere))
cat("\n--- Encoding: Memory by hemisphere ---\n")
print(tapply(enc$memory, enc$trial_hemisphere, function(x) {
  sprintf("n=%d, rem=%d (%.1f%%)", length(x), sum(x), mean(x)*100)
}))

# Factor: L = reference, Bilateral first contrast, Right second
enc$hemi_cat <- factor(enc$trial_hemisphere, levels = c("L", "Bilateral", "R"))

##############################################################################
# MODEL 1: Main Effects — Laterality + Stim (Encoding, All Patients)
##############################################################################
cat("\n\n========== MODEL 1: Main Effects (Encoding, All Patients) ==========\n")

m1 <- glmer(
  memory ~ hemi_cat + stim + (1 | patient_id),
  data = enc, family = binomial, control = ctrl
)
cat("\n--- Model 1 summary ---\n")
print(summary(m1))
td1 <- save_tidy(m1, "h3c_allsub_model1_odds_ratios.csv")

##############################################################################
# MODEL 2: Interactions — Laterality x Stim (Encoding, All Patients)
##############################################################################
cat("\n\n========== MODEL 2: Interactions (Encoding, All Patients) ==========\n")

m2 <- glmer(
  memory ~ hemi_cat * stim + (1 | patient_id),
  data = enc, family = binomial, control = ctrl
)
cat("\n--- Model 2 summary ---\n")
print(summary(m2))
td2 <- save_tidy(m2, "h3c_allsub_model2_odds_ratios.csv")

# LRT: Model 1 (main effects) vs Model 2 (interactions)
cat("\n--- LRT: Model 1 vs Model 2 ---\n")
lrt12 <- anova(m1, m2)
print(lrt12)
write.csv(as.data.frame(lrt12), file.path(out_dir, "h3c_allsub_lrt_m1_vs_m2.csv"))

##############################################################################
# MODEL 3: Bilateral vs Unilateral + Stim (Encoding, All Patients)
##############################################################################
cat("\n\n========== MODEL 3: Bilateral vs Unilateral (Encoding) ==========\n")

enc$bilateral_ied <- ifelse(enc$trial_hemisphere == "Bilateral", 1, 0)

m3 <- glmer(
  memory ~ bilateral_ied + stim + (1 | patient_id),
  data = enc, family = binomial, control = ctrl
)
cat("\n--- Model 3 summary ---\n")
print(summary(m3))
td3 <- save_tidy(m3, "h3c_allsub_model3_odds_ratios.csv")

##############################################################################
# MODEL 5: Timing Window on Bilateral Trials (bilateral-IED patients only)
##############################################################################
cat("\n\n========== MODEL 5: Timing Window (Bilateral Trials) ==========\n")

enc_bilat <- enc[enc$trial_hemisphere == "Bilateral", ]
enc_bilat$patient_id <- droplevels(factor(enc_bilat$patient_id))

cat(sprintf("Bilateral trials: %d trials, %d patients\n",
            nrow(enc_bilat), nlevels(enc_bilat$patient_id)))

cat("\n--- Timing window distributions ---\n")
for (v in c("ied_before_image", "ied_during_image",
            "ied_after_image", "ied_during_stim")) {
  n1 <- sum(enc_bilat[[v]])
  cat(sprintf("  %s: %d/%d (%.1f%%)\n", v, n1, nrow(enc_bilat),
              n1 / nrow(enc_bilat) * 100))
}

m5 <- glmer(
  memory ~ ied_before_image + ied_during_image +
           ied_after_image + ied_during_stim + (1 | patient_id),
  data = enc_bilat, family = binomial, control = ctrl
)
cat("\n--- Model 5 summary ---\n")
print(summary(m5))
td5 <- save_tidy(m5, "h3c_allsub_model5_odds_ratios.csv")

##############################################################################
# ENCODING FOREST PLOT (Model 1 main effects)
##############################################################################
td_plot <- tidy(m1, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
td_plot <- td_plot[td_plot$term != "(Intercept)", ]
labels_map <- c("hemi_catBilateral" = "Bilateral IED (vs Left)",
                "hemi_catR" = "Right IED (vs Left)",
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
       title = "Encoding: IED Laterality and Memory (All Subjects)",
       subtitle = "Reference = Left IED; GLMM with random intercept") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(out_dir, "h3c_allsub_encoding_forest.png"), p_enc,
       width = 8, height = 4.5, dpi = 300)

##############################################################################
# ═══════════════════════════ RETRIEVAL PHASE ═══════════════════════════════
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("RETRIEVAL PHASE — ALL SUBJECTS\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

ret <- read.csv(file.path(out_dir, "h3c_retrieval_all.csv"), stringsAsFactors = FALSE)
ret$patient_id <- factor(ret$patient_id)

cat(sprintf("All patients: %d trials, %d patients\n", nrow(ret), nlevels(ret$patient_id)))
cat("\n--- Retrieval: Trial hemisphere distribution ---\n")
print(table(ret$trial_hemisphere))
cat("\n--- Retrieval: Memory by hemisphere ---\n")
print(tapply(ret$memory, ret$trial_hemisphere, function(x) {
  sprintf("n=%d, rem=%d (%.1f%%)", length(x), sum(x), mean(x)*100)
}))

ret$hemi_cat <- factor(ret$trial_hemisphere, levels = c("L", "Bilateral", "R"))

##############################################################################
# MODEL 4: Main Effects — Laterality (Retrieval, All Patients)
##############################################################################
cat("\n\n========== MODEL 4: Main Effects (Retrieval) ==========\n")

m4 <- tryCatch({
  glmer(
    memory ~ hemi_cat + (1 | patient_id),
    data = ret, family = binomial, control = ctrl
  )
}, error = function(e) {
  cat("Model 4 failed:", conditionMessage(e), "\n")
  NULL
})

if (!is.null(m4)) {
  cat("\n--- Model 4 summary ---\n")
  print(summary(m4))
  td4 <- save_tidy(m4, "h3c_allsub_model4_odds_ratios.csv")
}

# Retrieval forest plot
if (!is.null(m4)) {
  td_ret <- tidy(m4, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  td_ret <- td_ret[td_ret$term != "(Intercept)", ]
  ret_labels <- c("hemi_catBilateral" = "Bilateral IED (vs Left)",
                  "hemi_catR" = "Right IED (vs Left)")
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
         title = "Retrieval: IED Laterality and Recall (All Subjects)",
         subtitle = "Reference = Left IED; GLMM with random intercept") +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(out_dir, "h3c_allsub_retrieval_forest.png"), p_ret,
         width = 8, height = 4, dpi = 300)
}

##############################################################################
# Save all summaries
##############################################################################
sink(file.path(out_dir, "h3c_allsub_summaries.txt"))

cat("========== MODEL 1: Main Effects (Encoding, All Patients) ==========\n\n")
print(summary(m1))

cat("\n\n========== MODEL 2: Interactions (Encoding, All Patients) ==========\n\n")
print(summary(m2))
cat("\nLRT: Model 1 vs Model 2:\n")
print(lrt12)

cat("\n\n========== MODEL 3: Bilateral vs Unilateral (Encoding) ==========\n\n")
print(summary(m3))

cat("\n\n========== MODEL 5: Timing Window (Bilateral Trials) ==========\n\n")
print(summary(m5))

if (!is.null(m4)) {
  cat("\n\n========== MODEL 4: Main Effects (Retrieval) ==========\n\n")
  print(summary(m4))
}

sink()

cat("\n\n===== ALL DONE (All-Subjects Laterality) =====\n")
