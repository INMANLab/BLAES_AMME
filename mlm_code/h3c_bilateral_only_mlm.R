##############################################################################
#  Hypothesis 3c (simplified) — Bilateral IED occurrence and memory
#  Single predictor: bilateral_ied (1 = IEDs in both hemispheres on a trial,
#  0 = unilateral). No L/R hemisphere terms. Bilateral-coverage patients only.
#  Encoding: memory ~ bilateral_ied + (1 | patient_id)
#  Encoding robustness: + stim covariate ("regardless of BLA stimulation")
#  Retrieval: memory ~ bilateral_ied + (1 | patient_id)
##############################################################################

suppressMessages({
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
})

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
# Canonical data/output dir (data + PDF live here after the topic-dir reorg)
out_dir <- normalizePath(file.path(script_dir, "..", "IED", "ied_timing_memory"))

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

save_tidy <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  td
}

##############################################################################
# ENCODING
##############################################################################
enc_bi <- read.csv(file.path(out_dir, "h3c_encoding_bilateral.csv"),
                   stringsAsFactors = FALSE)
enc_bi$patient_id <- factor(enc_bi$patient_id)

cat("\n===== ENCODING =====\n")
cat(sprintf("Bilateral-coverage subset: %d trials, %d patients\n",
            nrow(enc_bi), nlevels(enc_bi$patient_id)))
n_bi_trials <- sum(enc_bi$bilateral_ied == 1)
n_bi_pats   <- length(unique(enc_bi$patient_id[enc_bi$bilateral_ied == 1]))
cat(sprintf("Bilateral-IED trials: %d  (from %d patients)\n", n_bi_trials, n_bi_pats))
cat("\n--- Memory by bilateral_ied ---\n")
print(tapply(enc_bi$memory, enc_bi$bilateral_ied, function(x)
  sprintf("n=%d, rem=%d (%.1f%%)", length(x), sum(x), mean(x) * 100)))

# Primary model
mE <- glmer(memory ~ bilateral_ied + (1 | patient_id),
            data = enc_bi, family = binomial, control = ctrl)
cat("\n--- Encoding primary: memory ~ bilateral_ied ---\n")
print(summary(mE))
save_tidy(mE, "h3c_enc_bilonly_or.csv")

# ── Hierarchical entry: stimulation first, then add bilateral IED ────────────
# Step 1: stim only. Step 2: + bilateral_ied. LRT tests the increment from
# adding bilateral IED ABOVE AND BEYOND stimulation.
mE_step1 <- glmer(memory ~ stim + (1 | patient_id),
                  data = enc_bi, family = binomial, control = ctrl)
mEs <- glmer(memory ~ stim + bilateral_ied + (1 | patient_id),
             data = enc_bi, family = binomial, control = ctrl)
lrt_h <- anova(mE_step1, mEs)
cat("\n--- Step 1: stim only ---\n");          print(summary(mE_step1))
cat("\n--- Step 2: stim + bilateral_ied ---\n"); print(summary(mEs))
cat("\n--- Hierarchical LRT (adding bilateral_ied above stim) ---\n"); print(lrt_h)
save_tidy(mE_step1, "h3c_enc_stimonly_or.csv")
save_tidy(mEs, "h3c_enc_bilonly_stim_or.csv")
write.csv(
  data.frame(chisq = lrt_h$Chisq[2], df = lrt_h$Df[2], p.value = lrt_h$`Pr(>Chisq)`[2]),
  file.path(out_dir, "h3c_enc_hierarchical_lrt.csv"), row.names = FALSE)

# ── Interaction model (reported in a SEPARATE table from the main effect) ─────
mEx <- glmer(memory ~ bilateral_ied * stim + (1 | patient_id),
             data = enc_bi, family = binomial, control = ctrl)
mEx0 <- glmer(memory ~ bilateral_ied + stim + (1 | patient_id),
              data = enc_bi, family = binomial, control = ctrl)
lrt <- anova(mEx0, mEx)
cat("\n--- Encoding bilateral_ied x stim interaction ---\n")
print(summary(mEx))
cat("\nLRT for interaction:\n"); print(lrt)
save_tidy(mEx, "h3c_enc_bilxstim_or.csv")
write.csv(
  data.frame(chisq = lrt$Chisq[2], df = lrt$Df[2], p.value = lrt$`Pr(>Chisq)`[2]),
  file.path(out_dir, "h3c_enc_bilxstim_lrt.csv"), row.names = FALSE)

# ── Timing-of-bilateral-IED model: all four encoding windows ──────────────────
# Restricted to bilateral trials (IEDs in both hemispheres); tests whether the
# window in which the bilateral IEDs occur predicts memory, across all four
# encoding windows (before image, during image, after image, during stim).
enc_bilonly <- enc_bi[enc_bi$trial_hemisphere == "Bilateral", ]
enc_bilonly$patient_id <- droplevels(factor(enc_bilonly$patient_id))
cat("\n--- Bilateral-trial timing-window subset ---\n")
cat(sprintf("Bilateral trials: %d, patients: %d\n",
            nrow(enc_bilonly), nlevels(enc_bilonly$patient_id)))
for (w in c("ied_before_image", "ied_during_image", "ied_after_image", "ied_during_stim")) {
  cat(sprintf("%s present:\n", w))
  print(tapply(enc_bilonly$memory, enc_bilonly[[w]], function(x)
    sprintf("n=%d, rem=%d (%.1f%%)", length(x), sum(x), mean(x) * 100)))
}

mEw <- glmer(
  memory ~ ied_before_image + ied_during_image + ied_after_image +
           ied_during_stim + (1 | patient_id),
  data = enc_bilonly, family = binomial, control = ctrl)
cat("\n--- Bilateral IED timing windows (all four encoding windows) ---\n")
print(summary(mEw))
save_tidy(mEw, "h3c_enc_bilwindow_or.csv")

##############################################################################
# RETRIEVAL
##############################################################################
ret_bi <- read.csv(file.path(out_dir, "h3c_retrieval_bilateral.csv"),
                   stringsAsFactors = FALSE)
ret_bi$patient_id <- factor(ret_bi$patient_id)

cat("\n\n===== RETRIEVAL =====\n")
cat(sprintf("Bilateral-coverage subset: %d trials, %d patients\n",
            nrow(ret_bi), nlevels(ret_bi$patient_id)))
n_bi_trials_r <- sum(ret_bi$bilateral_ied == 1)
n_bi_pats_r   <- length(unique(ret_bi$patient_id[ret_bi$bilateral_ied == 1]))
cat(sprintf("Bilateral-IED trials: %d  (from %d patients)\n", n_bi_trials_r, n_bi_pats_r))
cat("\n--- Recall by bilateral_ied ---\n")
print(tapply(ret_bi$memory, ret_bi$bilateral_ied, function(x)
  sprintf("n=%d, rem=%d (%.1f%%)", length(x), sum(x), mean(x) * 100)))

mR <- glmer(memory ~ bilateral_ied + (1 | patient_id),
            data = ret_bi, family = binomial, control = ctrl)
cat("\n--- Retrieval: memory ~ bilateral_ied ---\n")
print(summary(mR))
save_tidy(mR, "h3c_ret_bilonly_or.csv")

# Save text summaries
sink(file.path(out_dir, "h3c_bilonly_summaries.txt"))
cat("===== ENCODING: memory ~ bilateral_ied + (1|patient_id) =====\n\n")
print(summary(mE))
cat("\n\n===== ENCODING robustness: memory ~ bilateral_ied + stim + (1|patient_id) =====\n\n")
print(summary(mEs))
cat("\n\n===== RETRIEVAL: memory ~ bilateral_ied + (1|patient_id) =====\n\n")
print(summary(mR))
sink()

cat("\n\n===== DONE =====\n")
