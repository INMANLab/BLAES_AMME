#!/usr/bin/env Rscript
# Difference in odds of remembering between trials WITH vs WITHOUT IEDs.
# memory ~ ied + (1|patient), binomial/logit; ied = 1 (IED) vs 0 (no-IED, ref).
# Usage: Rscript rem_diff_ied_vs_noied.R <data_dir> <out_csv>

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]; out_csv <- args[2]
suppressMessages(library(lme4))

res <- data.frame()
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
for (ph in c("enc", "ret")) {
  d <- read.csv(file.path(data_dir, paste0(ph, "_long.csv")))
  d$patient <- factor(d$patient)
  m  <- glmer(cbind(rem, forg) ~ ied + (1 | patient), data = d, family = binomial, control = ctrl)
  m0 <- glmer(cbind(rem, forg) ~ 1   + (1 | patient), data = d, family = binomial, control = ctrl)
  co <- summary(m)$coefficients
  # LRT of the ied fixed effect: full (~ ied) vs null (~ 1), both keeping the
  # patient random intercept.  anova() refits with ML and returns AIC/BIC/Chisq.
  a <- anova(m0, m, test = "LRT")
  res <- rbind(res, data.frame(
    phase = ph,
    intercept = co[1, 1], int_se = co[1, 2],
    ied_coef = co[2, 1], ied_se = co[2, 2], ied_z = co[2, 3], ied_p = co[2, 4],
    aic_null = a[["AIC"]][1], aic_full = a[["AIC"]][2],
    bic_null = a[["BIC"]][1], bic_full = a[["BIC"]][2],
    lrt_chisq = a[["Chisq"]][2], lrt_df = a[["Df"]][2], lrt_p = a[["Pr(>Chisq)"]][2]))
}
write.csv(res, out_csv, row.names = FALSE)
cat("wrote", out_csv, "\n")
