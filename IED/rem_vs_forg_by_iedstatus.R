#!/usr/bin/env Rscript
# Intercept-only mixed-effects logistic regression testing whether targets are
# remembered more than forgotten, separately for IED and no-IED trials,
# encoding and retrieval.  Mirrors mlm_code/ied_multiwindow_mlm.R (lme4::glmer).
# Usage: Rscript rem_vs_forg_by_iedstatus.R <data_dir> <out_csv>

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
out_csv <- args[2]
suppressMessages(library(lme4))

datasets <- list(
  c("Encoding", "IED", "enc_ied.csv"),
  c("Encoding", "no-IED", "enc_noied.csv"),
  c("Retrieval", "IED", "ret_ied.csv"),
  c("Retrieval", "no-IED", "ret_noied.csv")
)

res <- data.frame()
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
for (ds in datasets) {
  d <- read.csv(file.path(data_dir, ds[3]))
  d$patient <- factor(d$patient)
  m  <- glmer(cbind(rem, forg) ~ 1 + (1 | patient), data = d, family = binomial, control = ctrl)
  m0 <- glmer(cbind(rem, forg) ~ 0 + (1 | patient), data = d, family = binomial, control = ctrl)
  co <- summary(m)$coefficients
  # LRT of the fixed intercept: full (~ 1) vs null (~ 0, log-odds fixed at 0 =
  # 50% remembering), both keeping the patient random intercept.  Tests whether
  # remembering differs from chance.  anova() refits with ML, returns AIC/BIC/Chisq.
  a <- anova(m0, m, test = "LRT")
  res <- rbind(res, data.frame(
    phase = ds[1], group = ds[2],
    n = sum(d$rem) + sum(d$forg), rem = sum(d$rem), forg = sum(d$forg),
    npat = nrow(d),
    intercept = co[1, 1], se = co[1, 2], z = co[1, 3], p = co[1, 4],
    aic_null = a[["AIC"]][1], aic_full = a[["AIC"]][2],
    bic_null = a[["BIC"]][1], bic_full = a[["BIC"]][2],
    lrt_chisq = a[["Chisq"]][2], lrt_df = a[["Df"]][2], lrt_p = a[["Pr(>Chisq)"]][2]))
}
write.csv(res, out_csv, row.names = FALSE)
cat("wrote", out_csv, "\n")
