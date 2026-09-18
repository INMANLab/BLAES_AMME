##############################################################################
# Dedicated EC-PRC PAC slow-gamma encoding analysis comparing:
#   RI: Accuracy ~ band + StimCond + band:StimCond + (1 | Patient)
#   RS: Accuracy ~ band + StimCond + band:StimCond + (1 + StimCond | Patient)
#
# Refits both models, saves:
#   - per-model fixed-effect coefficient tables (for the report's APA tables)
#   - sequential model build (m0 -> m1 -> m2 -> m_full) for each
#   - cross-model build (m_full_RI -> m_full_RS) testing the random slope
#   - random-effect variance components for each
#   - per-subject empirical stim effect + band slope (for the figure)
#
# All outputs go to:
#   OUTPUTS/encoding_memory_reports/EC_PRC_PAC_random_slope_report/
##############################################################################

source("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES/mlm_code/posthoc_common.R")
suppressPackageStartupMessages({ library(broom.mixed) })

OUT <- file.path(REPO_ROOT, "OUTPUTS", "encoding_memory_reports",
                 "EC_PRC_PAC_random_slope_report")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# Data prep (all trials, both stim and nostim).
d <- panel_data("pac", "HPCrhinal", "EC_PRC", "slow_gamma", "all")
d$StimCond_num <- ifelse(d$StimCond == "stim", 1L, 0L)
N_TRIALS <- nrow(d); N_PATS <- length(unique(d$Patient))
cat(sprintf("EC-PRC PAC encoding (all): %d trials, %d patients\n",
            N_TRIALS, N_PATS))

# ── Models -- both share the same fixed-effect structure, differ in RE ─────
m0  <- safe_glmer(Accuracy ~ 1 + (1 | Patient), d)
m1  <- safe_glmer(Accuracy ~ band_c + (1 | Patient), d)
m2  <- safe_glmer(Accuracy ~ band_c + StimCond_num + (1 | Patient), d)
m_full_ri <- safe_glmer(
  Accuracy ~ band_c + StimCond_num + band_c:StimCond_num + (1 | Patient), d)

# Random-slope variants for the sequential build under RS:
m0_rs  <- safe_glmer(Accuracy ~ 1 + (1 + StimCond_num | Patient), d)
m1_rs  <- safe_glmer(Accuracy ~ band_c + (1 + StimCond_num | Patient), d)
m2_rs  <- safe_glmer(
  Accuracy ~ band_c + StimCond_num + (1 + StimCond_num | Patient), d)
m_full_rs <- safe_glmer(
  Accuracy ~ band_c + StimCond_num + band_c:StimCond_num +
            (1 + StimCond_num | Patient), d)

stopifnot(!is.null(m_full_ri), !is.null(m_full_rs))

# ── Save fixed-effect coefs ────────────────────────────────────────────────
save_coefs <- function(m, path) {
  td <- broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE,
                          exponentiate = TRUE)
  write.csv(td, path, row.names = FALSE)
}
save_coefs(m_full_ri, file.path(OUT, "coefs_RI.csv"))
save_coefs(m_full_rs, file.path(OUT, "coefs_RS.csv"))

# ── Sequential model build ─────────────────────────────────────────────────
build_anova <- function(fits, fmls) {
  rows <- list(); prev <- NULL
  for (nm in names(fits)) {
    m <- fits[[nm]]
    if (is.null(m)) next
    npar <- attr(logLik(m), "df"); ll <- as.numeric(logLik(m))
    chisq <- NA_real_; df_v <- NA_real_; pv <- NA_real_
    if (!is.null(prev)) {
      cmp <- tryCatch(anova(prev, m, refit = FALSE), error = function(e) NULL)
      if (is.null(cmp)) cmp <- tryCatch(anova(prev, m), error = function(e) NULL)
      if (!is.null(cmp)) {
        pick <- function(col) {
          v <- cmp[[col]]
          if (is.null(v) || length(v) < 2) NA_real_ else as.numeric(v[2])
        }
        chisq <- pick("Chisq")
        df_v  <- pick("Chi Df"); if (is.na(df_v)) df_v <- pick("Df")
        pv    <- pick("Pr(>Chisq)")
      }
    }
    rows[[length(rows) + 1]] <- data.frame(
      Model = nm, Formula = fmls[[nm]],
      npar = npar, AIC = AIC(m), BIC = BIC(m),
      logLik = ll, deviance = -2 * ll,
      Chisq = chisq, Df = df_v, p.value = pv,
      stringsAsFactors = FALSE
    )
    prev <- m
  }
  do.call(rbind, rows)
}

ri_fmls <- list(
  m0     = "Accuracy ~ 1 + (1 | Patient)",
  m1     = "Accuracy ~ band + (1 | Patient)",
  m2     = "Accuracy ~ band + StimCond + (1 | Patient)",
  m_full = "Accuracy ~ band + StimCond + band:StimCond + (1 | Patient)"
)
rs_fmls <- list(
  m0     = "Accuracy ~ 1 + (1 + StimCond | Patient)",
  m1     = "Accuracy ~ band + (1 + StimCond | Patient)",
  m2     = "Accuracy ~ band + StimCond + (1 + StimCond | Patient)",
  m_full = "Accuracy ~ band + StimCond + band:StimCond + (1 + StimCond | Patient)"
)

anova_ri <- build_anova(
  list(m0 = m0, m1 = m1, m2 = m2, m_full = m_full_ri), ri_fmls)
anova_rs <- build_anova(
  list(m0 = m0_rs, m1 = m1_rs, m2 = m2_rs, m_full = m_full_rs), rs_fmls)
write.csv(anova_ri, file.path(OUT, "anova_RI.csv"), row.names = FALSE)
write.csv(anova_rs, file.path(OUT, "anova_RS.csv"), row.names = FALSE)

# Cross-model LRT for "does the random StimCond slope add anything?".
cmp <- anova(m_full_ri, m_full_rs)
write.csv(data.frame(
  rownames(cmp), as.data.frame(unclass(cmp)), check.names = FALSE),
  file.path(OUT, "ri_vs_rs_lrt.csv"), row.names = FALSE)

# ── Random-effect variance components ──────────────────────────────────────
varcomp_row <- function(m, label) {
  v <- VarCorr(m)$Patient
  sd_int   <- attr(v, "stddev")[1]
  sd_stim  <- if (length(attr(v, "stddev")) >= 2) attr(v, "stddev")[2] else NA
  rho      <- if (!is.null(attr(v, "correlation")) &&
                  ncol(attr(v, "correlation")) >= 2)
                attr(v, "correlation")[1, 2] else NA
  data.frame(
    model = label,
    sd_intercept = unname(sd_int),
    sd_stim = unname(sd_stim),
    cor_int_stim = unname(rho),
    singular = isSingular(m),
    AIC = AIC(m), BIC = BIC(m),
    stringsAsFactors = FALSE
  )
}
vc <- rbind(varcomp_row(m_full_ri, "RI"),
            varcomp_row(m_full_rs, "RS"))
write.csv(vc, file.path(OUT, "variance_components.csv"), row.names = FALSE)

# Also pull fixed-effect SE/var for both models for the JN explanation.
fixed_summary <- function(m, label) {
  s <- summary(m)$coefficients
  v <- vcov(m)
  out <- data.frame(
    model = label,
    term = rownames(s),
    estimate = s[, "Estimate"],
    se = s[, "Std. Error"],
    z = s[, "z value"],
    p = s[, "Pr(>|z|)"],
    or = exp(s[, "Estimate"]),
    ci_low = exp(s[, "Estimate"] - 1.96 * s[, "Std. Error"]),
    ci_high = exp(s[, "Estimate"] + 1.96 * s[, "Std. Error"]),
    stringsAsFactors = FALSE
  )
  row.names(out) <- NULL
  out
}
fx <- rbind(fixed_summary(m_full_ri, "RI"),
            fixed_summary(m_full_rs, "RS"))
write.csv(fx, file.path(OUT, "fixed_effect_summary.csv"), row.names = FALSE)

# ── Per-subject empirical signals (for the supporting figure) ──────────────
library(dplyr)
per_subj <- d %>% group_by(Patient) %>% summarise(
  n_trials = n(),
  n_stim   = sum(StimCond_num == 1L),
  n_nostim = sum(StimCond_num == 0L),
  rem_rate_stim   = mean(Accuracy[StimCond_num == 1L]),
  rem_rate_nostim = mean(Accuracy[StimCond_num == 0L]),
  band_cor = cor(band_c, Accuracy),
  .groups = "drop"
)
# Smoothed logit difference (add tiny epsilon to avoid log(0/0))
eps <- 0.5 / 80
per_subj$stim_effect_logit <- with(per_subj,
  log((rem_rate_stim + eps) / (1 - rem_rate_stim + eps) /
      ((rem_rate_nostim + eps) / (1 - rem_rate_nostim + eps))))
write.csv(per_subj, file.path(OUT, "per_subject_summary.csv"),
          row.names = FALSE)

# ── Per-subject random-effect BLUPs from the RS model ─────────────────────
# Gives each subject's deviation from the population intercept and the
# population StimCond effect (partial-pooled shrinkage estimates).
re <- ranef(m_full_rs, condVar = TRUE)$Patient
re_df <- as.data.frame(re)
re_df$Patient <- rownames(re_df)
re_df <- re_df[, c("Patient", "(Intercept)", "StimCond_num")]
names(re_df)[2:3] <- c("u_intercept", "u_StimCond")

# Get conditional SEs from the postVar attribute (one variance matrix per
# subject). For each subject, sqrt of the diagonal.
pv <- attr(re, "postVar")
se_int  <- sqrt(pv[1, 1, ])
se_stim <- sqrt(pv[2, 2, ])
re_df$se_u_intercept <- se_int
re_df$se_u_StimCond  <- se_stim

# Per-subject *total* stim effect = fixed b2 + subject's u_StimCond.
fix_b2 <- fixef(m_full_rs)["StimCond_num"]
re_df$subject_stim_effect <- fix_b2 + re_df$u_StimCond
# 95% CI on subject_stim_effect uses only the BLUP's conditional SE
# (treats the fixed effect as a population constant for visualization).
re_df$ci_lo <- re_df$subject_stim_effect - 1.96 * re_df$se_u_StimCond
re_df$ci_hi <- re_df$subject_stim_effect + 1.96 * re_df$se_u_StimCond

write.csv(re_df, file.path(OUT, "per_subject_blups_rs.csv"),
          row.names = FALSE)

# Save the population-mean stim effect with its SE as a separate row.
pop_se <- summary(m_full_rs)$coefficients["StimCond_num", "Std. Error"]
write.csv(data.frame(
  fixed_b2 = unname(fix_b2),
  fixed_b2_se = unname(pop_se),
  fixed_b2_ci_lo = unname(fix_b2 - 1.96 * pop_se),
  fixed_b2_ci_hi = unname(fix_b2 + 1.96 * pop_se),
  sigma_StimCond = unname(attr(VarCorr(m_full_rs)$Patient, "stddev")[2])
), file.path(OUT, "population_stim_effect_rs.csv"), row.names = FALSE)


# ── Per-subject NAIVE GLM (no random effects, that subject's trials only) ──
# Direct model-free counterpart to the BLUPs. If this distribution matches
# the BLUP distribution in shape and direction, the heterogeneity isn't an
# artifact of the random-slope parameterization.
cat("\nFitting per-subject naive GLMs...\n")
naive_rows <- list()
for (s in unique(d$Patient)) {
  sub <- d[d$Patient == s, ]
  if (length(unique(sub$StimCond_num)) < 2 ||
      length(unique(sub$Accuracy)) < 2) next
  m <- tryCatch(
    suppressWarnings(glm(
      Accuracy ~ band_c + StimCond_num + band_c:StimCond_num,
      data = sub, family = binomial)),
    error = function(e) NULL
  )
  if (is.null(m)) next
  s_coef <- summary(m)$coefficients
  if (!"StimCond_num" %in% rownames(s_coef)) next
  est <- s_coef["StimCond_num", "Estimate"]
  se  <- s_coef["StimCond_num", "Std. Error"]
  z   <- s_coef["StimCond_num", "z value"]
  pv  <- s_coef["StimCond_num", "Pr(>|z|)"]
  conv <- m$converged
  naive_rows[[length(naive_rows) + 1]] <- data.frame(
    Patient = as.character(s),
    n_trials = nrow(sub),
    naive_stim_effect = est, se = se,
    ci_lo = est - 1.96 * se, ci_hi = est + 1.96 * se,
    z = z, p = pv, converged = conv,
    stringsAsFactors = FALSE
  )
}
naive_df <- do.call(rbind, naive_rows)
write.csv(naive_df, file.path(OUT, "per_subject_naive_stim_effect.csv"),
          row.names = FALSE)

cat("Saved all CSVs ->", OUT, "\n")
