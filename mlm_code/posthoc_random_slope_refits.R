##############################################################################
# Post-hoc: refit every encoding GLMM panel with a random slope variant and
# compare to the random-intercept-only baseline (LRT, AIC, BIC, convergence).
#
# Models per analysis type:
#
#   alltrials / endogenous_memory_effect / stim_effect (main-effect models):
#     RI: Accuracy ~ band_c + (1 | Patient)
#     RS: Accuracy ~ band_c + (1 + band_c | Patient)
#
#   GLMM_interactions_model_variations (the band_c x StimCond interaction):
#     RI: Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)
#     RS: Accuracy ~ band_c + StimCond + band_c:StimCond + (1 + StimCond | Patient)
#
# Outputs (one row per panel, all 70 per analysis type):
#   {analysis_dir}/post_hoc_testing/random_slope_comparison.csv
#
# Also saves the random-slope fit's coefficients per panel:
#   {analysis_dir}/post_hoc_testing/rs_coefs_{measure}_{unit}_{band}.csv
##############################################################################

source("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES/mlm_code/posthoc_common.R")

is_interaction <- function(analysis) {
  identical(analysis, "GLMM_interactions_model_variations")
}

ri_formula <- function(analysis) {
  if (is_interaction(analysis))
    as.formula("Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)")
  else
    as.formula("Accuracy ~ band_c + (1 | Patient)")
}

rs_formula <- function(analysis) {
  if (is_interaction(analysis))
    as.formula("Accuracy ~ band_c + StimCond + band_c:StimCond + (1 + StimCond | Patient)")
  else
    as.formula("Accuracy ~ band_c + (1 + band_c | Patient)")
}

fit_or_centered <- function(formula, data) {
  m <- safe_glmer(formula, data)
  if (is.null(m)) {
    data$band_c <- data$band_c - mean(data$band_c, na.rm = TRUE)
    m <- safe_glmer(formula, data)
    if (!is.null(m)) attr(m, "centered") <- TRUE
  }
  m
}

fixed_summary <- function(m, term) {
  if (is.null(m)) return(c(estimate = NA, se = NA, z = NA, p = NA))
  s <- summary(m)$coefficients
  if (!term %in% rownames(s)) return(c(estimate = NA, se = NA, z = NA, p = NA))
  c(estimate = unname(s[term, "Estimate"]),
    se       = unname(s[term, "Std. Error"]),
    z        = unname(s[term, "z value"]),
    p        = unname(s[term, "Pr(>|z|)"]))
}

panel_lrt <- function(ri, rs) {
  if (is.null(ri) || is.null(rs)) return(list(chisq = NA, df = NA, p = NA))
  cmp <- tryCatch(anova(ri, rs, refit = FALSE), error = function(e) NULL)
  if (is.null(cmp)) cmp <- tryCatch(anova(ri, rs), error = function(e) NULL)
  if (is.null(cmp)) return(list(chisq = NA, df = NA, p = NA))
  list(
    chisq = as.numeric(cmp[["Chisq"]][2]),
    df    = as.numeric(cmp[["Df"]][2]),
    p     = as.numeric(cmp[["Pr(>Chisq)"]][2])
  )
}

run_analysis <- function(analysis) {
  out_dir <- posthoc_out_dir(ANALYSIS_DIR[[analysis]])
  panels_csv <- file.path(out_dir, "all_panels.csv")
  if (!file.exists(panels_csv)) stop("Missing all_panels.csv: ", panels_csv)
  panels <- read.csv(panels_csv, stringsAsFactors = FALSE)
  contrast <- ANALYSIS_CONTRAST[[analysis]]
  focal_term <- if (is_interaction(analysis)) "band_c:StimCondstim" else "band_c"

  cat(sprintf("\n========== %s (%d panels, contrast=%s, focal=%s) ==========\n",
              analysis, nrow(panels), contrast, focal_term))

  rows <- list()
  for (i in seq_len(nrow(panels))) {
    pr <- panels[i, ]
    measure <- pr$measure; scope <- pr$scope
    unit <- pr$unit; band <- pr$band
    cat(sprintf("[%d/%d] %s | %s | %s | %s\n",
                i, nrow(panels), measure, scope, unit, band))

    d <- panel_data(measure, scope, unit, band, contrast)
    n_pat <- length(unique(d$Patient))
    if (nrow(d) < MIN_TRIALS || n_pat < MIN_PATIENTS) {
      cat(sprintf("   SKIP (n=%d trials, %d patients)\n", nrow(d), n_pat))
      next
    }

    ri <- fit_or_centered(ri_formula(analysis), d)
    rs <- fit_or_centered(rs_formula(analysis), d)

    ri_fx <- fixed_summary(ri, focal_term)
    rs_fx <- fixed_summary(rs, focal_term)
    lrt <- panel_lrt(ri, rs)

    ri_aic <- if (!is.null(ri)) AIC(ri) else NA
    ri_bic <- if (!is.null(ri)) BIC(ri) else NA
    rs_aic <- if (!is.null(rs)) AIC(rs) else NA
    rs_bic <- if (!is.null(rs)) BIC(rs) else NA

    # Singular-fit check: random-effect var/cov matrix near-singular?
    rs_singular <- if (!is.null(rs)) tryCatch(isSingular(rs),
                                              error = function(e) NA) else NA

    # Save RS coefs CSV (for downstream JN reading).
    if (!is.null(rs)) {
      td <- broom.mixed::tidy(rs, effects = "fixed", conf.int = TRUE,
                              exponentiate = TRUE)
      td$converged_with_warning <- isTRUE(attr(rs, "converged_with_warning"))
      td$centered <- isTRUE(attr(rs, "centered"))
      coef_path <- file.path(out_dir,
        sprintf("rs_coefs_%s_%s_%s.csv", measure, unit, band))
      write.csv(td, coef_path, row.names = FALSE)
    }

    rows[[length(rows) + 1]] <- data.frame(
      measure = measure, scope = scope, unit = unit, band = band,
      n_trials = nrow(d), n_patients = n_pat,
      ri_fit = !is.null(ri), rs_fit = !is.null(rs),
      ri_AIC = ri_aic, rs_AIC = rs_aic, dAIC = rs_aic - ri_aic,
      ri_BIC = ri_bic, rs_BIC = rs_bic, dBIC = rs_bic - ri_bic,
      LRT_chisq = lrt$chisq, LRT_df = lrt$df, LRT_p = lrt$p,
      ri_focal_est = ri_fx["estimate"], ri_focal_se = ri_fx["se"],
      ri_focal_z = ri_fx["z"], ri_focal_p = ri_fx["p"],
      rs_focal_est = rs_fx["estimate"], rs_focal_se = rs_fx["se"],
      rs_focal_z = rs_fx["z"], rs_focal_p = rs_fx["p"],
      rs_singular = rs_singular,
      stringsAsFactors = FALSE
    )
  }

  out <- do.call(rbind, rows)
  out_path <- file.path(out_dir, "random_slope_comparison.csv")
  write.csv(out, out_path, row.names = FALSE)
  cat(sprintf("Saved: %s\n", out_path))
  invisible(out)
}

# CLI: if a single analysis name is passed, run just that one; else run all.
.args <- commandArgs(trailingOnly = TRUE)
analyses <- if (length(.args) >= 1) .args[1] else names(ANALYSIS_DIR)

for (a in analyses) run_analysis(a)
cat("\n===== RANDOM-SLOPE REFITS DONE =====\n")
