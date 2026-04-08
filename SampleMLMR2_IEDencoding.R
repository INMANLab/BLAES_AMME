suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(lme4)
  library(lmerTest)
})

# ---- Helper functions ----

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  path
}

format_p_value <- function(x) {
  ifelse(is.na(x), "", ifelse(x < 0.001, "<0.001", sprintf("%.3f", x)))
}

term_label <- function(term) {
  labels <- c(
    "(Intercept)" = "(Intercept)",
    "BeforeImgITI" = "BeforeImgITI",
    "DuringImg" = "DuringImg",
    "DuringStim" = "DuringStim",
    "AfterImgITI" = "AfterImgITI",
    "BeforeImgITI_prop" = "BeforeImgITI prop",
    "DuringImg_prop" = "DuringImg prop",
    "DuringStim_prop" = "DuringStim prop",
    "AfterImgITI_prop" = "AfterImgITI prop"
  )
  if (term %in% names(labels)) labels[[term]] else term
}

make_table_grob <- function(df_display, first_col_width = 2.4, base_size = 16,
                            body_fontsize = NULL, header_fontsize = NULL, column_widths = NULL) {
  if (is.null(body_fontsize)) body_fontsize <- max(12, base_size - 1)
  if (is.null(header_fontsize)) header_fontsize <- base_size
  table_theme <- ttheme_minimal(
    base_size = base_size,
    core = list(
      fg_params = list(fontsize = body_fontsize, x = 0.02, hjust = 0.02),
      bg_params = list(fill = rep(c("#f4f4f4", "white"), length.out = nrow(df_display)), col = NA),
      padding = unit(c(6, 5), "mm")
    ),
    colhead = list(
      fg_params = list(fontsize = header_fontsize, fontface = "bold", x = 0.02, hjust = 0.02),
      bg_params = list(fill = "#d9d9d9", col = NA),
      padding = unit(c(6, 5), "mm")
    )
  )
  tbl <- tableGrob(df_display, rows = NULL, theme = table_theme)
  if (!is.null(column_widths) && length(column_widths) == ncol(df_display)) {
    tbl$widths <- unit(column_widths, "null")
  } else if (ncol(tbl) >= 2) {
    tbl$widths <- unit.c(unit(first_col_width, "null"), unit(rep(1, ncol(tbl) - 1), "null"))
  }
  tbl
}

save_table_png <- function(df_display, path, title = NULL, width = 9, height = 5) {
  png(filename = path, width = width, height = height, units = "in", res = 220)
  grid.newpage()
  if (!is.null(title) && nzchar(title)) {
    pushViewport(viewport(layout = grid.layout(2, 1, heights = unit(c(0.12, 0.88), "npc"))))
    grid.draw(textGrob(title, gp = gpar(fontsize = 18, fontface = "bold"), vp = viewport(layout.pos.row = 1)))
    tbl <- make_table_grob(df_display)
    grid.draw(editGrob(tbl, vp = viewport(layout.pos.row = 2)))
  } else {
    grid.draw(make_table_grob(df_display))
  }
  dev.off()
}

save_plot <- function(plot_obj, path, width = 8, height = 4, dpi = 300) {
  ggsave(filename = path, plot = plot_obj, width = width, height = height, dpi = dpi, units = "in")
}

# ---- Data preparation ----

prepare_ied_encoding_data <- function() {
  # Read trial-level IED data (collapsed across channels/regions to one row per patient-trial)
  trial_csv <- file.path("IED", "ied_trial_level_summary", "collapsed_trial_level.csv")
  trial_dat <- read.csv(trial_csv, stringsAsFactors = FALSE)

  # Convert Y/N IED indicators to binary 0/1
  trial_dat <- trial_dat %>%
    mutate(
      BeforeImgITI = ifelse(BeforeImgITI == "Y", 1, 0),
      DuringImg    = ifelse(DuringImg == "Y", 1, 0),
      DuringStim   = ifelse(DuringStim == "Y", 1, 0),
      AfterImgITI  = ifelse(AfterImgITI == "Y", 1, 0),
      StimCond     = ifelse(StimCond == "S", "stim", "nostim")
    )

  # Read patient-level memory data (avg_dprime_diff = memory modulation)
  memory_csv <- file.path("IED", "ied_memory_relationships", "memory_vs_ied_merged.csv")
  memory_dat <- read.csv(memory_csv, stringsAsFactors = FALSE) %>%
    select(Patient, avg_stim_dprime_diff)
  colnames(memory_dat)[2] <- "avg_dprime_diff"

  list(trial_dat = trial_dat, memory_dat = memory_dat)
}

# ---- Patient-level data: aggregate IED proportions per patient ----

build_patient_level_ied <- function(trial_dat, memory_dat) {
  patient_agg <- trial_dat %>%
    group_by(Patient) %>%
    summarise(
      BeforeImgITI_prop = mean(BeforeImgITI, na.rm = TRUE),
      DuringImg_prop    = mean(DuringImg, na.rm = TRUE),
      DuringStim_prop   = mean(DuringStim, na.rm = TRUE),
      AfterImgITI_prop  = mean(AfterImgITI, na.rm = TRUE),
      N_trials          = n(),
      .groups = "drop"
    )

  dat <- inner_join(patient_agg, memory_dat, by = "Patient")
  dat
}

# ---- Trial-level data: merge trial IED indicators with patient memory modulation ----

build_trial_level_ied <- function(trial_dat, memory_dat) {
  dat <- inner_join(trial_dat, memory_dat, by = "Patient")
  dat
}

# ---- Model fitting ----

# Patient-level models (linear regression, no random effect — one row per patient)
fit_patient_models_ied <- function(dat) {
  list(
    m0     = lm(avg_dprime_diff ~ 1, data = dat),
    m1     = lm(avg_dprime_diff ~ DuringImg_prop, data = dat),
    m2     = lm(avg_dprime_diff ~ DuringImg_prop + BeforeImgITI_prop, data = dat),
    m3     = lm(avg_dprime_diff ~ DuringImg_prop + BeforeImgITI_prop + DuringStim_prop, data = dat),
    m_full = lm(avg_dprime_diff ~ DuringImg_prop + BeforeImgITI_prop + DuringStim_prop + AfterImgITI_prop,
                data = dat)
  )
}

# Trial-level models (linear mixed model with Patient random intercept)
# DV = avg_dprime_diff (patient-level, repeated across trials)
# IVs = binary IED indicators per trial
fit_trial_models_ied <- function(dat) {
  list(
    m0     = lmer(avg_dprime_diff ~ 1 + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m1     = lmer(avg_dprime_diff ~ DuringImg + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m2     = lmer(avg_dprime_diff ~ DuringImg + BeforeImgITI + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m3     = lmer(avg_dprime_diff ~ DuringImg + BeforeImgITI + DuringStim + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m_full = lmer(avg_dprime_diff ~ DuringImg + BeforeImgITI + DuringStim + AfterImgITI + (1 | Patient),
                  data = dat, na.action = na.exclude, REML = FALSE)
  )
}

# ---- Coefficient extraction ----

extract_coef_table_lm <- function(model) {
  coef_mat <- as.data.frame(summary(model)$coefficients)
  coef_mat$term <- rownames(coef_mat)

  fixed_terms <- coef_mat$term
  conf <- confint(model, parm = fixed_terms, level = 0.95)
  conf <- conf[fixed_terms, , drop = FALSE]

  statistic_col <- if ("t value" %in% names(coef_mat)) "t value" else "z value"
  p_col <- grep("^Pr\\(", names(coef_mat), value = TRUE)
  p_values <- if (length(p_col) == 1) coef_mat[[p_col]] else rep(NA_real_, nrow(coef_mat))

  data.frame(
    Predictor   = vapply(coef_mat$term, term_label, character(1)),
    Estimate    = coef_mat$Estimate,
    std_error   = coef_mat$`Std. Error`,
    conf_low    = conf[, 1],
    conf_high   = conf[, 2],
    statistic   = coef_mat[[statistic_col]],
    p_value     = p_values,
    stringsAsFactors = FALSE
  )
}

extract_coef_table_lmer <- function(model) {
  coef_mat <- as.data.frame(summary(model)$coefficients)
  coef_mat$term <- rownames(coef_mat)

  fixed_terms <- coef_mat$term
  conf <- suppressMessages(confint(model, parm = fixed_terms, method = "Wald"))
  conf <- conf[fixed_terms, , drop = FALSE]

  statistic_col <- if ("t value" %in% names(coef_mat)) "t value" else "z value"
  p_col <- grep("^Pr\\(", names(coef_mat), value = TRUE)
  p_values <- if (length(p_col) == 1) coef_mat[[p_col]] else rep(NA_real_, nrow(coef_mat))

  data.frame(
    Predictor   = vapply(coef_mat$term, term_label, character(1)),
    Estimate    = coef_mat$Estimate,
    std_error   = coef_mat$`Std. Error`,
    conf_low    = conf[, 1],
    conf_high   = conf[, 2],
    statistic   = coef_mat[[statistic_col]],
    p_value     = p_values,
    stringsAsFactors = FALSE
  )
}

display_coef_table <- function(df) {
  data.frame(
    Predictors   = df$Predictor,
    Estimate     = sprintf("%.3f", df$Estimate),
    `std. Error`  = sprintf("%.3f", df$std_error),
    CI           = sprintf("%.3f to %.3f", df$conf_low, df$conf_high),
    Statistic    = sprintf("%.3f", df$statistic),
    p            = format_p_value(df$p_value),
    check.names  = FALSE,
    stringsAsFactors = FALSE
  )
}

# ---- Model comparison ----

extract_model_compare_table_lm <- function(model_list) {
  comp <- as.data.frame(anova(model_list$m0, model_list$m1, model_list$m2, model_list$m3, model_list$m_full))
  comp$Model <- c("m0", "m1", "m2", "m3", "m_full")
  rownames(comp) <- NULL

  # Column names differ between lm and lmer anova output
  rss_col <- if ("RSS" %in% names(comp)) comp$RSS else rep(NA_real_, nrow(comp))
  f_col <- if ("F" %in% names(comp)) comp$F else rep(NA_real_, nrow(comp))
  p_col <- grep("^Pr\\(", names(comp), value = TRUE)
  p_vals <- if (length(p_col) == 1) comp[[p_col]] else rep(NA_real_, nrow(comp))

  data.frame(
    Model    = comp$Model,
    Res.Df   = comp$Res.Df,
    RSS      = rss_col,
    Df       = if ("Df" %in% names(comp)) comp$Df else rep(NA_real_, nrow(comp)),
    `Sum of Sq` = if ("Sum of Sq" %in% names(comp)) comp$`Sum of Sq` else rep(NA_real_, nrow(comp)),
    F_value  = f_col,
    p_value  = p_vals,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

extract_model_compare_table_lmer <- function(model_list) {
  comp <- as.data.frame(anova(model_list$m0, model_list$m1, model_list$m2, model_list$m3, model_list$m_full))
  comp$Model <- c("m0", "m1", "m2", "m3", "m_full")
  rownames(comp) <- NULL

  p_col <- grep("^Pr", names(comp), value = TRUE)
  p_vals <- if (length(p_col) == 1) comp[[p_col]] else rep(NA_real_, nrow(comp))
  chisq_vals <- if ("Chisq" %in% names(comp)) comp$Chisq else rep(NA_real_, nrow(comp))
  df_vals <- if ("Df" %in% names(comp)) comp$Df else rep(NA_real_, nrow(comp))
  deviance_vals <- if ("deviance" %in% names(comp)) comp$deviance else rep(NA_real_, nrow(comp))

  data.frame(
    Model     = comp$Model,
    npar      = comp$npar,
    AIC       = comp$AIC,
    BIC       = comp$BIC,
    logLik    = comp$logLik,
    minus_2logL = deviance_vals,
    Chisq     = chisq_vals,
    Df        = df_vals,
    p_value   = p_vals,
    stringsAsFactors = FALSE
  )
}

display_model_compare_table_lm <- function(df) {
  data.frame(
    Model      = df$Model,
    Res.Df     = ifelse(is.na(df$Res.Df), "", sprintf("%.0f", df$Res.Df)),
    RSS        = ifelse(is.na(df$RSS), "", sprintf("%.3f", df$RSS)),
    Df         = ifelse(is.na(df$Df), "", sprintf("%.0f", df$Df)),
    `Sum of Sq` = ifelse(is.na(df$`Sum of Sq`), "", sprintf("%.4f", df$`Sum of Sq`)),
    F          = ifelse(is.na(df$F_value), "", sprintf("%.4f", df$F_value)),
    `Pr(>F)`   = format_p_value(df$p_value),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

display_model_compare_table_lmer <- function(df) {
  data.frame(
    Model       = df$Model,
    npar        = ifelse(is.na(df$npar), "", sprintf("%.0f", df$npar)),
    AIC         = ifelse(is.na(df$AIC), "", sprintf("%.3f", df$AIC)),
    BIC         = ifelse(is.na(df$BIC), "", sprintf("%.3f", df$BIC)),
    logLik      = ifelse(is.na(df$logLik), "", sprintf("%.3f", df$logLik)),
    `-2*log(L)` = ifelse(is.na(df$minus_2logL), "", sprintf("%.3f", df$minus_2logL)),
    Chisq       = ifelse(is.na(df$Chisq), "", sprintf("%.4f", df$Chisq)),
    Df          = ifelse(is.na(df$Df), "", sprintf("%.0f", df$Df)),
    `Pr(>Chi)`  = format_p_value(df$p_value),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

# ---- Fit summary ----

build_fit_summary_patient <- function(model_list, dat, compare_df) {
  m_full <- model_list$m_full
  m0 <- model_list$m0
  r2_full <- summary(m_full)$r.squared
  adj_r2_full <- summary(m_full)$adj.r.squared
  full_row <- compare_df[compare_df$Model == "m_full", , drop = FALSE]

  data.frame(
    Metric = c(
      "N patients",
      "R-squared (full)",
      "Adj R-squared (full)",
      "Residual Std Error (full)",
      "F-statistic (full)",
      "Full model p (vs m3)"
    ),
    Value = c(
      sprintf("%d", nrow(dat)),
      sprintf("%.3f", r2_full),
      sprintf("%.3f", adj_r2_full),
      sprintf("%.3f", sigma(m_full)),
      sprintf("%.3f", summary(m_full)$fstatistic[1]),
      if (nrow(full_row) == 1) format_p_value(full_row$p_value) else ""
    ),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

build_fit_summary_trial <- function(model_list, dat, compare_df) {
  m_full <- model_list$m_full
  m0 <- model_list$m0
  varcorr_null <- as.data.frame(VarCorr(m0))
  patient_var <- sum(varcorr_null$vcov[varcorr_null$grp == "Patient"], na.rm = TRUE)
  residual_var <- sum(varcorr_null$vcov[varcorr_null$grp == "Residual"], na.rm = TRUE)
  icc <- if (patient_var + residual_var > 0) patient_var / (patient_var + residual_var) else NA_real_

  # Marginal and conditional R2 for the full model
  varcorr_full <- as.data.frame(VarCorr(m_full))
  random_var_full <- sum(varcorr_full$vcov[varcorr_full$grp != "Residual"], na.rm = TRUE)
  residual_var_full <- sum(varcorr_full$vcov[varcorr_full$grp == "Residual"], na.rm = TRUE)

  fixed_formula <- lme4::nobars(formula(m_full))
  term_matrix <- delete.response(terms(fixed_formula))
  model_matrix <- model.matrix(term_matrix, dat)
  beta <- fixef(m_full)
  model_matrix <- model_matrix[, names(beta), drop = FALSE]
  fixed_linear <- as.vector(model_matrix %*% beta)
  fixed_var <- stats::var(fixed_linear, na.rm = TRUE)
  if (!is.finite(fixed_var)) fixed_var <- 0

  total_var <- fixed_var + random_var_full + residual_var_full
  marginal_r2 <- if (total_var > 0) fixed_var / total_var else NA_real_
  conditional_r2 <- if (total_var > 0) (fixed_var + random_var_full) / total_var else NA_real_

  full_row <- compare_df[compare_df$Model == "m_full", , drop = FALSE]

  data.frame(
    Metric = c(
      "Null patient variance",
      "Residual variance",
      "Null ICC",
      "N patients",
      "Observations (trials)",
      "Full marginal R2",
      "Full conditional R2",
      "Full model AIC",
      "Full model BIC",
      "m_full vs prior p"
    ),
    Value = c(
      sprintf("%.3f", patient_var),
      sprintf("%.3f", residual_var),
      sprintf("%.3f", icc),
      sprintf("%d", dplyr::n_distinct(dat$Patient)),
      sprintf("%d", nrow(dat)),
      sprintf("%.3f", marginal_r2),
      sprintf("%.3f", conditional_r2),
      if (nrow(full_row) == 1) sprintf("%.3f", full_row$AIC) else "",
      if (nrow(full_row) == 1) sprintf("%.3f", full_row$BIC) else "",
      if (nrow(full_row) == 1) format_p_value(full_row$p_value) else ""
    ),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

# ---- Formula display ----

build_formula_display_patient <- function() {
  data.frame(
    Model = c("m0", "m1", "m2", "m3", "m_full"),
    Formula = c(
      'lm("avg_dprime_diff ~ 1")',
      'lm("avg_dprime_diff ~ DuringImg_prop")',
      'lm("avg_dprime_diff ~ DuringImg_prop + BeforeImgITI_prop")',
      'lm("avg_dprime_diff ~ DuringImg_prop + BeforeImgITI_prop + DuringStim_prop")',
      'lm("avg_dprime_diff ~ DuringImg_prop + BeforeImgITI_prop + DuringStim_prop + AfterImgITI_prop")'
    ),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

build_formula_display_trial <- function() {
  data.frame(
    Model = c("m0", "m1", "m2", "m3", "m_full"),
    Formula = c(
      'lmer("avg_dprime_diff ~ 1 + (1 | Patient)")',
      'lmer("avg_dprime_diff ~ DuringImg + (1 | Patient)")',
      'lmer("avg_dprime_diff ~ DuringImg + BeforeImgITI + (1 | Patient)")',
      'lmer("avg_dprime_diff ~ DuringImg + BeforeImgITI + DuringStim + (1 | Patient)")',
      'lmer("avg_dprime_diff ~ DuringImg + BeforeImgITI + DuringStim + AfterImgITI + (1 | Patient)")'
    ),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

# ---- Random effect extraction ----

extract_random_effect_df <- function(model, group_name = "Patient") {
  re_list <- ranef(model, condVar = TRUE)
  if (!(group_name %in% names(re_list))) {
    stop(sprintf("Random effect group %s not found", group_name))
  }
  re_df <- re_list[[group_name]]
  effect_name <- names(re_df)[1]
  estimates <- re_df[[effect_name]]
  post_var <- attr(re_df, "postVar")
  se_vals <- if (!is.null(post_var)) sqrt(post_var[1, 1, ]) else rep(NA_real_, length(estimates))

  out <- data.frame(
    group    = rownames(re_df),
    estimate = estimates,
    se       = se_vals,
    stringsAsFactors = FALSE
  )
  out$lower <- out$estimate - 1.96 * out$se
  out$upper <- out$estimate + 1.96 * out$se
  out
}

make_random_effect_plot <- function(re_df, title_text) {
  plot_df <- re_df %>%
    arrange(estimate) %>%
    mutate(group = factor(group, levels = group))

  ggplot(plot_df, aes(x = estimate, y = group)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.6) +
    geom_segment(aes(x = lower, xend = upper, y = group, yend = group), linewidth = 0.7, color = "#333333") +
    geom_point(size = 2.1, color = "#111111") +
    labs(
      title = title_text,
      x = "Random intercept (deviation from grand mean)",
      y = "Patient"
    ) +
    theme_bw(base_size = 15) +
    theme(
      plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 12),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    )
}

# ---- Effect plots ----

make_effect_plot_patient <- function(model, data, predictor, title_text) {
  pred_min <- min(data[[predictor]], na.rm = TRUE)
  pred_max <- max(data[[predictor]], na.rm = TRUE)
  if (pred_min == pred_max) { pred_min <- pred_min - 0.05; pred_max <- pred_max + 0.05 }

  newdata <- data.frame(x_value = seq(pred_min, pred_max, length.out = 100))
  # Set other predictors to their means
  for (p in c("BeforeImgITI_prop", "DuringImg_prop", "DuringStim_prop", "AfterImgITI_prop")) {
    newdata[[p]] <- mean(data[[p]], na.rm = TRUE)
  }
  newdata[[predictor]] <- newdata$x_value

  preds <- predict(model, newdata = newdata, se.fit = TRUE)
  newdata$fit   <- preds$fit
  newdata$lower <- preds$fit - 1.96 * preds$se.fit
  newdata$upper <- preds$fit + 1.96 * preds$se.fit

  ggplot(newdata, aes(x = x_value, y = fit)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#a9c7e5", alpha = 0.45) +
    geom_line(color = "#0a78c9", linewidth = 1) +
    geom_rug(data = data, aes(x = .data[[predictor]]), sides = "b", inherit.aes = FALSE, alpha = 0.4) +
    geom_point(data = data, aes(x = .data[[predictor]], y = avg_dprime_diff),
               inherit.aes = FALSE, alpha = 0.6, size = 2) +
    labs(x = predictor, y = "Memory Modulation (avg_dprime_diff)", title = title_text) +
    theme_bw(base_size = 16) +
    theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      panel.grid = element_blank()
    )
}

make_effect_plot_trial <- function(model, data, predictor, title_text) {
  # For binary predictors, show group means with CI
  newdata <- expand.grid(x_value = c(0, 1))
  for (p in c("BeforeImgITI", "DuringImg", "DuringStim", "AfterImgITI")) {
    newdata[[p]] <- 0
  }
  newdata[[predictor]] <- newdata$x_value

  # Fixed-effects prediction
  fixed_formula <- lme4::nobars(formula(model))
  term_matrix <- delete.response(terms(fixed_formula))
  model_matrix <- model.matrix(term_matrix, newdata)
  beta <- fixef(model)
  model_matrix <- model_matrix[, names(beta), drop = FALSE]
  fit_link <- as.vector(model_matrix %*% beta)
  se_link <- sqrt(pmax(0, diag(model_matrix %*% as.matrix(vcov(model)) %*% t(model_matrix))))

  newdata$fit   <- fit_link
  newdata$lower <- fit_link - 1.96 * se_link
  newdata$upper <- fit_link + 1.96 * se_link
  newdata$IED_present <- factor(newdata$x_value, levels = c(0, 1), labels = c("No IED", "IED"))

  ggplot(newdata, aes(x = IED_present, y = fit)) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2, linewidth = 0.8) +
    geom_point(size = 3, color = "#0a78c9") +
    labs(x = predictor, y = "Memory Modulation (avg_dprime_diff)", title = title_text) +
    theme_bw(base_size = 16) +
    theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      panel.grid = element_blank()
    )
}

# ---- Summary panel ----

save_summary_panel <- function(output_path, level_label, n_patients, coef_display, compare_display, plots) {
  title_text <- sprintf("IED Encoding Effects on Memory Modulation (N=%d) - %s", n_patients, level_label)

  coef_grob    <- make_table_grob(coef_display, first_col_width = 2.7)
  compare_grob <- make_table_grob(compare_display, first_col_width = 1.8)

  left_panel <- arrangeGrob(coef_grob, compare_grob, ncol = 1, heights = c(0.55, 0.45))
  right_panel <- arrangeGrob(grobs = plots, ncol = 1, heights = rep(1 / length(plots), length(plots)))

  summary_grob <- arrangeGrob(
    textGrob(title_text, gp = gpar(fontsize = 22, fontface = "bold")),
    arrangeGrob(left_panel, right_panel, ncol = 2, widths = c(1.38, 1.12)),
    ncol = 1,
    heights = c(0.06, 0.94)
  )

  png(filename = output_path, width = 13.333, height = 7.5, units = "in", res = 260, bg = "white")
  grid.newpage()
  pushViewport(viewport(x = 0.5, y = 0.5, width = unit(0.997, "npc"), height = unit(0.992, "npc")))
  grid.draw(summary_grob)
  upViewport()
  dev.off()
}

# ---- Overview panel ----

save_overview_panel <- function(output_path, level_label, formula_display, compare_display,
                                fit_display, random_plot = NULL) {
  title_text <- sprintf("IED Encoding - %s model fit", level_label)
  subtitle_text <- "IED encoding time-window effects on memory modulation (avg_dprime_diff)"

  formula_grob <- make_table_grob(formula_display, first_col_width = 0.7, base_size = 13,
                                  body_fontsize = 12, header_fontsize = 13)
  compare_grob <- make_table_grob(compare_display, first_col_width = 1.05, base_size = 13,
                                  body_fontsize = 12, header_fontsize = 13)
  fit_grob <- make_table_grob(fit_display, first_col_width = 1.55, base_size = 14,
                              body_fontsize = 13, header_fontsize = 14)
  note_grob <- textGrob(
    "Models compare nested IED time-window predictors of memory modulation.",
    x = 0, hjust = 0, gp = gpar(fontsize = 13)
  )

  top_row <- arrangeGrob(formula_grob, compare_grob, ncol = 2, widths = c(1.55, 1.05))

  if (!is.null(random_plot)) {
    bottom_row <- arrangeGrob(
      random_plot,
      arrangeGrob(fit_grob, note_grob, ncol = 1, heights = c(0.8, 0.2)),
      ncol = 2, widths = c(1.45, 0.9)
    )
  } else {
    bottom_row <- arrangeGrob(fit_grob, note_grob, ncol = 1, heights = c(0.8, 0.2))
  }

  overview_grob <- arrangeGrob(
    textGrob(title_text, gp = gpar(fontsize = 24, fontface = "bold")),
    textGrob(subtitle_text, gp = gpar(fontsize = 15)),
    top_row,
    bottom_row,
    ncol = 1,
    heights = c(0.06, 0.03, 0.36, 0.55)
  )

  png(filename = output_path, width = 13.333, height = 7.5, units = "in", res = 260, bg = "white")
  grid.newpage()
  pushViewport(viewport(x = 0.5, y = 0.5, width = unit(0.997, "npc"), height = unit(0.992, "npc")))
  grid.draw(overview_grob)
  upViewport()
  dev.off()
}

# ============================================================
# Main analysis pipeline
# ============================================================

run_ied_encoding_analysis <- function() {
  output_dir <- ensure_dir(file.path("outputs", "stats", "ied_encoding"))
  overview_dir <- ensure_dir(file.path(output_dir, "overview"))
  patient_dir <- ensure_dir(file.path(output_dir, "patient_level"))
  trial_dir <- ensure_dir(file.path(output_dir, "trial_level"))

  # Prepare data
  raw <- prepare_ied_encoding_data()
  patient_dat <- build_patient_level_ied(raw$trial_dat, raw$memory_dat)
  trial_dat <- build_trial_level_ied(raw$trial_dat, raw$memory_dat)

  message(sprintf("Patient-level data: %d patients", nrow(patient_dat)))
  message(sprintf("Trial-level data: %d trials from %d patients",
                  nrow(trial_dat), dplyr::n_distinct(trial_dat$Patient)))

  # ---- Patient-level analysis ----
  message("Fitting patient-level models...")
  patient_models <- fit_patient_models_ied(patient_dat)

  patient_coef_df <- extract_coef_table_lm(patient_models$m_full)
  patient_coef_display <- display_coef_table(patient_coef_df)
  patient_compare_df <- extract_model_compare_table_lm(patient_models)
  patient_compare_display <- display_model_compare_table_lm(patient_compare_df)
  patient_fit_display <- build_fit_summary_patient(patient_models, patient_dat, patient_compare_df)
  patient_formula_display <- build_formula_display_patient()

  # Save patient-level outputs
  write.csv(patient_coef_df, file.path(patient_dir, "coefficients.csv"), row.names = FALSE)
  write.csv(patient_compare_df, file.path(patient_dir, "model_comparison.csv"), row.names = FALSE)
  write.csv(patient_fit_display, file.path(patient_dir, "fit_summary.csv"), row.names = FALSE)
  save_table_png(patient_coef_display, file.path(patient_dir, "coefficients_table.png"),
                 title = "Patient-level coefficients (full model)")
  save_table_png(patient_compare_display, file.path(patient_dir, "model_comparison_table.png"),
                 title = "Patient-level model comparison")

  # Patient-level effect plots
  p1 <- make_effect_plot_patient(patient_models$m_full, patient_dat, "BeforeImgITI_prop", "BeforeImgITI proportion")
  p2 <- make_effect_plot_patient(patient_models$m_full, patient_dat, "DuringImg_prop", "DuringImg proportion")
  p3 <- make_effect_plot_patient(patient_models$m_full, patient_dat, "DuringStim_prop", "DuringStim proportion")
  p4 <- make_effect_plot_patient(patient_models$m_full, patient_dat, "AfterImgITI_prop", "AfterImgITI proportion")

  save_plot(p1, file.path(patient_dir, "BeforeImgITI_effect.png"))
  save_plot(p2, file.path(patient_dir, "DuringImg_effect.png"))
  save_plot(p3, file.path(patient_dir, "DuringStim_effect.png"))
  save_plot(p4, file.path(patient_dir, "AfterImgITI_effect.png"))

  save_summary_panel(
    file.path(patient_dir, "summary_panel.png"),
    level_label = "Patient-level (lm)",
    n_patients = nrow(patient_dat),
    coef_display = patient_coef_display,
    compare_display = patient_compare_display,
    plots = list(p1, p2, p3, p4)
  )

  # Patient-level overview
  save_overview_panel(
    file.path(overview_dir, "patient_level_overview.png"),
    level_label = "patient-level",
    formula_display = patient_formula_display,
    compare_display = patient_compare_display,
    fit_display = patient_fit_display,
    random_plot = NULL
  )
  write.csv(patient_compare_df, file.path(overview_dir, "patient_level_model_comparison.csv"), row.names = FALSE)
  write.csv(patient_fit_display, file.path(overview_dir, "patient_level_fit_summary.csv"), row.names = FALSE)

  # ---- Trial-level analysis ----
  message("Fitting trial-level models...")
  trial_models <- fit_trial_models_ied(trial_dat)

  trial_coef_df <- extract_coef_table_lmer(trial_models$m_full)
  trial_coef_display <- display_coef_table(trial_coef_df)
  trial_compare_df <- extract_model_compare_table_lmer(trial_models)
  trial_compare_display <- display_model_compare_table_lmer(trial_compare_df)
  trial_fit_display <- build_fit_summary_trial(trial_models, trial_dat, trial_compare_df)
  trial_formula_display <- build_formula_display_trial()

  # Random effect plot
  re_df <- extract_random_effect_df(trial_models$m0, "Patient")
  re_plot <- make_random_effect_plot(re_df, "Patient random intercepts (trial-level m0)")

  # Save trial-level outputs
  write.csv(trial_coef_df, file.path(trial_dir, "coefficients.csv"), row.names = FALSE)
  write.csv(trial_compare_df, file.path(trial_dir, "model_comparison.csv"), row.names = FALSE)
  write.csv(trial_fit_display, file.path(trial_dir, "fit_summary.csv"), row.names = FALSE)
  save_table_png(trial_coef_display, file.path(trial_dir, "coefficients_table.png"),
                 title = "Trial-level coefficients (full model)")
  save_table_png(trial_compare_display, file.path(trial_dir, "model_comparison_table.png"),
                 title = "Trial-level model comparison")
  save_plot(re_plot, file.path(trial_dir, "random_intercepts.png"), width = 8, height = 6)

  # Trial-level effect plots
  t1 <- make_effect_plot_trial(trial_models$m_full, trial_dat, "BeforeImgITI", "BeforeImgITI (trial-level)")
  t2 <- make_effect_plot_trial(trial_models$m_full, trial_dat, "DuringImg", "DuringImg (trial-level)")
  t3 <- make_effect_plot_trial(trial_models$m_full, trial_dat, "DuringStim", "DuringStim (trial-level)")
  t4 <- make_effect_plot_trial(trial_models$m_full, trial_dat, "AfterImgITI", "AfterImgITI (trial-level)")

  save_plot(t1, file.path(trial_dir, "BeforeImgITI_effect.png"))
  save_plot(t2, file.path(trial_dir, "DuringImg_effect.png"))
  save_plot(t3, file.path(trial_dir, "DuringStim_effect.png"))
  save_plot(t4, file.path(trial_dir, "AfterImgITI_effect.png"))

  save_summary_panel(
    file.path(trial_dir, "summary_panel.png"),
    level_label = "Trial-level (lmer)",
    n_patients = dplyr::n_distinct(trial_dat$Patient),
    coef_display = trial_coef_display,
    compare_display = trial_compare_display,
    plots = list(t1, t2, t3, t4)
  )

  # Trial-level overview
  save_overview_panel(
    file.path(overview_dir, "trial_level_overview.png"),
    level_label = "trial-level",
    formula_display = trial_formula_display,
    compare_display = trial_compare_display,
    fit_display = trial_fit_display,
    random_plot = re_plot
  )
  write.csv(trial_compare_df, file.path(overview_dir, "trial_level_model_comparison.csv"), row.names = FALSE)
  write.csv(trial_fit_display, file.path(overview_dir, "trial_level_fit_summary.csv"), row.names = FALSE)

  message("IED encoding analysis complete. Outputs saved to: ", output_dir)
}

# Run the analysis
run_ied_encoding_analysis()
