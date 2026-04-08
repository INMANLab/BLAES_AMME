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
    "(Intercept)"          = "(Intercept)",
    "ChannelSpread_z"      = "Channel Spread (z)",
    "RegionSpread_z"       = "Region Spread (z)",
    "HemisphereBilateral"  = "Hemisphere [Bilateral]",
    "HemisphereRight"      = "Hemisphere [Right]",
    "GrayMatterProp_z"     = "Gray Matter Prop (z)",
    "TrialChannelSpread_z" = "Trial Channel Spread (z)",
    "TrialRegionSpread_z"  = "Trial Region Spread (z)"
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

derive_patient_hemisphere <- function() {
  raw_csv <- file.path("IED", "AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned.csv")
  raw_dat <- read.csv(raw_csv, stringsAsFactors = FALSE)

  hemi_dat <- raw_dat %>%
    filter(Hemisphere %in% c("L", "R")) %>%
    group_by(Patient) %>%
    summarise(
      hemispheres = paste(sort(unique(Hemisphere)), collapse = ","),
      .groups = "drop"
    ) %>%
    mutate(
      Hemisphere = case_when(
        hemispheres == "L"   ~ "Left",
        hemispheres == "R"   ~ "Right",
        hemispheres == "L,R" ~ "Bilateral",
        TRUE ~ NA_character_
      )
    ) %>%
    select(Patient, Hemisphere)

  hemi_dat
}

prepare_ied_features_data <- function() {
  # Patient-level memory data
  memory_csv <- file.path("IED", "ied_memory_relationships", "memory_vs_ied_merged.csv")
  memory_dat <- read.csv(memory_csv, stringsAsFactors = FALSE) %>%
    select(Patient, avg_stim_dprime_diff)
  colnames(memory_dat)[2] <- "avg_dprime_diff"

  # Patient-level channel spread
  cs_csv <- file.path("IED", "ied_trial_level_summary", "patient_channelspread_summary.csv")
  cs_dat <- read.csv(cs_csv, stringsAsFactors = FALSE)

  # Patient-level region spread (proxy for LeadSpread)
  rs_csv <- file.path("IED", "ied_trial_level_summary", "patient_regionspread_summary.csv")
  rs_dat <- read.csv(rs_csv, stringsAsFactors = FALSE)

  # Patient-level gray matter proportion
  gm_csv <- file.path("IED", "ied_trial_level_summary", "patient_graymatter_gw_summary.csv")
  gm_dat <- read.csv(gm_csv, stringsAsFactors = FALSE) %>%
    rename(GrayMatterProp = G) %>%
    select(Patient, GrayMatterProp)

  # Patient-level hemisphere
  hemi_dat <- derive_patient_hemisphere()

  # Merge patient-level data
  patient_dat <- memory_dat %>%
    inner_join(cs_dat, by = "Patient") %>%
    inner_join(rs_dat, by = "Patient") %>%
    left_join(gm_dat, by = "Patient") %>%
    left_join(hemi_dat, by = "Patient") %>%
    rename(
      ChannelSpread = PatientChannelSpread,
      RegionSpread  = PatientRegionSpread
    ) %>%
    filter(!is.na(GrayMatterProp), !is.na(Hemisphere))

  # Z-score continuous predictors
  patient_dat <- patient_dat %>%
    mutate(
      ChannelSpread_z  = as.numeric(scale(ChannelSpread)),
      RegionSpread_z   = as.numeric(scale(RegionSpread)),
      GrayMatterProp_z = as.numeric(scale(GrayMatterProp)),
      Hemisphere       = factor(Hemisphere, levels = c("Left", "Right", "Bilateral"))
    )

  # Trial-level data
  trial_cs_csv <- file.path("IED", "ied_trial_level_summary", "trial_channelspread_summary.csv")
  trial_rs_csv <- file.path("IED", "ied_trial_level_summary", "trial_regionspread_summary.csv")
  trial_cs <- read.csv(trial_cs_csv, stringsAsFactors = FALSE)
  trial_rs <- read.csv(trial_rs_csv, stringsAsFactors = FALSE)

  collapsed_csv <- file.path("IED", "ied_trial_level_summary", "collapsed_trial_level.csv")
  collapsed <- read.csv(collapsed_csv, stringsAsFactors = FALSE)

  trial_dat <- collapsed %>%
    inner_join(trial_cs, by = c("Patient", "Trial")) %>%
    inner_join(trial_rs, by = c("Patient", "Trial")) %>%
    inner_join(memory_dat, by = "Patient") %>%
    left_join(gm_dat, by = "Patient") %>%
    left_join(hemi_dat, by = "Patient") %>%
    filter(!is.na(GrayMatterProp), !is.na(Hemisphere)) %>%
    mutate(
      TrialChannelSpread_z = as.numeric(scale(TrialChannelSpread)),
      TrialRegionSpread_z  = as.numeric(scale(TrialRegionSpread)),
      GrayMatterProp_z     = as.numeric(scale(GrayMatterProp)),
      Hemisphere           = factor(Hemisphere, levels = c("Left", "Right", "Bilateral"))
    )

  list(patient_dat = patient_dat, trial_dat = trial_dat)
}

# ---- Model fitting ----

fit_patient_models_features <- function(dat) {
  list(
    m0     = lm(avg_dprime_diff ~ 1, data = dat),
    m1     = lm(avg_dprime_diff ~ ChannelSpread_z, data = dat),
    m2     = lm(avg_dprime_diff ~ ChannelSpread_z + RegionSpread_z, data = dat),
    m3     = lm(avg_dprime_diff ~ ChannelSpread_z + RegionSpread_z + Hemisphere, data = dat),
    m_full = lm(avg_dprime_diff ~ ChannelSpread_z + RegionSpread_z + Hemisphere + GrayMatterProp_z,
                data = dat)
  )
}

fit_trial_models_features <- function(dat) {
  ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  list(
    m0     = lmer(avg_dprime_diff ~ 1 + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE, control = ctrl),
    m1     = lmer(avg_dprime_diff ~ TrialChannelSpread_z + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE, control = ctrl),
    m2     = lmer(avg_dprime_diff ~ TrialChannelSpread_z + TrialRegionSpread_z + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE, control = ctrl),
    m3     = lmer(avg_dprime_diff ~ TrialChannelSpread_z + TrialRegionSpread_z + Hemisphere + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE, control = ctrl),
    m_full = lmer(avg_dprime_diff ~ TrialChannelSpread_z + TrialRegionSpread_z + Hemisphere + GrayMatterProp_z + (1 | Patient),
                  data = dat, na.action = na.exclude, REML = FALSE, control = ctrl)
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
    Predictor = vapply(coef_mat$term, term_label, character(1)),
    Estimate  = coef_mat$Estimate,
    std_error = coef_mat$`Std. Error`,
    conf_low  = conf[, 1],
    conf_high = conf[, 2],
    statistic = coef_mat[[statistic_col]],
    p_value   = p_values,
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
    Predictor = vapply(coef_mat$term, term_label, character(1)),
    Estimate  = coef_mat$Estimate,
    std_error = coef_mat$`Std. Error`,
    conf_low  = conf[, 1],
    conf_high = conf[, 2],
    statistic = coef_mat[[statistic_col]],
    p_value   = p_values,
    stringsAsFactors = FALSE
  )
}

display_coef_table <- function(df) {
  data.frame(
    Predictors  = df$Predictor,
    Estimate    = sprintf("%.3f", df$Estimate),
    `std. Error` = sprintf("%.3f", df$std_error),
    CI          = sprintf("%.3f to %.3f", df$conf_low, df$conf_high),
    Statistic   = sprintf("%.3f", df$statistic),
    p           = format_p_value(df$p_value),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

# ---- Model comparison ----

extract_model_compare_table_lm <- function(model_list) {
  comp <- as.data.frame(anova(model_list$m0, model_list$m1, model_list$m2, model_list$m3, model_list$m_full))
  comp$Model <- c("m0", "m1", "m2", "m3", "m_full")
  rownames(comp) <- NULL

  rss_col <- if ("RSS" %in% names(comp)) comp$RSS else rep(NA_real_, nrow(comp))
  f_col <- if ("F" %in% names(comp)) comp$F else rep(NA_real_, nrow(comp))
  p_col <- grep("^Pr\\(", names(comp), value = TRUE)
  p_vals <- if (length(p_col) == 1) comp[[p_col]] else rep(NA_real_, nrow(comp))

  data.frame(
    Model       = comp$Model,
    Res.Df      = comp$Res.Df,
    RSS         = rss_col,
    Df          = if ("Df" %in% names(comp)) comp$Df else rep(NA_real_, nrow(comp)),
    `Sum of Sq` = if ("Sum of Sq" %in% names(comp)) comp$`Sum of Sq` else rep(NA_real_, nrow(comp)),
    F_value     = f_col,
    p_value     = p_vals,
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
    Model       = comp$Model,
    npar        = comp$npar,
    AIC         = comp$AIC,
    BIC         = comp$BIC,
    logLik      = comp$logLik,
    minus_2logL = deviance_vals,
    Chisq       = chisq_vals,
    Df          = df_vals,
    p_value     = p_vals,
    stringsAsFactors = FALSE
  )
}

display_model_compare_table_lm <- function(df) {
  data.frame(
    Model       = df$Model,
    Res.Df      = ifelse(is.na(df$Res.Df), "", sprintf("%.0f", df$Res.Df)),
    RSS         = ifelse(is.na(df$RSS), "", sprintf("%.3f", df$RSS)),
    Df          = ifelse(is.na(df$Df), "", sprintf("%.0f", df$Df)),
    `Sum of Sq` = ifelse(is.na(df$`Sum of Sq`), "", sprintf("%.4f", df$`Sum of Sq`)),
    F           = ifelse(is.na(df$F_value), "", sprintf("%.4f", df$F_value)),
    `Pr(>F)`    = format_p_value(df$p_value),
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
      'lm("avg_dprime_diff ~ ChannelSpread_z")',
      'lm("avg_dprime_diff ~ ChannelSpread_z + RegionSpread_z")',
      'lm("avg_dprime_diff ~ ChannelSpread_z + RegionSpread_z + Hemisphere")',
      'lm("avg_dprime_diff ~ ChannelSpread_z + RegionSpread_z + Hemisphere + GrayMatterProp_z")'
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
      'lmer("avg_dprime_diff ~ TrialChannelSpread_z + (1 | Patient)")',
      'lmer("avg_dprime_diff ~ TrialChannelSpread_z + TrialRegionSpread_z + (1 | Patient)")',
      'lmer("avg_dprime_diff ~ TrialChannelSpread_z + TrialRegionSpread_z + Hemisphere + (1 | Patient)")',
      'lmer("avg_dprime_diff ~ TrialChannelSpread_z + TrialRegionSpread_z + Hemisphere + GrayMatterProp_z + (1 | Patient)")'
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

make_effect_plot_continuous_lm <- function(model, data, predictor, raw_predictor, title_text) {
  pred_min <- min(data[[predictor]], na.rm = TRUE)
  pred_max <- max(data[[predictor]], na.rm = TRUE)
  if (pred_min == pred_max) { pred_min <- pred_min - 0.5; pred_max <- pred_max + 0.5 }

  newdata <- data.frame(x_value = seq(pred_min, pred_max, length.out = 100))
  for (p in c("ChannelSpread_z", "RegionSpread_z", "GrayMatterProp_z")) {
    newdata[[p]] <- 0
  }
  newdata[["Hemisphere"]] <- factor("Left", levels = levels(data$Hemisphere))
  newdata[[predictor]] <- newdata$x_value

  preds <- predict(model, newdata = newdata, se.fit = TRUE)
  newdata$fit   <- preds$fit
  newdata$lower <- preds$fit - 1.96 * preds$se.fit
  newdata$upper <- preds$fit + 1.96 * preds$se.fit

  ggplot(newdata, aes(x = x_value, y = fit)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#a9c7e5", alpha = 0.45) +
    geom_line(color = "#0a78c9", linewidth = 1) +
    geom_point(data = data, aes(x = .data[[predictor]], y = avg_dprime_diff),
               inherit.aes = FALSE, alpha = 0.6, size = 2) +
    labs(x = paste0(raw_predictor, " (z-scored)"), y = "Memory Modulation (avg_dprime_diff)",
         title = title_text) +
    theme_bw(base_size = 16) +
    theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      panel.grid = element_blank()
    )
}

make_effect_plot_hemisphere_lm <- function(model, data, title_text) {
  newdata <- expand.grid(
    Hemisphere = factor(c("Left", "Right", "Bilateral"), levels = levels(data$Hemisphere)),
    stringsAsFactors = FALSE
  )
  for (p in c("ChannelSpread_z", "RegionSpread_z", "GrayMatterProp_z")) {
    newdata[[p]] <- 0
  }

  preds <- predict(model, newdata = newdata, se.fit = TRUE)
  newdata$fit   <- preds$fit
  newdata$lower <- preds$fit - 1.96 * preds$se.fit
  newdata$upper <- preds$fit + 1.96 * preds$se.fit

  ggplot(newdata, aes(x = Hemisphere, y = fit)) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2, linewidth = 0.8) +
    geom_point(size = 3, color = "#0a78c9") +
    geom_jitter(data = data, aes(x = Hemisphere, y = avg_dprime_diff),
                inherit.aes = FALSE, alpha = 0.4, width = 0.1, size = 2) +
    labs(x = "IED Hemisphere", y = "Memory Modulation (avg_dprime_diff)", title = title_text) +
    theme_bw(base_size = 16) +
    theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      panel.grid = element_blank()
    )
}

make_effect_plot_continuous_lmer <- function(model, data, predictor, raw_predictor, title_text) {
  pred_min <- min(data[[predictor]], na.rm = TRUE)
  pred_max <- max(data[[predictor]], na.rm = TRUE)
  if (pred_min == pred_max) { pred_min <- pred_min - 0.5; pred_max <- pred_max + 0.5 }

  newdata <- data.frame(x_value = seq(pred_min, pred_max, length.out = 100))
  for (p in c("TrialChannelSpread_z", "TrialRegionSpread_z", "GrayMatterProp_z")) {
    newdata[[p]] <- 0
  }
  newdata[["Hemisphere"]] <- factor("Left", levels = levels(data$Hemisphere))
  newdata[[predictor]] <- newdata$x_value

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

  ggplot(newdata, aes(x = x_value, y = fit)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#a9c7e5", alpha = 0.45) +
    geom_line(color = "#0a78c9", linewidth = 1) +
    labs(x = paste0(raw_predictor, " (z-scored)"), y = "Memory Modulation (avg_dprime_diff)",
         title = title_text) +
    theme_bw(base_size = 16) +
    theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      panel.grid = element_blank()
    )
}

make_effect_plot_hemisphere_lmer <- function(model, data, title_text) {
  newdata <- expand.grid(
    Hemisphere = factor(c("Left", "Right", "Bilateral"), levels = levels(data$Hemisphere)),
    stringsAsFactors = FALSE
  )
  for (p in c("TrialChannelSpread_z", "TrialRegionSpread_z", "GrayMatterProp_z")) {
    newdata[[p]] <- 0
  }

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

  ggplot(newdata, aes(x = Hemisphere, y = fit)) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2, linewidth = 0.8) +
    geom_point(size = 3, color = "#0a78c9") +
    labs(x = "IED Hemisphere", y = "Memory Modulation (avg_dprime_diff)", title = title_text) +
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
  title_text <- sprintf("IED Features Effects on Memory Modulation (N=%d) - %s", n_patients, level_label)

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
  title_text <- sprintf("IED Features - %s model fit", level_label)
  subtitle_text <- "IED spatial/structural feature effects on memory modulation (avg_dprime_diff)"

  formula_grob <- make_table_grob(formula_display, first_col_width = 0.7, base_size = 13,
                                  body_fontsize = 12, header_fontsize = 13)
  compare_grob <- make_table_grob(compare_display, first_col_width = 1.05, base_size = 13,
                                  body_fontsize = 12, header_fontsize = 13)
  fit_grob <- make_table_grob(fit_display, first_col_width = 1.55, base_size = 14,
                              body_fontsize = 13, header_fontsize = 14)
  note_grob <- textGrob(
    "Models compare nested IED feature predictors of memory modulation. Continuous predictors z-scored.",
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

run_ied_features_analysis <- function() {
  output_dir  <- ensure_dir(file.path("outputs", "stats", "ied_features"))
  overview_dir <- ensure_dir(file.path(output_dir, "overview"))
  patient_dir <- ensure_dir(file.path(output_dir, "patient_level"))
  trial_dir   <- ensure_dir(file.path(output_dir, "trial_level"))

  # Prepare data
  dat <- prepare_ied_features_data()
  patient_dat <- dat$patient_dat
  trial_dat   <- dat$trial_dat

  message(sprintf("Patient-level data: %d patients", nrow(patient_dat)))
  message(sprintf("Trial-level data: %d trials from %d patients",
                  nrow(trial_dat), dplyr::n_distinct(trial_dat$Patient)))

  # ---- Patient-level analysis ----
  message("Fitting patient-level models...")
  patient_models <- fit_patient_models_features(patient_dat)

  patient_coef_df <- extract_coef_table_lm(patient_models$m_full)
  patient_coef_display <- display_coef_table(patient_coef_df)
  patient_compare_df <- extract_model_compare_table_lm(patient_models)
  patient_compare_display <- display_model_compare_table_lm(patient_compare_df)
  patient_fit_display <- build_fit_summary_patient(patient_models, patient_dat, patient_compare_df)
  patient_formula_display <- build_formula_display_patient()

  write.csv(patient_coef_df, file.path(patient_dir, "coefficients.csv"), row.names = FALSE)
  write.csv(patient_compare_df, file.path(patient_dir, "model_comparison.csv"), row.names = FALSE)
  write.csv(patient_fit_display, file.path(patient_dir, "fit_summary.csv"), row.names = FALSE)
  save_table_png(patient_coef_display, file.path(patient_dir, "coefficients_table.png"),
                 title = "Patient-level coefficients (full model)")
  save_table_png(patient_compare_display, file.path(patient_dir, "model_comparison_table.png"),
                 title = "Patient-level model comparison")

  p1 <- make_effect_plot_continuous_lm(patient_models$m_full, patient_dat, "ChannelSpread_z", "ChannelSpread", "Channel Spread")
  p2 <- make_effect_plot_continuous_lm(patient_models$m_full, patient_dat, "RegionSpread_z", "RegionSpread", "Region Spread (LeadSpread proxy)")
  p3 <- make_effect_plot_hemisphere_lm(patient_models$m_full, patient_dat, "Hemisphere")
  p4 <- make_effect_plot_continuous_lm(patient_models$m_full, patient_dat, "GrayMatterProp_z", "GrayMatterProp", "Gray Matter Proportion")

  save_plot(p1, file.path(patient_dir, "ChannelSpread_effect.png"))
  save_plot(p2, file.path(patient_dir, "RegionSpread_effect.png"))
  save_plot(p3, file.path(patient_dir, "Hemisphere_effect.png"))
  save_plot(p4, file.path(patient_dir, "GrayMatterProp_effect.png"))

  save_summary_panel(
    file.path(patient_dir, "summary_panel.png"),
    level_label = "Patient-level (lm)",
    n_patients = nrow(patient_dat),
    coef_display = patient_coef_display,
    compare_display = patient_compare_display,
    plots = list(p1, p2, p3, p4)
  )

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
  trial_models <- fit_trial_models_features(trial_dat)

  trial_coef_df <- extract_coef_table_lmer(trial_models$m_full)
  trial_coef_display <- display_coef_table(trial_coef_df)
  trial_compare_df <- extract_model_compare_table_lmer(trial_models)
  trial_compare_display <- display_model_compare_table_lmer(trial_compare_df)
  trial_fit_display <- build_fit_summary_trial(trial_models, trial_dat, trial_compare_df)
  trial_formula_display <- build_formula_display_trial()

  re_df <- extract_random_effect_df(trial_models$m0, "Patient")
  re_plot <- make_random_effect_plot(re_df, "Patient random intercepts (trial-level m0)")

  write.csv(trial_coef_df, file.path(trial_dir, "coefficients.csv"), row.names = FALSE)
  write.csv(trial_compare_df, file.path(trial_dir, "model_comparison.csv"), row.names = FALSE)
  write.csv(trial_fit_display, file.path(trial_dir, "fit_summary.csv"), row.names = FALSE)
  save_table_png(trial_coef_display, file.path(trial_dir, "coefficients_table.png"),
                 title = "Trial-level coefficients (full model)")
  save_table_png(trial_compare_display, file.path(trial_dir, "model_comparison_table.png"),
                 title = "Trial-level model comparison")
  save_plot(re_plot, file.path(trial_dir, "random_intercepts.png"), width = 8, height = 6)

  t1 <- make_effect_plot_continuous_lmer(trial_models$m_full, trial_dat, "TrialChannelSpread_z", "TrialChannelSpread", "Trial Channel Spread")
  t2 <- make_effect_plot_continuous_lmer(trial_models$m_full, trial_dat, "TrialRegionSpread_z", "TrialRegionSpread", "Trial Region Spread")
  t3 <- make_effect_plot_hemisphere_lmer(trial_models$m_full, trial_dat, "Hemisphere")
  t4 <- make_effect_plot_continuous_lmer(trial_models$m_full, trial_dat, "GrayMatterProp_z", "GrayMatterProp", "Gray Matter Proportion")

  save_plot(t1, file.path(trial_dir, "ChannelSpread_effect.png"))
  save_plot(t2, file.path(trial_dir, "RegionSpread_effect.png"))
  save_plot(t3, file.path(trial_dir, "Hemisphere_effect.png"))
  save_plot(t4, file.path(trial_dir, "GrayMatterProp_effect.png"))

  save_summary_panel(
    file.path(trial_dir, "summary_panel.png"),
    level_label = "Trial-level (lmer)",
    n_patients = dplyr::n_distinct(trial_dat$Patient),
    coef_display = trial_coef_display,
    compare_display = trial_compare_display,
    plots = list(t1, t2, t3, t4)
  )

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

  message("IED features analysis complete. Outputs saved to: ", output_dir)
}

# Run the analysis
run_ied_features_analysis()
