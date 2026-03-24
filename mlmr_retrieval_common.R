suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(lme4)
  library(lmerTest)
})

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  path
}

sort_freq_cols <- function(df, prefix = "diff_Freq_") {
  cols <- names(df)[startsWith(names(df), prefix)]
  cols[order(as.numeric(sub(prefix, "", cols, fixed = TRUE)))]
}

sanitize_name <- function(x) {
  gsub("[^A-Za-z0-9]+", "_", x)
}

display_region_name <- function(region) {
  gsub("_", " - ", region, fixed = TRUE)
}

normalize_region_label <- function(region) {
  if (is.na(region)) {
    return(NA_character_)
  }

  parts <- strsplit(trimws(region), "_", fixed = TRUE)[[1]]
  parts <- trimws(parts)
  parts <- parts[nzchar(parts)]

  if (length(parts) == 0) {
    return(NA_character_)
  }

  parts[parts == "ER"] <- "EC"
  if (length(parts) == 2) {
    parts <- sort(parts)
  }

  paste(parts, collapse = "_")
}

normalize_region_labels <- function(regions) {
  vapply(regions, normalize_region_label, character(1), USE.NAMES = FALSE)
}

format_p_value <- function(x) {
  ifelse(is.na(x), "", ifelse(x < 0.001, "<0.001", sprintf("%.3f", x)))
}

term_label <- function(term) {
  labels <- c(
    "(Intercept)" = "(Intercept)",
    "theta_c" = "theta c",
    "slow_gamma_c" = "slow gamma c",
    "fast_gamma_c" = "fast gamma c",
    "StimCondstim" = "StimCond [stim]",
    "theta_c:StimCondstim" = "theta c x StimCond [stim]",
    "slow_gamma_c:StimCondstim" = "slow gamma c x StimCond [stim]",
    "fast_gamma_c:StimCondstim" = "fast gamma c x StimCond [stim]"
  )
  if (term %in% names(labels)) {
    labels[[term]]
  } else {
    term
  }
}

prepare_model_data <- function(input_csv) {
  dat <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)

  required_cols <- c("Measure", "Patient", "Region", "trial_type", "yes_or_no")
  missing_cols <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns in %s: %s", input_csv, paste(missing_cols, collapse = ", ")))
  }

  diff_cols <- sort_freq_cols(dat, "diff_Freq_")
  if (length(diff_cols) == 0) {
    stop(sprintf("No diff_Freq_ columns found in %s", input_csv))
  }

  dat$Region <- normalize_region_labels(dat$Region)

  dat <- dat %>%
    mutate(
      Memory = case_when(
        yes_or_no == "yes" & trial_type != "new" ~ "remembered",
        yes_or_no == "no" & trial_type != "new" ~ "forgotten",
        TRUE ~ NA_character_
      ),
      Accuracy = case_when(
        yes_or_no == "yes" & trial_type != "new" ~ 1,
        yes_or_no == "no" & trial_type != "new" ~ 0,
        TRUE ~ NA_real_
      ),
      StimCond = ifelse(trial_type == "nostim", "nostim", "stim")
    ) %>%
    filter(trial_type != "new", yes_or_no %in% c("yes", "no")) %>%
    group_by(Measure, Patient, Region) %>%
    mutate(trial_idx = row_number()) %>%
    ungroup()

  datmodel <- dat %>%
    select(Measure, Patient, Region, trial_type, trial_idx, yes_or_no, Memory, Accuracy, StimCond, all_of(diff_cols)) %>%
    pivot_longer(
      cols = all_of(diff_cols),
      names_to = "Freqs",
      values_to = "Metric_vals"
    ) %>%
    mutate(Freqs = as.numeric(sub("diff_Freq_", "", Freqs, fixed = TRUE)))

  theta_band_range <- c(4, 8)
  slow_gamma_band_range <- c(30, 54.9999)
  fast_gamma_band_range <- c(55, 100)

  datmodel <- datmodel %>%
    group_by(Measure, Patient, Region, StimCond, trial_idx, Accuracy, Memory) %>%
    summarise(
      theta = mean(Metric_vals[Freqs >= theta_band_range[1] & Freqs <= theta_band_range[2]], na.rm = TRUE),
      slow_gamma = mean(Metric_vals[Freqs >= slow_gamma_band_range[1] & Freqs <= slow_gamma_band_range[2]], na.rm = TRUE),
      fast_gamma = mean(Metric_vals[Freqs >= fast_gamma_band_range[1] & Freqs <= fast_gamma_band_range[2]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      StimCond = factor(StimCond, levels = c("nostim", "stim")),
      theta_c = theta - mean(theta, na.rm = TRUE),
      slow_gamma_c = slow_gamma - mean(slow_gamma, na.rm = TRUE),
      fast_gamma_c = fast_gamma - mean(fast_gamma, na.rm = TRUE)
    )

  datmodel
}

build_patient_level_data <- function(region_data) {
  region_data %>%
    group_by(Measure, Patient, Region, StimCond) %>%
    summarise(
      Accuracy = mean(Accuracy, na.rm = TRUE),
      theta = mean(theta, na.rm = TRUE),
      slow_gamma = mean(slow_gamma, na.rm = TRUE),
      fast_gamma = mean(fast_gamma, na.rm = TRUE),
      N = n(),
      .groups = "drop"
    ) %>%
    mutate(
      StimCond = factor(StimCond, levels = c("nostim", "stim")),
      theta_c = theta - mean(theta, na.rm = TRUE),
      slow_gamma_c = slow_gamma - mean(slow_gamma, na.rm = TRUE),
      fast_gamma_c = fast_gamma - mean(fast_gamma, na.rm = TRUE)
    )
}

fit_patient_models <- function(dat) {
  list(
    m_full = lmer(
      Accuracy ~ (theta_c + slow_gamma_c + fast_gamma_c) * StimCond + (1 | Patient),
      data = dat,
      na.action = na.exclude,
      REML = FALSE
    ),
    m0 = lmer(Accuracy ~ 1 + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m1 = lmer(Accuracy ~ StimCond + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m2 = lmer(Accuracy ~ StimCond + theta_c + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m3 = lmer(Accuracy ~ StimCond + slow_gamma_c + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m4 = lmer(
      Accuracy ~ StimCond + theta_c + slow_gamma_c + fast_gamma_c + (1 | Patient),
      data = dat,
      na.action = na.exclude,
      REML = FALSE
    )
  )
}

fit_trial_models <- function(dat) {
  control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  list(
    m_full = glmer(
      Accuracy ~ (theta_c + slow_gamma_c + fast_gamma_c) * StimCond + (1 | Patient),
      data = dat,
      family = binomial,
      na.action = na.exclude,
      control = control
    ),
    m0 = glmer(Accuracy ~ 1 + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = control),
    m1 = glmer(Accuracy ~ StimCond + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = control),
    m2 = glmer(Accuracy ~ StimCond + theta_c + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = control),
    m3 = glmer(
      Accuracy ~ StimCond + slow_gamma_c + (1 | Patient),
      data = dat,
      family = binomial,
      na.action = na.exclude,
      control = control
    ),
    m4 = glmer(
      Accuracy ~ StimCond + theta_c + slow_gamma_c + fast_gamma_c + (1 | Patient),
      data = dat,
      family = binomial,
      na.action = na.exclude,
      control = control
    )
  )
}

extract_coef_table <- function(model, logistic = FALSE) {
  coef_mat <- as.data.frame(summary(model)$coefficients)
  coef_mat$term <- rownames(coef_mat)

  fixed_terms <- coef_mat$term
  conf <- suppressMessages(confint(model, parm = fixed_terms, method = "Wald"))
  conf <- conf[fixed_terms, , drop = FALSE]

  statistic_col <- if ("t value" %in% names(coef_mat)) "t value" else "z value"
  p_col <- grep("^Pr\\(", names(coef_mat), value = TRUE)
  p_values <- if (length(p_col) == 1) coef_mat[[p_col]] else rep(NA_real_, nrow(coef_mat))

  estimate_vals <- coef_mat$Estimate
  conf_low <- conf[, 1]
  conf_high <- conf[, 2]
  estimate_label <- "Estimate"

  if (logistic) {
    estimate_vals <- exp(estimate_vals)
    conf_low <- exp(conf_low)
    conf_high <- exp(conf_high)
    estimate_label <- "Odds Ratio"
  }

  data.frame(
    Predictor = vapply(coef_mat$term, term_label, character(1)),
    estimate = estimate_vals,
    std_error = coef_mat$`Std. Error`,
    conf_low = conf_low,
    conf_high = conf_high,
    statistic = coef_mat[[statistic_col]],
    p_value = p_values,
    estimate_label = estimate_label,
    stringsAsFactors = FALSE
  )
}

display_coef_table <- function(df) {
  estimate_heading <- unique(df$estimate_label)[1]
  data.frame(
    Predictors = df$Predictor,
    setNames(list(sprintf("%.2f", df$estimate)), estimate_heading),
    `std. Error` = sprintf("%.2f", df$std_error),
    CI = sprintf("%.2f to %.2f", df$conf_low, df$conf_high),
    Statistic = sprintf("%.2f", df$statistic),
    p = format_p_value(df$p_value),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

extract_model_compare_table <- function(model_list) {
  comp <- as.data.frame(anova(model_list$m0, model_list$m1, model_list$m2, model_list$m3, model_list$m4, model_list$m_full))
  comp$Model <- c("m0", "m1", "m2", "m3", "m4", "m_full")
  rownames(comp) <- NULL

  p_col <- grep("^Pr", names(comp), value = TRUE)
  p_vals <- if (length(p_col) == 1) comp[[p_col]] else rep(NA_real_, nrow(comp))
  chisq_vals <- if ("Chisq" %in% names(comp)) comp$Chisq else rep(NA_real_, nrow(comp))
  df_vals <- if ("Df" %in% names(comp)) comp$Df else rep(NA_real_, nrow(comp))
  deviance_vals <- if ("deviance" %in% names(comp)) comp$deviance else rep(NA_real_, nrow(comp))

  data.frame(
    Model = comp$Model,
    npar = comp$npar,
    AIC = comp$AIC,
    BIC = comp$BIC,
    logLik = comp$logLik,
    minus_2logL = deviance_vals,
    Chisq = chisq_vals,
    Df = df_vals,
    p_value = p_vals,
    stringsAsFactors = FALSE
  )
}

display_model_compare_table <- function(df) {
  data.frame(
    Model = df$Model,
    npar = ifelse(is.na(df$npar), "", sprintf("%.0f", df$npar)),
    AIC = ifelse(is.na(df$AIC), "", sprintf("%.3f", df$AIC)),
    BIC = ifelse(is.na(df$BIC), "", sprintf("%.3f", df$BIC)),
    logLik = ifelse(is.na(df$logLik), "", sprintf("%.3f", df$logLik)),
    `-2*log(L)` = ifelse(is.na(df$minus_2logL), "", sprintf("%.3f", df$minus_2logL)),
    Chisq = ifelse(is.na(df$Chisq), "", sprintf("%.4f", df$Chisq)),
    Df = ifelse(is.na(df$Df), "", sprintf("%.0f", df$Df)),
    `Pr(>Chi)` = format_p_value(df$p_value),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

predict_fixed_ci <- function(model, newdata, logistic = FALSE) {
  fixed_formula <- lme4::nobars(formula(model))
  term_matrix <- delete.response(terms(fixed_formula))
  model_matrix <- model.matrix(term_matrix, newdata)
  beta <- fixef(model)
  model_matrix <- model_matrix[, names(beta), drop = FALSE]
  fit_link <- as.vector(model_matrix %*% beta)
  se_link <- sqrt(pmax(0, diag(model_matrix %*% as.matrix(vcov(model)) %*% t(model_matrix))))

  if (logistic) {
    data.frame(
      fit = plogis(fit_link),
      lower = plogis(fit_link - 1.96 * se_link),
      upper = plogis(fit_link + 1.96 * se_link)
    )
  } else {
    data.frame(
      fit = fit_link,
      lower = fit_link - 1.96 * se_link,
      upper = fit_link + 1.96 * se_link
    )
  }
}

make_effect_plot <- function(model, data, predictor, title_text, logistic = FALSE) {
  pred_min <- min(data[[predictor]], na.rm = TRUE)
  pred_max <- max(data[[predictor]], na.rm = TRUE)
  if (!is.finite(pred_min) || !is.finite(pred_max)) {
    stop(sprintf("Non-finite predictor range for %s", predictor))
  }
  if (pred_min == pred_max) {
    pred_min <- pred_min - 0.5
    pred_max <- pred_max + 0.5
  }

  newdata <- expand.grid(
    StimCond = factor(c("nostim", "stim"), levels = levels(data$StimCond)),
    x_value = seq(pred_min, pred_max, length.out = 100),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  newdata$theta_c <- 0
  newdata$slow_gamma_c <- 0
  newdata$fast_gamma_c <- 0
  newdata[[predictor]] <- newdata$x_value

  preds <- predict_fixed_ci(model, newdata, logistic = logistic)
  plot_df <- cbind(newdata, preds)

  p <- ggplot(plot_df, aes(x = x_value, y = fit)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#a9c7e5", alpha = 0.45) +
    geom_line(color = "#0a78c9", linewidth = 1) +
    geom_rug(data = data, aes(x = .data[[predictor]]), sides = "b", inherit.aes = FALSE, alpha = 0.4) +
    facet_wrap(~StimCond, nrow = 1) +
    labs(x = NULL, y = "Accuracy", title = title_text) +
    theme_bw(base_size = 16) +
    theme(
      strip.text = element_text(size = 14),
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      axis.title.y = element_text(size = 18, face = "bold"),
      axis.text = element_text(size = 12),
      panel.grid = element_blank()
    )

  if (logistic) {
    p <- p + coord_cartesian(ylim = c(0, 1))
  }

  p
}

save_plot <- function(plot_obj, path, width = 8, height = 4, dpi = 300) {
  ggsave(filename = path, plot = plot_obj, width = width, height = height, dpi = dpi, units = "in")
}

make_table_grob <- function(df_display, first_col_width = 2.4, base_size = 16, body_fontsize = NULL, header_fontsize = NULL, column_widths = NULL) {
  if (is.null(body_fontsize)) {
    body_fontsize <- max(12, base_size - 1)
  }
  if (is.null(header_fontsize)) {
    header_fontsize <- base_size
  }
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

save_summary_panel <- function(
    output_path,
    measure_name,
    region_name,
    n_patients,
    level_label,
    phase_name,
    coef_display,
    compare_display,
    plots
) {
  title_text <- sprintf("%s in the %s (N=%d) - %s MLMM", measure_name, display_region_name(region_name), n_patients, level_label)
  phase_text <- tolower(phase_name)
  subtitle_text <- if (level_label == "patient-level") {
    sprintf("Patient-level mixed model from trial-level %s spectra", phase_text)
  } else {
    sprintf("Trial-level mixed model from baseline-corrected %s spectra", phase_text)
  }

  coef_grob <- make_table_grob(coef_display, first_col_width = 2.7)
  compare_grob <- make_table_grob(compare_display, first_col_width = 1.8)

  left_panel <- arrangeGrob(
    coef_grob,
    compare_grob,
    ncol = 1,
    heights = c(0.62, 0.38)
  )

  right_panel <- arrangeGrob(
    grobs = plots,
    ncol = 1,
    heights = c(0.34, 0.33, 0.33)
  )

  summary_grob <- arrangeGrob(
    textGrob(title_text, gp = gpar(fontsize = 24, fontface = "bold")),
    textGrob(subtitle_text, gp = gpar(fontsize = 15)),
    arrangeGrob(left_panel, right_panel, ncol = 2, widths = c(1.38, 1.12)),
    ncol = 1,
    heights = c(0.055, 0.03, 0.915)
  )

  png(filename = output_path, width = 13.333, height = 7.5, units = "in", res = 260, bg = "white")
  grid.newpage()
  pushViewport(viewport(x = 0.5, y = 0.5, width = unit(0.997, "npc"), height = unit(0.992, "npc")))
  grid.draw(summary_grob)
  upViewport()
  dev.off()
}

build_formula_display <- function(level_slug) {
  model_fn <- if (identical(level_slug, "trial_level")) "glmer" else "lmer"
  suffix <- if (identical(level_slug, "trial_level")) ", family = binomial" else ""
  data.frame(
    Model = c("m0", "m1", "m2", "m3", "m4", "m_full"),
    Formula = c(
      sprintf('%s("Accuracy ~ 1 + (1 | Patient)"%s)', model_fn, suffix),
      sprintf('%s("Accuracy ~ StimCond + (1 | Patient)"%s)', model_fn, suffix),
      sprintf('%s("Accuracy ~ StimCond + theta_c + (1 | Patient)"%s)', model_fn, suffix),
      sprintf('%s("Accuracy ~ StimCond + slow_gamma_c + (1 | Patient)"%s)', model_fn, suffix),
      sprintf('%s("Accuracy ~ StimCond + theta_c + slow_gamma_c + fast_gamma_c + (1 | Patient)"%s)', model_fn, suffix),
      sprintf('%s("Accuracy ~ (theta_c + slow_gamma_c + fast_gamma_c) * StimCond + (1 | Patient)"%s)', model_fn, suffix)
    ),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

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
    group = rownames(re_df),
    estimate = estimates,
    se = se_vals,
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

compute_model_fit_metrics <- function(model, data, logistic = FALSE) {
  varcorr_df <- as.data.frame(VarCorr(model))
  random_var <- sum(varcorr_df$vcov[varcorr_df$grp != "Residual"], na.rm = TRUE)
  patient_var <- sum(varcorr_df$vcov[varcorr_df$grp == "Patient"], na.rm = TRUE)
  residual_var <- if (logistic) {
    (pi ^ 2) / 3
  } else if (any(varcorr_df$grp == "Residual")) {
    sum(varcorr_df$vcov[varcorr_df$grp == "Residual"], na.rm = TRUE)
  } else {
    sigma(model) ^ 2
  }

  fixed_formula <- lme4::nobars(formula(model))
  term_matrix <- delete.response(terms(fixed_formula))
  model_matrix <- model.matrix(term_matrix, data)
  beta <- fixef(model)
  model_matrix <- model_matrix[, names(beta), drop = FALSE]
  fixed_linear <- as.vector(model_matrix %*% beta)
  fixed_var <- stats::var(fixed_linear, na.rm = TRUE)
  if (!is.finite(fixed_var)) {
    fixed_var <- 0
  }

  total_var <- fixed_var + random_var + residual_var
  list(
    patient_var = patient_var,
    residual_var = residual_var,
    icc = if (patient_var + residual_var > 0) patient_var / (patient_var + residual_var) else NA_real_,
    marginal_r2 = if (total_var > 0) fixed_var / total_var else NA_real_,
    conditional_r2 = if (total_var > 0) (fixed_var + random_var) / total_var else NA_real_
  )
}

build_fit_summary_display <- function(null_model, full_model, analysis_data, compare_df, region_count, logistic = FALSE) {
  null_metrics <- compute_model_fit_metrics(null_model, analysis_data, logistic = logistic)
  full_metrics <- compute_model_fit_metrics(full_model, analysis_data, logistic = logistic)
  full_row <- compare_df[compare_df$Model == "m_full", , drop = FALSE]

  data.frame(
    Metric = c(
      "Null patient variance",
      "Residual variance",
      "Null ICC",
      "N patients",
      "N regions",
      "Observations",
      "Full marginal R2",
      "Full conditional R2",
      "Full model AIC",
      "Full model BIC",
      "m_full vs prior p"
    ),
    Value = c(
      sprintf("%.3f", null_metrics$patient_var),
      sprintf("%.3f", null_metrics$residual_var),
      sprintf("%.3f", null_metrics$icc),
      sprintf("%d", dplyr::n_distinct(analysis_data$Patient)),
      sprintf("%d", region_count),
      sprintf("%d", nrow(analysis_data)),
      sprintf("%.3f", full_metrics$marginal_r2),
      sprintf("%.3f", full_metrics$conditional_r2),
      if (nrow(full_row) == 1) sprintf("%.3f", full_row$AIC) else "",
      if (nrow(full_row) == 1) sprintf("%.3f", full_row$BIC) else "",
      if (nrow(full_row) == 1) format_p_value(full_row$p_value) else ""
    ),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

save_overview_panel <- function(
    output_path,
    measure_name,
    phase_name,
    level_slug,
    formula_display,
    compare_display,
    fit_display,
    random_plot
) {
  level_label <- if (identical(level_slug, "patient_level")) "patient-level" else "trial-level"
  title_text <- sprintf("%s %s - overall %s model fit", phase_name, measure_name, level_label)
  subtitle_text <- "Pooled across all regions using the same model set as the region-level analyses"

  formula_grob <- make_table_grob(formula_display, first_col_width = 0.7, base_size = 13, body_fontsize = 12, header_fontsize = 13)
  compare_grob <- make_table_grob(compare_display, first_col_width = 1.05, base_size = 13, body_fontsize = 12, header_fontsize = 13)
  fit_grob <- make_table_grob(fit_display, first_col_width = 1.55, base_size = 14, body_fontsize = 13, header_fontsize = 14)
  note_grob <- textGrob(
    "Lower AIC/BIC indicates better relative fit. The final p-value compares m_full to the prior model in the tested sequence.",
    x = 0,
    hjust = 0,
    gp = gpar(fontsize = 13)
  )

  top_row <- arrangeGrob(
    formula_grob,
    compare_grob,
    ncol = 2,
    widths = c(1.55, 1.05)
  )

  bottom_row <- arrangeGrob(
    random_plot,
    arrangeGrob(fit_grob, note_grob, ncol = 1, heights = c(0.8, 0.2)),
    ncol = 2,
    widths = c(1.45, 0.9)
  )

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

write_overview_outputs <- function(datmodel, measure_name, output_dir, phase_name) {
  overview_dir <- ensure_dir(file.path(output_dir, "overview"))
  region_count <- dplyr::n_distinct(datmodel$Region)

  for (level_slug in c("patient_level", "trial_level")) {
    logistic <- identical(level_slug, "trial_level")
    analysis_data <- if (logistic) datmodel else build_patient_level_data(datmodel)
    analysis_data <- analysis_data %>%
      filter(is.finite(theta_c), is.finite(slow_gamma_c), is.finite(fast_gamma_c)) %>%
      mutate(StimCond = factor(StimCond, levels = c("nostim", "stim")))

    model_list <- if (logistic) fit_trial_models(analysis_data) else fit_patient_models(analysis_data)
    compare_df <- extract_model_compare_table(model_list)
    compare_display <- display_model_compare_table(compare_df)
    fit_display <- build_fit_summary_display(
      null_model = model_list$m0,
      full_model = model_list$m_full,
      analysis_data = analysis_data,
      compare_df = compare_df,
      region_count = region_count,
      logistic = logistic
    )
    formula_display <- build_formula_display(level_slug)
    random_plot <- make_random_effect_plot(
      extract_random_effect_df(model_list$m0, "Patient"),
      sprintf("Patient random intercepts from %s", if (logistic) "trial-level m0" else "patient-level m0")
    )

    write.csv(compare_df, file.path(overview_dir, sprintf("%s_model_comparison.csv", level_slug)), row.names = FALSE)
    write.csv(fit_display, file.path(overview_dir, sprintf("%s_fit_summary.csv", level_slug)), row.names = FALSE)
    save_overview_panel(
      output_path = file.path(overview_dir, sprintf("%s_overview.png", level_slug)),
      measure_name = measure_name,
      phase_name = phase_name,
      level_slug = level_slug,
      formula_display = formula_display,
      compare_display = compare_display,
      fit_display = fit_display,
      random_plot = random_plot
    )
  }
}

analyze_region_level <- function(region_name, region_data, measure_name, level_slug, output_root, phase_name) {
  level_label <- if (level_slug == "patient_level") "patient-level" else "trial-level"
  logistic <- identical(level_slug, "trial_level")
  region_dir <- ensure_dir(file.path(output_root, level_slug, sanitize_name(region_name)))

  analysis_data <- if (logistic) region_data else build_patient_level_data(region_data)
  analysis_data <- analysis_data %>%
    filter(is.finite(theta_c), is.finite(slow_gamma_c), is.finite(fast_gamma_c)) %>%
    mutate(StimCond = factor(StimCond, levels = c("nostim", "stim")))

  if (nrow(analysis_data) < 6 ||
      dplyr::n_distinct(analysis_data$Patient) < 2 ||
      dplyr::n_distinct(analysis_data$StimCond) < 2) {
    return(list(success = FALSE, reason = "insufficient_data"))
  }

  model_list <- tryCatch(
    if (logistic) fit_trial_models(analysis_data) else fit_patient_models(analysis_data),
    error = function(e) e
  )
  if (inherits(model_list, "error")) {
    return(list(success = FALSE, reason = conditionMessage(model_list)))
  }

  coef_df <- extract_coef_table(model_list$m_full, logistic = logistic)
  compare_df <- extract_model_compare_table(model_list)
  coef_display <- display_coef_table(coef_df)
  compare_display <- display_model_compare_table(compare_df)

  coef_csv <- file.path(region_dir, "coefficients.csv")
  compare_csv <- file.path(region_dir, "model_comparison.csv")
  coef_png <- file.path(region_dir, "coefficients_table.png")
  compare_png <- file.path(region_dir, "model_comparison_table.png")
  summary_png <- file.path(region_dir, "summary_panel.png")
  theta_png <- file.path(region_dir, "theta_effect.png")
  slow_png <- file.path(region_dir, "slow_gamma_effect.png")
  fast_png <- file.path(region_dir, "fast_gamma_effect.png")

  write.csv(coef_df, coef_csv, row.names = FALSE)
  write.csv(compare_df, compare_csv, row.names = FALSE)
  save_table_png(coef_display, coef_png, title = sprintf("%s coefficients", tools::toTitleCase(level_label)))
  save_table_png(compare_display, compare_png, title = "Model comparison")

  theta_plot <- make_effect_plot(model_list$m_full, analysis_data, "theta_c", "Theta (4-8Hz)", logistic = logistic)
  slow_plot <- make_effect_plot(model_list$m_full, analysis_data, "slow_gamma_c", "Slow gamma (30-55Hz)", logistic = logistic)
  fast_plot <- make_effect_plot(model_list$m_full, analysis_data, "fast_gamma_c", "HFA (55-100Hz)", logistic = logistic)

  save_plot(theta_plot, theta_png)
  save_plot(slow_plot, slow_png)
  save_plot(fast_plot, fast_png)
  save_summary_panel(
    summary_png,
    measure_name = measure_name,
    region_name = region_name,
    n_patients = dplyr::n_distinct(region_data$Patient),
    level_label = level_label,
    phase_name = phase_name,
    coef_display = coef_display,
    compare_display = compare_display,
    plots = list(theta_plot, slow_plot, fast_plot)
  )

  list(
    success = TRUE,
    region = region_name,
    level = level_slug,
    n_patients = dplyr::n_distinct(region_data$Patient),
    n_rows = nrow(analysis_data),
    summary_panel = summary_png,
    coefficients_csv = coef_csv,
    model_comparison_csv = compare_csv,
    theta_plot = theta_png,
    slow_gamma_plot = slow_png,
    fast_gamma_plot = fast_png
  )
}

run_mlmr_analysis <- function(measure_name, input_csv, output_dir, phase_name = "Retrieval") {
  ensure_dir(output_dir)
  datmodel <- prepare_model_data(input_csv)
  write_overview_outputs(datmodel, measure_name, output_dir, phase_name)
  regions <- sort(unique(datmodel$Region))

  manifest_rows <- list()
  skipped_rows <- list()

  for (region_name in regions) {
    message(sprintf("Analyzing %s: %s", measure_name, region_name))
    region_data <- datmodel %>% filter(Region == region_name)

    for (level_slug in c("patient_level", "trial_level")) {
      result <- analyze_region_level(region_name, region_data, measure_name, level_slug, output_dir, phase_name)
      if (isTRUE(result$success)) {
        manifest_rows[[length(manifest_rows) + 1]] <- result
      } else {
        skipped_rows[[length(skipped_rows) + 1]] <- data.frame(
          region = region_name,
          level = level_slug,
          reason = result$reason,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(manifest_rows) > 0) {
    manifest_df <- bind_rows(lapply(manifest_rows, as.data.frame))
    write.csv(manifest_df, file.path(output_dir, "manifest.csv"), row.names = FALSE)
  }

  if (length(skipped_rows) > 0) {
    skipped_df <- bind_rows(skipped_rows)
    write.csv(skipped_df, file.path(output_dir, "skipped_regions.csv"), row.names = FALSE)
  }
}
