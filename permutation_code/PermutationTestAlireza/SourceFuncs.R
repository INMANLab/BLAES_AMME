################################ From mlmr_retrieval_common.R ------
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
      .groups = "drop")
    # ) %>%
    # mutate(
    #   StimCond = factor(StimCond, levels = c("nostim", "stim")),
    #   theta_c = theta - mean(theta, na.rm = TRUE),
    #   slow_gamma_c = slow_gamma - mean(slow_gamma, na.rm = TRUE),
    #   fast_gamma_c = fast_gamma - mean(fast_gamma, na.rm = TRUE)
    # )
  
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
  # Build anova call dynamically based on available models
  ordered_names <- c("m0", "m1", "m2", "m3", "m4", "m_full")
  available <- ordered_names[ordered_names %in% names(model_list)]
  available_models <- unname(model_list[available])
  comp <- as.data.frame(do.call(anova, available_models))
  comp$Model <- available
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

run_mlmr_analysis <- function(measure_name, input_csv, output_dir, phase_name = "Retrieval", region_filter = NULL) {
  ensure_dir(output_dir)
  datmodel <- prepare_model_data(input_csv)
  if (!is.null(region_filter)) {
    datmodel <- region_filter(datmodel)
  }
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



################################ From mlmr_band_region_common.R ------
BAND_SPECS <- list(
  theta = list(column = "theta", centered = "theta_c", label = "Theta (4-8Hz)", short = "theta c", slug = "theta"),
  slow_gamma = list(column = "slow_gamma", centered = "slow_gamma_c", label = "Slow gamma (30-55Hz)", short = "slow gamma c", slug = "slow_gamma"),
  fast_gamma = list(column = "fast_gamma", centered = "fast_gamma_c", label = "HFA (55-100Hz)", short = "fast gamma c", slug = "fast_gamma"),
  slow_gamma_pac = list(column = "slow_gamma_pac", centered = "slow_gamma_pac_c", label = "Theta Phase x Slow Gamma Amp (30-50Hz)", short = "Slow Gamma PAC", slug = "slow_gamma_pac"),
  hfa_pac = list(column = "hfa_pac", centered = "hfa_pac_c", label = "Theta Phase x HFA Amp (55-100Hz)", short = "HFA PAC", slug = "hfa_pac")
)

label_band_region_term <- function(term, band_key) {
  spec <- BAND_SPECS[[band_key]]
  components <- strsplit(term, ":", fixed = TRUE)[[1]]
  labeled <- vapply(
    components,
    function(component) {
      if (identical(component, spec$centered)) {
        return(spec$short)
      }
      if (identical(component, "StimCondstim")) {
        return("StimCond [stim]")
      }
      if (startsWith(component, "Region")) {
        region_name <- sub("^Region", "", component)
        return(sprintf("Region [%s]", display_region_name(region_name)))
      }
      component
    },
    character(1)
  )
  paste(labeled, collapse = " x ")
}

extract_band_region_coef_table <- function(model, band_key, logistic = FALSE) {
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
    Predictor = vapply(coef_mat$term, label_band_region_term, character(1), band_key = band_key),
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

prepare_band_analysis_data <- function(datmodel, band_key, level_slug) {
  spec <- BAND_SPECS[[band_key]]
  if (!(spec$column %in% names(datmodel))) return(data.frame())
  if (!(spec$centered %in% names(datmodel))) return(data.frame())
  
  dat <- if (identical(level_slug, "trial_level")) datmodel else build_patient_level_data(datmodel)
  if (!(spec$column %in% names(dat))) return(data.frame())
  
  dat %>%
    mutate(
      Region = factor(Region, levels = sort(unique(Region))),
      StimCond = factor(StimCond, levels = c("nostim", "stim")),
      band_value = .data[[spec$column]],
      band_c = .data[[spec$centered]]
    ) %>%
    filter(is.finite(band_value), is.finite(band_c))
}

fit_patient_band_region_models <- function(dat) {
  list(
    m_full = lmer(
      Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
      data = dat,
      na.action = na.exclude,
      REML = FALSE
    ),
    m0 = lmer(Accuracy ~ 1 + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m1 = lmer(Accuracy ~ StimCond + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m2 = lmer(Accuracy ~ StimCond + band_c + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m3 = lmer(Accuracy ~ StimCond + Region + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m4 = lmer(Accuracy ~ StimCond + band_c + Region + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE)
  )
}

fit_trial_band_region_models <- function(dat) {
  control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  list(
    m_full = glmer(
      Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
      data = dat,
      family = binomial,
      na.action = na.exclude,
      control = control
    ),
    m0 = glmer(Accuracy ~ 1 + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = control),
    m1 = glmer(Accuracy ~ StimCond + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = control),
    m2 = glmer(Accuracy ~ StimCond + band_c + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = control),
    m3 = glmer(Accuracy ~ StimCond + Region + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = control),
    m4 = glmer(Accuracy ~ StimCond + band_c + Region + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = control)
  )
}

build_band_region_formula_display <- function(level_slug, band_key) {
  model_fn <- if (identical(level_slug, "trial_level")) "glmer" else "lmer"
  suffix <- if (identical(level_slug, "trial_level")) ", family = binomial" else ""
  spec <- BAND_SPECS[[band_key]]
  
  data.frame(
    Model = c("m0", "m1", "m2", "m3", "m4", "m_full"),
    Formula = c(
      sprintf('%s("Accuracy ~ 1 + (1 | Patient)"%s)', model_fn, suffix),
      sprintf('%s("Accuracy ~ StimCond + (1 | Patient)"%s)', model_fn, suffix),
      sprintf('%s("Accuracy ~ StimCond + %s + (1 | Patient)"%s)', model_fn, spec$centered, suffix),
      sprintf('%s("Accuracy ~ StimCond + Region + (1 | Patient)"%s)', model_fn, suffix),
      sprintf('%s("Accuracy ~ StimCond + %s + Region + (1 | Patient)"%s)', model_fn, spec$centered, suffix),
      sprintf('%s("Accuracy ~ StimCond + Region + %s + %s:StimCond + %s:Region + (1 | Patient)"%s)',
              model_fn, spec$centered, spec$centered, spec$centered, suffix)
    ),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

make_band_region_effect_plot <- function(model, data, band_key, logistic = FALSE) {
  spec <- BAND_SPECS[[band_key]]
  pred_min <- min(data$band_c, na.rm = TRUE)
  pred_max <- max(data$band_c, na.rm = TRUE)
  if (pred_min == pred_max) {
    pred_min <- pred_min - 0.5
    pred_max <- pred_max + 0.5
  }
  
  newdata <- expand.grid(
    StimCond = factor(c("nostim", "stim"), levels = levels(data$StimCond)),
    Region = factor(levels(data$Region), levels = levels(data$Region)),
    band_c = seq(pred_min, pred_max, length.out = 100),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  
  preds <- predict_fixed_ci(model, newdata, logistic = logistic)
  plot_df <- cbind(newdata, preds)
  
  p <- ggplot(plot_df, aes(x = band_c, y = fit)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#a9c7e5", alpha = 0.35) +
    geom_line(color = "#0a78c9", linewidth = 0.8) +
    facet_grid(StimCond ~ Region, labeller = labeller(Region = function(x) vapply(x, display_region_name, character(1)))) +
    labs(x = spec$label, y = "Accuracy", title = NULL) +
    theme_bw(base_size = 13) +
    theme(
      strip.text = element_text(size = 8, face = "bold"),
      axis.title = element_text(size = 11.5),
      axis.text = element_text(size = 7.8),
      panel.grid = element_blank(),
      panel.spacing = unit(0.45, "lines")
    )
  
  if (logistic) {
    p <- p + coord_cartesian(ylim = c(0, 1))
  }
  
  p
}

save_band_region_summary_panel <- function(
    output_path,
    measure_name,
    band_key,
    level_label,
    phase_name,
    n_patients,
    n_regions,
    coef_display,
    compare_display,
    effect_plot
) {
  spec <- BAND_SPECS[[band_key]]
  title_text <- sprintf("%s - %s - %s", measure_name, spec$label, tools::toTitleCase(level_label))
  subtitle_text <- sprintf("%s mixed model across regions within %s (N=%d patients, %d regions)",
                           phase_name, spec$label, n_patients, n_regions)
  
  coef_grob <- make_table_grob(
    coef_display,
    base_size = 9.8,
    body_fontsize = 9.1,
    header_fontsize = 10,
    column_widths = c(3.6, 1.0, 1.05, 1.7, 0.9, 0.8)
  )
  compare_grob <- make_table_grob(
    compare_display,
    base_size = 9.2,
    body_fontsize = 8.6,
    header_fontsize = 9.4,
    column_widths = c(1.0, 0.7, 0.9, 0.9, 0.95, 1.1, 0.8, 0.55, 0.95)
  )
  
  left_panel <- arrangeGrob(
    coef_grob,
    compare_grob,
    ncol = 1,
    heights = c(0.68, 0.32)
  )
  
  summary_grob <- arrangeGrob(
    textGrob(title_text, gp = gpar(fontsize = 16.5, fontface = "bold")),
    textGrob(subtitle_text, gp = gpar(fontsize = 10.2)),
    arrangeGrob(left_panel, ggplotGrob(effect_plot), ncol = 2, widths = c(1.08, 1.24)),
    ncol = 1,
    heights = c(0.05, 0.025, 0.925)
  )
  
  png(filename = output_path, width = 13.333, height = 7.5, units = "in", res = 260, bg = "white")
  grid.newpage()
  pushViewport(viewport(x = 0.5, y = 0.5, width = unit(0.997, "npc"), height = unit(0.992, "npc")))
  grid.draw(summary_grob)
  upViewport()
  dev.off()
}

save_band_region_overview_panel <- function(
    output_path,
    measure_name,
    phase_name,
    level_slug,
    band_key,
    formula_display,
    compare_display,
    fit_display,
    random_plot
) {
  spec <- BAND_SPECS[[band_key]]
  level_label <- if (identical(level_slug, "patient_level")) "patient-level" else "trial-level"
  title_text <- sprintf("%s %s - %s %s model fit", phase_name, measure_name, spec$label, level_label)
  subtitle_text <- sprintf("Pooled across all regions using the same %s model set as the band-by-region analyses", spec$label)
  
  formula_grob <- make_table_grob(
    formula_display,
    base_size = 10.4,
    body_fontsize = 9.7,
    header_fontsize = 10.4,
    column_widths = c(0.8, 3.5)
  )
  compare_grob <- make_table_grob(
    compare_display,
    base_size = 10.0,
    body_fontsize = 9.2,
    header_fontsize = 10.0,
    column_widths = c(1.0, 0.7, 0.9, 0.9, 0.95, 1.05, 0.8, 0.55, 0.95)
  )
  fit_grob <- make_table_grob(
    fit_display,
    base_size = 10.4,
    body_fontsize = 9.8,
    header_fontsize = 10.4,
    column_widths = c(1.8, 1.0)
  )
  note_grob <- textGrob(
    "Lower AIC/BIC indicates better relative fit. The final p-value compares m_full to the prior model in the tested sequence.",
    x = 0,
    hjust = 0,
    gp = gpar(fontsize = 9.5)
  )
  
  top_row <- arrangeGrob(
    formula_grob,
    compare_grob,
    ncol = 2,
    widths = c(1.25, 1.05)
  )
  
  bottom_row <- arrangeGrob(
    random_plot,
    arrangeGrob(fit_grob, note_grob, ncol = 1, heights = c(0.82, 0.18)),
    ncol = 2,
    widths = c(1.38, 0.95)
  )
  
  overview_grob <- arrangeGrob(
    textGrob(title_text, gp = gpar(fontsize = 19, fontface = "bold")),
    textGrob(subtitle_text, gp = gpar(fontsize = 11)),
    top_row,
    bottom_row,
    ncol = 1,
    heights = c(0.055, 0.03, 0.36, 0.555)
  )
  
  png(filename = output_path, width = 13.333, height = 7.5, units = "in", res = 260, bg = "white")
  grid.newpage()
  pushViewport(viewport(x = 0.5, y = 0.5, width = unit(0.997, "npc"), height = unit(0.992, "npc")))
  grid.draw(overview_grob)
  upViewport()
  dev.off()
}

write_band_region_overview_outputs <- function(datmodel, measure_name, output_dir, phase_name) {
  overview_dir <- ensure_dir(file.path(output_dir, "overview"))
  region_count <- dplyr::n_distinct(datmodel$Region)
  overview_paths <- list()
  
  for (band_key in names(BAND_SPECS)) {
    for (level_slug in c("patient_level", "trial_level")) {
      logistic <- identical(level_slug, "trial_level")
      analysis_data <- prepare_band_analysis_data(datmodel, band_key, level_slug)
      
      if (nrow(analysis_data) < 12 ||
          dplyr::n_distinct(analysis_data$Patient) < 2 ||
          dplyr::n_distinct(analysis_data$StimCond) < 2 ||
          dplyr::n_distinct(analysis_data$Region) < 2) {
        next
      }
      
      model_list <- tryCatch(
        if (logistic) fit_trial_band_region_models(analysis_data) else fit_patient_band_region_models(analysis_data),
        error = function(e) e
      )
      if (inherits(model_list, "error")) {
        next
      }
      
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
      formula_display <- build_band_region_formula_display(level_slug, band_key)
      random_plot <- make_random_effect_plot(
        extract_random_effect_df(model_list$m0, "Patient"),
        sprintf("Patient random intercepts from %s", if (logistic) "trial-level m0" else "patient-level m0")
      )
      
      compare_csv <- file.path(overview_dir, sprintf("%s_%s_model_comparison.csv", level_slug, BAND_SPECS[[band_key]]$slug))
      fit_csv <- file.path(overview_dir, sprintf("%s_%s_fit_summary.csv", level_slug, BAND_SPECS[[band_key]]$slug))
      formula_csv <- file.path(overview_dir, sprintf("%s_%s_formulas.csv", level_slug, BAND_SPECS[[band_key]]$slug))
      overview_png <- file.path(overview_dir, sprintf("%s_%s_overview.png", level_slug, BAND_SPECS[[band_key]]$slug))
      
      write.csv(compare_df, compare_csv, row.names = FALSE)
      write.csv(fit_display, fit_csv, row.names = FALSE)
      write.csv(formula_display, formula_csv, row.names = FALSE)
      save_band_region_overview_panel(
        output_path = overview_png,
        measure_name = measure_name,
        phase_name = phase_name,
        level_slug = level_slug,
        band_key = band_key,
        formula_display = formula_display,
        compare_display = compare_display,
        fit_display = fit_display,
        random_plot = random_plot
      )
      overview_paths[[paste(band_key, level_slug, sep = "::")]] <- overview_png
    }
  }
  
  overview_paths
}

analyze_band_region_level <- function(band_key, datmodel, measure_name, level_slug, output_root, phase_name) {
  level_label <- if (identical(level_slug, "patient_level")) "patient_level" else "trial_level"
  logistic <- identical(level_slug, "trial_level")
  band_dir <- ensure_dir(file.path(output_root, level_slug, BAND_SPECS[[band_key]]$slug))
  analysis_data <- prepare_band_analysis_data(datmodel, band_key, level_slug)
  
  if (nrow(analysis_data) < 12 ||
      dplyr::n_distinct(analysis_data$Patient) < 2 ||
      dplyr::n_distinct(analysis_data$StimCond) < 2 ||
      dplyr::n_distinct(analysis_data$Region) < 2) {
    return(list(success = FALSE, reason = "insufficient_data"))
  }
  
  model_list <- tryCatch(
    if (logistic) fit_trial_band_region_models(analysis_data) else fit_patient_band_region_models(analysis_data),
    error = function(e) e
  )
  if (inherits(model_list, "error")) {
    return(list(success = FALSE, reason = conditionMessage(model_list)))
  }
  
  coef_df <- extract_band_region_coef_table(model_list$m_full, band_key, logistic = logistic)
  compare_df <- extract_model_compare_table(model_list)
  coef_display <- display_coef_table(coef_df)
  compare_display <- display_model_compare_table(compare_df)
  effect_plot <- make_band_region_effect_plot(model_list$m_full, analysis_data, band_key, logistic = logistic)
  
  coef_csv <- file.path(band_dir, "coefficients.csv")
  compare_csv <- file.path(band_dir, "model_comparison.csv")
  effect_png <- file.path(band_dir, "band_effect_by_region.png")
  summary_png <- file.path(band_dir, "summary_panel.png")
  
  write.csv(coef_df, coef_csv, row.names = FALSE)
  write.csv(compare_df, compare_csv, row.names = FALSE)
  save_plot(effect_plot, effect_png, width = 12, height = 6)
  save_band_region_summary_panel(
    output_path = summary_png,
    measure_name = measure_name,
    band_key = band_key,
    level_label = gsub("_", "-", level_label, fixed = TRUE),
    phase_name = phase_name,
    n_patients = dplyr::n_distinct(analysis_data$Patient),
    n_regions = dplyr::n_distinct(analysis_data$Region),
    coef_display = coef_display,
    compare_display = compare_display,
    effect_plot = effect_plot
  )
  
  list(
    success = TRUE,
    band = band_key,
    level = level_slug,
    n_patients = dplyr::n_distinct(analysis_data$Patient),
    n_regions = dplyr::n_distinct(analysis_data$Region),
    n_rows = nrow(analysis_data),
    summary_panel = summary_png,
    coefficients_csv = coef_csv,
    model_comparison_csv = compare_csv,
    effect_plot = effect_png
  )
}

run_band_region_mlmr_analysis <- function(measure_name, input_csv, output_dir, phase_name = "Retrieval", region_filter = NULL) {
  ensure_dir(output_dir)
  datmodel <- prepare_model_data(input_csv)
  if (!is.null(region_filter)) {
    datmodel <- region_filter(datmodel)
  }
  overview_paths <- write_band_region_overview_outputs(datmodel, measure_name, output_dir, phase_name)
  
  manifest_rows <- list()
  skipped_rows <- list()
  
  for (band_key in names(BAND_SPECS)) {
    message(sprintf("Analyzing %s band comparison: %s", measure_name, BAND_SPECS[[band_key]]$label))
    for (level_slug in c("patient_level", "trial_level")) {
      result <- analyze_band_region_level(band_key, datmodel, measure_name, level_slug, output_dir, phase_name)
      if (isTRUE(result$success)) {
        overview_key <- paste(band_key, level_slug, sep = "::")
        if (!is.null(overview_paths[[overview_key]])) {
          result$overview_panel <- overview_paths[[overview_key]]
        }
        manifest_rows[[length(manifest_rows) + 1]] <- result
      } else {
        skipped_rows[[length(skipped_rows) + 1]] <- data.frame(
          band = band_key,
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
    write.csv(skipped_df, file.path(output_dir, "skipped_bands.csv"), row.names = FALSE)
  }
}


################################ From mlmr_pac_common.R-----

# ---------------------------------------------------------------------------
# PAC band definitions (amplitude frequency ranges)
# ---------------------------------------------------------------------------
PAC_SLOW_GAMMA_RANGE <- c(30, 50)
PAC_HFA_RANGE <- c(55, 100)

# ---------------------------------------------------------------------------
# Prepare PAC model data from CSV
# ---------------------------------------------------------------------------
prepare_pac_model_data <- function(input_csv, phase = "encoding") {
  dat <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
  
  required_cols <- c("Measure", "Patient", "Region", "trial_type", "yes_or_no")
  missing <- setdiff(required_cols, names(dat))
  if (length(missing) > 0) {
    stop(sprintf("Missing columns: %s", paste(missing, collapse = ", ")))
  }
  
  diff_cols <- sort_freq_cols(dat, "diff_Freq_")
  if (length(diff_cols) == 0) stop("No diff_Freq_ columns found")
  
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
  
  # Compute band means from diff_Freq columns
  freq_vals <- as.numeric(sub("diff_Freq_", "", diff_cols, fixed = TRUE))
  
  slow_gamma_cols <- diff_cols[freq_vals >= PAC_SLOW_GAMMA_RANGE[1] & freq_vals <= PAC_SLOW_GAMMA_RANGE[2]]
  hfa_cols <- diff_cols[freq_vals >= PAC_HFA_RANGE[1] & freq_vals <= PAC_HFA_RANGE[2]]
  
  datmodel <- dat %>%
    select(Measure, Patient, Region, trial_type, trial_idx, yes_or_no, Memory, Accuracy, StimCond, all_of(diff_cols))
  
  # Compute band means row-wise
  datmodel$slow_gamma_pac <- rowMeans(datmodel[, slow_gamma_cols, drop = FALSE], na.rm = TRUE)
  if (tolower(phase) == "encoding" && length(hfa_cols) > 0) {
    datmodel$hfa_pac <- rowMeans(datmodel[, hfa_cols, drop = FALSE], na.rm = TRUE)
  }
  
  # Keep only needed columns
  keep_cols <- c("Measure", "Patient", "Region", "trial_type", "trial_idx",
                 "yes_or_no", "Memory", "Accuracy", "StimCond", "slow_gamma_pac")
  if ("hfa_pac" %in% names(datmodel)) keep_cols <- c(keep_cols, "hfa_pac")
  datmodel <- datmodel[, keep_cols]
  
  # Center predictors
  datmodel <- datmodel %>%
    mutate(
      StimCond = factor(StimCond, levels = c("nostim", "stim")),
      slow_gamma_pac_c = slow_gamma_pac - mean(slow_gamma_pac, na.rm = TRUE)
    )
  if ("hfa_pac" %in% names(datmodel)) {
    datmodel$hfa_pac_c <- datmodel$hfa_pac - mean(datmodel$hfa_pac, na.rm = TRUE)
  }
  
  datmodel
}

build_pac_patient_level_data <- function(datmodel) {
  has_hfa <- "hfa_pac" %in% names(datmodel)
  
  result <- datmodel %>%
    group_by(Measure, Patient, Region, StimCond) %>%
    summarise(
      Accuracy = mean(Accuracy, na.rm = TRUE),
      slow_gamma_pac = mean(slow_gamma_pac, na.rm = TRUE),
      N = n(),
      .groups = "drop"
    )
  
  if (has_hfa) {
    hfa_means <- datmodel %>%
      group_by(Measure, Patient, Region, StimCond) %>%
      summarise(hfa_pac = mean(hfa_pac, na.rm = TRUE), .groups = "drop")
    result <- left_join(result, hfa_means, by = c("Measure", "Patient", "Region", "StimCond"))
  }
  
  result <- result %>%
    mutate(
      StimCond = factor(StimCond, levels = c("nostim", "stim")),
      slow_gamma_pac_c = slow_gamma_pac - mean(slow_gamma_pac, na.rm = TRUE)
    )
  if (has_hfa) {
    result$hfa_pac_c <- result$hfa_pac - mean(result$hfa_pac, na.rm = TRUE)
  }
  
  result
}

# ---------------------------------------------------------------------------
# Region-by-band models (one model per region, bands as predictors)
# ---------------------------------------------------------------------------
pac_term_label <- function(term, has_hfa) {
  labels <- c(
    "(Intercept)" = "(Intercept)",
    "slow_gamma_pac_c" = "Slow Gamma PAC (30-50 Hz)",
    "StimCondstim" = "StimCond [stim]",
    "slow_gamma_pac_c:StimCondstim" = "Slow Gamma PAC x StimCond [stim]"
  )
  if (has_hfa) {
    labels <- c(labels, c(
      "hfa_pac_c" = "HFA PAC (55-100 Hz)",
      "hfa_pac_c:StimCondstim" = "HFA PAC x StimCond [stim]"
    ))
  }
  if (term %in% names(labels)) labels[[term]] else term
}

extract_pac_coef_table <- function(model, has_hfa, logistic = FALSE) {
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
    Predictor = vapply(coef_mat$term, pac_term_label, character(1), has_hfa = has_hfa),
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

fit_pac_patient_models_encoding <- function(dat) {
  list(
    m0 = lmer(Accuracy ~ 1 + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m1 = lmer(Accuracy ~ StimCond + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m2 = lmer(Accuracy ~ StimCond + slow_gamma_pac_c + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m3 = lmer(Accuracy ~ StimCond + hfa_pac_c + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m4 = lmer(Accuracy ~ StimCond + slow_gamma_pac_c + hfa_pac_c + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m_full = lmer(
      Accuracy ~ (slow_gamma_pac_c + hfa_pac_c) * StimCond + (1 | Patient),
      data = dat, na.action = na.exclude, REML = FALSE
    )
  )
}

fit_pac_trial_models_encoding <- function(dat) {
  ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  list(
    m0 = glmer(Accuracy ~ 1 + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = ctrl),
    m1 = glmer(Accuracy ~ StimCond + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = ctrl),
    m2 = glmer(Accuracy ~ StimCond + slow_gamma_pac_c + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = ctrl),
    m3 = glmer(Accuracy ~ StimCond + hfa_pac_c + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = ctrl),
    m4 = glmer(Accuracy ~ StimCond + slow_gamma_pac_c + hfa_pac_c + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = ctrl),
    m_full = glmer(
      Accuracy ~ (slow_gamma_pac_c + hfa_pac_c) * StimCond + (1 | Patient),
      data = dat, family = binomial, na.action = na.exclude, control = ctrl
    )
  )
}

fit_pac_patient_models_retrieval <- function(dat) {
  list(
    m0 = lmer(Accuracy ~ 1 + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m1 = lmer(Accuracy ~ StimCond + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m2 = lmer(Accuracy ~ StimCond + slow_gamma_pac_c + (1 | Patient), data = dat, na.action = na.exclude, REML = FALSE),
    m_full = lmer(
      Accuracy ~ slow_gamma_pac_c * StimCond + (1 | Patient),
      data = dat, na.action = na.exclude, REML = FALSE
    )
  )
}

fit_pac_trial_models_retrieval <- function(dat) {
  ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  list(
    m0 = glmer(Accuracy ~ 1 + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = ctrl),
    m1 = glmer(Accuracy ~ StimCond + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = ctrl),
    m2 = glmer(Accuracy ~ StimCond + slow_gamma_pac_c + (1 | Patient), data = dat, family = binomial, na.action = na.exclude, control = ctrl),
    m_full = glmer(
      Accuracy ~ slow_gamma_pac_c * StimCond + (1 | Patient),
      data = dat, family = binomial, na.action = na.exclude, control = ctrl
    )
  )
}

# ---------------------------------------------------------------------------
# Band-by-region models (one model per band, region as predictor)
# ---------------------------------------------------------------------------

PAC_BAND_SPECS <- list(
  slow_gamma_pac = list(column = "slow_gamma_pac", centered = "slow_gamma_pac_c",
                        label = "Theta Phase x Slow Gamma Amplitude (30-50 Hz)",
                        short = "Slow Gamma PAC", slug = "slow_gamma_pac"),
  hfa_pac = list(column = "hfa_pac", centered = "hfa_pac_c",
                 label = "Theta Phase x HFA Amplitude (55-100 Hz)",
                 short = "HFA PAC", slug = "hfa_pac")
)

prepare_pac_band_analysis_data <- function(datmodel, band_key, level_slug) {
  spec <- PAC_BAND_SPECS[[band_key]]
  if (is.null(spec)) stop(sprintf("Unknown PAC band: %s", band_key))
  if (!(spec$column %in% names(datmodel))) return(data.frame())
  
  dat <- if (identical(level_slug, "trial_level")) datmodel else build_pac_patient_level_data(datmodel)
  if (!(spec$centered %in% names(dat))) return(data.frame())
  
  dat %>%
    mutate(
      Region = factor(Region, levels = sort(unique(Region))),
      StimCond = factor(StimCond, levels = c("nostim", "stim")),
      band_value = .data[[spec$column]],
      band_c = .data[[spec$centered]]
    ) %>%
    filter(is.finite(band_value), is.finite(band_c))
}

# ---------------------------------------------------------------------------
# Run region-by-band PAC analysis
# ---------------------------------------------------------------------------
run_pac_region_by_band <- function(input_csv, output_dir, phase = "encoding") {
  ensure_dir(output_dir)
  datmodel <- prepare_pac_model_data(input_csv, phase = phase)
  has_hfa <- "hfa_pac" %in% names(datmodel)
  is_encoding <- tolower(phase) == "encoding"
  phase_name <- tools::toTitleCase(phase)
  
  regions <- sort(unique(datmodel$Region))
  region_count <- length(regions)
  
  manifest_rows <- list()
  
  for (level_slug in c("patient_level", "trial_level")) {
    logistic <- identical(level_slug, "trial_level")
    
    for (region in regions) {
      region_data <- datmodel %>% filter(Region == region)
      n_patients <- dplyr::n_distinct(region_data$Patient)
      if (n_patients < 3) next
      if (dplyr::n_distinct(region_data$StimCond) < 2) next
      
      if (identical(level_slug, "patient_level")) {
        analysis_data <- build_pac_patient_level_data(region_data)
      } else {
        analysis_data <- region_data
      }
      if (nrow(analysis_data) < 8) next
      
      region_dir <- ensure_dir(file.path(output_dir, level_slug, sanitize_name(region)))
      
      model_list <- tryCatch({
        if (logistic) {
          if (is_encoding && has_hfa) fit_pac_trial_models_encoding(analysis_data)
          else fit_pac_trial_models_retrieval(analysis_data)
        } else {
          if (is_encoding && has_hfa) fit_pac_patient_models_encoding(analysis_data)
          else fit_pac_patient_models_retrieval(analysis_data)
        }
      }, error = function(e) e)
      if (inherits(model_list, "error")) next
      
      coef_df <- extract_pac_coef_table(model_list$m_full, has_hfa = is_encoding && has_hfa, logistic = logistic)
      compare_df <- extract_model_compare_table(model_list)
      
      write.csv(coef_df, file.path(region_dir, "coefficients.csv"), row.names = FALSE)
      write.csv(compare_df, file.path(region_dir, "model_comparison.csv"), row.names = FALSE)
      
      manifest_rows[[length(manifest_rows) + 1]] <- data.frame(
        success = TRUE, region = region, level = level_slug,
        n_patients = n_patients, n_rows = nrow(analysis_data),
        stringsAsFactors = FALSE
      )
    }
  }
  
  # Overview fit summary
  overview_dir <- ensure_dir(file.path(output_dir, "overview"))
  for (level_slug in c("patient_level", "trial_level")) {
    logistic <- identical(level_slug, "trial_level")
    if (identical(level_slug, "patient_level")) {
      analysis_data <- build_pac_patient_level_data(datmodel)
    } else {
      analysis_data <- datmodel
    }
    if (nrow(analysis_data) < 8) next
    
    model_list <- tryCatch({
      if (logistic) {
        if (is_encoding && has_hfa) fit_pac_trial_models_encoding(analysis_data)
        else fit_pac_trial_models_retrieval(analysis_data)
      } else {
        if (is_encoding && has_hfa) fit_pac_patient_models_encoding(analysis_data)
        else fit_pac_patient_models_retrieval(analysis_data)
      }
    }, error = function(e) e)
    if (inherits(model_list, "error")) next
    
    compare_df <- extract_model_compare_table(model_list)
    fit_display <- build_fit_summary_display(
      null_model = model_list$m0, full_model = model_list$m_full,
      analysis_data = analysis_data, compare_df = compare_df,
      region_count = region_count, logistic = logistic
    )
    write.csv(compare_df, file.path(overview_dir, sprintf("%s_model_comparison.csv", level_slug)), row.names = FALSE)
    write.csv(fit_display, file.path(overview_dir, sprintf("%s_fit_summary.csv", level_slug)), row.names = FALSE)
  }
  
  if (length(manifest_rows) > 0) {
    write.csv(bind_rows(manifest_rows), file.path(output_dir, "manifest.csv"), row.names = FALSE)
  }
  message(sprintf("PAC region-by-band complete: %s (%d regions)", output_dir, length(regions)))
}

# ---------------------------------------------------------------------------
# Run band-by-region PAC analysis
# ---------------------------------------------------------------------------
run_pac_band_by_region <- function(input_csv, output_dir, phase = "encoding") {
  ensure_dir(output_dir)
  datmodel <- prepare_pac_model_data(input_csv, phase = phase)
  has_hfa <- "hfa_pac" %in% names(datmodel)
  is_encoding <- tolower(phase) == "encoding"
  phase_name <- tools::toTitleCase(phase)
  
  band_keys <- if (is_encoding && has_hfa) c("slow_gamma_pac", "hfa_pac") else c("slow_gamma_pac")
  
  region_count <- dplyr::n_distinct(datmodel$Region)
  manifest_rows <- list()
  
  for (band_key in band_keys) {
    spec <- PAC_BAND_SPECS[[band_key]]
    message(sprintf("  Analyzing PAC band: %s", spec$label))
    
    for (level_slug in c("patient_level", "trial_level")) {
      logistic <- identical(level_slug, "trial_level")
      analysis_data <- prepare_pac_band_analysis_data(datmodel, band_key, level_slug)
      if (nrow(analysis_data) < 12) next
      if (dplyr::n_distinct(analysis_data$Patient) < 3) next
      if (dplyr::n_distinct(analysis_data$StimCond) < 2) next
      if (dplyr::n_distinct(analysis_data$Region) < 2) next
      
      band_dir <- ensure_dir(file.path(output_dir, level_slug, spec$slug))
      
      model_list <- tryCatch(
        if (logistic) fit_trial_band_region_models(analysis_data) else fit_patient_band_region_models(analysis_data),
        error = function(e) e
      )
      if (inherits(model_list, "error")) next
      
      coef_df <- extract_band_region_coef_table(model_list$m_full, band_key, logistic = logistic)
      compare_df <- extract_model_compare_table(model_list)
      
      write.csv(coef_df, file.path(band_dir, "coefficients.csv"), row.names = FALSE)
      write.csv(compare_df, file.path(band_dir, "model_comparison.csv"), row.names = FALSE)
      
      manifest_rows[[length(manifest_rows) + 1]] <- data.frame(
        success = TRUE, band = band_key, level = level_slug,
        n_patients = dplyr::n_distinct(analysis_data$Patient),
        n_regions = dplyr::n_distinct(analysis_data$Region),
        n_rows = nrow(analysis_data),
        stringsAsFactors = FALSE
      )
    }
  }
  
  # Overview
  overview_dir <- ensure_dir(file.path(output_dir, "overview"))
  for (band_key in band_keys) {
    spec <- PAC_BAND_SPECS[[band_key]]
    for (level_slug in c("patient_level", "trial_level")) {
      logistic <- identical(level_slug, "trial_level")
      analysis_data <- prepare_pac_band_analysis_data(datmodel, band_key, level_slug)
      if (nrow(analysis_data) < 12) next
      
      model_list <- tryCatch(
        if (logistic) fit_trial_band_region_models(analysis_data) else fit_patient_band_region_models(analysis_data),
        error = function(e) e
      )
      if (inherits(model_list, "error")) next
      
      compare_df <- extract_model_compare_table(model_list)
      fit_display <- build_fit_summary_display(
        null_model = model_list$m0, full_model = model_list$m_full,
        analysis_data = analysis_data, compare_df = compare_df,
        region_count = region_count, logistic = logistic
      )
      write.csv(compare_df, file.path(overview_dir, sprintf("%s_%s_model_comparison.csv", level_slug, spec$slug)), row.names = FALSE)
      write.csv(fit_display, file.path(overview_dir, sprintf("%s_%s_fit_summary.csv", level_slug, spec$slug)), row.names = FALSE)
    }
  }
  
  if (length(manifest_rows) > 0) {
    write.csv(bind_rows(manifest_rows), file.path(output_dir, "manifest.csv"), row.names = FALSE)
  }
  message(sprintf("PAC band-by-region complete: %s", output_dir))
}

# ============================================================================
# Region filter functions (exclude PHG everywhere)
# ============================================================================
filter_mtl_cortical <- function(datmodel) {
  # BLA, HPC, EC, PRC (broader MTL cortical regions)
  allowed <- c("BLA", "HPC", "EC", "PRC")
  datmodel %>% filter(Region %in% allowed)
}

filter_hpc_subfields <- function(datmodel) {
  # BLA, CA, DG (hippocampal subfields + amygdala)
  allowed <- c("BLA", "CA", "DG")
  datmodel %>% filter(Region %in% allowed)
}

filter_no_phg <- function(datmodel) {
  datmodel %>% filter(Region != "PHG")
}

filter_coherence_mtl_cortical <- function(datmodel) {
  # Keep pairs where BOTH regions are in {BLA, HPC, EC, PRC}
  allowed <- c("BLA", "HPC", "EC", "PRC")
  datmodel %>% filter({
    parts <- strsplit(Region, "_", fixed = TRUE)
    sapply(parts, function(p) all(p %in% allowed))
  })
}

filter_coherence_hpc_subfields <- function(datmodel) {
  # Keep pairs where BOTH regions are in {BLA, CA, DG}
  allowed <- c("BLA", "CA", "DG")
  datmodel %>% filter({
    parts <- strsplit(Region, "_", fixed = TRUE)
    sapply(parts, function(p) all(p %in% allowed))
  })
}

