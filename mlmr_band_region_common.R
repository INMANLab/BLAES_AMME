source("mlmr_retrieval_common.R")

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
