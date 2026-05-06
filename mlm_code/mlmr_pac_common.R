## PAC (Phase-Amplitude Coupling) MLM Analysis
## Theta phase x amplitude coupling at slow gamma (30-50 Hz) and HFA (55-100 Hz)
##
## Encoding: both slow_gamma_pac and hfa_pac as predictors
## Retrieval: slow_gamma_pac only

source("mlm/mlmr_retrieval_common.R")

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
