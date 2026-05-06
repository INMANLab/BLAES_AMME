suppressPackageStartupMessages({
  library(dplyr)
  library(lme4)
  library(lmerTest)
})

SCRIPT_DIR <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg)))
    else getwd()
  }
)

OUTPUT_ROOT <- file.path(SCRIPT_DIR, "outputs", "retrieval_memory_reports")
STATS_ROOT <- file.path(OUTPUT_ROOT, "stats")
dir.create(STATS_ROOT, recursive = TRUE, showWarnings = FALSE)

MIN_TRIALS <- 10
CTRL_LMER <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
CTRL_GLMER <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA_RANGE <- c(4, 8)
SLOW_GAMMA_RANGE <- c(30, 55)
PAC_SLOW_GAMMA_RANGE <- c(30, 50)
PAC_HFA_RANGE <- c(70, 100)

POWER_FAMILIES <- list(
  bla_hpc_subregions = list(
    label = "BLA with Hippocampal Subregions",
    regions = c("BLA", "CA", "DG", "HPC")
  ),
  subregions_ec_prc = list(
    label = "Hippocampal Subregions with EC and PRC",
    regions = c("CA", "DG", "HPC", "EC", "PRC")
  )
)

PAIR_FAMILIES <- list(
  bla_hpc_subregions = list(
    label = "BLA with Hippocampal Subregions",
    regions = c("BLA_CA", "BLA_DG", "BLA_HPC")
  ),
  subregions_ec_prc = list(
    label = "Hippocampal Subregions with EC and PRC",
    regions = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC")
  )
)

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  path
}

resolve_behavior_csv <- function() {
  candidates <- c(
    file.path(SCRIPT_DIR, "AMMEBLAES_includedpts_firstsession_behavioral.csv"),
    file.path(dirname(SCRIPT_DIR), "AMMEBLAES_includedpts_firstsession_behavioral.csv"),
    file.path(SCRIPT_DIR, "behavioral figures", "AMMEBLAES_includedpts_firstsession_behavioral.csv")
  )
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0) {
    stop("Could not locate AMMEBLAES_includedpts_firstsession_behavioral.csv")
  }
  existing[[1]]
}

sort_freq_cols <- function(df, prefix = "diff_Freq_") {
  cols <- names(df)[startsWith(names(df), prefix)]
  cols[order(as.numeric(sub(prefix, "", cols, fixed = TRUE)))]
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

sanitize_name <- function(x) {
  gsub("[^A-Za-z0-9]+", "_", x)
}

format_p_value <- function(x) {
  ifelse(is.na(x), "", ifelse(x < 0.001, "<0.001", sprintf("%.3f", x)))
}

load_behavior <- function() {
  beh <- read.csv(resolve_behavior_csv(), stringsAsFactors = FALSE)
  beh <- beh[, c("Patient", "avg_stim_dprime_diff")]
  beh$Patient <- as.character(beh$Patient)
  beh$avg_stim_dprime_diff <- as.numeric(beh$avg_stim_dprime_diff)
  beh <- beh[is.finite(beh$avg_stim_dprime_diff), , drop = FALSE]
  beh[!duplicated(beh$Patient), , drop = FALSE]
}

filter_balanced <- function(df, measure_label) {
  patients <- unique(df$Patient)
  counts <- data.frame(Patient = patients, stringsAsFactors = FALSE)
  counts$n_regions <- sapply(patients, function(p) length(unique(df$Region[df$Patient == p])))
  counts$n_rem <- sapply(patients, function(p) sum(df$Patient == p & df$yes_or_no == "yes"))
  counts$n_forg <- sapply(patients, function(p) sum(df$Patient == p & df$yes_or_no == "no"))
  counts$n_rem_unique <- round(counts$n_rem / pmax(counts$n_regions, 1))
  counts$n_forg_unique <- round(counts$n_forg / pmax(counts$n_regions, 1))
  included <- counts$Patient[counts$n_rem_unique >= MIN_TRIALS & counts$n_forg_unique >= MIN_TRIALS]
  message(sprintf("[%s] Balanced filter retained %d / %d patients", measure_label, length(included), nrow(counts)))
  df[df$Patient %in% included, , drop = FALSE]
}

extract_fixed_effects <- function(model, logistic = FALSE) {
  coef_mat <- as.data.frame(summary(model)$coefficients)
  coef_mat$term <- rownames(coef_mat)
  fixed_terms <- coef_mat$term
  conf <- suppressMessages(confint(model, parm = fixed_terms, method = "Wald"))
  conf <- conf[fixed_terms, , drop = FALSE]
  statistic_col <- if ("t value" %in% names(coef_mat)) "t value" else "z value"
  p_col <- grep("^Pr\\(", names(coef_mat), value = TRUE)
  p_vals <- if (length(p_col) == 1) coef_mat[[p_col]] else rep(NA_real_, nrow(coef_mat))

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
    term = coef_mat$term,
    estimate = estimate_vals,
    std_error = coef_mat$`Std. Error`,
    conf_low = conf_low,
    conf_high = conf_high,
    statistic = coef_mat[[statistic_col]],
    p_value = p_vals,
    estimate_label = estimate_label,
    stringsAsFactors = FALSE
  )
}

extract_model_compare_table <- function(model_list) {
  ordered_names <- c("m0", "m1", "m2", "m3", "m4", "m_full")
  available <- ordered_names[ordered_names %in% names(model_list)]
  available_models <- unname(model_list[available])
  comp <- as.data.frame(do.call(anova, available_models))
  comp$Model <- available
  rownames(comp) <- NULL
  p_col <- grep("^Pr", names(comp), value = TRUE)
  p_vals <- if (length(p_col) == 1) comp[[p_col]] else rep(NA_real_, nrow(comp))
  data.frame(
    Model = comp$Model,
    npar = comp$npar,
    AIC = comp$AIC,
    BIC = comp$BIC,
    logLik = comp$logLik,
    Chisq = if ("Chisq" %in% names(comp)) comp$Chisq else NA_real_,
    Df = if ("Df" %in% names(comp)) comp$Df else NA_real_,
    p_value = p_vals,
    stringsAsFactors = FALSE
  )
}

prepare_power_data <- function() {
  path <- file.path(SCRIPT_DIR, "outputs", "csvs", "combined_retrieval_power_all_mlmr_input.csv")
  dat <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  dat$Region <- normalize_region_labels(dat$Region)
  dat <- dat[dat$trial_type != "new" & dat$yes_or_no %in% c("yes", "no"), , drop = FALSE]

  freq_cols <- sort_freq_cols(dat)
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols, fixed = TRUE))
  dat$theta <- rowMeans(dat[, freq_cols[freqs >= THETA_RANGE[1] & freqs <= THETA_RANGE[2]], drop = FALSE], na.rm = TRUE)
  dat$slow_gamma <- rowMeans(dat[, freq_cols[freqs >= SLOW_GAMMA_RANGE[1] & freqs <= SLOW_GAMMA_RANGE[2]], drop = FALSE], na.rm = TRUE)
  dat$Accuracy <- ifelse(dat$yes_or_no == "yes", 1, 0)
  dat$StimCond <- factor(ifelse(dat$trial_type == "nostim", "nostim", "stim"), levels = c("nostim", "stim"))
  dat$Patient <- as.character(dat$Patient)
  dat
}

prepare_coherence_data <- function() {
  path <- file.path(SCRIPT_DIR, "outputs", "csvs", "combined_retrieval_coherence_all_mlmr_input.csv")
  dat <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  dat$Region <- normalize_region_labels(dat$Region)
  dat <- dat[dat$trial_type != "new" & dat$yes_or_no %in% c("yes", "no"), , drop = FALSE]

  freq_cols <- sort_freq_cols(dat)
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols, fixed = TRUE))
  dat$theta <- rowMeans(dat[, freq_cols[freqs >= THETA_RANGE[1] & freqs <= THETA_RANGE[2]], drop = FALSE], na.rm = TRUE)
  dat$slow_gamma <- rowMeans(dat[, freq_cols[freqs >= SLOW_GAMMA_RANGE[1] & freqs <= SLOW_GAMMA_RANGE[2]], drop = FALSE], na.rm = TRUE)
  dat$Accuracy <- ifelse(dat$yes_or_no == "yes", 1, 0)
  dat$StimCond <- factor(ifelse(dat$trial_type == "nostim", "nostim", "stim"), levels = c("nostim", "stim"))
  dat$Patient <- as.character(dat$Patient)
  dat
}

prepare_pac_data <- function() {
  path <- file.path(SCRIPT_DIR, "outputs", "csvs", "combined_retrieval_pac_all_mlmr_input.csv")
  dat <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  dat$Region <- normalize_region_labels(dat$Region)
  dat <- dat[dat$trial_type != "new" & dat$yes_or_no %in% c("yes", "no"), , drop = FALSE]

  freq_cols <- sort_freq_cols(dat)
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols, fixed = TRUE))
  dat$slow_gamma_pac <- rowMeans(dat[, freq_cols[freqs >= PAC_SLOW_GAMMA_RANGE[1] & freqs <= PAC_SLOW_GAMMA_RANGE[2]], drop = FALSE], na.rm = TRUE)
  dat$hfa_pac <- rowMeans(dat[, freq_cols[freqs >= PAC_HFA_RANGE[1] & freqs <= PAC_HFA_RANGE[2]], drop = FALSE], na.rm = TRUE)
  dat$Accuracy <- ifelse(dat$yes_or_no == "yes", 1, 0)
  dat$StimCond <- factor(ifelse(dat$trial_type == "nostim", "nostim", "stim"), levels = c("nostim", "stim"))
  dat$Patient <- as.character(dat$Patient)
  dat
}

build_patient_level_rows <- function(df, measure_name, behavior_df) {
  agg <- df %>%
    group_by(Patient, Region, StimCond) %>%
    summarise(
      theta = if ("theta" %in% names(df)) mean(theta, na.rm = TRUE) else NA_real_,
      slow_gamma = if ("slow_gamma" %in% names(df)) mean(slow_gamma, na.rm = TRUE) else NA_real_,
      slow_gamma_pac = if ("slow_gamma_pac" %in% names(df)) mean(slow_gamma_pac, na.rm = TRUE) else NA_real_,
      hfa_pac = if ("hfa_pac" %in% names(df)) mean(hfa_pac, na.rm = TRUE) else NA_real_,
      n_trials = n(),
      .groups = "drop"
    )
  out <- left_join(agg, behavior_df, by = "Patient")
  out <- out[is.finite(out$avg_stim_dprime_diff), , drop = FALSE]
  out$measure_name <- measure_name
  out
}

center_column <- function(x) {
  x - mean(x, na.rm = TRUE)
}

standardize_column <- function(x) {
  centered <- x - mean(x, na.rm = TRUE)
  sd_x <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(sd_x) || sd_x == 0) {
    return(centered)
  }
  centered / sd_x
}

fit_models <- function(data, formula_map, outcome_type) {
  logistic <- identical(outcome_type, "glmm")
  fitter <- if (logistic) {
    function(formula_text) glmer(as.formula(formula_text), data = data, family = binomial, na.action = na.exclude, control = CTRL_GLMER)
  } else {
    function(formula_text) lmer(as.formula(formula_text), data = data, na.action = na.exclude, REML = FALSE, control = CTRL_LMER)
  }
  models <- list()
  for (name in names(formula_map)) {
    models[[name]] <- fitter(formula_map[[name]])
  }
  models
}

fit_freq_across_regions <- function(data, outcome_type) {
  response <- if (identical(outcome_type, "glmm")) "Accuracy" else "avg_stim_dprime_diff"
  formula_map <- c(
    m0 = sprintf("%s ~ 1 + (1 | Patient)", response),
    m1 = sprintf("%s ~ StimCond + (1 | Patient)", response),
    m2 = sprintf("%s ~ StimCond + Region + (1 | Patient)", response),
    m3 = sprintf("%s ~ StimCond + Region + band_c + (1 | Patient)", response),
    m4 = sprintf("%s ~ StimCond + Region + band_c + band_c:Region + (1 | Patient)", response),
    m_full = sprintf("%s ~ StimCond + Region + band_c + band_c:Region + band_c:StimCond + (1 | Patient)", response)
  )
  fit_models(data, formula_map, outcome_type)
}

fit_region_across_bands <- function(data, outcome_type, predictor_a, predictor_b) {
  response <- if (identical(outcome_type, "glmm")) "Accuracy" else "avg_stim_dprime_diff"
  formula_map <- c(
    m0 = sprintf("%s ~ 1 + (1 | Patient)", response),
    m1 = sprintf("%s ~ StimCond + (1 | Patient)", response),
    m2 = sprintf("%s ~ StimCond + %s + (1 | Patient)", response, predictor_a),
    m3 = sprintf("%s ~ StimCond + %s + %s + (1 | Patient)", response, predictor_a, predictor_b),
    m4 = sprintf("%s ~ StimCond + %s + %s + %s:StimCond + (1 | Patient)", response, predictor_a, predictor_b, predictor_a),
    m_full = sprintf("%s ~ StimCond + %s + %s + %s:StimCond + %s:StimCond + (1 | Patient)", response, predictor_a, predictor_b, predictor_a, predictor_b)
  )
  fit_models(data, formula_map, outcome_type)
}

save_model_outputs <- function(model_list, model_dir, metadata_row, interaction_terms, outcome_type) {
  ensure_dir(model_dir)
  logistic <- identical(outcome_type, "glmm")
  coef_df <- extract_fixed_effects(model_list$m_full, logistic = logistic)
  compare_df <- extract_model_compare_table(model_list)
  summary_rows <- coef_df[coef_df$term %in% interaction_terms, , drop = FALSE]
  summary_rows$model_id <- metadata_row$model_id
  summary_rows$analysis_type <- metadata_row$analysis_type
  summary_rows$family_slug <- metadata_row$family_slug
  summary_rows$family_label <- metadata_row$family_label
  summary_rows$target <- metadata_row$target
  summary_rows$outcome_type <- outcome_type
  summary_rows$n_patients <- metadata_row$n_patients
  summary_rows$n_rows <- metadata_row$n_rows

  write.csv(coef_df, file.path(model_dir, "coefficients.csv"), row.names = FALSE)
  write.csv(compare_df, file.path(model_dir, "model_comparison.csv"), row.names = FALSE)
  write.csv(as.data.frame(metadata_row, stringsAsFactors = FALSE), file.path(model_dir, "metadata.csv"), row.names = FALSE)
  write.csv(summary_rows, file.path(model_dir, "interaction_summary.csv"), row.names = FALSE)

  cbind(as.data.frame(metadata_row, stringsAsFactors = FALSE), data.frame(
    coefficient_path = file.path(model_dir, "coefficients.csv"),
    model_comparison_path = file.path(model_dir, "model_comparison.csv"),
    interaction_summary_path = file.path(model_dir, "interaction_summary.csv"),
    stringsAsFactors = FALSE
  ))
}

run_frequency_across_regions <- function(measure_slug, family_slug, family_cfg, patient_df, trial_df, predictor_specs) {
  manifest_rows <- list()
  for (band_name in names(predictor_specs)) {
    predictor_col <- predictor_specs[[band_name]]

    patient_sub <- patient_df[patient_df$Region %in% family_cfg$regions, , drop = FALSE]
    patient_sub <- patient_sub[is.finite(patient_sub[[predictor_col]]), , drop = FALSE]
    patient_sub$Region <- factor(patient_sub$Region, levels = family_cfg$regions[family_cfg$regions %in% unique(patient_sub$Region)])
    if (identical(measure_slug, "pac")) {
      patient_sub$band_c <- standardize_column(patient_sub[[predictor_col]])
    } else {
      patient_sub$band_c <- center_column(patient_sub[[predictor_col]])
    }

    trial_sub <- trial_df[trial_df$Region %in% family_cfg$regions, , drop = FALSE]
    trial_sub <- trial_sub[is.finite(trial_sub[[predictor_col]]), , drop = FALSE]
    trial_sub$Region <- factor(trial_sub$Region, levels = family_cfg$regions[family_cfg$regions %in% unique(trial_sub$Region)])
    if (identical(measure_slug, "pac")) {
      trial_sub$band_c <- standardize_column(trial_sub[[predictor_col]])
    } else {
      trial_sub$band_c <- center_column(trial_sub[[predictor_col]])
    }

    for (outcome_type in c("mlm", "glmm")) {
      dat <- if (identical(outcome_type, "mlm")) patient_sub else trial_sub
      if (nrow(dat) < 12 || dplyr::n_distinct(dat$Patient) < 3 || dplyr::n_distinct(dat$StimCond) < 2 || dplyr::n_distinct(dat$Region) < 2) {
        next
      }

      model_list <- tryCatch(
        fit_freq_across_regions(dat, outcome_type),
        error = function(e) e
      )
      if (inherits(model_list, "error")) {
        message(sprintf("Skipped %s %s %s %s: %s", measure_slug, family_slug, band_name, outcome_type, conditionMessage(model_list)))
        next
      }

      model_id <- paste(measure_slug, outcome_type, family_slug, "frequency_across_regions", band_name, sep = "__")
      metadata_row <- list(
        model_id = model_id,
        measure_slug = measure_slug,
        analysis_type = "frequency_across_regions",
        family_slug = family_slug,
        family_label = family_cfg$label,
        target = band_name,
        n_patients = dplyr::n_distinct(dat$Patient),
        n_rows = nrow(dat)
      )
      model_dir <- file.path(STATS_ROOT, measure_slug, outcome_type, sanitize_name(model_id))
      manifest_rows[[length(manifest_rows) + 1]] <- save_model_outputs(
        model_list = model_list,
        model_dir = model_dir,
        metadata_row = metadata_row,
        interaction_terms = c("band_c:StimCondstim", "StimCondstim:band_c"),
        outcome_type = outcome_type
      )
    }
  }
  manifest_rows
}

run_region_across_bands <- function(measure_slug, family_slug, family_cfg, patient_df, trial_df, predictor_a, predictor_b) {
  manifest_rows <- list()
  for (region_name in family_cfg$regions) {
    patient_sub <- patient_df[patient_df$Region == region_name, , drop = FALSE]
    trial_sub <- trial_df[trial_df$Region == region_name, , drop = FALSE]

    for (outcome_type in c("mlm", "glmm")) {
      dat <- if (identical(outcome_type, "mlm")) patient_sub else trial_sub
      if (nrow(dat) < 8 || dplyr::n_distinct(dat$Patient) < 3 || dplyr::n_distinct(dat$StimCond) < 2) {
        next
      }
      if (identical(measure_slug, "pac")) {
        dat[[paste0(predictor_a, "_c")]] <- standardize_column(dat[[predictor_a]])
        dat[[paste0(predictor_b, "_c")]] <- standardize_column(dat[[predictor_b]])
      } else {
        dat[[paste0(predictor_a, "_c")]] <- center_column(dat[[predictor_a]])
        dat[[paste0(predictor_b, "_c")]] <- center_column(dat[[predictor_b]])
      }
      dat <- dat[is.finite(dat[[paste0(predictor_a, "_c")]]) & is.finite(dat[[paste0(predictor_b, "_c")]]), , drop = FALSE]
      if (nrow(dat) < 8) {
        next
      }

      model_list <- tryCatch(
        fit_region_across_bands(dat, outcome_type, paste0(predictor_a, "_c"), paste0(predictor_b, "_c")),
        error = function(e) e
      )
      if (inherits(model_list, "error")) {
        message(sprintf("Skipped %s %s %s %s: %s", measure_slug, family_slug, region_name, outcome_type, conditionMessage(model_list)))
        next
      }

      model_id <- paste(measure_slug, outcome_type, family_slug, "region_across_bands", region_name, sep = "__")
      metadata_row <- list(
        model_id = model_id,
        measure_slug = measure_slug,
        analysis_type = "region_across_bands",
        family_slug = family_slug,
        family_label = family_cfg$label,
        target = region_name,
        n_patients = dplyr::n_distinct(dat$Patient),
        n_rows = nrow(dat)
      )
      model_dir <- file.path(STATS_ROOT, measure_slug, outcome_type, sanitize_name(model_id))
      manifest_rows[[length(manifest_rows) + 1]] <- save_model_outputs(
        model_list = model_list,
        model_dir = model_dir,
        metadata_row = metadata_row,
        interaction_terms = c(
          paste0(predictor_a, "_c:StimCondstim"),
          paste0("StimCondstim:", predictor_a, "_c"),
          paste0(predictor_b, "_c:StimCondstim"),
          paste0("StimCondstim:", predictor_b, "_c")
        ),
        outcome_type = outcome_type
      )
    }
  }
  manifest_rows
}

write_measure_manifest <- function(measure_slug, rows) {
  if (length(rows) == 0) {
    return(invisible(NULL))
  }
  manifest_df <- bind_rows(rows)
  measure_dir <- ensure_dir(file.path(STATS_ROOT, measure_slug))
  write.csv(manifest_df, file.path(measure_dir, "manifest.csv"), row.names = FALSE)
}

run_measure_power <- function(behavior_df) {
  trial_df <- prepare_power_data()
  all_rows <- list()
  for (family_slug in names(POWER_FAMILIES)) {
    family_cfg <- POWER_FAMILIES[[family_slug]]
    family_trial_df <- trial_df[trial_df$Region %in% family_cfg$regions, , drop = FALSE]
    family_trial_df <- filter_balanced(family_trial_df, sprintf("power_%s", family_slug))
    patient_df <- build_patient_level_rows(family_trial_df, "power", behavior_df)
    family_rows <- c(
      run_frequency_across_regions("power", family_slug, family_cfg, patient_df, family_trial_df, c(theta = "theta", slow_gamma = "slow_gamma")),
      run_region_across_bands("power", family_slug, family_cfg, patient_df, family_trial_df, "theta", "slow_gamma")
    )
    all_rows <- c(all_rows, family_rows)
  }
  write_measure_manifest("power", all_rows)
}

run_measure_coherence <- function(behavior_df) {
  trial_df <- prepare_coherence_data()
  all_rows <- list()
  for (family_slug in names(PAIR_FAMILIES)) {
    family_cfg <- PAIR_FAMILIES[[family_slug]]
    family_trial_df <- trial_df[trial_df$Region %in% family_cfg$regions, , drop = FALSE]
    family_trial_df <- filter_balanced(family_trial_df, sprintf("coherence_%s", family_slug))
    patient_df <- build_patient_level_rows(family_trial_df, "coherence", behavior_df)
    family_rows <- c(
      run_frequency_across_regions("coherence", family_slug, family_cfg, patient_df, family_trial_df, c(theta = "theta", slow_gamma = "slow_gamma")),
      run_region_across_bands("coherence", family_slug, family_cfg, patient_df, family_trial_df, "theta", "slow_gamma")
    )
    all_rows <- c(all_rows, family_rows)
  }
  write_measure_manifest("coherence", all_rows)
}

run_measure_pac <- function(behavior_df) {
  trial_df <- prepare_pac_data()
  all_rows <- list()
  for (family_slug in names(PAIR_FAMILIES)) {
    family_cfg <- PAIR_FAMILIES[[family_slug]]
    family_trial_df <- trial_df[trial_df$Region %in% family_cfg$regions, , drop = FALSE]
    family_trial_df <- filter_balanced(family_trial_df, sprintf("pac_%s", family_slug))
    patient_df <- build_patient_level_rows(family_trial_df, "pac", behavior_df)
    family_rows <- c(
      run_frequency_across_regions("pac", family_slug, family_cfg, patient_df, family_trial_df, c(slow_gamma_pac = "slow_gamma_pac", hfa_pac = "hfa_pac")),
      run_region_across_bands("pac", family_slug, family_cfg, patient_df, family_trial_df, "slow_gamma_pac", "hfa_pac")
    )
    all_rows <- c(all_rows, family_rows)
  }
  write_measure_manifest("pac", all_rows)
}

main <- function() {
  behavior_df <- load_behavior()
  run_measure_power(behavior_df)
  run_measure_coherence(behavior_df)
  run_measure_pac(behavior_df)
  message(sprintf("Wrote retrieval model stats to %s", STATS_ROOT))
}

main()
