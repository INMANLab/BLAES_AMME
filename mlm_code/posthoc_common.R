##############################################################################
# Shared helpers for the encoding post-hoc analyses (random-slope refits,
# Johnson-Neyman, and stability testing). Mirrors the data prep logic in
# run_h1abc_maineffect.R and run_h1abc_full_retrieval.R but is self-contained
# so we don't touch the existing code.
##############################################################################

suppressPackageStartupMessages({
  library(lme4); library(broom.mixed); library(dplyr)
})

REPO_ROOT <- "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES"
CSV_IN <- file.path(REPO_ROOT, "OUTPUTS", "csvs")

MIN_PATIENTS <- 4
MIN_TRIALS   <- 30

THETA          <- c(4.88, 7.81)
SLOW_GAMMA     <- c(30.27, 54.69)
PAC_SLOW_GAMMA <- c(30, 50)

ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# Output directory token per analysis type.
ANALYSIS_DIR <- list(
  alltrials                          = "alltrials",
  endogenous_memory_effect           = "endogenous_memory_effect",
  stim_effect                        = "stim_effect",
  GLMM_interactions_model_variations = "GLMM_interactions_model_variations"
)

ANALYSIS_CONTRAST <- list(
  alltrials                          = "all",
  endogenous_memory_effect           = "nostim",
  stim_effect                        = "stim",
  GLMM_interactions_model_variations = "all"  # uses both stim+nostim
)

SCOPES <- list(
  BLAMTL = list(
    power_regions = c("BLA", "ALLHPC", "EC", "PRC"),
    coh_pairs     = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    pac_pairs     = c("BLA_ALLHPC", "BLA_EC", "BLA_PRC"),
    needs_allhpc  = TRUE
  ),
  HPCrhinal = list(
    power_regions = c("ALLHPC", "EC", "PRC"),
    coh_pairs     = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
    pac_pairs     = c("ALLHPC_EC", "ALLHPC_PRC", "EC_PRC"),
    needs_allhpc  = TRUE
  ),
  HippSubBLA = list(
    power_regions = c("BLA", "CA", "DG", "HPC"),
    coh_pairs     = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    pac_pairs     = c("BLA_CA", "BLA_DG", "BLA_HPC"),
    needs_allhpc  = FALSE
  ),
  HippSubRhinal = list(
    power_regions = c("CA", "DG", "HPC"),
    coh_pairs     = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"),
    pac_pairs     = c("CA_EC", "DG_EC", "EC_HPC", "CA_PRC", "DG_PRC", "HPC_PRC"),
    needs_allhpc  = FALSE
  )
)

HPC_SUB <- c("HPC", "CA", "DG")
ALLHPC_PAIR_SOURCES <- list(
  BLA_ALLHPC  = c("BLA_HPC", "BLA_CA", "BLA_DG"),
  ALLHPC_EC   = c("EC_HPC",  "CA_EC",  "DG_EC"),
  ALLHPC_PRC  = c("HPC_PRC", "CA_PRC", "DG_PRC")
)

ensure_dir <- function(p) { dir.create(p, showWarnings = FALSE, recursive = TRUE); p }

safe_glmer <- function(formula, data) {
  tryCatch(
    glmer(formula, data = data, family = binomial, control = ctrl_glmer),
    error   = function(e) NULL,
    warning = function(w) {
      m <- tryCatch(
        glmer(formula, data = data, family = binomial, control = ctrl_glmer),
        error = function(e) NULL)
      if (is.null(m)) return(NULL)
      attr(m, "converged_with_warning") <- TRUE
      m
    }
  )
}

load_measure_df <- function(measure) {
  fname <- switch(measure,
                  power     = "combined_encoding_power_all_mlmr_input.csv",
                  coherence = "combined_encoding_coherence_all_mlmr_input.csv",
                  pac       = "combined_encoding_pac_all_mlmr_input.csv")
  d <- read.csv(file.path(CSV_IN, fname), stringsAsFactors = FALSE,
                check.names = FALSE)
  d <- d[d$yes_or_no %in% c("yes", "no") &
         d$trial_type %in% c("nostim", "stim"), ]
  d$Patient[d$Patient == "BJH033"] <- "BJH032"
  d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
  d$StimCond <- factor(ifelse(d$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))
  d
}

build_allhpc_power <- function(d) {
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  meta_cols <- intersect(c("Patient", "Region", "trial_type", "yes_or_no",
                           "Accuracy", "StimCond"), names(d))
  sub <- d[d$Region %in% HPC_SUB, c(meta_cols, freq_cols)]
  if (nrow(sub) == 0) return(NULL)
  sub <- sub %>% group_by(Patient, Region) %>%
    mutate(trial_id = row_number()) %>% ungroup()
  agg <- sub %>%
    group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy, StimCond) %>%
    summarise(across(all_of(freq_cols), ~ mean(.x, na.rm = TRUE)),
              .groups = "drop") %>%
    mutate(Region = "ALLHPC") %>% select(-trial_id)
  as.data.frame(agg)
}

build_allhpc_pair <- function(d, target_label, source_pairs) {
  freq_cols <- grep("^diff_Freq_", names(d), value = TRUE)
  meta_cols <- intersect(c("Patient", "Region", "trial_type", "yes_or_no",
                           "Accuracy", "StimCond"), names(d))
  sub <- d[d$Region %in% source_pairs, c(meta_cols, freq_cols)]
  if (nrow(sub) == 0) return(NULL)
  sub <- sub %>% group_by(Patient, Region) %>%
    mutate(trial_id = row_number()) %>% ungroup()
  agg <- sub %>%
    group_by(Patient, trial_id, trial_type, yes_or_no, Accuracy, StimCond) %>%
    summarise(across(all_of(freq_cols), ~ mean(.x, na.rm = TRUE)),
              .groups = "drop") %>%
    mutate(Region = target_label) %>% select(-trial_id)
  as.data.frame(agg)
}

prep_with_allhpc <- function(d, measure) {
  if (measure == "power") {
    extra <- build_allhpc_power(d)
    if (!is.null(extra)) d <- bind_rows(d, extra)
  } else {
    for (target in names(ALLHPC_PAIR_SOURCES)) {
      sources <- ALLHPC_PAIR_SOURCES[[target]]
      extra <- build_allhpc_pair(d, target, sources)
      if (!is.null(extra)) d <- bind_rows(d, extra)
    }
  }
  d
}

add_band_columns <- function(d, measure) {
  freq_cols <- sort(grep("^diff_Freq_", names(d), value = TRUE))
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
  if (measure == "pac") {
    sg_cols <- freq_cols[freqs >= PAC_SLOW_GAMMA[1] & freqs <= PAC_SLOW_GAMMA[2]]
    d$slow_gamma <- rowMeans(d[, sg_cols, drop = FALSE], na.rm = TRUE)
  } else {
    th_cols <- freq_cols[freqs >= THETA[1]      & freqs <= THETA[2]]
    sg_cols <- freq_cols[freqs >= SLOW_GAMMA[1] & freqs <= SLOW_GAMMA[2]]
    d$theta      <- rowMeans(d[, th_cols, drop = FALSE], na.rm = TRUE)
    d$slow_gamma <- rowMeans(d[, sg_cols, drop = FALSE], na.rm = TRUE)
  }
  d
}

apply_contrast <- function(d, contrast) {
  if (contrast == "all")    return(d)
  if (contrast == "nostim") return(d[d$StimCond == "nostim", ])
  if (contrast == "stim")   return(d[d$StimCond == "stim", ])
  stop("Unknown contrast: ", contrast)
}

panel_data <- function(measure, scope_name, unit, band, contrast) {
  scope <- SCOPES[[scope_name]]
  d <- load_measure_df(measure)
  if (scope$needs_allhpc) d <- prep_with_allhpc(d, measure)
  d <- add_band_columns(d, measure)
  d <- apply_contrast(d, contrast)
  d <- d[d$Region == unit, ]
  d <- d[is.finite(d[[band]]), ]
  d$band_c <- d[[band]]
  d$Patient <- factor(d$Patient)
  d
}

posthoc_out_dir <- function(analysis_dir_token) {
  ensure_dir(file.path(
    REPO_ROOT, "OUTPUTS", "encoding_memory_reports",
    analysis_dir_token, "post_hoc_testing"
  ))
}
