##############################################################################
# Cluster-based permutation test of endogenous memory effects (Remembered
# vs Forgotten) on no-stim retrieval trials, using HMLET.
#
# Modalities: power, coherence, PAC.
# Permutations: 5000 per modality.
# Subjects: BJH032/BJH033 folded -> single subject (54 unique).
#
# Outputs to outputs/PermutationOutputsAlireza/endogenous_memory/<modality>/.
#
# Usage:
#   Rscript run_permutation_endogenous_memory.R           # all 3 modalities
#   Rscript run_permutation_endogenous_memory.R power     # specific modality
##############################################################################

suppressPackageStartupMessages({
  if (!require("pacman"))   install.packages("pacman",
                                              repos = "https://cloud.r-project.org")
  library(pacman)
  p_load(data.table, dplyr, tidyr, ggplot2)
  if (!require("HMLET", quietly = TRUE)) {
    if (!require("devtools")) install.packages("devtools",
                                                repos = "https://cloud.r-project.org")
    devtools::install_github("Alireza-Kazemi/HMLET",
                             subdir = "RPackage", upgrade = "never")
    library(HMLET)
  }
})

args <- commandArgs(trailingOnly = TRUE)
modalities <- if (length(args) > 0) args else c("power", "coherence", "pac")

script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    a <- commandArgs(trailingOnly = FALSE)
    f <- grep("--file=", a, value = TRUE)
    if (length(f) > 0) dirname(normalizePath(sub("--file=", "", f))) else getwd()
  }
)

INPUT_DIR <- file.path(script_dir, "outputs", "csvs")
OUT_BASE  <- file.path(script_dir, "outputs", "PermutationOutputsAlireza",
                       "endogenous_memory")
dir.create(OUT_BASE, recursive = TRUE, showWarnings = FALSE)

N_SAMPLES   <- 5000
SEED        <- 100
MIN_SUBJECTS <- 5

run_modality <- function(modality) {
  input_csv <- switch(modality,
    "power"     = "combined_retrieval_power_all_mlmr_input.csv",
    "coherence" = "combined_retrieval_coherence_all_mlmr_input.csv",
    "pac"       = "combined_retrieval_pac_all_mlmr_input.csv")

  cat(sprintf("\n========== %s | reading %s ==========\n", modality, input_csv))

  dat <- read.csv(file.path(INPUT_DIR, input_csv),
                  stringsAsFactors = FALSE, check.names = FALSE)

  # Fold BJH033 into BJH032
  dat$Patient[dat$Patient == "BJH033"] <- "BJH032"

  # Endogenous memory: no-stim retrieval trials only, valid memory response
  dat <- dat[dat$yes_or_no %in% c("yes", "no") &
             dat$trial_type == "nostim", ]
  if (nrow(dat) == 0) { cat("  No nostim trials\n"); return(invisible()) }
  dat$Memory <- ifelse(dat$yes_or_no == "yes", "Remembered", "Forgotten")

  freq_cols <- grep("_Freq_", names(dat), value = TRUE)
  if (length(freq_cols) == 0) {
    cat("  No _Freq_ columns; skipping\n"); return(invisible())
  }

  datL <- pivot_longer(dat, cols = all_of(freq_cols),
                       names_to = "Freqs", values_to = "value")
  datL$ValueType <- gsub("_Freq_.*$", "", datL$Freqs)
  datL$Freqs     <- as.numeric(gsub("[^0-9.]", "", datL$Freqs))

  vts <- unique(datL$ValueType)
  cat("  ValueTypes found:", paste(vts, collapse = ", "), "\n")
  if ("diff" %in% vts) datL <- datL[datL$ValueType == "diff", ]

  if (!"Measure" %in% names(datL)) datL$Measure <- modality

  datAvg <- datL %>%
    group_by(Measure, Patient, Region, Memory, Freqs, ValueType) %>%
    summarise(value = mean(value, na.rm = TRUE), N = n(), .groups = "drop") %>%
    as.data.frame()

  # Paired test requires both Memory levels per Patient x Region
  paired <- datAvg %>%
    distinct(Patient, Region, Memory) %>%
    count(Patient, Region) %>%
    filter(n == 2L) %>%
    select(Patient, Region)
  datAvg <- datAvg %>% semi_join(paired, by = c("Patient", "Region"))

  region_n <- datAvg %>%
    distinct(Patient, Region) %>%
    count(Region, name = "n_subjects") %>%
    arrange(desc(n_subjects))
  cat("  Subjects with paired data per Region:\n")
  print(region_n, n = nrow(region_n))

  # Drop regions with < MIN_SUBJECTS subjects
  drop_regs <- region_n$Region[region_n$n_subjects < MIN_SUBJECTS]
  if (length(drop_regs) > 0) {
    cat(sprintf("  Dropping (n < %d): %s\n",
                MIN_SUBJECTS, paste(drop_regs, collapse = ", ")))
    datAvg <- datAvg[!datAvg$Region %in% drop_regs, ]
  }
  if (nrow(datAvg) == 0) { cat("  Nothing to test\n"); return(invisible()) }

  datAvg$Tests <- paste(datAvg$Measure, datAvg$ValueType, datAvg$Region,
                        sep = "_")
  datAvg$trial <- 1

  cat(sprintf("  Running cluster perm test on %d Region(s), %d freqs, %d subjects total\n",
              length(unique(datAvg$Region)),
              length(unique(datAvg$Freqs)),
              length(unique(datAvg$Patient))))

  datP <- PermutationTestDataPrep_HMLET(
    data = datAvg, ID = "Patient", trial = NULL,
    timePoint = "Freqs",
    condition = "Memory", conditionLevels = c("Remembered", "Forgotten"),
    gazeMeasure = "value", testName = "Tests")

  set.seed(SEED)
  total_n <- length(unique(datP$ID))
  resPerm <- PermutationTest_HMLET(
    datP, samples = N_SAMPLES, paired = TRUE,
    threshold_t = stats::qt(p = 1 - .05, df = total_n - 1),
    permuteTrialsWithinSubject = FALSE)

  out_dir <- file.path(OUT_BASE, modality)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # Cluster stats
  cs <- resPerm[[1]]
  if (!is.null(cs)) {
    cs_path <- file.path(out_dir, sprintf("%s_cluster_stats.csv", modality))
    write.csv(cs, cs_path, row.names = FALSE)
    cat(sprintf("  Saved %s\n", cs_path))
  }

  # Save processed data for traceability
  write.csv(datAvg,
            file.path(out_dir, sprintf("%s_processed_data.csv", modality)),
            row.names = FALSE)

  # Time-series facet plot with cluster overlay
  ylab <- switch(modality,
    "power"     = "Power (dB, baseline-subtracted)",
    "coherence" = "Coherence (baseline-subtracted)",
    "pac"       = "PAC (baseline-subtracted)")

  n_facets <- length(unique(datP$testName))
  ncol <- min(4, max(2, ceiling(sqrt(n_facets))))
  nrow <- ceiling(n_facets / ncol)

  p <- PlotTimeSeries_HMLET(
    resPerm, showDataPointProp = FALSE,
    showOverallMean = "Line",
    gazePropRibbonAlpha = 0.2,
    clusterFillAlpha    = 0.4,
    alphaOverallMean    = 0.9,
    yLabel = ylab, pointSize = 1) +
    facet_wrap(~ testName, ncol = ncol) +
    theme(strip.text = element_text(size = 8),
          axis.text  = element_text(size = 7))

  ggsave(file.path(out_dir,
                   sprintf("%s_endogenous_memory.png", modality)),
         plot = p,
         width = ncol * 3.5, height = nrow * 3.0,
         dpi = 160, limitsize = FALSE)

  pn <- PlotNullDistribution_HMLET(resPerm)
  ggsave(file.path(out_dir,
                   sprintf("%s_null_distribution.png", modality)),
         plot = pn, width = 8, height = 6, dpi = 160, limitsize = FALSE)

  cat(sprintf("  Done: %s\n", modality))
}

for (m in modalities) run_modality(m)
cat("\n===== ALL DONE =====\n")
