##############################################################################
# Encoding MLM v2: Per-Region + Across-Region Power/Coherence + PAC
#
# Usage: Rscript run_v2_encoding.R --balanced
#        Rscript run_v2_encoding.R --imbalanced
#
# Per-region: Accuracy ~ band_c * StimCond + (1|Patient)
# Across-region: Accuracy ~ StimCond + Region + band_c + band_c:StimCond
#                + band_c:Region + (1|Patient)
# PAC (MTL: BLA_HPC, BLA_EC, BLA_PRC; HPC: BLA_HPC, BLA_CA, BLA_DG):
#   Accuracy ~ pac_z * StimCond + Region + (1|Patient)
#   Separate models for SG PAC and HFA PAC.
#
# Encoding bands: theta, slow gamma, HFA
##############################################################################

if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")

library(lme4)
library(lmerTest)
library(broom.mixed)

# ── Parse args ──
args <- commandArgs(trailingOnly = TRUE)
balanced <- "--balanced" %in% args
MIN_TRIALS <- 10

script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    a <- commandArgs(trailingOnly = FALSE)
    f <- grep("--file=", a, value = TRUE)
    if (length(f) > 0) dirname(normalizePath(sub("--file=", "", f)))
    else getwd()
  }
)

label <- if (balanced) "balanced" else "imbalanced"
out_dir <- file.path(script_dir, "outputs", paste0(label, "_encoding_mlm"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA <- c(4.88, 7.81)
SLOW_GAMMA <- c(30.27, 54.69)
HFA <- c(70.31, 99.61)
PAC_SLOW_GAMMA <- c(30, 50)
PAC_HFA <- c(70, 100)

BANDS <- list(
  theta = list(range = THETA, label = "Theta"),
  slow_gamma = list(range = SLOW_GAMMA, label = "Slow Gamma"),
  fast_gamma = list(range = HFA, label = "HFA")
)

save_coefs <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  cat(sprintf("  -> Saved %s\n", filename))
  return(td)
}

print_interaction <- function(td, pattern) {
  row <- td[grepl(pattern, td$term), ]
  if (nrow(row) > 0) {
    cat(sprintf("    %s: OR=%.4f [%.4f, %.4f], p=%.4f\n",
                row$term[1], row$estimate[1], row$conf.low[1], row$conf.high[1],
                row$p.value[1]))
  }
}

# ── Balanced-trials filter ──
filter_balanced <- function(df, measure_label) {
  if (!balanced) {
    cat(sprintf("\n  [%s] No filter (imbalanced): %d patients\n",
                measure_label, length(unique(df$Patient))))
    return(df)
  }
  patients <- unique(df$Patient)
  counts <- data.frame(Patient = patients, stringsAsFactors = FALSE)
  counts$n_regions <- sapply(patients, function(p) length(unique(df$Region[df$Patient == p])))
  counts$n_rem <- sapply(patients, function(p) sum(df$Patient == p & df$yes_or_no == "yes"))
  counts$n_forg <- sapply(patients, function(p) sum(df$Patient == p & df$yes_or_no == "no"))
  counts$n_rem_unique <- round(counts$n_rem / counts$n_regions)
  counts$n_forg_unique <- round(counts$n_forg / counts$n_regions)

  included <- counts$Patient[counts$n_rem_unique >= MIN_TRIALS & counts$n_forg_unique >= MIN_TRIALS]
  excluded <- counts[!(counts$Patient %in% included), ]

  cat(sprintf("\n  [%s] Balanced filter (>=%d per condition):\n", measure_label, MIN_TRIALS))
  cat(sprintf("    Included: %d / %d patients\n", length(included), nrow(counts)))
  if (nrow(excluded) > 0) {
    for (i in seq_len(nrow(excluded))) {
      cat(sprintf("    Excluded: %s (rem=%d, forg=%d)\n",
                  excluded$Patient[i], excluded$n_rem_unique[i], excluded$n_forg_unique[i]))
    }
  }
  df[df$Patient %in% included, ]
}

##############################################################################
# POWER
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("ENCODING POWER - %s\n", toupper(label)))
cat(paste(rep("=", 70), collapse=""), "\n")

pw_csv <- file.path(script_dir, "outputs", "csvs", "combined_encoding_power_all_mlmr_input.csv")
pw <- read.csv(pw_csv, stringsAsFactors = FALSE, check.names = FALSE)
pw <- pw[pw$yes_or_no %in% c("yes", "no") & pw$trial_type != "new", ]
pw <- filter_balanced(pw, "Power")

freq_cols <- sort(grep("^diff_Freq_", names(pw), value = TRUE))
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))

for (b in names(BANDS)) {
  r <- BANDS[[b]]$range
  pw[[b]] <- rowMeans(pw[, freq_cols[freqs >= r[1] & freqs <= r[2]], drop=FALSE], na.rm=TRUE)
}

pw$Accuracy <- ifelse(pw$yes_or_no == "yes", 1, 0)
pw$StimCond <- factor(ifelse(pw$trial_type == "nostim", "nostim", "stim"),
                      levels = c("nostim", "stim"))

mtl_regions <- c("BLA", "HPC", "EC", "PRC")
hpc_regions <- c("BLA", "CA", "DG")
all_regions <- unique(c(mtl_regions, hpc_regions))

# ── Per-region power models ──
cat("\n--- Per-Region Power Models ---\n")
for (reg in all_regions) {
  sub <- pw[pw$Region == reg, ]
  sub$Patient <- factor(sub$Patient)
  if (nrow(sub) < 30 || nlevels(sub$Patient) < 3) next

  cat(sprintf("\n  Region %s: %d trials, %d patients\n", reg, nrow(sub), nlevels(sub$Patient)))

  for (b in names(BANDS)) {
    sub$band_c <- sub[[b]] - mean(sub[[b]], na.rm=TRUE)
    cat(sprintf("    %s %s: ", reg, BANDS[[b]]$label))

    m <- tryCatch({
      glmer(Accuracy ~ band_c * StimCond + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("FAILED\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("power_region_%s_%s_coefs.csv", reg, b))
      print_interaction(td, "band_c:StimCond")
    }
  }
}

# ── Across-region power models ──
cat("\n--- Across-Region Power Models ---\n")
for (region_set_name in c("MTL", "HPC_subfields")) {
  allowed <- if (region_set_name == "MTL") mtl_regions else hpc_regions
  sub <- pw[pw$Region %in% allowed, ]
  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n  %s: %d trials, %d patients, regions: %s\n",
              region_set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))

  for (b in names(BANDS)) {
    sub$band_c <- sub[[b]] - mean(sub[[b]], na.rm=TRUE)
    cat(sprintf("    %s %s: ", region_set_name, BANDS[[b]]$label))

    m <- tryCatch({
      glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("FAILED\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("power_across_%s_%s_coefs.csv", region_set_name, b))
      print_interaction(td, "StimCondstim:band_c")
    }
  }
}

##############################################################################
# COHERENCE
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("ENCODING COHERENCE - %s\n", toupper(label)))
cat(paste(rep("=", 70), collapse=""), "\n")

coh_csv <- file.path(script_dir, "outputs", "csvs", "combined_encoding_coherence_all_mlmr_input.csv")
coh <- read.csv(coh_csv, stringsAsFactors = FALSE, check.names = FALSE)
coh <- coh[coh$yes_or_no %in% c("yes", "no") & coh$trial_type != "new", ]
coh <- filter_balanced(coh, "Coherence")

freq_cols_c <- sort(grep("^diff_Freq_", names(coh), value = TRUE))
freqs_c <- as.numeric(sub("diff_Freq_", "", freq_cols_c))

for (b in names(BANDS)) {
  r <- BANDS[[b]]$range
  coh[[b]] <- rowMeans(coh[, freq_cols_c[freqs_c >= r[1] & freqs_c <= r[2]], drop=FALSE], na.rm=TRUE)
}

coh$Accuracy <- ifelse(coh$yes_or_no == "yes", 1, 0)
coh$StimCond <- factor(ifelse(coh$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

coh_mtl_pairs_fn <- function(df) {
  allowed <- c("BLA", "HPC", "EC", "PRC")
  df[sapply(strsplit(df$Region, "_", fixed=TRUE), function(p) all(p %in% allowed)), ]
}
coh_hpc_pairs_fn <- function(df) {
  allowed <- c("BLA", "CA", "DG")
  df[sapply(strsplit(df$Region, "_", fixed=TRUE), function(p) all(p %in% allowed)), ]
}

# ── Per-region-pair coherence models ──
cat("\n--- Per-Region-Pair Coherence Models ---\n")
all_coh_regions <- sort(unique(coh$Region))
# Filter to only MTL or HPC subfield pairs
mtl_coh <- coh_mtl_pairs_fn(coh)
hpc_coh <- coh_hpc_pairs_fn(coh)
valid_pairs <- unique(c(mtl_coh$Region, hpc_coh$Region))

for (reg in sort(valid_pairs)) {
  sub <- coh[coh$Region == reg, ]
  sub$Patient <- factor(sub$Patient)
  if (nrow(sub) < 30 || nlevels(sub$Patient) < 3) {
    cat(sprintf("\n  Skipping %s: %d trials, %d patients\n", reg, nrow(sub), nlevels(sub$Patient)))
    next
  }

  cat(sprintf("\n  Pair %s: %d trials, %d patients\n", reg, nrow(sub), nlevels(sub$Patient)))

  for (b in names(BANDS)) {
    sub$band_c <- sub[[b]] - mean(sub[[b]], na.rm=TRUE)
    cat(sprintf("    %s %s: ", reg, BANDS[[b]]$label))

    m <- tryCatch({
      glmer(Accuracy ~ band_c * StimCond + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("FAILED\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("coherence_region_%s_%s_coefs.csv", reg, b))
      print_interaction(td, "band_c:StimCond")
    }
  }
}

# ── Across-region coherence models ──
cat("\n--- Across-Region Coherence Models ---\n")
for (region_set_name in c("MTL", "HPC_subfields")) {
  sub <- if (region_set_name == "MTL") coh_mtl_pairs_fn(coh) else coh_hpc_pairs_fn(coh)
  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n  %s: %d trials, %d patients, pairs: %s\n",
              region_set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))

  for (b in names(BANDS)) {
    sub$band_c <- sub[[b]] - mean(sub[[b]], na.rm=TRUE)
    cat(sprintf("    %s %s: ", region_set_name, BANDS[[b]]$label))

    m <- tryCatch({
      glmer(Accuracy ~ StimCond + Region + band_c + band_c:StimCond + band_c:Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("FAILED\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("coherence_across_%s_%s_coefs.csv", region_set_name, b))
      print_interaction(td, "StimCondstim:band_c")
    }
  }
}

##############################################################################
# PAC (Z-scored, separate SG and HFA models)
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat(sprintf("ENCODING PAC - %s\n", toupper(label)))
cat(paste(rep("=", 70), collapse=""), "\n")

pac_csv <- file.path(script_dir, "outputs", "csvs", "combined_encoding_pac_all_mlmr_input.csv")
pac <- read.csv(pac_csv, stringsAsFactors = FALSE, check.names = FALSE)
pac <- pac[pac$yes_or_no %in% c("yes", "no") & pac$trial_type != "new", ]
pac <- filter_balanced(pac, "PAC")

freq_cols_p <- sort(grep("^diff_Freq_", names(pac), value = TRUE))
freqs_p <- as.numeric(sub("diff_Freq_", "", freq_cols_p))

sg_cols <- freq_cols_p[freqs_p >= PAC_SLOW_GAMMA[1] & freqs_p <= PAC_SLOW_GAMMA[2]]
hfa_cols <- freq_cols_p[freqs_p >= PAC_HFA[1] & freqs_p <= PAC_HFA[2]]

pac$slow_gamma_pac <- rowMeans(pac[, sg_cols, drop=FALSE], na.rm=TRUE)
pac$hfa_pac <- rowMeans(pac[, hfa_cols, drop=FALSE], na.rm=TRUE)

pac$Accuracy <- ifelse(pac$yes_or_no == "yes", 1, 0)
pac$StimCond <- factor(ifelse(pac$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

# PAC region sets
pac_sets <- list(
  MTL = c("BLA_HPC", "BLA_EC", "BLA_PRC"),
  HPC_subfields = c("BLA_HPC", "BLA_CA", "BLA_DG")
)

for (set_name in names(pac_sets)) {
  pairs <- pac_sets[[set_name]]
  sub <- pac[pac$Region %in% pairs, ]
  sub$Region <- factor(sub$Region, levels = sort(unique(sub$Region)))
  sub$Patient <- factor(sub$Patient)

  cat(sprintf("\n--- PAC %s: %d trials, %d patients, pairs: %s ---\n",
              set_name, nrow(sub), nlevels(sub$Patient),
              paste(levels(sub$Region), collapse=", ")))

  for (pac_type in c("slow_gamma", "hfa")) {
    pac_col <- paste0(pac_type, "_pac")
    pac_label <- if (pac_type == "slow_gamma") "SG" else "HFA"

    # Z-score within the pooled set
    pac_sd <- sd(sub[[pac_col]], na.rm = TRUE)
    if (pac_sd < 1e-10) {
      cat(sprintf("  Skipping %s %s PAC: near-zero SD\n", set_name, pac_label))
      next
    }
    sub$pac_z <- as.numeric(scale(sub[[pac_col]]))

    cat(sprintf("  %s %s PAC (SD=%.6f): ", set_name, pac_label, pac_sd))

    m <- tryCatch({
      glmer(Accuracy ~ pac_z * StimCond + Region + (1 | Patient),
            data = sub, family = binomial, control = ctrl)
    }, error = function(e) { cat("FAILED\n"); NULL })

    if (!is.null(m)) {
      td <- save_coefs(m, sprintf("pac_%s_%s_coefs.csv", set_name, pac_type))
      print_interaction(td, "pac_z:StimCond")
      # Also print full summary
      cat("\n")
      print(summary(m))
    }
  }
}

cat(sprintf("\n\n===== ENCODING MLM V2 (%s) COMPLETE =====\n", toupper(label)))
