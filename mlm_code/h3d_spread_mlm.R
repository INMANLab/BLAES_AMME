##############################################################################
#  Hypothesis 3d — IED Spread, Neural Dynamics, and Memory
#
#  Part 1: Spread → Memory (4 spread metrics x stim)
#  Part 2: Spread x Stim x BLA-HPC Coherence → Memory
#           4 spread metrics x 2 bands = 8 models
#  Part 3: Spread x Stim x BLA-HPC PAC → Memory
#           4 spread metrics x 2 PAC metrics = 8 models
#
#  Spread metrics (patient-level, z-scored):
#    channel_spread  = mean channels (contacts) per trial
#    region_spread   = mean regions per trial
#    cross_region    = proportion of cross-region IED trials
#    within_region   = mean leads (within-region detections) per trial
##############################################################################

if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
if (!require("ggplot2"))     install.packages("ggplot2",      repos = "https://cloud.r-project.org")
if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")

library(lme4)
library(lmerTest)
library(ggplot2)
library(broom.mixed)

script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg)))
    else getwd()
  }
)
out_dir <- file.path(script_dir, "outputs", "ied_timing_memory")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

save_tidy <- function(model, filename) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  return(td)
}

##############################################################################
# ═══════════════════════ PART 1: SPREAD → MEMORY ══════════════════════════
##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 1: IED SPREAD → MEMORY\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

spread <- read.csv(file.path(out_dir, "h3d_spread_trials.csv"), stringsAsFactors = FALSE)
spread$patient_id <- factor(spread$patient_id)

cat(sprintf("Spread data: %d trials, %d patients\n", nrow(spread), nlevels(spread$patient_id)))

# Center trial-level predictors
spread$channel_spread_c <- scale(spread$n_channels, scale = FALSE)
spread$region_spread_c <- scale(spread$n_regions, scale = FALSE)
spread$within_region_c <- scale(spread$n_leads, scale = FALSE)
# cross_region is already binary 0/1

spread_models <- list(
  "S1_channel" = list(label = "Channel Spread x Stim",
                      formula = memory ~ channel_spread_c * stim + (1 | patient_id)),
  "S2_region" = list(label = "Region Spread x Stim",
                     formula = memory ~ region_spread_c * stim + (1 | patient_id)),
  "S3_crossregion" = list(label = "Cross-Region IED x Stim",
                          formula = memory ~ cross_region * stim + (1 | patient_id)),
  "S4_withinregion" = list(label = "Within-Region Spread x Stim",
                           formula = memory ~ within_region_c * stim + (1 | patient_id))
)

for (nm in names(spread_models)) {
  info <- spread_models[[nm]]
  cat(sprintf("\n========== MODEL %s: %s ==========\n", nm, info$label))
  m <- glmer(info$formula, data = spread, family = binomial, control = ctrl)
  print(summary(m))
  save_tidy(m, sprintf("h3d_model%s_odds_ratios.csv", nm))
  spread_models[[nm]]$model <- m
}

# Forest plot from additive-only models for main effects
td_all <- data.frame()
for (nm in names(spread_models)) {
  td <- tidy(spread_models[[nm]]$model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  td <- td[td$term != "(Intercept)" & !grepl("stim", td$term), ]
  if (nrow(td) > 0) {
    td$spread_type <- spread_models[[nm]]$label
    td_all <- rbind(td_all, td)
  }
}
if (nrow(td_all) > 0) {
  td_all$label <- td_all$spread_type
  td_all$label <- factor(td_all$label, levels = rev(td_all$label))
  p_spread <- ggplot(td_all, aes(x = estimate, y = label)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    geom_point(size = 3) +
    geom_errorbar(aes(xmin = conf.low, xmax = conf.high), width = 0.2, orientation = "y") +
    geom_text(aes(label = sprintf("OR=%.2f, p=%s", estimate,
                                  ifelse(p.value < .001, "<.001", sprintf("%.3f", p.value)))),
              hjust = -0.1, size = 3) +
    scale_x_continuous(limits = c(0.2, 3.5)) +
    labs(x = "Odds Ratio (95% CI)", y = "",
         title = "IED Spread Predictors of Memory",
         subtitle = "GLMM main effects (centered), random intercept for patient") +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold"))
  ggsave(file.path(out_dir, "h3d_spread_forest.png"), p_spread, width = 8, height = 4, dpi = 300)
}

##############################################################################
# Load patient-level spread metrics for Parts 2 & 3
##############################################################################
spread_desc <- read.csv(file.path(out_dir, "h3d_spread_descriptives.csv"),
                        stringsAsFactors = FALSE)
# Rename for clarity
pat_spread <- data.frame(
  patient_id = spread_desc$patient_id,
  pat_channel_spread = spread_desc$mean_channels,
  pat_region_spread = spread_desc$mean_regions,
  pat_cross_region = spread_desc$pct_cross_region / 100,  # proportion
  pat_within_region = spread_desc$mean_leads
)

# Define the 4 spread metrics for looping
spread_metrics <- list(
  "channel" = list(col = "pat_channel_spread", label = "Channel Spread"),
  "region"  = list(col = "pat_region_spread",  label = "Region Spread"),
  "crossreg" = list(col = "pat_cross_region",  label = "Cross-Region"),
  "withinreg" = list(col = "pat_within_region", label = "Within-Region Spread")
)

elig <- read.csv(file.path(out_dir, "h3d_eligible_subjects.csv"), stringsAsFactors = FALSE)
elig_coh <- elig$patient_id[tolower(elig$has_coherence) == "true"]
elig_pac <- elig$patient_id[tolower(elig$has_pac) == "true"]

##############################################################################
# ═══════════ PART 2: SPREAD x STIM x BLA COHERENCE → MEMORY ═════════════
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 2: SPREAD x STIM x BLA COHERENCE → MEMORY\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

coh_all <- read.csv(file.path(out_dir, "h3d_coherence_bla_trials.csv"), stringsAsFactors = FALSE)
coh_all <- coh_all[coh_all$patient_id %in% elig_coh, ]
coh_all <- merge(coh_all, pat_spread, by = "patient_id", all.x = TRUE)
coh_all$patient_id <- factor(coh_all$patient_id)

bla_pairs <- c("BLA_HPC")
bands <- c("theta", "slow_gamma")
band_labels <- c("Theta", "Slow Gamma")

coh_model_list <- list()

for (pair in bla_pairs) {
  coh_pair <- coh_all[coh_all$region_pair == pair, ]
  coh_pair$patient_id <- droplevels(coh_pair$patient_id)
  n_pat <- nlevels(coh_pair$patient_id)
  cat(sprintf("\n--- %s: %d trials, %d patients ---\n", pair, nrow(coh_pair), n_pat))

  if (nrow(coh_pair) < 20 || n_pat < 3) {
    cat("  Too few data, skipping.\n")
    next
  }

  for (i in seq_along(bands)) {
    band <- bands[i]
    bl <- band_labels[i]
    coh_col <- paste0("coh_", band)
    zcol <- paste0(coh_col, "_z")
    coh_pair[[zcol]] <- as.numeric(scale(coh_pair[[coh_col]]))

    for (sm_key in names(spread_metrics)) {
      sm <- spread_metrics[[sm_key]]
      sp_zcol <- paste0(sm$col, "_z")
      coh_pair[[sp_zcol]] <- as.numeric(scale(coh_pair[[sm$col]]))

      model_name <- sprintf("%s_%s_%s", pair, bl, sm$label)
      csv_name <- sprintf("h3d_coh_%s_%s_%s_odds_ratios.csv",
                          gsub("_", "", pair), tolower(band), sm_key)

      cat(sprintf("\n========== %s ==========\n", model_name))

      m <- tryCatch({
        f <- as.formula(paste0("memory ~ ", sp_zcol, " * stim * ", zcol, " + (1 | patient_id)"))
        glmer(f, data = coh_pair, family = binomial, control = ctrl)
      }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })

      if (!is.null(m)) {
        print(summary(m))
        save_tidy(m, csv_name)
        coh_model_list[[model_name]] <- m
      }
    }
  }
}

# Coherence forest plot: three-way interaction terms only
if (length(coh_model_list) > 0) {
  coh_3way <- data.frame()
  for (nm in names(coh_model_list)) {
    td <- tidy(coh_model_list[[nm]], effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
    # Only the three-way term (has two colons)
    three <- td[grepl(".*:.*:.*", td$term), ]
    if (nrow(three) > 0) {
      three$model <- nm
      three$label <- nm
      coh_3way <- rbind(coh_3way, three)
    }
  }
  if (nrow(coh_3way) > 0) {
    coh_3way$label <- factor(coh_3way$label, levels = rev(coh_3way$label))
    coh_3way$sig <- ifelse(coh_3way$p.value < 0.05, "sig", "ns")

    p_coh <- ggplot(coh_3way, aes(x = estimate, y = label, shape = sig)) +
      geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
      geom_point(size = 2.5, color = "steelblue") +
      geom_errorbar(aes(xmin = conf.low, xmax = conf.high), width = 0.2,
                    orientation = "y", color = "steelblue") +
      geom_text(aes(label = sprintf("p=%s", ifelse(p.value < .001, "<.001",
                                                    sprintf("%.3f", p.value)))),
                hjust = -0.1, size = 2.5, show.legend = FALSE) +
      scale_shape_manual(values = c("sig" = 17, "ns" = 16)) +
      labs(x = "Odds Ratio (95% CI)", y = "", shape = "",
           title = "Three-Way Interactions: Spread x Stim x BLA-HPC Coherence",
           subtitle = "Theta & slow gamma, 4 spread metrics") +
      theme_minimal(base_size = 11) +
      theme(plot.title = element_text(face = "bold"))
    ggsave(file.path(out_dir, "h3d_coherence_forest.png"), p_coh,
           width = 10, height = 5, dpi = 300)
  }
}

##############################################################################
# ═══════════ PART 3: SPREAD x STIM x BLA PAC → MEMORY ═══════════════════
##############################################################################
cat("\n\n", paste(rep("=", 70), collapse=""), "\n")
cat("PART 3: SPREAD x STIM x BLA PAC → MEMORY\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

pac_all <- read.csv(file.path(out_dir, "h3d_pac_bla_patient.csv"), stringsAsFactors = FALSE)

pac_model_list <- list()

for (pair in bla_pairs) {
  pac_pair <- pac_all[pac_all$region_pair == pair, ]
  pac_pair <- pac_pair[pac_pair$patient_id %in% elig_pac, ]
  pac_sub <- pac_pair[pac_pair$band == "Slow gamma",
                      c("patient_id", "pac_interaction", "pac_stim_rem")]

  if (nrow(pac_sub) < 3) {
    cat(sprintf("\n--- %s PAC: only %d patients, skipping ---\n", pair, nrow(pac_sub)))
    next
  }

  for (pac_metric in c("pac_interaction", "pac_stim_rem")) {
    pac_label <- ifelse(pac_metric == "pac_interaction", "PACint", "PACrem")
    pac_full <- ifelse(pac_metric == "pac_interaction",
                       "PAC Stim x Memory", "PAC Stim on Rem")

    for (sm_key in names(spread_metrics)) {
      sm <- spread_metrics[[sm_key]]

      sp <- spread[spread$patient_id %in% pac_sub$patient_id, ]
      sp <- merge(sp, pac_sub, by = "patient_id", all.x = TRUE)
      sp <- merge(sp, pat_spread[, c("patient_id", sm$col)], by = "patient_id", all.x = TRUE)
      sp$patient_id <- droplevels(factor(sp$patient_id))

      sp_zcol <- paste0(sm$col, "_z")
      pac_zcol <- paste0(pac_metric, "_z")
      sp[[sp_zcol]] <- as.numeric(scale(sp[[sm$col]]))
      sp[[pac_zcol]] <- as.numeric(scale(sp[[pac_metric]]))

      model_name <- sprintf("%s_%s_%s", pair, pac_full, sm$label)
      csv_name <- sprintf("h3d_pac_%s_%s_%s_odds_ratios.csv",
                          gsub("_", "", pair), pac_label, sm_key)

      cat(sprintf("\n========== %s ==========\n", model_name))
      cat(sprintf("  %d trials, %d patients\n", nrow(sp), nlevels(sp$patient_id)))

      m <- tryCatch({
        f <- as.formula(paste0("memory ~ ", sp_zcol, " * stim * ", pac_zcol, " + (1 | patient_id)"))
        glmer(f, data = sp, family = binomial, control = ctrl)
      }, error = function(e) { cat("  Failed:", conditionMessage(e), "\n"); NULL })

      if (!is.null(m)) {
        print(summary(m))
        save_tidy(m, csv_name)
        pac_model_list[[model_name]] <- m
      }
    }
  }
}

# PAC forest plot: three-way terms
if (length(pac_model_list) > 0) {
  pac_3way <- data.frame()
  for (nm in names(pac_model_list)) {
    td <- tidy(pac_model_list[[nm]], effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
    three <- td[grepl(".*:.*:.*", td$term), ]
    if (nrow(three) > 0) {
      three$model <- nm
      three$label <- nm
      pac_3way <- rbind(pac_3way, three)
    }
  }
  if (nrow(pac_3way) > 0) {
    pac_3way$label <- factor(pac_3way$label, levels = rev(pac_3way$label))
    pac_3way$sig <- ifelse(pac_3way$p.value < 0.05, "sig", "ns")

    p_pac <- ggplot(pac_3way, aes(x = estimate, y = label, shape = sig)) +
      geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
      geom_point(size = 2.5, color = "steelblue") +
      geom_errorbar(aes(xmin = conf.low, xmax = conf.high), width = 0.2,
                    orientation = "y", color = "steelblue") +
      geom_text(aes(label = sprintf("p=%s", ifelse(p.value < .001, "<.001",
                                                    sprintf("%.3f", p.value)))),
                hjust = -0.1, size = 2.5, show.legend = FALSE) +
      scale_shape_manual(values = c("sig" = 17, "ns" = 16)) +
      labs(x = "Odds Ratio (95% CI)", y = "", shape = "",
           title = "Three-Way Interactions: Spread x Stim x BLA-HPC PAC",
           subtitle = "Slow gamma, 4 spread metrics x 2 PAC metrics") +
      theme_minimal(base_size = 11) +
      theme(plot.title = element_text(face = "bold"))
    ggsave(file.path(out_dir, "h3d_pac_forest.png"), p_pac,
           width = 10, height = 5, dpi = 300)
  }
}

##############################################################################
# Save all summaries
##############################################################################
sink(file.path(out_dir, "h3d_all_summaries.txt"))
for (nm in names(spread_models)) {
  cat(sprintf("\n========== %s ==========\n\n", nm))
  print(summary(spread_models[[nm]]$model))
}
for (nm in names(coh_model_list)) {
  cat(sprintf("\n========== Coh: %s ==========\n\n", nm))
  print(summary(coh_model_list[[nm]]))
}
for (nm in names(pac_model_list)) {
  cat(sprintf("\n========== PAC: %s ==========\n\n", nm))
  print(summary(pac_model_list[[nm]]))
}
sink()

cat("\n\n===== ALL DONE =====\n")
