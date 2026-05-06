##############################################################################
# Refit each significant H1b LMM interaction panel and save:
#   - per-subject means of (feature, mem_mod_z, group) for scatter overlays
#   - LMM-predicted feature values for the continuous moderator on a mem_mod_z
#     grid (no-stim and stim), with model-based 95% CI from the fixed-effects
#     vcov.
#   - LMM-predicted feature values for the quad categorical moderator at each
#     responder group x stim condition combination, with 95% CI.
##############################################################################

suppressPackageStartupMessages({
  library(lme4); library(lmerTest)
})

script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    a <- commandArgs(trailingOnly = FALSE); f <- grep("--file=", a, value = TRUE)
    if (length(f) > 0) dirname(normalizePath(sub("--file=", "", f))) else getwd()
  }
)

out_dir <- file.path(script_dir, "outputs", "h1b_subregions_quad_lmm_imbalanced_retrieval_mlm")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

THETA <- c(4.88, 7.81); SLOW_GAMMA <- c(30.27, 54.69); PAC_SLOW_GAMMA <- c(30, 50)

# Responder
resp <- read.csv(file.path(script_dir, "outputs", "csvs",
                           "AMMEBLAES_responder_status.csv"), stringsAsFactors=FALSE)
names(resp)[names(resp) == "Responder.status"] <- "ResponderStatus"
resp$Patient[resp$Patient == "BJH033"] <- "BJH032"
resp <- resp[!duplicated(resp$Patient), ]
resp$QuadResponderGroup <- factor(
  resp$ResponderStatus,
  levels = c("Non-responders", "Anti-responders", "Moderate responders", "Strong responders"))
levels(resp$QuadResponderGroup) <- c("NonResp", "AntiResp", "Moderate", "Strong")
resp$mem_mod_z <- as.numeric(scale(resp$avg_stim_dprime_diff))

prep <- function(csv_path) {
  d <- read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)
  d <- d[d$yes_or_no %in% c("yes","no") & d$trial_type != "new", ]
  d$Accuracy <- ifelse(d$yes_or_no == "yes", 1L, 0L)
  d$StimCond <- factor(ifelse(d$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))
  merge(d, resp[, c("Patient","QuadResponderGroup","mem_mod_z")], by="Patient", all.x=TRUE)
}

add_band <- function(d, band, range) {
  freq_cols <- sort(grep("^diff_Freq_", names(d), value = TRUE))
  freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))
  cols <- freq_cols[freqs >= range[1] & freqs <= range[2]]
  d[[band]] <- rowMeans(d[, cols, drop=FALSE], na.rm=TRUE)
  d
}

# Build trial-level data per modality
pw <- prep(file.path(script_dir,"outputs","csvs","combined_retrieval_power_all_mlmr_input.csv"))
pw <- add_band(pw, "theta", THETA); pw <- add_band(pw, "slow_gamma", SLOW_GAMMA)

coh <- prep(file.path(script_dir,"outputs","csvs","combined_retrieval_coherence_all_mlmr_input.csv"))
coh <- add_band(coh, "theta", THETA); coh <- add_band(coh, "slow_gamma", SLOW_GAMMA)

pac <- prep(file.path(script_dir,"outputs","csvs","combined_retrieval_pac_all_mlmr_input.csv"))
pac <- add_band(pac, "slow_gamma", PAC_SLOW_GAMMA)  # PAC SG only

get_data <- function(modality, unit, band_label) {
  src <- list(power = pw, coherence = coh, pac = pac)[[modality]]
  feature_col <- if (modality == "pac") "slow_gamma" else band_label
  sub <- src[src$Region == unit, ]
  sub <- sub[is.finite(sub[[feature_col]]) & is.finite(sub$mem_mod_z) &
             !is.na(sub$QuadResponderGroup), ]
  sub$Patient <- factor(sub$Patient)
  sub$feature <- as.numeric(sub[[feature_col]])
  sub
}

# Panels to process: union of continuous-significant and categorical-significant.
PANELS <- list(
  list("coherence","BLA_CA","theta"),
  list("coherence","BLA_CA","slow_gamma"),
  list("coherence","BLA_DG","slow_gamma"),
  list("coherence","BLA_EC","slow_gamma"),
  list("coherence","CA_PRC","theta"),
  list("coherence","EC_HPC","theta"),
  list("coherence","HPC_PRC","slow_gamma"),
  list("pac","BLA_PRC","slow_gamma"),
  list("pac","CA_PRC","slow_gamma"),
  list("pac","EC_HPC","slow_gamma"),
  list("pac","HPC_PRC","slow_gamma"),
  list("power","DG","slow_gamma"),
  list("power","DG","theta")
)

predict_with_ci <- function(model, X) {
  beta <- fixef(model); V <- vcov(model)
  X <- X[, names(beta), drop = FALSE]
  yhat <- as.numeric(X %*% beta)
  se <- sqrt(rowSums((X %*% V) * X))
  data.frame(yhat = yhat, lo = yhat - 1.96 * se, hi = yhat + 1.96 * se)
}

for (p in PANELS) {
  modality <- p[[1]]; unit <- p[[2]]; band <- p[[3]]
  cat(sprintf("\n--- %s %s %s ---\n", modality, unit, band))
  sub <- get_data(modality, unit, band)
  if (nrow(sub) < 30 || nlevels(sub$Patient) < 5) { cat("  skipped\n"); next }

  panel_tag <- sprintf("%s_%s_%s", modality, unit, band)

  # Per-subject means (used for both overlays)
  per_subj <- aggregate(cbind(feature, mem_mod_z) ~ Patient + StimCond + QuadResponderGroup,
                        data = sub, FUN = mean)
  per_subj$n_trials <- as.integer(table(sub$Patient, sub$StimCond)[
    cbind(as.character(per_subj$Patient), as.character(per_subj$StimCond))])
  write.csv(per_subj, file.path(out_dir, sprintf("plot_subjects_%s.csv", panel_tag)),
            row.names = FALSE)

  # Continuous moderator model
  m_cont <- tryCatch(lmer(feature ~ StimCond + mem_mod_z + StimCond:mem_mod_z + (1 | Patient),
                          data = sub, REML = TRUE, control = ctrl),
                     error = function(e) { cat("    cont fit FAILED\n"); NULL })
  if (!is.null(m_cont)) {
    z_grid <- seq(min(sub$mem_mod_z), max(sub$mem_mod_z), length.out = 200)
    grid_rows <- list()
    for (sv in c(0, 1)) {
      cond <- if (sv == 0) "nostim" else "stim"
      X <- cbind(`(Intercept)` = 1, StimCondstim = sv, mem_mod_z = z_grid,
                 `StimCondstim:mem_mod_z` = sv * z_grid)
      pr <- predict_with_ci(m_cont, X)
      grid_rows[[length(grid_rows)+1]] <- data.frame(
        StimCond = cond, mem_mod_z = z_grid, yhat = pr$yhat, lo = pr$lo, hi = pr$hi)
    }
    write.csv(do.call(rbind, grid_rows),
              file.path(out_dir, sprintf("plot_cont_pred_%s.csv", panel_tag)),
              row.names = FALSE)
  }

  # Categorical (quad) model
  m_cat <- tryCatch(lmer(feature ~ StimCond + QuadResponderGroup +
                                  StimCond:QuadResponderGroup + (1 | Patient),
                         data = sub, REML = TRUE, control = ctrl),
                    error = function(e) { cat("    cat fit FAILED\n"); NULL })
  if (!is.null(m_cat)) {
    grid_rows <- list()
    groups <- levels(sub$QuadResponderGroup)
    for (g in groups) {
      gA <- as.numeric(g == "AntiResp")
      gM <- as.numeric(g == "Moderate")
      gS <- as.numeric(g == "Strong")
      for (sv in c(0, 1)) {
        cond <- if (sv == 0) "nostim" else "stim"
        X <- cbind(
          `(Intercept)` = 1,
          StimCondstim = sv,
          QuadResponderGroupAntiResp = gA,
          QuadResponderGroupModerate = gM,
          QuadResponderGroupStrong   = gS,
          `StimCondstim:QuadResponderGroupAntiResp` = sv * gA,
          `StimCondstim:QuadResponderGroupModerate` = sv * gM,
          `StimCondstim:QuadResponderGroupStrong`   = sv * gS
        )
        pr <- predict_with_ci(m_cat, X)
        grid_rows[[length(grid_rows)+1]] <- data.frame(
          Group = g, StimCond = cond, yhat = pr$yhat, lo = pr$lo, hi = pr$hi)
      }
    }
    write.csv(do.call(rbind, grid_rows),
              file.path(out_dir, sprintf("plot_cat_pred_%s.csv", panel_tag)),
              row.names = FALSE)
  }
  cat("  done\n")
}

cat("\n===== DONE =====\n")
