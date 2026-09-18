##############################################################################
# Johnson-Neyman plots (v2) using interactions::sim_slopes(), per the bioRxiv
# Pruessner et al. 2023 recipe. Replaces the rough JN PNGs from
# posthoc_johnson_neyman.R for the 8 trending encoding interaction panels.
#
# For each panel, fits both:
#   RI: Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)
#   RS: Accuracy ~ band_c + StimCond + band_c:StimCond + (1 + StimCond | Patient)
# (StimCond is coded 0/1 numeric for clean sim_slopes() handling.)
#
# Then calls:
#   sim_slopes(M, pred = StimCond_num, modx = band_c,
#              confint = FALSE, jnplot = TRUE)
# captures the JN plot, recolors the slope line gray, and saves to PNG at the
# same path the PDF report builder already expects:
#
#   {out_dir}/jn_<measure>_<unit>_<band>_<model>.png
#
# Output:
#   PNGs (overwriting v1) + jn_summary_v2.csv with sim_slopes() output.
##############################################################################

source("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES/mlm_code/posthoc_common.R")

suppressPackageStartupMessages({
  library(interactions); library(ggplot2)
})

ANALYSIS <- "GLMM_interactions_model_variations"

# Model formulas use StimCond_num (numeric 0/1) so sim_slopes() treats it as
# a continuous predictor (slope is identical to the StimCondstim contrast).
RI_FORM <- as.formula(
  "Accuracy ~ band_c + StimCond_num + band_c:StimCond_num + (1 | Patient)")
RS_FORM <- as.formula(
  "Accuracy ~ band_c + StimCond_num + band_c:StimCond_num + (1 + StimCond_num | Patient)")

fit_or_centered <- function(formula, data) {
  m <- safe_glmer(formula, data)
  if (is.null(m)) {
    data$band_c <- data$band_c - mean(data$band_c, na.rm = TRUE)
    m <- safe_glmer(formula, data)
    if (!is.null(m)) attr(m, "centered") <- TRUE
  }
  m
}

run_jn_plot <- function(m, panel_str, model_label, out_png) {
  if (is.null(m)) {
    cat("  ", model_label, " model is NULL; skipping\n")
    return(NULL)
  }
  A <- tryCatch(
    sim_slopes(m, pred = StimCond_num, modx = band_c,
               confint = FALSE, jnplot = TRUE,
               johnson_neyman = TRUE),
    error = function(e) {
      cat("  sim_slopes() FAILED (", model_label, "): ",
          conditionMessage(e), "\n", sep = "")
      NULL
    }
  )
  if (is.null(A)) return(NULL)

  p <- NULL
  if (!is.null(A$jn) && length(A$jn) >= 1 && !is.null(A$jn[[1]]$plot)) {
    p <- A$jn[[1]]$plot
    # Keep the default sim_slopes() palette (cyan = p<.05, pink = n.s.,
    # dashed teal = JN boundary, thick black bar = observed data range).
    # We only override title and axes for clarity.
    bounds <- A$jn[[1]]$bounds
    bounds_lbl <- if (!is.null(bounds))
      sprintf("JN boundaries: [%.3g, %.3g]", bounds[1], bounds[2])
    else "JN: no boundaries inside region"
    p <- p +
      ggtitle(sprintf("Johnson-Neyman: %s\n%s   |   %s",
                      panel_str, model_label, bounds_lbl)) +
      labs(x = "Band value (raw, uncentered) - moderator",
           y = "Simple slope of Stim - No-stim on logit(P(remember))",
           fill = "Significance", colour = "Significance") +
      theme_minimal(base_size = 11) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5))
    ggsave(out_png, plot = p, width = 9, height = 6, dpi = 200)
    cat("  Saved:", out_png, "\n")
  } else {
    cat("  No JN plot returned for ", model_label, " (no boundary)\n")
  }

  list(sim = A, plot = p)
}

format_summary_row <- function(measure, scope, unit, band, model_tag,
                               sim_out, n_trials, n_patients, centered) {
  data.frame(
    measure = measure, scope = scope, unit = unit, band = band,
    model = model_tag, n_trials = n_trials, n_patients = n_patients,
    centered = centered,
    jn_alpha = if (!is.null(sim_out$jn[[1]]$alpha)) sim_out$jn[[1]]$alpha else NA,
    jn_bounds_lo = if (!is.null(sim_out$jn[[1]]$bounds))
                     sim_out$jn[[1]]$bounds[1] else NA,
    jn_bounds_hi = if (!is.null(sim_out$jn[[1]]$bounds))
                     sim_out$jn[[1]]$bounds[2] else NA,
    has_jn_plot = !is.null(sim_out$jn[[1]]$plot),
    stringsAsFactors = FALSE
  )
}

main <- function() {
  out_dir <- posthoc_out_dir(ANALYSIS_DIR[[ANALYSIS]])
  trending_csv <- file.path(out_dir, "trending_panels.csv")
  trending <- read.csv(trending_csv, stringsAsFactors = FALSE)
  cat(sprintf("Running JN-v2 plots on %d trending interaction panels\n",
              nrow(trending)))

  summaries <- list()
  contrast <- ANALYSIS_CONTRAST[[ANALYSIS]]

  for (i in seq_len(nrow(trending))) {
    r <- trending[i, ]
    measure <- r$measure; scope <- r$scope; unit <- r$unit; band <- r$band
    panel_str <- sprintf("%s | %s | %s | %s", measure, scope, unit, band)
    cat(sprintf("\n[%d/%d] %s\n", i, nrow(trending), panel_str))

    d <- panel_data(measure, scope, unit, band, contrast)
    if (nrow(d) < MIN_TRIALS ||
        length(unique(d$Patient)) < MIN_PATIENTS) {
      cat("  SKIP (not enough data)\n"); next
    }
    d$StimCond_num <- ifelse(d$StimCond == "stim", 1L, 0L)

    for (mv in c("ri", "rs")) {
      formula <- if (mv == "ri") RI_FORM else RS_FORM
      m <- fit_or_centered(formula, d)
      label <- if (mv == "ri") "Random intercept only"
               else "+ random StimCond slope"
      png <- file.path(out_dir,
                       sprintf("jn_%s_%s_%s_%s.png",
                               measure, unit, band, mv))
      res <- run_jn_plot(m, panel_str, label, png)
      if (!is.null(res) && !is.null(res$sim)) {
        summaries[[length(summaries) + 1]] <- format_summary_row(
          measure, scope, unit, band, mv, res$sim,
          nrow(d), length(unique(d$Patient)),
          isTRUE(attr(m, "centered")))
      }
    }
  }

  if (length(summaries)) {
    out <- do.call(rbind, summaries)
    write.csv(out, file.path(out_dir, "jn_summary_v2.csv"),
              row.names = FALSE)
    cat("\nSaved: jn_summary_v2.csv\n")
  }
}

main()
cat("\n===== JN-v2 DONE =====\n")
