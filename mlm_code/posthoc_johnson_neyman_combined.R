##############################################################################
# Combined Johnson-Neyman figures.
#
# Two figures, one per region set, each with the theta and slow-gamma
# interaction panels grouped under clear band section headers:
#
#   jn_combined_main_regions.png    - main-region interactions (theta + slow gamma)
#   jn_combined_hippsubregions.png  - hipp.-subregion interactions (theta + slow gamma)
#
# Each sub-panel is the interactions::sim_slopes() JN plot for the random-
# intercept model (the RI fit gives interpretable JN boundaries; the RS fits
# mostly return degenerate -Inf/Inf regions, see jn_summary_v2.csv). Panels are
# tiled with cowplot::plot_grid(), one row per band with a bold band header.
#
# Display labels follow the dissertation convention: ALLHPC -> HPC, HPC -> SUB
# (figure text only; data column names are untouched).
##############################################################################

source("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES/mlm_code/posthoc_common.R")

suppressPackageStartupMessages({
  library(interactions); library(ggplot2); library(cowplot)
})

ANALYSIS <- "GLMM_interactions_model_variations"

RI_FORM <- as.formula(
  "Accuracy ~ band_c + StimCond_num + band_c:StimCond_num + (1 | Patient)")

# --- region sets ---------------------------------------------------------- #
# Each region set -> one output figure. Within it, panels are grouped by band
# (theta, slow gamma), each band shown as its own labelled row of sub-panels.
# scope only affects ALLHPC reconstruction; data is identical across the
# duplicate scopes in trending_panels.csv, so one representative scope is used.
REGIONS <- list(
  main_regions = list(
    title = "Main-region interactions",
    ncol  = 3L,
    out   = "jn_combined_main_regions.png",
    bands = list(
      list(label = "Theta", panels = list(
        list(measure = "coherence", scope = "HPCrhinal", unit = "ALLHPC_EC", band = "theta")
      )),
      list(label = "Slow gamma", panels = list(
        list(measure = "power",     scope = "HPCrhinal", unit = "PRC",        band = "slow_gamma"),
        list(measure = "coherence", scope = "HPCrhinal", unit = "ALLHPC_PRC", band = "slow_gamma"),
        list(measure = "pac",       scope = "HPCrhinal", unit = "EC_PRC",     band = "slow_gamma")
      ))
    )
  ),
  hippsubregions = list(
    title = "Hipp.-subregion interactions",
    ncol  = 2L,
    out   = "jn_combined_hippsubregions.png",
    bands = list(
      list(label = "Theta", panels = list(
        list(measure = "coherence", scope = "HippSubRhinal", unit = "EC_HPC", band = "theta"),
        list(measure = "power",     scope = "HippSubRhinal", unit = "DG",     band = "theta")
      )),
      list(label = "Slow gamma", panels = list(
        list(measure = "coherence", scope = "HippSubRhinal", unit = "CA_PRC", band = "slow_gamma"),
        list(measure = "power",     scope = "HippSubRhinal", unit = "DG",     band = "slow_gamma")
      ))
    )
  )
)

# --- display helpers ------------------------------------------------------ #
relabel_token <- function(tok) {
  if (tok == "ALLHPC") return("HPC")   # whole-hippocampus main region
  if (tok == "HPC")    return("SUB")   # hippocampal subfield
  tok
}
relabel_unit <- function(unit) {
  toks <- strsplit(unit, "_", fixed = TRUE)[[1]]
  paste(vapply(toks, relabel_token, character(1)), collapse = "-")
}
measure_disp <- function(m) c(power = "Power", coherence = "Coherence",
                              pac = "PAC")[[m]]
band_disp    <- function(b) c(theta = "Theta",
                              slow_gamma = "Slow gamma")[[b]]

# sim_slopes() maps the JN significance factor (levels "Significant" /
# "Insignificant") to fill+colour. Override so the *significant* stim-effect
# region is pink and the non-significant region is teal, with explicit labels.
SIG_VALUES <- c(Significant = "#F8766D", Insignificant = "#00BFC4")
SIG_BREAKS <- c("Significant", "Insignificant")
SIG_LABELS <- c("Stim effect significant", "No stim effect")

# --- one JN sub-panel ----------------------------------------------------- #
make_subpanel <- function(p) {
  d <- panel_data(p$measure, p$scope, p$unit, p$band,
                  ANALYSIS_CONTRAST[[ANALYSIS]])
  if (nrow(d) < MIN_TRIALS || length(unique(d$Patient)) < MIN_PATIENTS) {
    cat("  SKIP (not enough data):", p$measure, p$unit, p$band, "\n")
    return(NULL)
  }
  d$StimCond_num <- ifelse(d$StimCond == "stim", 1L, 0L)

  m <- safe_glmer(RI_FORM, d)
  if (is.null(m)) {
    d$band_c <- d$band_c - mean(d$band_c, na.rm = TRUE)
    m <- safe_glmer(RI_FORM, d)
  }
  if (is.null(m)) { cat("  model NULL:", p$unit, p$band, "\n"); return(NULL) }

  A <- tryCatch(
    sim_slopes(m, pred = StimCond_num, modx = band_c,
               confint = FALSE, jnplot = TRUE, johnson_neyman = TRUE),
    error = function(e) { cat("  sim_slopes FAILED:", conditionMessage(e), "\n"); NULL })
  if (is.null(A) || is.null(A$jn) || is.null(A$jn[[1]]$plot)) {
    cat("  no JN plot:", p$unit, p$band, "\n"); return(NULL)
  }

  bounds <- A$jn[[1]]$bounds
  bounds_lbl <- if (!is.null(bounds) && all(is.finite(bounds)))
    sprintf("JN: [%.3g, %.3g]", bounds[1], bounds[2])
  else "JN: no boundary in range"

  ttl <- sprintf("%s  %s", measure_disp(p$measure), relabel_unit(p$unit))
  A$jn[[1]]$plot +
    scale_fill_manual(values = SIG_VALUES, breaks = SIG_BREAKS,
                      labels = SIG_LABELS, name = NULL) +
    scale_colour_manual(values = SIG_VALUES, breaks = SIG_BREAKS,
                        labels = SIG_LABELS, name = NULL) +
    ggtitle(ttl, subtitle = bounds_lbl) +
    labs(x = sprintf("%s %s (moderator)", measure_disp(p$measure),
                     band_disp(p$band)),
         y = "Simple slope: Stim - No-stim\non logit(P(remember))") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
          plot.subtitle = element_text(size = 9, hjust = 0.5),
          legend.position = "bottom")
}

# --- one band row (header + padded panel grid) ---------------------------- #
band_header <- function(label) {
  ggdraw() +
    draw_label(label, fontface = "bold", size = 13, x = 0.012, hjust = 0,
               colour = "#333333")
}

band_row <- function(panels, ncol, label_offset) {
  plots <- lapply(panels, make_subpanel)
  plots <- Filter(Negate(is.null), plots)
  if (length(plots) == 0) return(NULL)
  labels <- LETTERS[label_offset + seq_along(plots)]
  n_real <- length(plots)
  # pad to a full ncol-wide row so panel widths match across bands, splitting
  # empty cells on both sides so a partial row is centered (e.g. lone theta).
  pad <- ncol - n_real
  if (pad > 0) {
    left <- pad %/% 2L; right <- pad - left
    plots  <- c(rep(list(NULL), left), plots, rep(list(NULL), right))
    labels <- c(rep("", left), labels, rep("", right))
  }
  list(grid = plot_grid(plotlist = plots, ncol = ncol,
                        labels = labels, label_size = 13),
       n = n_real)
}

# --- assemble one region figure ------------------------------------------- #
build_region <- function(rs, out_dir) {
  cat(sprintf("\n=== %s ===\n", rs$title))
  elems <- list(ggdraw() +
                  draw_label(rs$title, fontface = "bold", size = 16,
                             x = 0.5, hjust = 0.5))
  rel   <- c(0.55)
  label_offset <- 0L
  for (bd in rs$bands) {
    row <- band_row(bd$panels, rs$ncol, label_offset)
    if (is.null(row)) next
    label_offset <- label_offset + row$n
    elems <- c(elems, list(band_header(bd$label), row$grid))
    rel   <- c(rel, 0.4, 4.4)
  }
  if (length(elems) <= 1) { cat("  (no panels)\n"); return(invisible()) }

  combined <- plot_grid(plotlist = elems, ncol = 1, rel_heights = rel)
  out_png  <- file.path(out_dir, rs$out)
  ggsave(out_png, plot = combined,
         width = rs$ncol * 4.6, height = sum(rel) + 0.3, dpi = 200,
         limitsize = FALSE)
  cat("  Saved:", out_png, "\n")
}

main <- function() {
  out_dir <- posthoc_out_dir(ANALYSIS_DIR[[ANALYSIS]])
  for (key in names(REGIONS)) build_region(REGIONS[[key]], out_dir)
}

main()
cat("\n===== JN COMBINED DONE =====\n")
