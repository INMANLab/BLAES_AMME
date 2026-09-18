##############################################################################
# Johnson-Neyman test for the trending encoding GLMM interactions
# (GLMM_interactions_model_variations). For each trending panel
# (band_c:StimCondstim raw p < .10), compute the simple slope of StimCondstim
# as a function of band_c (the continuous moderator), and identify the JN
# region of band_c where stim vs nostim differs in P(remembered) on the
# logit scale.
#
# Two model versions per panel:
#   RI: Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)
#   RS: Accuracy ~ band_c + StimCond + band_c:StimCond + (1 + StimCond | Patient)
#
# Math (on the logit scale):
#   logit(P) = b0 + b1*band_c + b2*StimCondstim + b3*band_c*StimCondstim
#   Simple slope of StimCondstim at band_c = w:  SS(w) = b2 + b3*w
#   Var(SS(w))                              =  Var(b2) + 2*w*Cov(b2,b3) + w^2*Var(b3)
#   z(w)                                    =  SS(w) / sqrt(Var(SS(w)))
#   JN boundary(ies) solve:  z(w)^2 = z_crit^2
#     => (b3^2 - z_crit^2 * Var(b3)) * w^2
#        + 2 * (b2*b3 - z_crit^2 * Cov(b2,b3)) * w
#        + (b2^2 - z_crit^2 * Var(b2))         = 0
#
# Outputs:
#   jn_<measure>_<unit>_<band>_<model>.csv  -- 200-point grid (band_c, SS, SE, z, p, sig)
#   jn_summary.csv                          -- one row per (panel, model) with JN roots / region
##############################################################################

source("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES/mlm_code/posthoc_common.R")

ANALYSIS <- "GLMM_interactions_model_variations"
Z_CRIT <- qnorm(0.975)   # two-sided alpha = .05

INT_RI <- as.formula("Accuracy ~ band_c + StimCond + band_c:StimCond + (1 | Patient)")
INT_RS <- as.formula("Accuracy ~ band_c + StimCond + band_c:StimCond + (1 + StimCond | Patient)")

fit_or_centered <- function(formula, data) {
  m <- safe_glmer(formula, data)
  if (is.null(m)) {
    data$band_c <- data$band_c - mean(data$band_c, na.rm = TRUE)
    m <- safe_glmer(formula, data)
    if (!is.null(m)) attr(m, "centered") <- TRUE
  }
  m
}

jn_for_model <- function(m, band_grid) {
  beta <- fixef(m); V <- as.matrix(vcov(m))
  b2 <- beta["StimCondstim"]; b3 <- beta["band_c:StimCondstim"]
  v2 <- V["StimCondstim", "StimCondstim"]
  v3 <- V["band_c:StimCondstim", "band_c:StimCondstim"]
  c23 <- V["StimCondstim", "band_c:StimCondstim"]

  ss  <- b2 + b3 * band_grid
  vss <- v2 + 2 * band_grid * c23 + (band_grid^2) * v3
  se  <- sqrt(pmax(vss, 0))
  zv  <- ss / se
  pv  <- 2 * (1 - pnorm(abs(zv)))

  # JN boundaries: solve quadratic in w
  a <- b3^2 - Z_CRIT^2 * v3
  b <- 2 * (b2 * b3 - Z_CRIT^2 * c23)
  cc <- b2^2 - Z_CRIT^2 * v2
  disc <- b^2 - 4 * a * cc
  roots <- if (a == 0) {
    if (b == 0) numeric(0) else -cc / b
  } else if (disc < 0) {
    numeric(0)
  } else {
    s <- sqrt(disc)
    c((-b - s) / (2 * a), (-b + s) / (2 * a))
  }

  # Direction at each root
  list(
    grid_df = data.frame(
      band_c = band_grid, simple_slope = ss, se = se,
      z = zv, p = pv, significant = pv < 0.05
    ),
    roots = sort(roots),
    b2 = b2, b3 = b3, v2 = v2, v3 = v3, cov23 = c23,
    grid_range = range(band_grid),
    n_sig = sum(pv < 0.05, na.rm = TRUE),
    n_grid = length(band_grid)
  )
}

run_panel <- function(measure, scope, unit, band, out_dir) {
  contrast <- ANALYSIS_CONTRAST[[ANALYSIS]]
  d <- panel_data(measure, scope, unit, band, contrast)
  n_pat <- length(unique(d$Patient))
  if (nrow(d) < MIN_TRIALS || n_pat < MIN_PATIENTS) {
    cat(sprintf("  SKIP %s/%s/%s/%s (n=%d trials, %d patients)\n",
                measure, scope, unit, band, nrow(d), n_pat))
    return(list())
  }

  qs <- quantile(d$band_c, probs = c(0.02, 0.98), na.rm = TRUE)
  band_grid <- seq(qs[1], qs[2], length.out = 200)

  results <- list()
  for (mv in c("ri", "rs")) {
    formula <- if (mv == "ri") INT_RI else INT_RS
    m <- fit_or_centered(formula, d)
    if (is.null(m)) {
      cat(sprintf("  %s fit FAILED for %s/%s/%s/%s\n",
                  toupper(mv), measure, scope, unit, band))
      next
    }
    if (isTRUE(attr(m, "centered"))) {
      # Shift the grid to match the centered model's predictor scale.
      m_band_c <- d$band_c - mean(d$band_c, na.rm = TRUE)
      band_grid_use <- seq(quantile(m_band_c, 0.02), quantile(m_band_c, 0.98),
                           length.out = 200)
    } else {
      band_grid_use <- band_grid
    }
    jn <- jn_for_model(m, band_grid_use)

    grid_path <- file.path(out_dir,
      sprintf("jn_%s_%s_%s_%s.csv", measure, unit, band, mv))
    write.csv(jn$grid_df, grid_path, row.names = FALSE)

    results[[mv]] <- data.frame(
      measure = measure, scope = scope, unit = unit, band = band,
      model = mv,
      n_trials = nrow(d), n_patients = n_pat,
      centered = isTRUE(attr(m, "centered")),
      b2_estimate = unname(jn$b2), b3_estimate = unname(jn$b3),
      var_b2 = unname(jn$v2), var_b3 = unname(jn$v3),
      cov_b2_b3 = unname(jn$cov23),
      jn_root_lo = if (length(jn$roots) >= 1) jn$roots[1] else NA,
      jn_root_hi = if (length(jn$roots) >= 2) jn$roots[2] else NA,
      n_roots = length(jn$roots),
      n_sig_in_data_range = jn$n_sig,
      data_band_min = jn$grid_range[1], data_band_max = jn$grid_range[2],
      stringsAsFactors = FALSE
    )
  }
  results
}

main <- function() {
  out_dir <- posthoc_out_dir(ANALYSIS_DIR[[ANALYSIS]])
  trending_csv <- file.path(out_dir, "trending_panels.csv")
  if (!file.exists(trending_csv)) stop("Missing: ", trending_csv)
  trending <- read.csv(trending_csv, stringsAsFactors = FALSE)
  cat(sprintf("Running JN on %d trending interaction panels\n", nrow(trending)))

  all_summary <- list()
  for (i in seq_len(nrow(trending))) {
    r <- trending[i, ]
    cat(sprintf("[%d/%d] %s | %s | %s | %s\n",
                i, nrow(trending), r$measure, r$scope, r$unit, r$band))
    res <- run_panel(r$measure, r$scope, r$unit, r$band, out_dir)
    for (mv in names(res)) all_summary[[length(all_summary) + 1]] <- res[[mv]]
  }
  if (length(all_summary) > 0) {
    out <- do.call(rbind, all_summary)
    out_path <- file.path(out_dir, "jn_summary.csv")
    write.csv(out, out_path, row.names = FALSE)
    cat(sprintf("\nSaved: %s\n", out_path))
  }
}

main()
cat("\n===== JOHNSON-NEYMAN DONE =====\n")
