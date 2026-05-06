##############################################################################
# All-Subjects PAC with Z-scored Predictors - Encoding
#
# NO balanced-trials filter. Z-scores PAC predictors within each region pair.
##############################################################################

if (!require("lme4"))        install.packages("lme4",        repos = "https://cloud.r-project.org")
if (!require("lmerTest"))    install.packages("lmerTest",     repos = "https://cloud.r-project.org")
if (!require("broom.mixed")) install.packages("broom.mixed",  repos = "https://cloud.r-project.org")

library(lme4)
library(lmerTest)
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

out_dir <- file.path(script_dir, "outputs", "imbalanced_encoding_mlm")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

PAC_SLOW_GAMMA <- c(30, 50)
PAC_HFA <- c(70, 100)

save_coefs <- function(model, filename, logistic = TRUE) {
  td <- tidy(model, effects = "fixed", conf.int = TRUE, exponentiate = logistic)
  write.csv(td, file.path(out_dir, filename), row.names = FALSE)
  return(td)
}

##############################################################################
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("ENCODING PAC - ALL SUBJECTS - Z-SCORED PREDICTORS\n")
cat(paste(rep("=", 70), collapse=""), "\n")

pac_csv <- file.path(script_dir, "outputs", "csvs", "combined_encoding_pac_all_mlmr_input.csv")
pac <- read.csv(pac_csv, stringsAsFactors = FALSE, check.names = FALSE)
pac <- pac[pac$yes_or_no %in% c("yes", "no") & pac$trial_type != "new", ]

cat(sprintf("\n  All subjects: %d patients\n", length(unique(pac$Patient))))

freq_cols <- sort(grep("^diff_Freq_", names(pac), value = TRUE))
freqs <- as.numeric(sub("diff_Freq_", "", freq_cols))

sg_cols <- freq_cols[freqs >= PAC_SLOW_GAMMA[1] & freqs <= PAC_SLOW_GAMMA[2]]
hfa_cols <- freq_cols[freqs >= PAC_HFA[1] & freqs <= PAC_HFA[2]]

pac$slow_gamma_pac <- rowMeans(pac[, sg_cols, drop=FALSE], na.rm=TRUE)
pac$hfa_pac <- rowMeans(pac[, hfa_cols, drop=FALSE], na.rm=TRUE)

pac$Accuracy <- ifelse(pac$yes_or_no == "yes", 1, 0)
pac$StimCond <- factor(ifelse(pac$trial_type == "nostim", "nostim", "stim"),
                       levels = c("nostim", "stim"))

pac_regions <- sort(unique(pac$Region))
cat(sprintf("\n  PAC region pairs: %s\n", paste(pac_regions, collapse=", ")))

for (reg in pac_regions) {
  sub <- pac[pac$Region == reg, ]
  sub$Patient <- factor(sub$Patient)

  if (nrow(sub) < 30 || nlevels(sub$Patient) < 3) {
    cat(sprintf("\n  Skipping %s: %d trials, %d patients\n", reg, nrow(sub), nlevels(sub$Patient)))
    next
  }

  sg_sd <- sd(sub$slow_gamma_pac, na.rm = TRUE)
  hfa_sd <- sd(sub$hfa_pac, na.rm = TRUE)

  if (sg_sd < 1e-10 || hfa_sd < 1e-10) {
    cat(sprintf("\n  Skipping %s: near-zero SD in PAC\n", reg))
    next
  }

  sub$slow_gamma_pac_z <- as.numeric(scale(sub$slow_gamma_pac))
  sub$hfa_pac_z <- as.numeric(scale(sub$hfa_pac))

  cat(sprintf("\n--- PAC %s: %d trials, %d patients ---\n",
              reg, nrow(sub), nlevels(sub$Patient)))
  cat(sprintf("  Memory rate: %.1f%%, Stim rate: %.1f%%\n",
              mean(sub$Accuracy)*100, mean(sub$StimCond == "stim")*100))
  cat(sprintf("  SG PAC SD: %.6f, HFA PAC SD: %.6f\n", sg_sd, hfa_sd))

  cat(sprintf("  MODEL: PAC %s - (SG_z + HFA_z) x StimCond\n", reg))

  m <- tryCatch({
    glmer(Accuracy ~ (slow_gamma_pac_z + hfa_pac_z) * StimCond + (1 | Patient),
          data = sub, family = binomial, control = ctrl)
  }, error = function(e) { cat("    FAILED:", conditionMessage(e), "\n"); NULL })

  if (!is.null(m)) {
    td <- save_coefs(m, sprintf("pac_zscored_%s_coefs.csv", gsub("/", "_", reg)))
    for (term_pat in c("slow_gamma_pac_z:StimCond", "hfa_pac_z:StimCond")) {
      row <- td[grepl(term_pat, td$term, fixed=TRUE), ]
      if (nrow(row) > 0) {
        cat(sprintf("    %s: OR=%.4f [%.4f, %.4f], p=%.4f\n",
                    row$term[1], row$estimate[1], row$conf.low[1], row$conf.high[1],
                    row$p.value[1]))
      }
    }
    print(summary(m))
  }
}

cat("\n\n===== IMBALANCED ENCODING PAC Z-SCORED MLM COMPLETE =====\n")
