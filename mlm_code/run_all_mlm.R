## Master script: run all MLM analyses
## - Power & Coherence: region subsets (MTL cortical, HPC subfields), no PHG
## - PAC: encoding and retrieval
## Each analysis has both region-by-band and band-by-region variants

source("mlm/mlmr_retrieval_common.R")
source("mlm/mlmr_band_region_common.R")
source("mlm/mlmr_pac_common.R")

# ============================================================================
# Region filter functions (exclude PHG everywhere)
# ============================================================================
filter_mtl_cortical <- function(datmodel) {
  # BLA, HPC, EC, PRC (broader MTL cortical regions)
  allowed <- c("BLA", "HPC", "EC", "PRC")
  datmodel %>% filter(Region %in% allowed)
}

filter_hpc_subfields <- function(datmodel) {
  # BLA, CA, DG (hippocampal subfields + amygdala)
  allowed <- c("BLA", "CA", "DG")
  datmodel %>% filter(Region %in% allowed)
}

filter_no_phg <- function(datmodel) {
  datmodel %>% filter(Region != "PHG")
}

filter_coherence_mtl_cortical <- function(datmodel) {
  # Keep pairs where BOTH regions are in {BLA, HPC, EC, PRC}
  allowed <- c("BLA", "HPC", "EC", "PRC")
  datmodel %>% filter({
    parts <- strsplit(Region, "_", fixed = TRUE)
    sapply(parts, function(p) all(p %in% allowed))
  })
}

filter_coherence_hpc_subfields <- function(datmodel) {
  # Keep pairs where BOTH regions are in {BLA, CA, DG}
  allowed <- c("BLA", "CA", "DG")
  datmodel %>% filter({
    parts <- strsplit(Region, "_", fixed = TRUE)
    sapply(parts, function(p) all(p %in% allowed))
  })
}

# ============================================================================
# 1. POWER analyses (encoding + retrieval)
# ============================================================================
cat("\n========== ENCODING POWER ==========\n")

enc_power_csv <- file.path("outputs", "csvs", "combined_encoding_power_all_mlmr_input.csv")

# MTL cortical (BLA, HPC, EC, PRC) - region_by_band
message("Encoding Power - MTL cortical - region_by_band")
run_mlmr_analysis(
  measure_name = "Power", input_csv = enc_power_csv,
  output_dir = file.path("outputs", "stats", "enc_power_mtl_rbband"),
  phase_name = "Encoding",
  region_filter = function(d) filter_no_phg(filter_mtl_cortical(d))
)

# MTL cortical - band_by_region
message("Encoding Power - MTL cortical - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Power", input_csv = enc_power_csv,
  output_dir = file.path("outputs", "stats", "enc_power_mtl_bbreg"),
  phase_name = "Encoding",
  region_filter = function(d) filter_no_phg(filter_mtl_cortical(d))
)

# HPC subfields (BLA, CA, DG) - region_by_band
message("Encoding Power - HPC subfields - region_by_band")
run_mlmr_analysis(
  measure_name = "Power", input_csv = enc_power_csv,
  output_dir = file.path("outputs", "stats", "enc_power_hpc_rbband"),
  phase_name = "Encoding",
  region_filter = function(d) filter_no_phg(filter_hpc_subfields(d))
)

# HPC subfields - band_by_region
message("Encoding Power - HPC subfields - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Power", input_csv = enc_power_csv,
  output_dir = file.path("outputs", "stats", "enc_power_hpc_bbreg"),
  phase_name = "Encoding",
  region_filter = function(d) filter_no_phg(filter_hpc_subfields(d))
)

cat("\n========== RETRIEVAL POWER ==========\n")

ret_power_csv <- file.path("outputs", "csvs", "combined_retrieval_power_all_mlmr_input.csv")

# MTL cortical - region_by_band
message("Retrieval Power - MTL cortical - region_by_band")
run_mlmr_analysis(
  measure_name = "Power", input_csv = ret_power_csv,
  output_dir = file.path("outputs", "stats", "ret_power_mtl_rbband"),
  region_filter = function(d) filter_no_phg(filter_mtl_cortical(d))
)

# MTL cortical - band_by_region
message("Retrieval Power - MTL cortical - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Power", input_csv = ret_power_csv,
  output_dir = file.path("outputs", "stats", "ret_power_mtl_bbreg"),
  region_filter = function(d) filter_no_phg(filter_mtl_cortical(d))
)

# HPC subfields - region_by_band
message("Retrieval Power - HPC subfields - region_by_band")
run_mlmr_analysis(
  measure_name = "Power", input_csv = ret_power_csv,
  output_dir = file.path("outputs", "stats", "ret_power_hpc_rbband"),
  region_filter = function(d) filter_no_phg(filter_hpc_subfields(d))
)

# HPC subfields - band_by_region
message("Retrieval Power - HPC subfields - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Power", input_csv = ret_power_csv,
  output_dir = file.path("outputs", "stats", "ret_power_hpc_bbreg"),
  region_filter = function(d) filter_no_phg(filter_hpc_subfields(d))
)

# ============================================================================
# 2. COHERENCE analyses (encoding + retrieval)
# ============================================================================
cat("\n========== ENCODING COHERENCE ==========\n")

enc_coh_csv <- file.path("outputs", "csvs", "combined_encoding_coherence_all_mlmr_input.csv")

# MTL cortical - region_by_band
message("Encoding Coherence - MTL cortical - region_by_band")
run_mlmr_analysis(
  measure_name = "Coherence", input_csv = enc_coh_csv,
  output_dir = file.path("outputs", "stats", "enc_coh_mtl_rbband"),
  phase_name = "Encoding",
  region_filter = filter_coherence_mtl_cortical
)

# MTL cortical - band_by_region
message("Encoding Coherence - MTL cortical - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Coherence", input_csv = enc_coh_csv,
  output_dir = file.path("outputs", "stats", "enc_coh_mtl_bbreg"),
  phase_name = "Encoding",
  region_filter = filter_coherence_mtl_cortical
)

# HPC subfields - region_by_band
message("Encoding Coherence - HPC subfields - region_by_band")
run_mlmr_analysis(
  measure_name = "Coherence", input_csv = enc_coh_csv,
  output_dir = file.path("outputs", "stats", "enc_coh_hpc_rbband"),
  phase_name = "Encoding",
  region_filter = filter_coherence_hpc_subfields
)

# HPC subfields - band_by_region
message("Encoding Coherence - HPC subfields - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Coherence", input_csv = enc_coh_csv,
  output_dir = file.path("outputs", "stats", "enc_coh_hpc_bbreg"),
  phase_name = "Encoding",
  region_filter = filter_coherence_hpc_subfields
)

cat("\n========== RETRIEVAL COHERENCE ==========\n")

ret_coh_csv <- file.path("outputs", "csvs", "combined_retrieval_coherence_all_mlmr_input.csv")

# MTL cortical - region_by_band
message("Retrieval Coherence - MTL cortical - region_by_band")
run_mlmr_analysis(
  measure_name = "Coherence", input_csv = ret_coh_csv,
  output_dir = file.path("outputs", "stats", "ret_coh_mtl_rbband"),
  region_filter = filter_coherence_mtl_cortical
)

# MTL cortical - band_by_region
message("Retrieval Coherence - MTL cortical - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Coherence", input_csv = ret_coh_csv,
  output_dir = file.path("outputs", "stats", "ret_coh_mtl_bbreg"),
  region_filter = filter_coherence_mtl_cortical
)

# HPC subfields - region_by_band
message("Retrieval Coherence - HPC subfields - region_by_band")
run_mlmr_analysis(
  measure_name = "Coherence", input_csv = ret_coh_csv,
  output_dir = file.path("outputs", "stats", "ret_coh_hpc_rbband"),
  region_filter = filter_coherence_hpc_subfields
)

# HPC subfields - band_by_region
message("Retrieval Coherence - HPC subfields - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Coherence", input_csv = ret_coh_csv,
  output_dir = file.path("outputs", "stats", "ret_coh_hpc_bbreg"),
  region_filter = filter_coherence_hpc_subfields
)

# ============================================================================
# 3. PAC analyses (encoding + retrieval)
# ============================================================================
cat("\n========== ENCODING PAC ==========\n")

enc_pac_csv <- file.path("outputs", "csvs", "combined_encoding_pac_all_mlmr_input.csv")

# Region-by-band (one model per region pair, PAC bands as predictors)
message("Encoding PAC - region_by_band")
run_pac_region_by_band(
  input_csv = enc_pac_csv,
  output_dir = file.path("outputs", "stats", "enc_pac_rbband"),
  phase = "encoding"
)

# Band-by-region (one model per band, region as predictor)
message("Encoding PAC - band_by_region")
run_pac_band_by_region(
  input_csv = enc_pac_csv,
  output_dir = file.path("outputs", "stats", "enc_pac_bbreg"),
  phase = "encoding"
)

cat("\n========== RETRIEVAL PAC ==========\n")

ret_pac_csv <- file.path("outputs", "csvs", "combined_retrieval_pac_all_mlmr_input.csv")

# Region-by-band
message("Retrieval PAC - region_by_band")
run_pac_region_by_band(
  input_csv = ret_pac_csv,
  output_dir = file.path("outputs", "stats", "ret_pac_rbband"),
  phase = "retrieval"
)

# Band-by-region
message("Retrieval PAC - band_by_region")
run_pac_band_by_region(
  input_csv = ret_pac_csv,
  output_dir = file.path("outputs", "stats", "ret_pac_bbreg"),
  phase = "retrieval"
)

cat("\n========== ALL ANALYSES COMPLETE ==========\n")
