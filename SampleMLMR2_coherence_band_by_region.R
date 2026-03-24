source("mlmr_band_region_common.R")

input_csv <- file.path("outputs", "csvs", "combined_retrieval_coherence_all_mlmr_input.csv")
output_dir <- file.path("outputs", "stats", "coherence_band_by_region_bla")

run_band_region_mlmr_analysis(
  measure_name = "Coherence",
  input_csv = input_csv,
  output_dir = output_dir,
  region_filter = function(datmodel) {
    datmodel %>% filter(grepl("BLA", Region))
  }
)
