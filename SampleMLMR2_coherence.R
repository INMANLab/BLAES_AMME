source("mlmr_retrieval_common.R")

input_csv <- file.path("outputs", "csvs", "combined_retrieval_coherence_all_mlmr_input.csv")
output_dir <- file.path("outputs", "stats", "coherence")

run_mlmr_analysis(
  measure_name = "Coherence",
  input_csv = input_csv,
  output_dir = output_dir
)
