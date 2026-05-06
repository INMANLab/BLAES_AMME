source("mlm/mlmr_retrieval_common.R")

input_csv <- file.path("outputs", "csvs", "combined_encoding_coherence_all_mlmr_input.csv")
output_dir <- file.path("outputs", "stats", "encoding_coherence")

run_mlmr_analysis(
  measure_name = "Coherence",
  input_csv = input_csv,
  output_dir = output_dir,
  phase_name = "Encoding"
)
