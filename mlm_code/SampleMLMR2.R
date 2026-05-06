source("mlm/mlmr_retrieval_common.R")

input_csv <- file.path("outputs", "csvs", "combined_retrieval_power_all_mlmr_input.csv")
output_dir <- file.path("outputs", "stats", "power")

run_mlmr_analysis(
  measure_name = "Power",
  input_csv = input_csv,
  output_dir = output_dir
)
