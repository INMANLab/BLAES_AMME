import pandas as pd

# The summary says BJH017 has 110 remembered trials
# But I'm counting only 40 (22 + 18) per region in power CSV
# Let me check all columns

power = pd.read_csv('outputs/csvs/combined_encoding_power_blaes_mlmr_input.csv')
coherence = pd.read_csv('outputs/csvs/combined_encoding_coherence_blaes_mlmr_input.csv')

print("Checking what's being counted in balanced_memory_trials.py...")
print("\nThe CSV files have ALREADY been exported and FILTERED")
print("The counts in the summary come from BEFORE balanced filtering\n")

# Let me trace through the get_nostim_trial_counts function logic
power_nostim = power[power['trial_type'] == 'nostim'].copy()

# Replicate get_nostim_trial_counts logic
df = power_nostim
print("Step 1: Filter to nostim:")
print(f"  Total rows: {len(df)}")

# Line 66-67 of balanced_memory_trials.py
first_region = df.groupby('Patient')['Region'].first().to_dict()
df_filtered = df[df.apply(lambda r: r['Region'] == first_region[r['Patient']], axis=1)]
print(f"\nStep 2: After deduplication to first region:")
print(f"  Total rows: {len(df_filtered)}")

# Count per subject
print(f"\nStep 3: Count by subject (BJH017 only):")
bjh017_data = df_filtered[df_filtered['Patient'] == 'BJH017']
n_rem = (bjh017_data['yes_or_no'] == 'yes').sum()
n_forg = (bjh017_data['yes_or_no'] == 'no').sum()
print(f"  BJH017: rem={n_rem}, forg={n_forg}")
print(f"  Expected from summary: rem=110, forg=90")
print(f"  MISMATCH!")

print("\nThis suggests the summary was generated from UNFILTERED data")
print("before the balanced_memory_trials.py modifications.")
print("\nLet me check the summary generation timing...")

