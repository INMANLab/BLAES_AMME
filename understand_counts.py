import pandas as pd

# The get_nostim_trial_counts function:
# 1. Takes data['mlmr_export_frames']
# 2. Concatenates them
# 3. Filters to nostim
# 4. Deduplicates to first region per patient
# 5. Counts yes/no per patient

# So the counts in the summary represent:
# "How many trials does this subject have, after deduplicating by region?"

# For power, all subjects have multiple regions, so:
# - If a trial appears in 5 regions, it gets counted as 1 after dedup
# - get_nostim_trial_counts selects first region and counts

# But I'm seeing counts that DON'T match...
# Let me check if maybe the data passed is filtered DIFFERENTLY

# Actually, re-reading the code:
# Line 278: counts = get_nostim_trial_counts(data, measure='power')
# This is BEFORE filtering to balanced subjects
# So it counts ALL subjects in the data

# Let me count what we actually have in the csv:
power = pd.read_csv('outputs/csvs/combined_encoding_power_blaes_mlmr_input.csv')
blaes_subjects = ['BJH017', 'BJH024', 'BJH025', 'BJH026', 'BJH027', 'BJH029', 
                  'BJH030', 'BJH032', 'BJH035', 'BJH040', 'BJH041', 'BJH042',
                  'BJH046', 'BJH049', 'SLCH006', 'SLCH018', 'UIC202215',
                  'UIC202302', 'UIC202306', 'UIC202308', 'UIC202311',
                  'UIC202313', 'UIC202401']

# Filter to BLAES subjects and nostim
power_blaes = power[power['Patient'].isin(blaes_subjects) & (power['trial_type'] == 'nostim')]

print("Counting trials in CSV (all subjects):")
# Group by patient and count
counts_dict = {}
for pat in sorted(power_blaes['Patient'].unique()):
    pat_data = power_blaes[power_blaes['Patient'] == pat]
    n_rem = (pat_data['yes_or_no'] == 'yes').sum()
    n_forg = (pat_data['yes_or_no'] == 'no').sum()
    counts_dict[pat] = {'remembered': int(n_rem), 'forgotten': int(n_forg)}
    print(f"{pat:12}: rem={n_rem:3}, forg={n_forg:3}")

print("\nNow applying the deduplication logic from get_nostim_trial_counts:")
# Deduplicate to first region
first_region = power_blaes.groupby('Patient')['Region'].first().to_dict()
power_dedup = power_blaes[power_blaes.apply(lambda r: r['Region'] == first_region[r['Patient']], axis=1)]

print("\nAfter deduplication to first region:")
counts_dedup = {}
for pat in sorted(power_dedup['Patient'].unique()):
    pat_data = power_dedup[power_dedup['Patient'] == pat]
    n_rem = (pat_data['yes_or_no'] == 'yes').sum()
    n_forg = (pat_data['yes_or_no'] == 'no').sum()
    counts_dedup[pat] = {'remembered': int(n_rem), 'forgotten': int(n_forg)}
    print(f"{pat:12}: rem={n_rem:3}, forg={n_forg:3}")

print("\n\nSummary file says BJH017: rem=110, forg=90")
print(f"CSV + dedup gives BJH017: rem={counts_dedup['BJH017']['remembered']}, forg={counts_dedup['BJH017']['forgotten']}")
print("\nThey don't match!")
print("This means the summary is NOT directly from the CSV files,")
print("or the counts are from mlmr_export_frames BEFORE export to CSV")

