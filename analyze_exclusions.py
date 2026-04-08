import pandas as pd

# Load the COMPLETE CSV files before any balanced filtering
power = pd.read_csv('outputs/csvs/combined_encoding_power_blaes_mlmr_input.csv')
coherence = pd.read_csv('outputs/csvs/combined_encoding_coherence_blaes_mlmr_input.csv')

# Filter to nostim
power_nostim = power[power['trial_type'] == 'nostim']
coherence_nostim = coherence[coherence['trial_type'] == 'nostim']

# Check if BJH040 appears in coherence at all (before any filtering)
print("BJH040 in Power Encoding - Full CSV:")
bjh040_power = power[power['Patient'] == 'BJH040']
print(f"  Regions: {sorted(bjh040_power['Region'].unique())}")

print("\nBJH040 in Coherence Encoding - Full CSV:")
bjh040_coh = coherence[coherence['Patient'] == 'BJH040']
print(f"  Regions: {sorted(bjh040_coh['Region'].unique())}")

# Check what regions are present in coherence CSV at all
print("\nAll BLAES regions in Coherence CSV:")
print(f"  {sorted(coherence['Region'].unique())}")

# Check the raw source to understand this
print("\n\nAnalyzing: Does BJH040 have NO pair-based regions?")
# If BJH040 only has CA in Power, he'd only have pair regions like BLA_CA, CA_*, etc. in Coherence
# But if those pairs require BOTH regions to exist for that subject, he might have none

# Let's check which subjects have which single regions
print("\nBLAES Power - Subjects by region:")
for region in ['BLA', 'CA', 'DG', 'EC', 'HPC', 'PRC']:
    subjs = set(power_nostim[power_nostim['Region'] == region]['Patient'].unique())
    if region in ['BLA', 'CA']:  # Focus on the regions
        print(f"\n{region}: {len(subjs)} subjects")
        # For these subjects, check if they appear in coherence
        in_coh = []
        not_in_coh = []
        for s in subjs:
            if s in coherence_nostim['Patient'].unique():
                in_coh.append(s)
            else:
                not_in_coh.append(s)
        print(f"  In coherence: {in_coh}")
        if not_in_coh:
            print(f"  NOT in coherence: {not_in_coh}")

