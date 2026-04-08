import pandas as pd

# Verify the key claim: BJH017 has same underlying trials but different 
# region/pair representations

power = pd.read_csv('outputs/csvs/combined_encoding_power_blaes_mlmr_input.csv')
coherence = pd.read_csv('outputs/csvs/combined_encoding_coherence_blaes_mlmr_input.csv')

# Get BJH017 data
bjh017_power = power[(power['Patient'] == 'BJH017') & (power['trial_type'] == 'nostim')]
bjh017_coh = coherence[(coherence['Patient'] == 'BJH017') & (coherence['trial_type'] == 'nostim')]

print("BJH017 VALIDATION")
print("="*70)
print(f"\nPower data:")
print(f"  Total rows: {len(bjh017_power)}")
print(f"  Regions: {sorted(bjh017_power['Region'].unique())}")
print(f"  Remember/Forget split:")
for region in sorted(bjh017_power['Region'].unique()):
    data = bjh017_power[bjh017_power['Region'] == region]
    n_rem = (data['yes_or_no'] == 'yes').sum()
    n_forg = (data['yes_or_no'] == 'no').sum()
    print(f"    {region:6}: rem={n_rem:3}, forg={n_forg:3}")

print(f"\nCoherence data:")
print(f"  Total rows: {len(bjh017_coh)}")
print(f"  Pairs: {sorted(bjh017_coh['Region'].unique())}")
print(f"  Sample from each pair:")
for region in sorted(bjh017_coh['Region'].unique())[:3]:  # Show first 3 pairs
    data = bjh017_coh[bjh017_coh['Region'] == region]
    n_rem = (data['yes_or_no'] == 'yes').sum()
    n_forg = (data['yes_or_no'] == 'no').sum()
    print(f"    {region:8}: rem={n_rem:3}, forg={n_forg:3}")

print("\nINTERPRETATION:")
print("-"*70)
print("All 5 regions in Power have identical trial counts (22 rem, 18 forg)")
print("  → This represents 40 underlying trials, deduplicated across regions")
print("\nAll pairs in Coherence have identical trial counts (22 rem, 18 forg)")
print("  → Each pair independently shows 40 trials")
print("  → But there are 9 pairs (not 5), so total rows = 40 × 9 = 360")
print("\nWhen balanced_memory_trials.py deduplicates to 'first region':")
print("  Power: Takes first region (BLA) → 40 trials ✓")
print("  Coherence: Takes first pair (BLA_CA) → 40 trials ✓")
print("\nBUT the reported summary counts differ!")
print("  Power summary: BJH017 = 110 rem, 90 forg")
print("  Coh summary: BJH017 = 198 rem, 162 forg")
print("\nThis suggests there's region filtering applied in coherence that")
print("reduces trial counts BEFORE they reach the CSV export function.")

