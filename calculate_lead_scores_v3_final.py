"""
Calculate Lead Scores with Option 3: Piecewise Linear Recency
"""
import pandas as pd
import numpy as np
from datetime import datetime

print("Loading lead data...")
df = pd.read_csv('separate_processing/leads_enriched.csv', low_memory=False)

# Calculate days since activity
df['created_date_parsed'] = pd.to_datetime(df['created_date'], errors='coerce')
df['last_status_change_parsed'] = pd.to_datetime(df['last_status_change_date'], errors='coerce')
df['most_recent_date'] = df[['created_date_parsed', 'last_status_change_parsed']].max(axis=1)
df['days_since_activity'] = (datetime.now() - df['most_recent_date']).dt.days

print(f"Total leads: {len(df):,}")
print(f"Converted: {df['is_converted'].sum():,}")

# ===== OPTION 3: PIECEWISE LINEAR RECENCY SCORING =====
def piecewise_recency_score(days):
    """Piecewise linear recency scoring"""
    if pd.isna(days):
        return 0
    elif days <= 7:
        return 100  # Super fresh
    elif days <= 30:
        # Linear decline from 100 to 85 (7 to 30 days)
        return 100 - (days - 7) * (15 / 23)
    elif days <= 90:
        # Linear decline from 85 to 70 (30 to 90 days)
        return 85 - (days - 30) * (15 / 60)
    elif days <= 180:
        # Linear decline from 70 to 55 (90 to 180 days)
        return 70 - (days - 90) * (15 / 90)
    elif days <= 365:
        # Flatten: decline from 55 to 35 (180 to 365 days)
        return 55 - (days - 180) * (20 / 185)
    else:
        # Completely flat after 365 days
        return 30

print("\n" + "=" * 80)
print("OPTION 3: PIECEWISE LINEAR RECENCY SCORING")
print("=" * 80)

# Calculate recency score
df['recency_score'] = df['days_since_activity'].apply(piecewise_recency_score)

print("\nRecency Scoring Logic:")
print("-" * 80)
print("  0-7 days:     100 points (Ultra fresh)")
print("  7-30 days:    100 → 85 points (Steep drop)")
print("  30-90 days:   85 → 70 points (Moderate drop)")
print("  90-180 days:  70 → 55 points (Gradual decline)")
print("  180-365 days: 55 → 35 points (Flattening)")
print("  365+ days:    30 points (FLAT - all old leads same)")

# Calculate other components (unchanged)
print("\nCalculating other components...")

# Journey progression
journey_steps = {
    'kyc_step': 100,
    'v2_bank_details_step': 10,
    'final_step': 7,
    'v2_business_details_step': 2,
    'v2_personal_details_step': 2,
}

df['journey_points'] = 0
for step, points in journey_steps.items():
    if step in df.columns:
        df['journey_points'] += df[step].fillna(0) * points

df['journey_progression_score'] = df['journey_points'].clip(0, 100)

# Category
category_scores = {
    'Retail & Consumer Goods': 100,
    'Healthcare & Wellness': 88,
    'Software & Information Technology': 53,
    'Hospitality, Travel & Leisure': 44,
    'Public & Professional Services': 37,
    'Education / Educational Services': 13
}
df['category_score'] = df['L1'].map(category_scores).fillna(50)

# Digital presence
digital_features = {
    'has_website': 100,
    'has_maps': 69,
    'has_ecommerce': 40,
    'has_social_media': 0
}

df['digital_presence_score'] = 0
for feature, points in digital_features.items():
    if feature in df.columns:
        df['digital_presence_score'] += df[feature].fillna(0) * points

# Normalize to 0-100
max_digital = sum(digital_features.values())
df['digital_presence_score'] = (df['digital_presence_score'] / max_digital * 100).round(2)

# Calculate final score with weights
weights = {
    'recency_score': 0.25,
    'journey_progression_score': 0.35,
    'category_score': 0.15,
    'digital_presence_score': 0.25
}

df['final_score'] = 0
for component, weight in weights.items():
    df['final_score'] += df[component] * weight

df['final_score'] = df['final_score'].round(2)

# Add tiers (renamed Archive to Very Cold)
df['quality_tier'] = pd.cut(
    df['final_score'],
    bins=[-1, 15, 25, 35, 50, 100],
    labels=['Very Cold', 'Cold', 'Luke-warm', 'Warm', 'Hot']
)

# ===== ANALYSIS =====
print("\n" + "=" * 80)
print("SCORE DISTRIBUTION ANALYSIS")
print("=" * 80)

print("\nRecency Score Distribution:")
print(df['recency_score'].describe())

print("\nFinal Score Distribution:")
print(df['final_score'].describe())

print("\nComponent Contributions (average):")
for component, weight in weights.items():
    contribution = (df[component] * weight).mean()
    print(f"  {component:<30}: {contribution:>6.2f} points (weight: {weight*100:2.0f}%)")

print("\n" + "=" * 80)
print("TIER DISTRIBUTION")
print("=" * 80)

tier_order = ['Hot', 'Warm', 'Luke-warm', 'Cold', 'Very Cold']
tier_analysis = df.groupby('quality_tier').agg({
    'is_converted': ['count', 'sum', 'mean']
}).round(4)

print(f"\n{'Tier':<12} {'Leads':>10} {'Converted':>10} {'Conv%':>8} {'vs Base':>8}")
print("-" * 80)

baseline = df['is_converted'].mean()
for tier in tier_order:
    if tier in tier_analysis.index:
        count = tier_analysis.loc[tier, ('is_converted', 'count')]
        conv = tier_analysis.loc[tier, ('is_converted', 'sum')]
        conv_rate = tier_analysis.loc[tier, ('is_converted', 'mean')] * 100
        lift = conv_rate / (baseline * 100)

        pct_of_total = count / len(df) * 100

        print(f"{tier:<12} {count:>10,.0f} {conv:>10,.0f} {conv_rate:>7.2f}% {lift:>7.2f}x "
              f"({pct_of_total:>5.1f}%)")

# Conversion correlation
corr = df['final_score'].corr(df['is_converted'])
print(f"\nCorrelation with conversion: {corr:.4f}")

# ===== RECENCY SCORE BY AGE BUCKET =====
print("\n" + "=" * 80)
print("RECENCY SCORE BY AGE BUCKET")
print("=" * 80)

age_buckets = [
    (0, 7, '0-7 days'),
    (7, 30, '7-30 days'),
    (30, 90, '30-90 days'),
    (90, 180, '90-180 days'),
    (180, 365, '180-365 days'),
    (365, 730, '365-730 days'),
    (730, 10000, '>730 days')
]

print(f"\n{'Age Range':<20} {'Count':>10} {'Avg Recency':>13} {'Conv%':>8} {'Leads %':>10}")
print("-" * 80)

for min_age, max_age, label in age_buckets:
    bucket = df[(df['days_since_activity'] >= min_age) &
                (df['days_since_activity'] < max_age)]
    if len(bucket) > 0:
        count = len(bucket)
        avg_recency = bucket['recency_score'].mean()
        conv_rate = bucket['is_converted'].mean() * 100
        pct = count / len(df) * 100
        print(f"{label:<20} {count:>10,} {avg_recency:>12.1f} {conv_rate:>7.2f}% {pct:>9.1f}%")

# ===== COMPARISON WITH OLD SCORES =====
print("\n" + "=" * 80)
print("COMPARISON: OLD vs NEW (OPTION 3)")
print("=" * 80)

# Load old scores
df_old = pd.read_csv('separate_processing/scored_leads_v2.csv')

# Merge on lead_id
comparison = df[['lead_id', 'final_score', 'quality_tier', 'recency_score']].merge(
    df_old[['lead_id', 'final_score', 'quality_tier', 'recency_score']],
    on='lead_id',
    suffixes=('_new', '_old')
)

print(f"\nScore Changes:")
print(f"  Average old score: {comparison['final_score_old'].mean():.2f}")
print(f"  Average new score: {comparison['final_score_new'].mean():.2f}")
print(f"  Change: {comparison['final_score_new'].mean() - comparison['final_score_old'].mean():+.2f}")

improved = comparison[comparison['final_score_new'] > comparison['final_score_old']]
print(f"\nLeads with improved scores: {len(improved):,} ({len(improved)/len(comparison)*100:.1f}%)")
print(f"  Average improvement: {(improved['final_score_new'] - improved['final_score_old']).mean():+.2f} points")

worsened = comparison[comparison['final_score_new'] < comparison['final_score_old']]
print(f"\nLeads with lower scores: {len(worsened):,} ({len(worsened)/len(comparison)*100:.1f}%)")
print(f"  Average decline: {(worsened['final_score_new'] - worsened['final_score_old']).mean():+.2f} points")

# Tier changes
print(f"\n\nTier Changes:")
print(f"{'From → To':<25} {'Count':>10}")
print("-" * 40)

tier_changes = comparison.groupby(['quality_tier_old', 'quality_tier_new']).size()
for (old_tier, new_tier), count in tier_changes.sort_values(ascending=False).head(10).items():
    if old_tier != new_tier:
        print(f"{old_tier} → {new_tier:<15} {count:>10,}")

# Correlation comparison
corr_old = df_old['final_score'].corr(df_old['is_converted'])
corr_new = df['final_score'].corr(df['is_converted'])

print(f"\n\nModel Performance:")
print(f"  Old correlation: {corr_old:.4f}")
print(f"  New correlation: {corr_new:.4f}")
print(f"  Change: {corr_new - corr_old:+.4f}")

# ===== SAVE FINAL OUTPUT FILES =====
print("\n" + "=" * 80)
print("SAVING FINAL OUTPUT FILES")
print("=" * 80)

output_cols = [
    'lead_id', 'email', 'mobile', 'company_name', 'L1', 'L2', 'L3',
    'lead_status', 'business_type','contacted_status', 'journey_stage', 'onboarding_version',
    'created_date', 'last_status_change_date', 'days_since_activity',
    'recency_score', 'journey_progression_score', 'category_score',
    'digital_presence_score', 'final_score', 'quality_tier',
    'is_converted', 'has_website', 'has_maps', 'has_ecommerce', 'has_social_media'
]
output_cols = [col for col in output_cols if col in df.columns]

# Save all scored leads
all_leads_file = 'separate_processing/scored_leads_v3_final.csv'
df[output_cols].to_csv(all_leads_file, index=False)
print(f"\n✓ Saved all scored leads to: {all_leads_file}")

# Save tier-specific files (unresponsive only)
unresponsive = df[df['lead_status'] == 'Unresponsive']

for tier in tier_order:
    tier_leads = unresponsive[unresponsive['quality_tier'] == tier].sort_values('final_score', ascending=False)
    tier_file = f'all_tier_leads_csv/tier_{tier.lower().replace(" ", "_")}_leads.csv'
    tier_leads[output_cols].to_csv(tier_file, index=False)
    print(f"✓ Saved {len(tier_leads):,} {tier} tier leads to: {tier_file}")

# Save combined hot + warm
high_quality = unresponsive[unresponsive['quality_tier'].isin(['Hot', 'Warm'])].sort_values('final_score', ascending=False)
high_quality_file = 'separate_processing/high_quality_unresponsive_leads_v3_final.csv'
high_quality[output_cols].to_csv(high_quality_file, index=False)
print(f"✓ Saved {len(high_quality):,} Hot+Warm leads to: {high_quality_file}")

# Save top 1000
top_1000 = unresponsive.sort_values('final_score', ascending=False).head(1000)
top_1000_file = 'separate_processing/top_1000_leads_v3_final.csv'
top_1000[output_cols].to_csv(top_1000_file, index=False)
print(f"✓ Saved top 1000 leads to: {top_1000_file}")

# ===== EXTRACT EXAMPLE LEADS FOR PRESENTATION =====
print("\n" + "=" * 80)
print("EXTRACTING EXAMPLE LEADS FOR EACH TIER")
print("=" * 80)

example_cols = ['lead_id', 'company_name', 'L1', 'journey_stage', 'days_since_activity',
                'recency_score', 'journey_progression_score', 'digital_presence_score',
                'category_score', 'final_score', 'has_website', 'has_maps',
                'has_ecommerce', 'is_converted']
example_cols = [col for col in example_cols if col in df.columns]

examples_dict = {}

for tier in tier_order:
    tier_leads = unresponsive[unresponsive['quality_tier'] == tier].sort_values('final_score', ascending=False)

    # Get 5 diverse examples
    examples = tier_leads.head(5)
    examples_dict[tier] = examples[example_cols]

    print(f"\n{tier} Tier - Top 5 Examples:")
    print("-" * 80)
    for idx, row in examples.iterrows():
        company = str(row.get('company_name', 'N/A'))[:40]
        category = str(row.get('L1', 'N/A'))[:25]
        stage = str(row.get('journey_stage', 'N/A'))[:20]
        score = row.get('final_score', 0)
        days = row.get('days_since_activity', 0)
        website = '✓' if row.get('has_website', 0) == 1 else '✗'
        maps = '✓' if row.get('has_maps', 0) == 1 else '✗'

        print(f"  {company:<40} | {category:<25} | Score: {score:>5.1f} | {days:>3.0f} days | Web:{website} Maps:{maps}")

# Save examples to CSV for presentation
examples_file = 'separate_processing/tier_examples_for_presentation.csv'
all_examples = pd.concat([df.assign(tier=tier) for tier, df in examples_dict.items()])
all_examples.to_csv(examples_file, index=False)
print(f"\n✓ Saved tier examples to: {examples_file}")

print("\n" + "=" * 80)
print("ALL FILES GENERATED SUCCESSFULLY!")
print("=" * 80)
