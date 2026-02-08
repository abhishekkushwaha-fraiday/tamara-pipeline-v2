"""
Generate dashboard charts for each tier's lead CSV.
Usage: python3 generate_tier_charts.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import gaussian_kde
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.dirname(script_dir)

# --- Color palette ---
C = {
    'blue':    '#2563EB',
    'sky':     '#0EA5E9',
    'purple':  '#8B5CF6',
    'amber':   '#F59E0B',
    'green':   '#10B981',
    'red':     '#EF4444',
    'grey':    '#94A3B8',
    'light':   '#E2E8F0',
    'yes':     '#2563EB',
    'no':      '#CBD5E1',
    'teal':    '#14B8A6',
    'rose':    '#F43F5E',
}
PALETTE = [C['blue'], C['sky'], C['purple'], C['amber'], C['green'], C['red'], C['teal'], C['rose']]


def add_hbar_labels(ax, bars, total, fontsize=10):
    """Add 'count (pct%)' label to the right of each horizontal bar."""
    max_w = max(b.get_width() for b in bars) if bars else 1
    for bar in bars:
        count = int(bar.get_width())
        pct = count / total * 100 if total else 0
        ax.text(
            bar.get_width() + max_w * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f'{count:,}  ({pct:.1f}%)',
            ha='left', va='center', fontsize=fontsize, fontweight='bold',
        )


def generate_tier_chart(csv_path, output_path, tier_name):
    """Generate a dashboard chart for any tier CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    total = len(df)

    fig = plt.figure(figsize=(24, 36), facecolor='white')
    fig.suptitle(f'{tier_name}  \u2014  {total:,} leads',
                 fontsize=24, fontweight='bold', y=0.995)

    gs = gridspec.GridSpec(7, 4, figure=fig, hspace=0.42, wspace=0.35,
                           height_ratios=[1.0, 0.9, 0.8, 1.0, 1.0, 0.9, 1.1])

    # =================================================================
    # ROW 1 LEFT: Journey Stage  (horizontal bar, Converted removed)
    # =================================================================
    ax = fig.add_subplot(gs[0, :3])

    stage_map = {
        'KYB Submitted':   'KYB Submitted',
        'KYB In Progress': 'KYB In Progress',
        'Registered':      'Registered',
        'Onboarding: Kyc': 'KYC',
    }
    counts = {}
    for raw, label in stage_map.items():
        counts[label] = int((df['journey_stage'] == raw).sum())
    mapped_total = sum(counts.values())
    other_count = total - mapped_total
    # Include unmapped stages (except Converted) in Other
    converted_count = int((df['journey_stage'] == 'Converted').sum())
    other_count -= converted_count  # exclude Converted from total display
    if other_count > 0:
        counts['Other'] = other_count

    counts = {k: v for k, v in counts.items() if v > 0}
    display_total = total - converted_count  # denominator excludes Converted

    sorted_items = sorted(counts.items(), key=lambda x: x[1])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = PALETTE[:len(labels)]

    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.6)
    add_hbar_labels(ax, bars, display_total)
    ax.set_title('Journey Stage Breakdown', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(0, max(values) * 1.20)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)

    # =================================================================
    # ROW 1 RIGHT: Freelancer vs Registered  (donut chart)
    # =================================================================
    ax = fig.add_subplot(gs[0, 3])

    btype = df['business_type'].fillna('Unknown').str.strip().str.lower()
    btype_counts = btype.value_counts()
    bt_labels = [v.title() for v in btype_counts.index]
    bt_values = btype_counts.values
    bt_colors = [C['blue'], C['amber'], C['grey'], C['sky']]

    wedges, _ = ax.pie(
        bt_values,
        colors=bt_colors[:len(bt_values)],
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
    )
    for i, (wedge, count) in enumerate(zip(wedges, bt_values)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        pct = count / total * 100
        label_text = f'{bt_labels[i]}\n{count:,} ({pct:.1f}%)'
        ha = 'left' if x >= 0 else 'right'
        ax.annotate(
            label_text,
            xy=(0.78 * x, 0.78 * y),
            xytext=(1.30 * x, 1.15 * y),
            ha=ha, va='center', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=C['grey'], lw=0.8),
        )
    ax.set_title('Business Type', fontsize=14, fontweight='bold', pad=12)

    # =================================================================
    # ROW 2: Contacted Status  (clean binary donut + summary)
    # =================================================================
    ax = fig.add_subplot(gs[1, :])

    # Blank/NaN = Not Contacted, anything written = Contacted
    raw_status = df['contacted_status'].fillna('').str.strip()
    n_contacted = int((raw_status != '').sum())
    n_not_contacted = total - n_contacted
    pct_c = n_contacted / total * 100
    pct_nc = n_not_contacted / total * 100

    # Draw a horizontal split bar (full width, single stacked bar)
    bar_y = [0]
    ax.barh(bar_y, [n_contacted], height=0.5, color=C['green'], edgecolor='white', label='Contacted')
    ax.barh(bar_y, [n_not_contacted], height=0.5, left=[n_contacted], color=C['red'], edgecolor='white', label='Not Contacted')

    # Labels inside the bars
    if n_contacted > total * 0.08:
        ax.text(n_contacted / 2, 0, f'Contacted\n{n_contacted:,}  ({pct_c:.1f}%)',
                ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    if n_not_contacted > total * 0.08:
        ax.text(n_contacted + n_not_contacted / 2, 0, f'Not Contacted\n{n_not_contacted:,}  ({pct_nc:.1f}%)',
                ha='center', va='center', fontsize=13, fontweight='bold', color='white')

    ax.set_title('Contacted vs Not Contacted', fontsize=14, fontweight='bold', pad=12)
    ax.set_yticks([])
    ax.set_xlabel('Count')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # =================================================================
    # ROW 3: Digital Presence  (stacked horizontal bar)
    # =================================================================
    ax = fig.add_subplot(gs[2, :])

    digital_features = [
        ('has_website',      'Website'),
        ('has_maps',         'Google Maps'),
        ('has_ecommerce',    'E-commerce'),
        ('has_social_media', 'Social Media'),
    ]

    feature_labels = [lbl for _, lbl in digital_features]
    yes_vals = [int(df[col].sum()) for col, _ in digital_features]
    no_vals = [total - y for y in yes_vals]

    y_pos = np.arange(len(feature_labels))
    bar_h = 0.45

    ax.barh(y_pos, yes_vals, height=bar_h, color=C['yes'],
            edgecolor='white', label='Yes')
    ax.barh(y_pos, no_vals, height=bar_h, left=yes_vals,
            color=C['no'], edgecolor='white', label='No')

    for i in range(len(feature_labels)):
        yes_pct = yes_vals[i] / total * 100
        if yes_vals[i] > total * 0.08:
            ax.text(yes_vals[i] / 2, y_pos[i],
                    f'{yes_vals[i]:,}  ({yes_pct:.1f}%)',
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color='white')
        no_pct = no_vals[i] / total * 100
        if no_vals[i] > total * 0.08:
            ax.text(yes_vals[i] + no_vals[i] / 2, y_pos[i],
                    f'{no_vals[i]:,}  ({no_pct:.1f}%)',
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color='#475569')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=12)
    ax.set_title('Digital Presence', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Count')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis()

    # =================================================================
    # ROW 4: Leads Created by Month  (bar chart timeline)
    # =================================================================
    ax = fig.add_subplot(gs[3, :])

    df['created_date_parsed'] = pd.to_datetime(df['created_date'], errors='coerce')
    df['created_month'] = df['created_date_parsed'].dt.to_period('M')
    month_counts = df.dropna(subset=['created_month']).groupby('created_month').size()
    month_counts = month_counts.sort_index()

    month_labels = [str(p) for p in month_counts.index]
    month_values = month_counts.values

    bar_colors = [C['blue'] if v >= np.percentile(month_values, 75) else
                  C['sky'] if v >= np.percentile(month_values, 50) else
                  C['light'] for v in month_values]

    bars = ax.bar(range(len(month_labels)), month_values, color=bar_colors,
                  edgecolor='white', width=0.7)

    # Add count labels on top of bars
    for bar, val in zip(bars, month_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(month_values) * 0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
    ax.set_title('Leads Created by Month', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Count')
    ax.spines[['top', 'right']].set_visible(False)

    # =================================================================
    # ROW 5: Lead Age  (bucketed bar chart + stats panel)
    # =================================================================
    ax = fig.add_subplot(gs[4, :3])

    ages = df['days_since_activity'].dropna()
    mean_age = ages.mean()
    median_age = ages.median()

    # Clear bucketed ranges
    age_buckets = [
        ('0-30',    0,   30),
        ('31-60',   31,  60),
        ('61-90',   61,  90),
        ('91-120',  91,  120),
        ('121-180', 121, 180),
        ('181-270', 181, 270),
        ('271-365', 271, 365),
        ('365+',    365, float('inf')),
    ]
    bucket_labels = [b[0] for b in age_buckets]
    bucket_vals = []
    for _, lo, hi in age_buckets:
        if hi == float('inf'):
            bucket_vals.append(int((ages >= lo).sum()))
        else:
            bucket_vals.append(int(((ages >= lo) & (ages <= hi)).sum()))

    bucket_colors = [C['green'] if v == max(bucket_vals) else
                     C['blue'] if v >= np.percentile(bucket_vals, 50) else
                     C['sky'] for v in bucket_vals]

    bars = ax.bar(range(len(bucket_labels)), bucket_vals, color=bucket_colors,
                  edgecolor='white', width=0.7)
    for bar, val in zip(bars, bucket_vals):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bucket_vals) * 0.015,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(bucket_labels)))
    ax.set_xticklabels(bucket_labels, fontsize=10)
    ax.set_xlabel('Days Since Last Activity', fontsize=11)
    ax.set_ylabel('Count')
    ax.set_title('Lead Age Distribution', fontsize=14, fontweight='bold', pad=12)
    ax.spines[['top', 'right']].set_visible(False)

    # Stats panel
    ax_stats = fig.add_subplot(gs[4, 3])
    ax_stats.axis('off')
    stats_text = (
        f"Lead Age Stats\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"Mean:      {mean_age:,.0f} days\n"
        f"Median:    {median_age:,.0f} days\n"
        f"Min:       {ages.min():,.0f} days\n"
        f"Max:       {ages.max():,.0f} days\n"
        f"Std Dev:   {ages.std():,.0f} days\n"
        f"Total:     {len(ages):,} leads"
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=12, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor=C['light'], edgecolor=C['grey'], alpha=0.8))

    # =================================================================
    # ROW 6: Lead Score  (box plot with stats + frequency bar chart)
    # =================================================================
    scores = df['final_score'].dropna()
    q1 = scores.quantile(0.25)
    q2 = scores.median()
    q3 = scores.quantile(0.75)
    score_mean = scores.mean()
    score_min = scores.min()
    score_max = scores.max()

    # Box plot (left ~55%) with clear stats table below
    ax_box = fig.add_subplot(gs[5, :2])
    ax_box.boxplot(scores, vert=False, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=C['blue'], alpha=0.7, edgecolor=C['blue']),
                   medianprops=dict(color=C['red'], linewidth=3),
                   whiskerprops=dict(color=C['blue'], linewidth=1.5),
                   capprops=dict(color=C['blue'], linewidth=2),
                   flierprops=dict(marker='o', markersize=4, markerfacecolor=C['grey'], alpha=0.5))

    # Place labels — alternate above and below the box to avoid overlap
    # Above: Min, Median, Max   |   Below: Q1, Mean, Q3
    above = [
        (score_min, f'Min: {score_min:.1f}', C['grey']),
        (q2,        f'Median: {q2:.1f}',     C['red']),
        (score_max, f'Max: {score_max:.1f}',  C['grey']),
    ]
    below = [
        (q1,         f'Q1: {q1:.1f}',          C['sky']),
        (score_mean, f'Mean: {score_mean:.1f}', C['amber']),
        (q3,         f'Q3: {q3:.1f}',           C['purple']),
    ]
    for val, label, color in above:
        ax_box.plot(val, 1.0, marker='v', markersize=8, color=color, zorder=5)
        ax_box.text(val, 1.32, label, ha='center', va='bottom', fontsize=12,
                    fontweight='bold', color=color)
    for val, label, color in below:
        ax_box.plot(val, 1.0, marker='^', markersize=8, color=color, zorder=5)
        ax_box.text(val, 0.52, label, ha='center', va='top', fontsize=12,
                    fontweight='bold', color=color)

    ax_box.set_ylim(0.2, 1.9)
    ax_box.set_xlabel('Final Score', fontsize=12)
    ax_box.set_title('Lead Score Box Plot', fontsize=14, fontweight='bold', pad=30)
    ax_box.set_yticks([])
    ax_box.spines[['top', 'right', 'left']].set_visible(False)
    ax_box.tick_params(axis='x', labelsize=11)

    # Frequency curve (KDE) with histogram underlay
    ax_freq = fig.add_subplot(gs[5, 2:])

    # Light histogram in background for context
    counts_hist, bin_edges, _ = ax_freq.hist(
        scores, bins=20, color=C['light'], edgecolor='white', alpha=0.6)

    # KDE curve — compute manually with numpy
    score_range = np.linspace(scores.min() - 2, scores.max() + 2, 300)
    kde = gaussian_kde(scores, bw_method=0.3)
    kde_vals = kde(score_range)
    # Scale KDE to match histogram counts
    bin_width = bin_edges[1] - bin_edges[0]
    kde_scaled = kde_vals * len(scores) * bin_width

    ax_freq.plot(score_range, kde_scaled, color=C['blue'], linewidth=2.5, zorder=3)
    ax_freq.fill_between(score_range, kde_scaled, alpha=0.15, color=C['blue'], zorder=2)

    # Annotate key score buckets on the curve with count and %
    score_buckets = [
        ('0-20',  0,  20), ('21-40', 21, 40), ('41-60', 41, 60),
        ('61-80', 61, 80), ('81-100', 81, 100),
    ]
    for label, lo, hi in score_buckets:
        count = int(((scores >= lo) & (scores <= hi)).sum())
        if count == 0:
            continue
        pct = count / total * 100
        mid = (lo + hi) / 2
        y_val = kde(mid)[0] * len(scores) * bin_width
        ax_freq.annotate(
            f'{count:,} ({pct:.1f}%)',
            xy=(mid, y_val), xytext=(mid, y_val + max(kde_scaled) * 0.12),
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=C['blue'],
            arrowprops=dict(arrowstyle='->', color=C['blue'], lw=1.2),
        )

    # Mean and Median lines
    ax_freq.axvline(score_mean, color=C['amber'], linewidth=2, linestyle='--',
                    label=f'Mean: {score_mean:.1f}', zorder=4)
    ax_freq.axvline(q2, color=C['red'], linewidth=2, linestyle='--',
                    label=f'Median: {q2:.1f}', zorder=4)

    ax_freq.set_xlabel('Final Score', fontsize=12)
    ax_freq.set_ylabel('Frequency')
    ax_freq.set_title('Lead Score Frequency Distribution', fontsize=14, fontweight='bold', pad=12)
    ax_freq.legend(fontsize=11, loc='upper right')
    ax_freq.spines[['top', 'right']].set_visible(False)

    # =================================================================
    # ROW 7: Business Category L1  (horizontal bar, sorted)
    # =================================================================
    ax = fig.add_subplot(gs[6, :])

    cat_counts = df['L1'].value_counts().sort_values(ascending=True)
    cat_colors = PALETTE[:len(cat_counts)]
    cat_colors.reverse()

    bars = ax.barh(cat_counts.index, cat_counts.values,
                   color=cat_colors, edgecolor='white', height=0.55)
    add_hbar_labels(ax, bars, total)
    ax.set_title('Business Category (L1)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Count')
    ax.set_xlim(0, max(cat_counts.values) * 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_overall_chart(csv_path, output_path):
    """Generate an overall dashboard for all leads with tier distribution."""
    df = pd.read_csv(csv_path, low_memory=False)
    total = len(df)

    fig = plt.figure(figsize=(26, 44), facecolor='white')
    fig.suptitle(f'Overall Lead Scoring Dashboard  \u2014  {total:,} leads',
                 fontsize=26, fontweight='bold', y=0.995)

    gs = gridspec.GridSpec(8, 4, figure=fig, hspace=0.40, wspace=0.35,
                           height_ratios=[1.0, 1.0, 0.7, 0.8, 1.0, 1.0, 0.9, 1.1])

    # =================================================================
    # ROW 1: Tier Distribution  (horizontal bar + donut)
    # =================================================================
    ax = fig.add_subplot(gs[0, :3])

    tier_order = ['Hot', 'Warm', 'Luke-warm', 'Cold', 'Very Cold']
    tier_colors_map = {
        'Hot': C['red'], 'Warm': C['amber'], 'Luke-warm': C['sky'],
        'Cold': C['blue'], 'Very Cold': C['purple'],
    }
    tier_counts = df['quality_tier'].value_counts()
    t_labels = [t for t in tier_order if t in tier_counts.index]
    t_values = [tier_counts[t] for t in t_labels]
    t_colors = [tier_colors_map[t] for t in t_labels]

    bars = ax.barh(t_labels, t_values, color=t_colors, edgecolor='white', height=0.6)
    add_hbar_labels(ax, bars, total, fontsize=11)
    ax.set_title('Tier Distribution', fontsize=16, fontweight='bold', pad=14)
    ax.set_xlim(0, max(t_values) * 1.20)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=12)
    ax.invert_yaxis()

    # Tier donut (right)
    ax_donut = fig.add_subplot(gs[0, 3])
    wedges, _ = ax_donut.pie(
        t_values, colors=t_colors, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
    )
    for i, (wedge, count) in enumerate(zip(wedges, t_values)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        pct = count / total * 100
        ha = 'left' if x >= 0 else 'right'
        ax_donut.annotate(
            f'{t_labels[i]}\n{pct:.1f}%',
            xy=(0.78 * x, 0.78 * y), xytext=(1.30 * x, 1.15 * y),
            ha=ha, va='center', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=C['grey'], lw=0.8),
        )
    ax_donut.set_title('Tier Split', fontsize=14, fontweight='bold', pad=12)

    # =================================================================
    # ROW 2 LEFT: Journey Stage  (horizontal bar, Converted removed)
    # =================================================================
    ax = fig.add_subplot(gs[1, :3])

    stage_map = {
        'KYB Submitted':   'KYB Submitted',
        'KYB In Progress': 'KYB In Progress',
        'Registered':      'Registered',
        'Onboarding: Kyc': 'KYC',
    }
    counts = {}
    for raw, label in stage_map.items():
        counts[label] = int((df['journey_stage'] == raw).sum())
    mapped_total = sum(counts.values())
    converted_count = int((df['journey_stage'] == 'Converted').sum())
    other_count = total - mapped_total - converted_count
    if other_count > 0:
        counts['Other'] = other_count

    counts = {k: v for k, v in counts.items() if v > 0}
    display_total = total - converted_count

    sorted_items = sorted(counts.items(), key=lambda x: x[1])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = PALETTE[:len(labels)]

    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.6)
    add_hbar_labels(ax, bars, display_total)
    ax.set_title('Journey Stage Breakdown', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(0, max(values) * 1.20)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)

    # ROW 2 RIGHT: Freelancer vs Registered  (donut)
    ax = fig.add_subplot(gs[1, 3])

    btype = df['business_type'].fillna('Unknown').str.strip().str.lower()
    btype_counts = btype.value_counts()
    bt_labels = [v.title() for v in btype_counts.index]
    bt_values = btype_counts.values
    bt_colors = [C['blue'], C['amber'], C['grey'], C['sky']]

    wedges, _ = ax.pie(
        bt_values, colors=bt_colors[:len(bt_values)], startangle=90,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
    )
    for i, (wedge, count) in enumerate(zip(wedges, bt_values)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        pct = count / total * 100
        ha = 'left' if x >= 0 else 'right'
        ax.annotate(
            f'{bt_labels[i]}\n{count:,} ({pct:.1f}%)',
            xy=(0.78 * x, 0.78 * y), xytext=(1.30 * x, 1.15 * y),
            ha=ha, va='center', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=C['grey'], lw=0.8),
        )
    ax.set_title('Business Type', fontsize=14, fontweight='bold', pad=12)

    # =================================================================
    # ROW 3: Contacted vs Not Contacted
    # =================================================================
    ax = fig.add_subplot(gs[2, :])

    raw_status = df['contacted_status'].fillna('').str.strip()
    n_contacted = int((raw_status != '').sum())
    n_not_contacted = total - n_contacted
    pct_c = n_contacted / total * 100
    pct_nc = n_not_contacted / total * 100

    bar_y = [0]
    ax.barh(bar_y, [n_contacted], height=0.5, color=C['green'], edgecolor='white', label='Contacted')
    ax.barh(bar_y, [n_not_contacted], height=0.5, left=[n_contacted], color=C['red'], edgecolor='white', label='Not Contacted')

    if n_contacted > total * 0.08:
        ax.text(n_contacted / 2, 0, f'Contacted\n{n_contacted:,}  ({pct_c:.1f}%)',
                ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    if n_not_contacted > total * 0.08:
        ax.text(n_contacted + n_not_contacted / 2, 0, f'Not Contacted\n{n_not_contacted:,}  ({pct_nc:.1f}%)',
                ha='center', va='center', fontsize=13, fontweight='bold', color='white')

    ax.set_title('Contacted vs Not Contacted', fontsize=14, fontweight='bold', pad=12)
    ax.set_yticks([])
    ax.set_xlabel('Count')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # =================================================================
    # ROW 4: Digital Presence
    # =================================================================
    ax = fig.add_subplot(gs[3, :])

    digital_features = [
        ('has_website',      'Website'),
        ('has_maps',         'Google Maps'),
        ('has_ecommerce',    'E-commerce'),
        ('has_social_media', 'Social Media'),
    ]
    feature_labels = [lbl for _, lbl in digital_features]
    yes_vals = [int(df[col].sum()) for col, _ in digital_features]
    no_vals = [total - y for y in yes_vals]

    y_pos = np.arange(len(feature_labels))
    bar_h = 0.45
    ax.barh(y_pos, yes_vals, height=bar_h, color=C['yes'], edgecolor='white', label='Yes')
    ax.barh(y_pos, no_vals, height=bar_h, left=yes_vals, color=C['no'], edgecolor='white', label='No')

    for i in range(len(feature_labels)):
        yes_pct = yes_vals[i] / total * 100
        if yes_vals[i] > total * 0.08:
            ax.text(yes_vals[i] / 2, y_pos[i], f'{yes_vals[i]:,}  ({yes_pct:.1f}%)',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        no_pct = no_vals[i] / total * 100
        if no_vals[i] > total * 0.08:
            ax.text(yes_vals[i] + no_vals[i] / 2, y_pos[i], f'{no_vals[i]:,}  ({no_pct:.1f}%)',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='#475569')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=12)
    ax.set_title('Digital Presence', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Count')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis()

    # =================================================================
    # ROW 5: Leads Created by Month
    # =================================================================
    ax = fig.add_subplot(gs[4, :])

    df['created_date_parsed'] = pd.to_datetime(df['created_date'], errors='coerce')
    df['created_month'] = df['created_date_parsed'].dt.to_period('M')
    month_counts = df.dropna(subset=['created_month']).groupby('created_month').size().sort_index()

    month_labels = [str(p) for p in month_counts.index]
    month_values = month_counts.values

    bar_colors = [C['blue'] if v >= np.percentile(month_values, 75) else
                  C['sky'] if v >= np.percentile(month_values, 50) else
                  C['light'] for v in month_values]

    bars = ax.bar(range(len(month_labels)), month_values, color=bar_colors,
                  edgecolor='white', width=0.7)
    for bar, val in zip(bars, month_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(month_values) * 0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=8)
    ax.set_title('Leads Created by Month', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Count')
    ax.spines[['top', 'right']].set_visible(False)

    # =================================================================
    # ROW 6: Lead Age Distribution
    # =================================================================
    ax = fig.add_subplot(gs[5, :3])

    ages = df['days_since_activity'].dropna()
    mean_age = ages.mean()
    median_age = ages.median()

    age_buckets = [
        ('0-30',    0,   30),  ('31-60',   31,  60),  ('61-90',   61,  90),
        ('91-120',  91,  120), ('121-180', 121, 180),  ('181-270', 181, 270),
        ('271-365', 271, 365), ('365+',    365, float('inf')),
    ]
    bucket_labels = [b[0] for b in age_buckets]
    bucket_vals = []
    for _, lo, hi in age_buckets:
        if hi == float('inf'):
            bucket_vals.append(int((ages >= lo).sum()))
        else:
            bucket_vals.append(int(((ages >= lo) & (ages <= hi)).sum()))

    bucket_colors = [C['green'] if v == max(bucket_vals) else
                     C['blue'] if v >= np.percentile(bucket_vals, 50) else
                     C['sky'] for v in bucket_vals]

    bars = ax.bar(range(len(bucket_labels)), bucket_vals, color=bucket_colors,
                  edgecolor='white', width=0.7)
    for bar, val in zip(bars, bucket_vals):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bucket_vals) * 0.015,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(bucket_labels)))
    ax.set_xticklabels(bucket_labels, fontsize=10)
    ax.set_xlabel('Days Since Last Activity', fontsize=11)
    ax.set_ylabel('Count')
    ax.set_title('Lead Age Distribution', fontsize=14, fontweight='bold', pad=12)
    ax.spines[['top', 'right']].set_visible(False)

    # Stats panel
    ax_stats = fig.add_subplot(gs[5, 3])
    ax_stats.axis('off')
    stats_text = (
        f"Lead Age Stats\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"Mean:      {mean_age:,.0f} days\n"
        f"Median:    {median_age:,.0f} days\n"
        f"Min:       {ages.min():,.0f} days\n"
        f"Max:       {ages.max():,.0f} days\n"
        f"Std Dev:   {ages.std():,.0f} days\n"
        f"Total:     {len(ages):,} leads"
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=12, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor=C['light'], edgecolor=C['grey'], alpha=0.8))

    # =================================================================
    # ROW 7: Lead Score  (box plot + frequency curve)
    # =================================================================
    scores = df['final_score'].dropna()
    q1 = scores.quantile(0.25)
    q2 = scores.median()
    q3 = scores.quantile(0.75)
    score_mean = scores.mean()
    score_min = scores.min()
    score_max = scores.max()

    ax_box = fig.add_subplot(gs[6, :2])
    ax_box.boxplot(scores, vert=False, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=C['blue'], alpha=0.7, edgecolor=C['blue']),
                   medianprops=dict(color=C['red'], linewidth=3),
                   whiskerprops=dict(color=C['blue'], linewidth=1.5),
                   capprops=dict(color=C['blue'], linewidth=2),
                   flierprops=dict(marker='o', markersize=4, markerfacecolor=C['grey'], alpha=0.5))

    above = [
        (score_min, f'Min: {score_min:.1f}', C['grey']),
        (q2,        f'Median: {q2:.1f}',     C['red']),
        (score_max, f'Max: {score_max:.1f}',  C['grey']),
    ]
    below = [
        (q1,         f'Q1: {q1:.1f}',          C['sky']),
        (score_mean, f'Mean: {score_mean:.1f}', C['amber']),
        (q3,         f'Q3: {q3:.1f}',           C['purple']),
    ]
    for val, label, color in above:
        ax_box.plot(val, 1.0, marker='v', markersize=8, color=color, zorder=5)
        ax_box.text(val, 1.32, label, ha='center', va='bottom', fontsize=12,
                    fontweight='bold', color=color)
    for val, label, color in below:
        ax_box.plot(val, 1.0, marker='^', markersize=8, color=color, zorder=5)
        ax_box.text(val, 0.52, label, ha='center', va='top', fontsize=12,
                    fontweight='bold', color=color)

    ax_box.set_ylim(0.2, 1.9)
    ax_box.set_xlabel('Final Score', fontsize=12)
    ax_box.set_title('Lead Score Box Plot', fontsize=14, fontweight='bold', pad=30)
    ax_box.set_yticks([])
    ax_box.spines[['top', 'right', 'left']].set_visible(False)
    ax_box.tick_params(axis='x', labelsize=11)

    # Frequency curve
    ax_freq = fig.add_subplot(gs[6, 2:])
    counts_hist, bin_edges, _ = ax_freq.hist(
        scores, bins=25, color=C['light'], edgecolor='white', alpha=0.6)

    score_range = np.linspace(scores.min() - 2, scores.max() + 2, 300)
    kde = gaussian_kde(scores, bw_method=0.3)
    kde_vals = kde(score_range)
    bin_width = bin_edges[1] - bin_edges[0]
    kde_scaled = kde_vals * len(scores) * bin_width

    ax_freq.plot(score_range, kde_scaled, color=C['blue'], linewidth=2.5, zorder=3)
    ax_freq.fill_between(score_range, kde_scaled, alpha=0.15, color=C['blue'], zorder=2)

    score_buckets = [
        ('0-20',  0,  20), ('21-40', 21, 40), ('41-60', 41, 60),
        ('61-80', 61, 80), ('81-100', 81, 100),
    ]
    for label, lo, hi in score_buckets:
        count = int(((scores >= lo) & (scores <= hi)).sum())
        if count == 0:
            continue
        pct = count / total * 100
        mid = (lo + hi) / 2
        y_val = kde(mid)[0] * len(scores) * bin_width
        ax_freq.annotate(
            f'{count:,} ({pct:.1f}%)',
            xy=(mid, y_val), xytext=(mid, y_val + max(kde_scaled) * 0.12),
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=C['blue'],
            arrowprops=dict(arrowstyle='->', color=C['blue'], lw=1.2),
        )

    ax_freq.axvline(score_mean, color=C['amber'], linewidth=2, linestyle='--',
                    label=f'Mean: {score_mean:.1f}', zorder=4)
    ax_freq.axvline(q2, color=C['red'], linewidth=2, linestyle='--',
                    label=f'Median: {q2:.1f}', zorder=4)

    ax_freq.set_xlabel('Final Score', fontsize=12)
    ax_freq.set_ylabel('Frequency')
    ax_freq.set_title('Lead Score Frequency Distribution', fontsize=14, fontweight='bold', pad=12)
    ax_freq.legend(fontsize=11, loc='upper right')
    ax_freq.spines[['top', 'right']].set_visible(False)

    # =================================================================
    # ROW 8: Business Category L1
    # =================================================================
    ax = fig.add_subplot(gs[7, :])

    cat_counts = df['L1'].value_counts().sort_values(ascending=True)
    cat_colors = PALETTE[:len(cat_counts)]
    cat_colors.reverse()

    bars = ax.barh(cat_counts.index, cat_counts.values,
                   color=cat_colors, edgecolor='white', height=0.55)
    add_hbar_labels(ax, bars, total)
    ax.set_title('Business Category (L1)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Count')
    ax.set_xlim(0, max(cat_counts.values) * 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# --- All tiers to generate ---
TIERS = [
    ('tier_hot_leads.csv',       'tier_1_hot_leads_dashboard.png',       'Tier 1: Hot Leads'),
    ('tier_warm_leads.csv',      'tier_2_warm_leads_dashboard.png',      'Tier 2: Warm Leads'),
    ('tier_luke-warm_leads.csv', 'tier_3_luke_warm_leads_dashboard.png', 'Tier 3: Luke-warm Leads'),
    ('tier_cold_leads.csv',      'tier_4_cold_leads_dashboard.png',      'Tier 4: Cold Leads'),
    ('tier_very_cold_leads.csv', 'tier_5_very_cold_leads_dashboard.png', 'Tier 5: Very Cold Leads'),
]

if __name__ == '__main__':
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if arg == 'hot':
        tiers_to_run = [TIERS[0]]
    elif arg == 'overall':
        tiers_to_run = []
    else:
        tiers_to_run = TIERS

    for csv_name, png_name, tier_label in tiers_to_run:
        csv_file = os.path.join(csv_dir, csv_name)
        out_file = os.path.join(script_dir, png_name)
        if os.path.exists(csv_file):
            generate_tier_chart(csv_file, out_file, tier_label)
        else:
            print(f"Skipped (not found): {csv_name}")

    # Generate overall dashboard
    if arg in ('all', 'overall'):
        overall_csv = os.path.join(csv_dir, 'scored_leads_v3_final.csv')
        overall_out = os.path.join(script_dir, 'overall_leads_dashboard.png')
        if os.path.exists(overall_csv):
            generate_overall_chart(overall_csv, overall_out)
        else:
            print(f"Skipped (not found): scored_leads_v3_final.csv")
