import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Data
models = ['SSL + MLP (1%)', 'SSL + Linear (5%)', 'Supervised (100%)']
mf1_scores = [64.45, 63.2, 70.6]
percentages = [91, 89.5, 100]

# Colors matching the label efficiency plots
colors = ['#9467BD', '#1F77B4', '#FF6B6B']

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot bars
bars = ax.bar(models, mf1_scores, color=colors, width=0.6)

# Annotate bars with values
for bar, pct, val in zip(bars, percentages, mf1_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}\n({pct}%)', 
            ha='center', va='bottom', 
            fontsize=12, fontweight='bold')

# Add supervised reference line
ax.axhline(70.6, linestyle='--', color='gray', alpha=0.5, linewidth=1)
ax.text(-0.2, 71.5, 'Supervised upper bound', 
        ha='left', color='gray', fontsize=10, 
        style='italic')

# Styling
ax.set_ylabel('MF1 Score', fontsize=12, fontweight='bold')
ax.set_ylim(0, 80)
ax.set_title('SSL Models Recover Supervised Performance\nwith Minimal Labels', 
             fontsize=14, fontweight='bold', pad=20)

# Customize grid
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Customize ticks
plt.xticks(rotation=15)
ax.tick_params(axis='both', which='major', labelsize=12)

# Background and spines
ax.set_facecolor('#FAFAFA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.tight_layout()
plt.show()