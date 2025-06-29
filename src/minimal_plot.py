import matplotlib.pyplot as plt

# Data
models = ['SSL + MLP (1%)', 'SSL + Linear (5%)', 'Supervised (100%)']
mf1_scores = [64.45, 63.2, 70.6]
percentages = [91, 89.5, 100]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, mf1_scores, color=['#4c72b0', '#55a868', '#c44e52'])

# Annotate bars
for bar, pct, val in zip(bars, percentages, mf1_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, 
            f'{val:.1f} ({pct}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Dotted line for supervised reference
ax.axhline(70.6, linestyle='--', color='gray', linewidth=1)
ax.text(2, 71.5, 'Supervised upper bound', ha='center', color='gray', fontsize=9)

# Labels and aesthetics
ax.set_ylabel('Macro F1 Score')
ax.set_ylim(0, 80)
ax.set_title('SSL Models Recover Supervised Performance with Minimal Labels')
plt.xticks(rotation=15)
plt.tight_layout()

plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()
