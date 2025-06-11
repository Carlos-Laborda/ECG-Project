import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme for different models
COLORS = {
    # Supervised models
    'CNN': '#2E86C1',
    'TCN': '#E74C3C',
    'Transformer': '#6A5ACD',
    
    # SSL models
    'TS2Vec': '#2E86C1',
    'SoftTS2Vec': '#E74C3C',
    'TSTCC': '#6A5ACD',
    'SoftTSTCC': '#2ECC71',
    'SimCLR': '#FFA500'
}

# Marker styles for different models
MARKERS = {
    # Supervised models
    'CNN': 'o',
    'TCN': 's',
    'Transformer': '^',
    
    # SSL models
    'TS2Vec': 'o',
    'SoftTS2Vec': 's',
    'TSTCC': '^',
    'SoftTSTCC': 'D',
    'SimCLR': '*'
}

SIGNIFICANCE = {
    0.01: {('CNN', 'TCN'): True, ('CNN', 'Transformer'): False, ('Transformer', 'TCN'): True},
    0.05: {('CNN', 'TCN'): False, ('CNN', 'Transformer'): True, ('Transformer', 'TCN'): True},
    0.10: {('CNN', 'TCN'): False, ('CNN', 'Transformer'): True, ('Transformer', 'TCN'): True},
    0.50: {('CNN', 'TCN'): False, ('CNN', 'Transformer'): True, ('Transformer', 'TCN'): True},
    1.00: {('CNN', 'TCN'): False, ('CNN', 'Transformer'): False, ('Transformer', 'TCN'): False}
}

# Add statistical significance information for SSL models
SSL_LINEAR_SIGNIFICANCE = {
    0.01: {
        ('SimCLR', 'SoftTS2Vec'): False,
        ('SimCLR', 'TSTCC'): False,
        ('SimCLR', 'SoftTSTCC'): False,
        ('SimCLR', 'TS2Vec'): False,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): False
    },
    0.05: {
        ('SimCLR', 'SoftTS2Vec'): False,
        ('SimCLR', 'TSTCC'): False,
        ('SimCLR', 'SoftTSTCC'): False,
        ('SimCLR', 'TS2Vec'): True,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): False
    },
    0.10: {
        ('SimCLR', 'SoftTS2Vec'): True,
        ('SimCLR', 'TSTCC'): True,
        ('SimCLR', 'SoftTSTCC'): False,
        ('SimCLR', 'TS2Vec'): True,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): True
    },
    0.50: {
        ('SimCLR', 'SoftTS2Vec'): True,
        ('SimCLR', 'TSTCC'): True,
        ('SimCLR', 'SoftTSTCC'): True,
        ('SimCLR', 'TS2Vec'): True,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): False
    },
    1.00: {
        ('SimCLR', 'SoftTS2Vec'): True,
        ('SimCLR', 'TSTCC'): True,
        ('SimCLR', 'SoftTSTCC'): True,
        ('SimCLR', 'TS2Vec'): True,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): False
    }
}

SSL_MLP_SIGNIFICANCE = {
    0.01: {
        ('SimCLR', 'SoftTS2Vec'): False,
        ('SimCLR', 'TSTCC'): False,
        ('SimCLR', 'SoftTSTCC'): False,
        ('SimCLR', 'TS2Vec'): False,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): False
    },
    0.05: {
        ('SimCLR', 'SoftTS2Vec'): True,
        ('SimCLR', 'TSTCC'): True,
        ('SimCLR', 'SoftTSTCC'): True,
        ('SimCLR', 'TS2Vec'): True,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): False
    },
    0.10: {
        ('SimCLR', 'SoftTS2Vec'): True,
        ('SimCLR', 'TSTCC'): True,
        ('SimCLR', 'SoftTSTCC'): True,
        ('SimCLR', 'TS2Vec'): True,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): False
    },
    0.50: {
        ('SimCLR', 'SoftTS2Vec'): True,
        ('SimCLR', 'TSTCC'): True,
        ('SimCLR', 'SoftTSTCC'): True,
        ('SimCLR', 'TS2Vec'): True,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): False
    },
    1.00: {
        ('SimCLR', 'SoftTS2Vec'): True,
        ('SimCLR', 'TSTCC'): True,
        ('SimCLR', 'SoftTSTCC'): True,
        ('SimCLR', 'TS2Vec'): True,
        ('SoftTS2Vec', 'TSTCC'): False,
        ('SoftTS2Vec', 'SoftTSTCC'): False,
        ('SoftTS2Vec', 'TS2Vec'): False,
        ('TSTCC', 'SoftTSTCC'): False,
        ('TSTCC', 'TS2Vec'): False,
        ('SoftTSTCC', 'TS2Vec'): False
    }
}

# Load data
df = pd.read_csv(Path("../results/confidence_intervals/supervised_ci.csv"))
df["label_fraction"] = df["label_fraction"].astype(float)
fractions = sorted(df["label_fraction"].unique())
models = sorted(df["model"].unique())
n_models = len(models)
bar_width = 0.2
x = np.arange(len(fractions))

# Plot
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

# Plot each model
for i, model in enumerate(models):
    sub = df[df["model"] == model].sort_values("label_fraction")
    means = sub["mean"].values
    yerr_lower = means - sub["ci_lower"].values
    yerr_upper = sub["ci_upper"].values - means

    positions = x + i * bar_width
    ax.errorbar(
        positions, means, yerr=[yerr_lower, yerr_upper],
        fmt=MARKERS.get(model, 'o'),
        capsize=5,
        label=model,
        color=COLORS.get(model, f'C{i}'),
        linestyle='None',
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
        capthick=2,
        elinewidth=2
    )

# Add significance asterisks
for idx, frac in enumerate(fractions):
    comparisons = SIGNIFICANCE.get(frac, {})
    base_x = x[idx]
    height = max(df[df["label_fraction"] == frac]["ci_upper"]) + 2

    for j, (m1, m2) in enumerate(comparisons):
        if comparisons[(m1, m2)]:
            i1 = models.index(m1)
            i2 = models.index(m2)
            x1 = base_x + i1 * bar_width
            x2 = base_x + i2 * bar_width
            y = height + j * 3
            ax.plot([x1, x2], [y, y], color='black', linewidth=1)
            ax.text((x1 + x2)/2, y + 0.01, "*", ha='center', va='bottom', fontsize=14)

# Styling
ax.set_title("Supervised Models", fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Proportion of Labeled Training Data", fontsize=12, fontweight='bold')
ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
ax.set_xticks(x + bar_width * (n_models - 1) / 2)
ax.set_xticklabels([f'{int(f*100)}%' for f in fractions], fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12) 
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('#FAFAFA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, loc='lower right')

plt.tight_layout()
plt.savefig("../results/error_bars_supervised.pdf", dpi=300)
plt.savefig("../results/error_bars_supervised.png", dpi=300)
plt.show()


# --- Plot for SSL Models ---

# Create a single figure with two subplots
fig_ssl, (ax_ssl_linear, ax_ssl_mlp) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)

# Load data for SSL Linear
df_ssl_linear = pd.read_csv(Path("../results/confidence_intervals/ssl_linear_ci.csv"))
df_ssl_linear["label_fraction"] = df_ssl_linear["label_fraction"].astype(float)
fractions_ssl_linear = sorted(df_ssl_linear["label_fraction"].unique())

# Load data for SSL MLP
df_ssl_mlp = pd.read_csv(Path("../results/confidence_intervals/ssl_mlp_ci.csv"))
df_ssl_mlp["label_fraction"] = df_ssl_mlp["label_fraction"].astype(float)
fractions_ssl_mlp = sorted(df_ssl_mlp["label_fraction"].unique())

# Filter for SSL models in desired order
ssl_models = ['SimCLR', 'TS2Vec', 'SoftTS2Vec', 'TSTCC', 'SoftTSTCC']

# Process Linear data
df_ssl_linear = df_ssl_linear[df_ssl_linear["model"].isin(ssl_models)]
df_ssl_linear['model'] = pd.Categorical(df_ssl_linear['model'], categories=ssl_models, ordered=True)
df_ssl_linear = df_ssl_linear.sort_values('model')

# Process MLP data
df_ssl_mlp = df_ssl_mlp[df_ssl_mlp["model"].isin(ssl_models)]
df_ssl_mlp['model'] = pd.Categorical(df_ssl_mlp['model'], categories=ssl_models, ordered=True)
df_ssl_mlp = df_ssl_mlp.sort_values('model')

# Common variables
n_models = len(ssl_models)
x_ssl = np.arange(len(fractions_ssl_linear)) * 1.5

# Plot Linear subplot
for i, model in enumerate(ssl_models):
    sub = df_ssl_linear[df_ssl_linear["model"] == model].sort_values("label_fraction")
    means = sub["mean"].values
    yerr_lower = means - sub["ci_lower"].values
    yerr_upper = sub["ci_upper"].values - means

    positions = x_ssl + i * bar_width
    ax_ssl_linear.errorbar(
        positions, means, yerr=[yerr_lower, yerr_upper],
        fmt=MARKERS.get(model, 'o'),
        capsize=5,
        label=model,
        color=COLORS.get(model, f'C{i}'),
        linestyle='None',
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
        capthick=2,
        elinewidth=2
    )

# Plot MLP subplot
for i, model in enumerate(ssl_models):
    sub = df_ssl_mlp[df_ssl_mlp["model"] == model].sort_values("label_fraction")
    means = sub["mean"].values
    yerr_lower = means - sub["ci_lower"].values
    yerr_upper = sub["ci_upper"].values - means

    positions = x_ssl + i * bar_width
    ax_ssl_mlp.errorbar(
        positions, means, yerr=[yerr_lower, yerr_upper],
        fmt=MARKERS.get(model, 'o'),
        capsize=5,
        label=model,
        color=COLORS.get(model, f'C{i}'),
        linestyle='None',
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
        capthick=2,
        elinewidth=2
    )

# Add significance asterisks for both plots
for idx, frac in enumerate(fractions_ssl_linear):
    # Linear plot significance
    comparisons = SSL_LINEAR_SIGNIFICANCE.get(frac, {})
    base_x = x_ssl[idx]
    relevant_data = df_ssl_linear[df_ssl_linear["label_fraction"] == frac]
    if not relevant_data.empty:
        height = max(relevant_data["ci_upper"]) + 2 
    else:
        height = ax_ssl_linear.get_ylim()[1] * 0.1

    for j, (m1, m2) in enumerate(comparisons):
        if comparisons[(m1, m2)]:
            i1 = ssl_models.index(m1)
            i2 = ssl_models.index(m2)
            x1 = base_x + i1 * bar_width
            x2 = base_x + i2 * bar_width
            y = height + j * 3
            ax_ssl_linear.plot([x1, x2], [y, y], color='black', linewidth=1)
            ax_ssl_linear.text((x1 + x2)/2, y + 0.01, "*", ha='center', va='bottom', fontsize=14)

    # MLP plot significance
    comparisons = SSL_MLP_SIGNIFICANCE.get(frac, {})
    relevant_data = df_ssl_mlp[df_ssl_mlp["label_fraction"] == frac]
    if not relevant_data.empty:
        height = max(relevant_data["ci_upper"]) + 2
    else:
        height = ax_ssl_mlp.get_ylim()[1] * 0.1

    for j, (m1, m2) in enumerate(comparisons):
        if comparisons[(m1, m2)]:
            i1 = ssl_models.index(m1)
            i2 = ssl_models.index(m2)
            x1 = base_x + i1 * bar_width
            x2 = base_x + i2 * bar_width
            y = height + j * 3
            ax_ssl_mlp.plot([x1, x2], [y, y], color='black', linewidth=1)
            ax_ssl_mlp.text((x1 + x2)/2, y + 0.01, "*", ha='center', va='bottom', fontsize=14)

# Styling for both subplots
for ax, title in [(ax_ssl_linear, "SSL Models with Linear Classifier"), 
                  (ax_ssl_mlp, "SSL Models with MLP Classifier")]:
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Proportion of Labeled Training Data", fontsize=12, fontweight='bold')
    ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    ax.set_xticks(x_ssl + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([f'{int(f*100)}%' for f in fractions_ssl_linear], fontsize=12) 
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_facecolor('#FAFAFA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, loc='lower right')

# Add a common title
fig_ssl.suptitle('SSL Models Performance Comparison', fontsize=16, fontweight='bold', y=1.02)

# Modified saving approach
plt.tight_layout()
# Save with extra space at the top for the suptitle
plt.savefig("../results/error_bars_ssl.pdf", 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.3)  # Add padding
plt.savefig("../results/error_bars_ssl.png", 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.3)  # Add padding
plt.show()