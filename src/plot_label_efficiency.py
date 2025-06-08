"""
Plot label-efficiency curves (macro-F1 vs. label-fraction)
for all models in the `results/` directory.
"""
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib as mpl
import seaborn as sns
from glob import glob
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# --------------------------------------------------
# config ― edit to match your folder names if needed
# --------------------------------------------------
ROOT_DIR       = pathlib.Path("../results")
LABEL_FRACTIONS = [0.01, 0.05, 0.1, 0.5, 1.0]
RESULTS_FILE = "../results/all_results.csv"

MODEL_NAMES = [
    # Supervised models first
    "Supervised_CNN",
    "Supervised_TCN", 
    "Supervised_Transformer",

    # SSL models
    "TS2Vec",
    "SoftTS2Vec",
    "TSTCC",
    "SoftTSTCC",
    "SimCLR",
]

SUPERVISED_MODELS = [
    "Supervised_CNN",
    "Supervised_TCN",
    "Supervised_Transformer"
]

SSL_MODELS = [
    "TS2Vec",
    "SoftTS2Vec",
    "TSTCC",
    "SoftTSTCC",
    #"SimCLR"
]

GROUPS = {
    "Supervised": SUPERVISED_MODELS,
    "Self-Supervised": SSL_MODELS
}

# --------------------------------------------
# Load or build results DataFrame
# --------------------------------------------
if os.path.exists(RESULTS_FILE):
    print(f"[Loading cached metrics from {RESULTS_FILE}]")
    data = pd.read_csv(RESULTS_FILE)
else:
    print("Parsing metrics and building dataframe...")
    records = []

    for group_name, model_list in GROUPS.items():
        for model in model_list:
            pattern = str(ROOT_DIR / model / "*_*") 
            matching_dirs = glob(pattern)
            for exp_path_str in matching_dirs:
                exp_path = pathlib.Path(exp_path_str)
                exp_name = exp_path.name

                # Parse classifier model and label fraction
                parts = exp_name.split("_")
                if len(parts) < 2:
                    continue  # skip malformed names

                classifier_model = parts[0]
                try:
                    label_fraction = float(parts[-1])
                except ValueError:
                    continue  # skip if not a valid float

                csv_file = exp_path / "aggregated_metrics.csv"
                if not csv_file.exists():
                    continue

                df = pd.read_csv(csv_file)

                # Normalize AUC metric name
                auc_name = 'test_auc_roc' if 'test_auc_roc' in df.metric.values else (
                    'test_auroc' if 'test_auroc' in df.metric.values else None)
                if auc_name is None:
                    continue

                def get_metric(metric):
                    row = df[df.metric == metric]
                    return (row["mean"].values[0], row["std"].values[0]) if not row.empty else (np.nan, np.nan)

                acc_mean, acc_std = get_metric("test_accuracy")
                auc_mean, auc_std = get_metric(auc_name)
                pr_mean, pr_std   = get_metric("test_pr_auc")
                f1_mean, f1_std   = get_metric("test_f1")

                records.append({
                    "group": group_name,
                    "model": model,
                    "classifier_model": classifier_model,
                    "label_fraction": label_fraction,
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "auc_mean": auc_mean,
                    "auc_std": auc_std,
                    "pr_auc_mean": pr_mean,
                    "pr_auc_std": pr_std,
                    "f1_mean": f1_mean,
                    "f1_std": f1_std
                })

    data = pd.DataFrame(records)
    data.to_csv(RESULTS_FILE, index=False)
    print(f"[info] Saved full results to {RESULTS_FILE}")
    

# --------------------------------------------------
# Statistical Testing at Each Label Fraction SSL (linear) vs Supervised
# --------------------------------------------------
print("\nStatistical Testing (SSL vs Supervised)")
print("-" * 50)

for fraction in LABEL_FRACTIONS:
    # Get supervised scores for this fraction
    supervised_scores = data[
        (data.group == 'Supervised') & 
        (data.label_fraction == fraction)
    ]['f1_mean']
    
    # Get SSL scores for this fraction (linear classifier only)
    ssl_scores = data[
        (data.group == 'Self-Supervised') & 
        (data.classifier_model == 'LinearClassifier') &
        (data.model != 'SimCLR') &
        (data.label_fraction == fraction)
    ]['f1_mean']
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(supervised_scores, ssl_scores)
    
    print(f"\nLabel Fraction: {int(fraction*100)}%")
    print(f"Supervised mean: {supervised_scores.mean():.3f} ± {supervised_scores.std():.3f}")
    print(f"SSL mean: {ssl_scores.mean():.3f} ± {ssl_scores.std():.3f}")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant difference detected (p < 0.05)")
        if supervised_scores.mean() > ssl_scores.mean():
            print("Supervised performs better")
        else:
            print("Self-supervised performs better")
    else:
        print("No significant difference (p >= 0.05)")
        
# --------------------------------------------------
# Statistical Testing: Supervised vs TS2Vec-MLP
# --------------------------------------------------
print("\nStatistical Testing (Supervised vs TS2Vec-MLP)")
print("-" * 50)

for fraction in LABEL_FRACTIONS:
    # Get supervised scores for this fraction
    supervised_scores = data[
        (data.group == 'Supervised') & 
        (data.label_fraction == fraction)
    ]['f1_mean']
    
    # Get TS2Vec MLP scores and std from seeds for this fraction
    ts2vec_mlp_data = data[
        (data.model == 'TS2Vec') & 
        (data.classifier_model == 'MLPClassifier') &
        (data.label_fraction == fraction)
    ]
    ts2vec_mlp_mean = ts2vec_mlp_data['f1_mean'].mean()
    ts2vec_mlp_std = ts2vec_mlp_data['f1_std'].mean()  # Use std from seeds
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(supervised_scores, ts2vec_mlp_data['f1_mean'])
    
    print(f"\nLabel Fraction: {int(fraction*100)}%")
    print(f"Supervised mean: {supervised_scores.mean():.3f} ± {supervised_scores.std():.3f}")
    print(f"TS2Vec-MLP mean: {ts2vec_mlp_mean:.3f} ± {ts2vec_mlp_std:.3f}")  # Now using seed-level std
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant difference detected (p < 0.05)")
        if supervised_scores.mean() > ts2vec_mlp_mean:
            print("Supervised performs better")
        else:
            print("TS2Vec-MLP performs better")
    else:
        print("No significant difference (p >= 0.05)")

# --------------------------------------------------
# Supervised vs SSL Models: Average F1 Score Comparison
# --------------------------------------------------
# Filter and aggregate Supervised models
supervised_avg = (
    data[data.group == 'Supervised']
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models (linear classifier only)
ssl_avg = (
    data[(data.group == 'Self-Supervised') & 
         (data.classifier_model == 'LinearClassifier') &
         (data.model != 'SimCLR')] 
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

# Create publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot Supervised Average
ax.plot(supervised_avg.label_fraction, supervised_avg.f1_mean,
        marker='o', label='Supervised (Average)',
        color='#FF6B6B', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(supervised_avg.label_fraction,
                supervised_avg.f1_mean - supervised_avg.f1_std,
                supervised_avg.f1_mean + supervised_avg.f1_std,
                color='#FF6B6B', alpha=0.15)

# Plot SSL Average
ax.plot(ssl_avg.label_fraction, ssl_avg.f1_mean,
        marker='s', label='Self-Supervised (Average)',
        color='#1E90FF', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_avg.label_fraction,
                ssl_avg.f1_mean - ssl_avg.f1_std,
                ssl_avg.f1_mean + ssl_avg.f1_std,
                color='#1E90FF', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Average Performance: Supervised vs Self-Supervised', 
             fontsize=14, fontweight='bold', pad=20)

# Grid and background
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('#FAFAFA')

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Legend
ax.legend(frameon=True, 
         fancybox=True,
         shadow=False,
         fontsize=11,
         loc='lower right')

plt.tight_layout()

# Save high-resolution versions
plt.savefig('../results/supervised_vs_ssl_avg.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/supervised_vs_ssl_avg.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()


# --------------------------------------------------
# Three-way Comparison: Supervised vs SSL-Linear vs SSL-MLP
# --------------------------------------------------
# Filter and aggregate Supervised models
supervised_avg = (
    data[data.group == 'Supervised']
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models with Linear classifier
ssl_linear_avg = (
    data[(data.group == 'Self-Supervised') & 
         (data.classifier_model == 'LinearClassifier') &
         (data.model != 'SimCLR')]
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models with MLP classifier
ssl_mlp_avg = (
    data[(data.group == 'Self-Supervised') & 
         (data.classifier_model == 'MLPClassifier') &
         (data.model != 'SimCLR')]
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

# Create publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot Supervised Average
ax.plot(supervised_avg.label_fraction, supervised_avg.f1_mean,
        marker='o', label='Supervised (Average)',
        color='#FF6B6B', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(supervised_avg.label_fraction,
                supervised_avg.f1_mean - supervised_avg.f1_std,
                supervised_avg.f1_mean + supervised_avg.f1_std,
                color='#FF6B6B', alpha=0.15)

# Plot SSL Linear Average
ax.plot(ssl_linear_avg.label_fraction, ssl_linear_avg.f1_mean,
        marker='s', label='Self-Supervised (Linear)',
        color='#1E90FF', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_linear_avg.label_fraction,
                ssl_linear_avg.f1_mean - ssl_linear_avg.f1_std,
                ssl_linear_avg.f1_mean + ssl_linear_avg.f1_std,
                color='#1E90FF', alpha=0.15)

# Plot SSL MLP Average
ax.plot(ssl_mlp_avg.label_fraction, ssl_mlp_avg.f1_mean,
        marker='^', label='Self-Supervised (MLP)',
        color='#6A5ACD', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_mlp_avg.label_fraction,
                ssl_mlp_avg.f1_mean - ssl_mlp_avg.f1_std,
                ssl_mlp_avg.f1_mean + ssl_mlp_avg.f1_std,
                color='#6A5ACD', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Average Performance: Supervised vs Self-Supervised Methods', 
             fontsize=14, fontweight='bold', pad=20)

# Grid and background
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('#FAFAFA')

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Legend
ax.legend(frameon=True, 
         fancybox=True,
         shadow=False,
         fontsize=11,
         loc='lower right')

plt.tight_layout()

# Save high-resolution versions
plt.savefig('../results/three_way_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/three_way_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()


# --------------------------------------------------
# TSTCC vs SoftTSTCC Comparison
# --------------------------------------------------
# Filter data for the two models
tstcc_data = data[data.model == 'TSTCC'].sort_values('label_fraction')
soft_tstcc_data = data[data.model == 'SoftTSTCC'].sort_values('label_fraction')

# Create publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot TSTCC
ax.plot(tstcc_data.label_fraction, tstcc_data.f1_mean,
        marker='o', label='TSTCC',
        color='#2E86C1', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(tstcc_data.label_fraction,
                tstcc_data.f1_mean - tstcc_data.f1_std,
                tstcc_data.f1_mean + tstcc_data.f1_std,
                color='#2E86C1', alpha=0.15)

# Plot SoftTSTCC
ax.plot(soft_tstcc_data.label_fraction, soft_tstcc_data.f1_mean,
        marker='s', label='SoftTSTCC',
        color='#E74C3C', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(soft_tstcc_data.label_fraction,
                soft_tstcc_data.f1_mean - soft_tstcc_data.f1_std,
                soft_tstcc_data.f1_mean + soft_tstcc_data.f1_std,
                color='#E74C3C', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Label Efficiency: TSTCC vs SoftTSTCC', 
             fontsize=14, fontweight='bold', pad=20)

# Grid and background
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('#FAFAFA')

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Legend
ax.legend(frameon=True, 
          fancybox=True,
          shadow=False,
          fontsize=11,
          loc='lower right')

plt.tight_layout()

# Save high-resolution versions
plt.savefig('../results/tstcc_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/tstcc_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()

# --------------------------------------------------
# TS2Vec vs SoftTS2Vec Comparison (Linear Classifier)
# --------------------------------------------------
# Filter data for the two models with LinearClassifier only
ts2vec_data = data[(data.model == 'TS2Vec') & 
                   (data.classifier_model == 'LinearClassifier')].sort_values('label_fraction')
soft_ts2vec_data = data[(data.model == 'SoftTS2Vec') & 
                        (data.classifier_model == 'LinearClassifier')].sort_values('label_fraction')

# Create publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot TS2Vec
ax.plot(ts2vec_data.label_fraction, ts2vec_data.f1_mean,
        marker='o', label='TS2Vec',
        color='#2E86C1', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ts2vec_data.label_fraction,
                ts2vec_data.f1_mean - ts2vec_data.f1_std,
                ts2vec_data.f1_mean + ts2vec_data.f1_std,
                color='#2E86C1', alpha=0.15)

# Plot SoftTS2Vec
ax.plot(soft_ts2vec_data.label_fraction, soft_ts2vec_data.f1_mean,
        marker='s', label='SoftTS2Vec',
        color='#E74C3C', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(soft_ts2vec_data.label_fraction,
                soft_ts2vec_data.f1_mean - soft_ts2vec_data.f1_std,
                soft_ts2vec_data.f1_mean + soft_ts2vec_data.f1_std,
                color='#E74C3C', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Label Efficiency: TS2Vec vs SoftTS2Vec\n(Linear Classifier)', 
             fontsize=14, fontweight='bold', pad=20)

# Grid and background
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('#FAFAFA')

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Legend
ax.legend(frameon=True, 
         fancybox=True,
         shadow=False,
         fontsize=11,
         loc='lower right')

plt.tight_layout()

# Save high-resolution versions
plt.savefig('../results/ts2vec_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/ts2vec_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()

# --------------------------------------------------
# TS2Vec: Linear vs MLP Classifier Comparison
# --------------------------------------------------
# Filter data for TS2Vec with different classifiers
ts2vec_linear = data[(data.model == 'TS2Vec') & 
                     (data.classifier_model == 'LinearClassifier')].sort_values('label_fraction')
ts2vec_mlp = data[(data.model == 'TS2Vec') & 
                  (data.classifier_model == 'MLPClassifier')].sort_values('label_fraction')

# Create publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot TS2Vec + Linear
ax.plot(ts2vec_linear.label_fraction, ts2vec_linear.f1_mean,
        marker='o', label='TS2Vec (Linear)',
        color='#2E86C1', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ts2vec_linear.label_fraction,
                ts2vec_linear.f1_mean - ts2vec_linear.f1_std,
                ts2vec_linear.f1_mean + ts2vec_linear.f1_std,
                color='#2E86C1', alpha=0.15)

# Plot TS2Vec + MLP
ax.plot(ts2vec_mlp.label_fraction, ts2vec_mlp.f1_mean,
        marker='s', label='TS2Vec (MLP)',
        color='#E74C3C', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ts2vec_mlp.label_fraction,
                ts2vec_mlp.f1_mean - ts2vec_mlp.f1_std,
                ts2vec_mlp.f1_mean + ts2vec_mlp.f1_std,
                color='#E74C3C', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
ax.set_title('TS2Vec: Linear vs MLP Classifier Performance', 
             fontsize=14, fontweight='bold', pad=20)

# Grid and background
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('#FAFAFA')

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Legend
ax.legend(frameon=True, 
         fancybox=True,
         shadow=False,
         fontsize=11,
         loc='lower right')

plt.tight_layout()

# Save high-resolution versions
plt.savefig('../results/ts2vec_classifier_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/ts2vec_classifier_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()