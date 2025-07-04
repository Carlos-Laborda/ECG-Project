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

SSL_LINEAR_MODELS = [
    ("TS2Vec", "LinearClassifier"),
    ("SoftTS2Vec", "LinearClassifier"),
    ("TSTCC", "LinearClassifier"),
    ("SoftTSTCC", "LinearClassifier"),
    ("SimCLR", "LinearClassifier")
]

SSL_MLP_MODELS = [
    ("TS2Vec", "MLPClassifier"),
    ("SoftTS2Vec", "MLPClassifier"),
    ("TSTCC", "MLPClassifier"),
    ("SoftTSTCC", "MLPClassifier"),
    ("SimCLR", "MLPClassifier")
]

GROUPS = {
    "Supervised": SUPERVISED_MODELS,
    "Self-Supervised (Linear)": SSL_LINEAR_MODELS,
    "Self-Supervised (MLP)": SSL_MLP_MODELS
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

    # Handle Supervised Models
    for model in SUPERVISED_MODELS:
        pattern = str(ROOT_DIR / model / "*_*") 
        matching_dirs = glob(pattern)
        for exp_path_str in matching_dirs:
            exp_path = pathlib.Path(exp_path_str)
            exp_name = exp_path.name

            # Parse classifier model and label fraction
            parts = exp_name.split("_")
            if len(parts) < 2:
                continue
            
            classifier_model = parts[0]
            try:
                label_fraction = float(parts[-1])
            except ValueError:
                continue

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
                "group": "Supervised",
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
    
    # Handle SSL Models (both Linear and MLP)
    for ssl_model, classifier in SSL_LINEAR_MODELS + SSL_MLP_MODELS:
        pattern = str(ROOT_DIR / ssl_model / f"{classifier}_*")
        matching_dirs = glob(pattern)
        for exp_path_str in matching_dirs:
            exp_path = pathlib.Path(exp_path_str)
            exp_name = exp_path.name

            parts = exp_name.split("_")
            if len(parts) < 2:
                continue

            try:
                label_fraction = float(parts[-1])
            except ValueError:
                continue

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

            # Correct group assignment based on classifier type
            group = "Self-Supervised (Linear)" if classifier == "LinearClassifier" else "Self-Supervised (MLP)"
            
            records.append({
                "group": group,
                "model": ssl_model,
                "classifier_model": classifier,  
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
    # Add model type column for easier filtering
    data['model_type'] = data.apply(
        lambda x: 'Supervised' if x['group'] == 'Supervised' 
        else ('SSL-Linear' if x['classifier_model'] == 'LinearClassifier' 
              else 'SSL-MLP'), axis=1
    )
    data.to_csv(RESULTS_FILE, index=False)
    print(f"[info] Saved full results to {RESULTS_FILE}")

# --------------------------------------------------
# Supervised Models Comparison: CNN vs TCN vs Transformer
# --------------------------------------------------
# Filter data for each supervised model
cnn_data = data[
    data.model == 'Supervised_CNN'
].sort_values('label_fraction')

tcn_data = data[
    data.model == 'Supervised_TCN'
].sort_values('label_fraction')

transformer_data = data[
    data.model == 'Supervised_Transformer'
].sort_values('label_fraction')

# Create publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot CNN
ax.plot(cnn_data.label_fraction, cnn_data.f1_mean,
        marker='o', label='CNN',
        color='#2E86C1', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(cnn_data.label_fraction,
                cnn_data.f1_mean - cnn_data.f1_std,
                cnn_data.f1_mean + cnn_data.f1_std,
                color='#2E86C1', alpha=0.15)

# Plot TCN
ax.plot(tcn_data.label_fraction, tcn_data.f1_mean,
        marker='s', label='TCN',
        color='#E74C3C', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(tcn_data.label_fraction,
                tcn_data.f1_mean - tcn_data.f1_std,
                tcn_data.f1_mean + tcn_data.f1_std,
                color='#E74C3C', alpha=0.15)

# Plot Transformer
ax.plot(transformer_data.label_fraction, transformer_data.f1_mean,
        marker='^', label='Transformer',
        color='#6A5ACD', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(transformer_data.label_fraction,
                transformer_data.f1_mean - transformer_data.f1_std,
                transformer_data.f1_mean + transformer_data.f1_std,
                color='#6A5ACD', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Label Efficiency: Supervised Models Comparison', 
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
plt.savefig('../results/supervised_models_comparison.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig('../results/supervised_models_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------
# SSL Linear Models Comparison (including SimCLR)
# --------------------------------------------------
# Filter data for each SSL model with linear classifier
ts2vec_data = data[
    (data.model == 'TS2Vec') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

soft_ts2vec_data = data[
    (data.model == 'SoftTS2Vec') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

tstcc_data = data[
    (data.model == 'TSTCC') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

soft_tstcc_data = data[
    (data.model == 'SoftTSTCC') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

simclr_data = data[
    (data.model == 'SimCLR') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

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

# Plot TSTCC
ax.plot(tstcc_data.label_fraction, tstcc_data.f1_mean,
        marker='^', label='TSTCC',
        color='#6A5ACD', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(tstcc_data.label_fraction,
                tstcc_data.f1_mean - tstcc_data.f1_std,
                tstcc_data.f1_mean + tstcc_data.f1_std,
                color='#6A5ACD', alpha=0.15)

# Plot SoftTSTCC
ax.plot(soft_tstcc_data.label_fraction, soft_tstcc_data.f1_mean,
        marker='D', label='SoftTSTCC',
        color='#2ECC71', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(soft_tstcc_data.label_fraction,
                soft_tstcc_data.f1_mean - soft_tstcc_data.f1_std,
                soft_tstcc_data.f1_mean + soft_tstcc_data.f1_std,
                color='#2ECC71', alpha=0.15)

# Plot SimCLR
ax.plot(simclr_data.label_fraction, simclr_data.f1_mean,
        marker='*', label='SimCLR',
        color='#FFA500', linewidth=2.5, markersize=10,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(simclr_data.label_fraction,
                simclr_data.f1_mean - simclr_data.f1_std,
                simclr_data.f1_mean + simclr_data.f1_std,
                color='#FFA500', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Label Efficiency: SSL Models with Linear Classifier', 
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
plt.savefig('../results/ssl_linear_models_comparison.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig('../results/ssl_linear_models_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# Print statistical comparison
print("\nStatistical Comparison of SSL Models with Linear Classifier")
print("=" * 70)

for fraction in LABEL_FRACTIONS:
    print(f"\nLabel Fraction: {int(fraction*100)}%")
    print("-" * 30)
    
    ts2vec_scores = ts2vec_data[ts2vec_data.label_fraction == fraction]['f1_mean']
    soft_ts2vec_scores = soft_ts2vec_data[soft_ts2vec_data.label_fraction == fraction]['f1_mean']
    tstcc_scores = tstcc_data[tstcc_data.label_fraction == fraction]['f1_mean']
    soft_tstcc_scores = soft_tstcc_data[soft_tstcc_data.label_fraction == fraction]['f1_mean']
    simclr_scores = simclr_data[simclr_data.label_fraction == fraction]['f1_mean']
    
    print(f"TS2Vec:     {ts2vec_scores.mean():.3f} ± {ts2vec_scores.std():.3f}")
    print(f"SoftTS2Vec: {soft_ts2vec_scores.mean():.3f} ± {soft_ts2vec_scores.std():.3f}")
    print(f"TSTCC:      {tstcc_scores.mean():.3f} ± {tstcc_scores.std():.3f}")
    print(f"SoftTSTCC:  {soft_tstcc_scores.mean():.3f} ± {soft_tstcc_scores.std():.3f}")
    print(f"SimCLR:     {simclr_scores.mean():.3f} ± {simclr_scores.std():.3f}")

# --------------------------------------------------
# Supervised vs SSL Models (Linear): Average F1 Score Comparison
# --------------------------------------------------
# Filter and aggregate data
supervised_avg = (
    data[data.group == 'Supervised']
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

ssl_linear_avg = (
    data[(data.group == 'Self-Supervised (Linear)') & 
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
        marker='o', label='Supervised',
        color='#FF6B6B', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(supervised_avg.label_fraction,
                supervised_avg.f1_mean - supervised_avg.f1_std,
                supervised_avg.f1_mean + supervised_avg.f1_std,
                color='#FF6B6B', alpha=0.15)

# Plot SSL Average
ax.plot(ssl_linear_avg.label_fraction, ssl_linear_avg.f1_mean,
        marker='s', label='Self-Supervised (Linear)',
        color='#1E90FF', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_linear_avg.label_fraction,
                ssl_linear_avg.f1_mean - ssl_linear_avg.f1_std,
                ssl_linear_avg.f1_mean + ssl_linear_avg.f1_std,
                color='#1E90FF', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Averaged Supervised vs Self-Supervised Learning with Linear Classifier', 
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
plt.savefig('../results/supervised_vs_ssl_linear.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/supervised_vs_ssl_linear.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()


# --------------------------------------------------
# Supervised vs SSL Models (MLP): Average F1 Score Comparison
# --------------------------------------------------
# Filter and aggregate data
supervised_avg = (
    data[data.group == 'Supervised']
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

ssl_mlp_avg = (
    data[(data.group == 'Self-Supervised (MLP)') & 
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
        marker='o', label='Supervised',
        color='#FF6B6B', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(supervised_avg.label_fraction,
                supervised_avg.f1_mean - supervised_avg.f1_std,
                supervised_avg.f1_mean + supervised_avg.f1_std,
                color='#FF6B6B', alpha=0.15)

# Plot SSL MLP Average
ax.plot(ssl_mlp_avg.label_fraction, ssl_mlp_avg.f1_mean,
        marker='s', label='Self-Supervised (MLP)',
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
ax.set_title('Averaged Supervised vs Self-Supervised Learning with MLP Classifier', 
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
plt.savefig('../results/supervised_vs_ssl_mlp.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/supervised_vs_ssl_mlp.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()

# --------------------------------------------------
# Three-way Comparison: F1-scores Supervised vs SSL-Linear vs SSL-MLP
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
    data[(data.group == 'Self-Supervised (Linear)') & 
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
    data[(data.group == 'Self-Supervised (MLP)') & 
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
        marker='o', label='Supervised',
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
ax.set_title('Supervised vs Self-Supervised Learning Methods', 
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
plt.savefig('../results/f1_three_way_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/f1_three_way_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()


# --------------------------------------------------
# Three-way Comparison: F1-scores Supervised (CNN + TCN) vs SSL-Linear vs SSL-MLP (excluding SimCLR)
# --------------------------------------------------
# Filter and aggregate Supervised models
supervised_avg = (
    data[(data.group == 'Supervised') &
         (data.model != 'Supervised_Transformer')]
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models with Linear classifier
ssl_linear_avg = (
    data[(data.group == 'Self-Supervised (Linear)') & 
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
    data[(data.group == 'Self-Supervised (MLP)') & 
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
        marker='o', label='Supervised (CNN + TCN)',
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
ax.set_title('Supervised vs Self-Supervised Learning Methods', 
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
plt.savefig('../results/f1_three_way_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/f1_three_way_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()

# --------------------------------------------------
# Three-way Comparison: AUC-ROC Supervised vs SSL-Linear vs SSL-MLP
# --------------------------------------------------
# Filter and aggregate Supervised models
supervised_avg = (
    data[data.group == 'Supervised']
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models with Linear classifier
ssl_linear_avg = (
    data[(data.group == 'Self-Supervised (Linear)') & 
         (data.model != 'SimCLR')]
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models with MLP classifier
ssl_mlp_avg = (
    data[(data.group == 'Self-Supervised (MLP)') & 
         (data.model != 'SimCLR')]
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

# Create publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot Supervised Average
ax.plot(supervised_avg.label_fraction, supervised_avg.auc_mean,
        marker='o', label='Supervised',
        color='#FF6B6B', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(supervised_avg.label_fraction,
                supervised_avg.auc_mean - supervised_avg.auc_std,
                supervised_avg.auc_mean + supervised_avg.auc_std,
                color='#FF6B6B', alpha=0.15)

# Plot SSL Linear Average
ax.plot(ssl_linear_avg.label_fraction, ssl_linear_avg.auc_mean,
        marker='s', label='Self-Supervised (Linear)',
        color='#1E90FF', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_linear_avg.label_fraction,
                ssl_linear_avg.auc_mean - ssl_linear_avg.auc_std,
                ssl_linear_avg.auc_mean + ssl_linear_avg.auc_std,
                color='#1E90FF', alpha=0.15)

# Plot SSL MLP Average
ax.plot(ssl_mlp_avg.label_fraction, ssl_mlp_avg.auc_mean,
        marker='^', label='Self-Supervised (MLP)',
        color='#6A5ACD', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_mlp_avg.label_fraction,
                ssl_mlp_avg.auc_mean - ssl_mlp_avg.auc_std,
                ssl_mlp_avg.auc_mean + ssl_mlp_avg.auc_std,
                color='#6A5ACD', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
ax.set_title('Supervised vs Self-Supervised Learning Methods', 
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
plt.savefig('../results/auc_roc_three_way_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/auc_roc_three_way_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()

# --------------------------------------------------
# Three-way Comparison: AUC-ROC Supervised (CNN + TCN) vs SSL-Linear vs SSL-MLP (excluding SimCLR)
# --------------------------------------------------
# Filter and aggregate Supervised models
supervised_avg = (
    data[(data.group == 'Supervised') &
         (data.model != 'Supervised_Transformer')]
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models with Linear classifier
ssl_linear_avg = (
    data[(data.group == 'Self-Supervised (Linear)') & 
         (data.model != 'SimCLR')]
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models with MLP classifier
ssl_mlp_avg = (
    data[(data.group == 'Self-Supervised (MLP)') & 
         (data.model != 'SimCLR')]
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

# Create publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot Supervised Average
ax.plot(supervised_avg.label_fraction, supervised_avg.auc_mean,
        marker='o', label='Supervised (CNN + TCN)',
        color='#FF6B6B', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(supervised_avg.label_fraction,
                supervised_avg.auc_mean - supervised_avg.auc_std,
                supervised_avg.auc_mean + supervised_avg.auc_std,
                color='#FF6B6B', alpha=0.15)

# Plot SSL Linear Average
ax.plot(ssl_linear_avg.label_fraction, ssl_linear_avg.auc_mean,
        marker='s', label='Self-Supervised (Linear)',
        color='#1E90FF', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_linear_avg.label_fraction,
                ssl_linear_avg.auc_mean - ssl_linear_avg.auc_std,
                ssl_linear_avg.auc_mean + ssl_linear_avg.auc_std,
                color='#1E90FF', alpha=0.15)

# Plot SSL MLP Average
ax.plot(ssl_mlp_avg.label_fraction, ssl_mlp_avg.auc_mean,
        marker='^', label='Self-Supervised (MLP)',
        color='#6A5ACD', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_mlp_avg.label_fraction,
                ssl_mlp_avg.auc_mean - ssl_mlp_avg.auc_std,
                ssl_mlp_avg.auc_mean + ssl_mlp_avg.auc_std,
                color='#6A5ACD', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
ax.set_title('Supervised vs Self-Supervised Learning Methods', 
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
plt.savefig('../results/auc_roc_three_way_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/auc_roc_three_way_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()

# --------------------------------------------------
# Four-way Comparison: F1-scores Supervised (CNN + TCN) vs Transformer vs SSL-Linear vs SSL-MLP (excluding SimCLR)
# --------------------------------------------------
# Filter and aggregate Supervised models (CNN + TCN)
supervised_avg = (
    data[(data.group == 'Supervised') &
         (data.model != 'Supervised_Transformer')]
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

# Get Transformer data separately
transformer_avg = (
    data[data.model == 'Supervised_Transformer']
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models (Linear and MLP) - keeping existing code
ssl_linear_avg = (
    data[(data.group == 'Self-Supervised (Linear)') & 
         (data.model != 'SimCLR')]
    .groupby('label_fraction')
    .agg(
        f1_mean=('f1_mean', 'mean'),
        f1_std=('f1_mean', 'std')
    )
    .reset_index()
)

ssl_mlp_avg = (
    data[(data.group == 'Self-Supervised (MLP)') & 
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

# Plot Supervised Average (CNN + TCN)
ax.plot(supervised_avg.label_fraction, supervised_avg.f1_mean,
        marker='o', label='Supervised (CNN + TCN)',
        color='#FF6B6B', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(supervised_avg.label_fraction,
                supervised_avg.f1_mean - supervised_avg.f1_std,
                supervised_avg.f1_mean + supervised_avg.f1_std,
                color='#FF6B6B', alpha=0.15)

# Plot Transformer separately
ax.plot(transformer_avg.label_fraction, transformer_avg.f1_mean,
        marker='D', label='Transformer',
        color='#2ECC71', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(transformer_avg.label_fraction,
                transformer_avg.f1_mean - transformer_avg.f1_std,
                transformer_avg.f1_mean + transformer_avg.f1_std,
                color='#2ECC71', alpha=0.15)

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
ax.set_title('Comparison of Learning Methods', 
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
         shadow=True,
         fontsize=11,
         loc='lower right')


plt.tight_layout()

plt.savefig('../results/f1_four_way_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/f1_four_way_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()


# --------------------------------------------------
# Four-way Comparison: AUC-ROC scores Supervised (CNN + TCN) vs Transformer vs SSL-Linear vs SSL-MLP (excluding SimCLR)
# --------------------------------------------------
# Filter and aggregate Supervised models (CNN + TCN)
supervised_avg = (
    data[(data.group == 'Supervised') &
         (data.model != 'Supervised_Transformer')]
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

# Get Transformer data separately
transformer_avg = (
    data[data.model == 'Supervised_Transformer']
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

# Filter and aggregate SSL models (Linear and MLP) - keeping existing code
ssl_linear_avg = (
    data[(data.group == 'Self-Supervised (Linear)') & 
         (data.model != 'SimCLR')]
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

ssl_mlp_avg = (
    data[(data.group == 'Self-Supervised (MLP)') & 
         (data.model != 'SimCLR')]
    .groupby('label_fraction')
    .agg(
        auc_mean=('auc_mean', 'mean'),
        auc_std=('auc_mean', 'std')
    )
    .reset_index()
)

# Create publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot Supervised Average (CNN + TCN)
ax.plot(supervised_avg.label_fraction, supervised_avg.auc_mean,
        marker='o', label='Supervised (CNN + TCN)',
        color='#FF6B6B', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(supervised_avg.label_fraction,
                supervised_avg.auc_mean - supervised_avg.auc_std,
                supervised_avg.auc_mean + supervised_avg.auc_std,
                color='#FF6B6B', alpha=0.15)

# Plot Transformer separately
ax.plot(transformer_avg.label_fraction, transformer_avg.auc_mean,
        marker='D', label='Transformer',
        color='#2ECC71', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(transformer_avg.label_fraction,
                transformer_avg.auc_mean - transformer_avg.auc_std,
                transformer_avg.auc_mean + transformer_avg.auc_std,
                color='#2ECC71', alpha=0.15)

# Plot SSL Linear Average
ax.plot(ssl_linear_avg.label_fraction, ssl_linear_avg.auc_mean,
        marker='s', label='Self-Supervised (Linear)',
        color='#1E90FF', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_linear_avg.label_fraction,
                ssl_linear_avg.auc_mean - ssl_linear_avg.auc_std,
                ssl_linear_avg.auc_mean + ssl_linear_avg.auc_std,
                color='#1E90FF', alpha=0.15)

# Plot SSL MLP Average
ax.plot(ssl_mlp_avg.label_fraction, ssl_mlp_avg.auc_mean,
        marker='^', label='Self-Supervised (MLP)',
        color='#6A5ACD', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ssl_mlp_avg.label_fraction,
                ssl_mlp_avg.auc_mean - ssl_mlp_avg.auc_std,
                ssl_mlp_avg.auc_mean + ssl_mlp_avg.auc_std,
                color='#6A5ACD', alpha=0.15)

# Styling
ax.set_xscale('log')
ax.set_xticks(LABEL_FRACTIONS)
ax.set_xticklabels([f'{int(f*100)}%' for f in LABEL_FRACTIONS])
ax.set_xlabel('Proportion of Labeled Training Data', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Comparison of Learning Methods', 
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
         shadow=True,
         fontsize=11,
         loc='lower right')


plt.tight_layout()

plt.savefig('../results/f1_four_way_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/f1_four_way_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()


import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, pathlib, os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# ------------------------------------------------------------------
# 3-way comparison of F1-scores: Supervised (CNN + TCN) vs SSL-Linear vs SSL-MLP (excluding SimCLR)
# Pooled Bootstrap confidence intervals
# ------------------------------------------------------------------
root = pathlib.Path("../results/confidence_intervals")
sup_file   = root / "supervised_ci.csv"
lin_file   = root / "ssl_linear_ci.csv"
mlp_file   = root / "ssl_mlp_ci.csv"

sup_df  = pd.read_csv(sup_file )
lin_df  = pd.read_csv(lin_file )
mlp_df  = pd.read_csv(mlp_file )

# drop SimCLR everywhere
lin_df  = lin_df[lin_df.model != "SimCLR"]
mlp_df  = mlp_df[mlp_df.model != "SimCLR"]


def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Calculate bootstrapped confidence interval for the mean."""
    boot_means = []
    for _ in range(n_boot):
        # Sample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    # Calculate percentile intervals
    lower = np.percentile(boot_means, ((1-ci)/2) * 100)
    upper = np.percentile(boot_means, (1-(1-ci)/2) * 100)
    return lower, upper

def pooled_ci(df):
    """Compute pooled mean and bootstrapped CIs for each label fraction."""
    results = []
    
    for frac, group in df.groupby("label_fraction"):
        scores = group["mean"].values  # Get all F1 scores for this fraction
        mean_score = np.mean(scores)
        ci_lower, ci_upper = bootstrap_ci(scores, n_boot=10000, ci=0.95)
        
        results.append({
            "label_fraction": frac,
            "mean": mean_score,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
        print(f"Processed label fraction: {frac}, Mean: {mean_score:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return pd.DataFrame(results).sort_values("label_fraction")

# Supervised-group (CNN + TCN)
print("Supervised Group:")
print("" + "="*30)
cnn_tcn = sup_df[sup_df.model.isin(["CNN", "TCN"])]
sup_grp = pooled_ci(cnn_tcn)

# SSL groups
print("SSL Linear Group:")
print("" + "="*30)
ssl_lin  = pooled_ci(lin_df)
print("SSL MLP Group:")
print("" + "="*30)
ssl_mlp  = pooled_ci(mlp_df)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Modified draw function
def draw(group_df, color, marker, label):
    ax.plot(group_df.label_fraction, group_df["mean"], marker=marker,
            color=color, linewidth=2.5, markersize=7,
            markerfacecolor='white', markeredgewidth=1.8,
            label=label)
    ax.fill_between(group_df.label_fraction, 
                    group_df["ci_lower"], 
                    group_df["ci_upper"],
                    color=color, alpha=0.15)

draw(ssl_mlp  , "#9467BD", "^", "SSL MLP")
draw(ssl_lin  , "#1F77B4", "s", "SSL Linear")
draw(sup_grp , "#FF6B6B", "o", "Supervised - CNN + TCN")

# Styling
ax.set_xscale('log')
fractions = [0.01, 0.05, 0.10, 0.50, 1.00]
ax.set_xticks(fractions)
ax.set_xticklabels([f"{int(f*100)}%" for f in fractions], fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12) 
ax.set_xlabel("Proportion of labeled training data", fontweight="bold", fontsize=12)
ax.set_ylabel("MF1 score", fontweight="bold", fontsize=12)
ax.set_title("Label-efficiency comparison: Supervised vs SSL", fontweight="bold", pad=18, fontsize=14)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, loc='lower right')

ax.set_facecolor("#FAFAFA")
for spine in ("top", "right"): ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("../results/label_efficiency_ci.pdf", dpi=300)
plt.savefig("../results/label_efficiency_ci.png", dpi=300)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# ------------------------------------------------------------------
# Load data from new confidence interval files
# ------------------------------------------------------------------
root = pathlib.Path("../results/confidence_intervals")
sup_file = root / "supervised_ci.csv"
lin_file = root / "ssl_linear_ci.csv"
mlp_file = root / "ssl_mlp_ci.csv"

sup_df = pd.read_csv(sup_file)
lin_df = pd.read_csv(lin_file)
mlp_df = pd.read_csv(mlp_file)

# Drop SimCLR as requested
lin_df = lin_df[lin_df.model != "SimCLR"]
mlp_df = mlp_df[mlp_df.model != "SimCLR"]

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Helper function to calculate bootstrapped confidence interval."""
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 - (1 - ci) / 2) * 100)
    return lower, upper

def pooled_ci(df, metric_prefix="f1"):
    """
    computes pooled mean and bootstrapped CIs for each label fraction.
    It pools the mean scores from the models in the group and then runs bootstrap.
    """
    results = []
    mean_col = f"{metric_prefix}_mean"
    
    for frac, group in df.groupby("label_fraction"):
        # Pool the mean scores from all models in the group for this fraction
        scores = group[mean_col].dropna().values
        if len(scores) == 0:
            continue
            
        # Calculate the overall mean and the bootstrapped CI on the pooled scores
        mean_score = np.mean(scores)
        ci_lower, ci_upper = bootstrap_ci(scores)
        
        results.append({
            "label_fraction": frac,
            "mean": mean_score,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
        
    return pd.DataFrame(results).sort_values("label_fraction")

def draw(ax, group_df, color, marker, label):
    """Generic plotting function for a model group."""
    ax.plot(group_df.label_fraction, group_df["mean"], marker=marker,
            color=color, linewidth=2.5, markersize=7,
            markerfacecolor='white', markeredgewidth=1.8,
            label=label)
    ax.fill_between(group_df.label_fraction, 
                    group_df["ci_lower"], 
                    group_df["ci_upper"],
                    color=color, alpha=0.15)

# --- Define metrics to plot ---
metrics_to_plot = {
    "f1": "MF1",
    "auc": "AUC-ROC",
    "accuracy": "Accuracy",
    "pr_auc": "PR-AUC"
}

# --- Loop through metrics and generate a plot for each ---
for metric_prefix, metric_name in metrics_to_plot.items():
    
    print(f"\n--- Generating plot for {metric_name} score ---")
    
    # --- Process data for the current metric using the corrected pooled_ci function ---
    # Supervised-group (CNN + TCN)
    cnn_tcn = sup_df[sup_df.model.isin(["CNN", "TCN"])]
    sup_grp = pooled_ci(cnn_tcn, metric_prefix=metric_prefix)

    # SSL groups
    ssl_lin = pooled_ci(lin_df, metric_prefix=metric_prefix)
    ssl_mlp = pooled_ci(mlp_df, metric_prefix=metric_prefix)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Draw the three groups
    draw(ax, ssl_mlp, "#9467BD", "^", "SSL MLP")
    draw(ax, ssl_lin, "#1F77B4", "s", "SSL Linear")
    draw(ax, sup_grp, "#FF6B6B", "o", "Supervised - CNN + TCN")

    # Styling
    ax.set_xscale('log')
    fractions = [0.01, 0.05, 0.10, 0.50, 1.00]
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{int(f*100)}%" for f in fractions], fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12) 
    ax.set_xlabel("Proportion of labeled training data", fontweight="bold", fontsize=12)
    ax.set_ylabel(f"{metric_name} score", fontweight="bold", fontsize=12)
    ax.set_title("Label-efficiency comparison: Supervised vs SSL", fontweight="bold", pad=18, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, loc='lower right')

    ax.set_facecolor("#FAFAFA")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    
    # Save the plot with a metric-specific name
    output_path = root.parent / f"label_efficiency_{metric_prefix}_ci"
    plt.savefig(f"{output_path}.pdf", dpi=300)
    plt.savefig(f"{output_path}.png", dpi=300)
    print(f"Saved plot to {output_path}.png")
    
    plt.show()