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
# TSTCC vs SoftTSTCC Comparison (Linear Classifier)
# --------------------------------------------------
# Filter data for the two models with LinearClassifier only
tstcc_data = data[
    (data.model == 'TSTCC') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

soft_tstcc_data = data[
    (data.model == 'SoftTSTCC') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

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
ax.set_title('Label Efficiency: TSTCC vs SoftTSTCC (Linear Classifier)', 
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
plt.savefig('../results/tstcc_linear_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('../results/tstcc_linear_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.show()

# --------------------------------------------------
# TS2Vec vs SoftTS2Vec Comparison (Linear Classifier)
# --------------------------------------------------
# Filter data for both models with LinearClassifier
ts2vec_data = data[
    (data.model == 'TS2Vec') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

soft_ts2vec_data = data[
    (data.model == 'SoftTS2Vec') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

# Create plot
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
ax.set_title('TS2Vec vs SoftTS2Vec with Linear Classifier', 
             fontsize=14, fontweight='bold', pad=20)

ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('#FAFAFA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.legend(frameon=True, fancybox=True, shadow=False, fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('../results/ts2vec_vs_soft_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../results/ts2vec_vs_soft_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------
# TS2Vec: Linear vs MLP Classifier Comparison
# --------------------------------------------------
# Filter data for TS2Vec with different classifiers
ts2vec_linear = data[
    (data.model == 'TS2Vec') & 
    (data.classifier_model == 'LinearClassifier')
].sort_values('label_fraction')

ts2vec_mlp = data[
    (data.model == 'TS2Vec') & 
    (data.classifier_model == 'MLPClassifier')
].sort_values('label_fraction')

# Create plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot TS2Vec with Linear classifier
ax.plot(ts2vec_linear.label_fraction, ts2vec_linear.f1_mean,
        marker='o', label='TS2Vec (Linear)',
        color='#2E86C1', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(ts2vec_linear.label_fraction,
                ts2vec_linear.f1_mean - ts2vec_linear.f1_std,
                ts2vec_linear.f1_mean + ts2vec_linear.f1_std,
                color='#2E86C1', alpha=0.15)

# Plot TS2Vec with MLP classifier
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
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('TS2Vec: Linear vs MLP Classifier Performance', 
             fontsize=14, fontweight='bold', pad=20)

ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('#FAFAFA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.legend(frameon=True, fancybox=True, shadow=False, fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('../results/ts2vec_classifier_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../results/ts2vec_classifier_comparison.png', dpi=300, bbox_inches='tight')
plt.show()