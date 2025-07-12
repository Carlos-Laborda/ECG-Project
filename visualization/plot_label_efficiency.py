import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# ------------------------------------------------------------------
# Load data from confidence interval files
# ------------------------------------------------------------------
root = pathlib.Path("../results/confidence_intervals")
sup_file = root / "supervised_ci.csv"
lin_file = root / "ssl_linear_ci.csv"
mlp_file = root / "ssl_mlp_ci.csv"

sup_df = pd.read_csv(sup_file)
lin_df = pd.read_csv(lin_file)
mlp_df = pd.read_csv(mlp_file)

# Drop SimCLR
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

# Loop through metrics and generate a plot for each
for metric_prefix, metric_name in metrics_to_plot.items():
    
    print(f"\n--- Generating plot for {metric_name} score ---")
    
    # Process data for the current metric using the pooled_ci function
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