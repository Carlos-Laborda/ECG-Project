import numpy as np
import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

# Bootstrapped 95% CI function
def bootstrap_ci(data, n_boot=10000, ci=95):
    """Calculate bootstrapped confidence interval for the mean."""
    boot_means = []
    data = np.array(data)
    if len(data) == 0:
        return np.nan, (np.nan, np.nan)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), (lower, upper)

# Configuration
ROOT = Path("/Users/carlitos/Desktop/ECG-Project/results")
SSL_MODELS = {"TS2Vec", "SoftTS2Vec", "TSTCC", "SoftTSTCC", "SimCLR"}
SUPERVISED_MODELS = ["Supervised_CNN", "Supervised_TCN", "Supervised_Transformer"]

# Define metrics to extract
METRIC_MAPPING = {
    "f1": ["test_f1"],
    "auc": ["test_auroc", "test_auc_roc"],
    "accuracy": ["test_accuracy"],
    "pr_auc": ["test_pr_auc"]
}

def save_results_to_csv(supervised_results, ssl_linear_results, ssl_mlp_results):
    """Save confidence interval results for all metrics to CSV files."""
    results_dir = Path("../results/confidence_intervals")
    results_dir.mkdir(exist_ok=True)
    
    def create_df(results_dict, models):
        records = []
        all_fractions = set()
        for metric_data in results_dict.values():
            all_fractions.update(metric_data.keys())

        for frac in sorted(list(all_fractions)):
            for model in models:
                has_data = any(model in results_dict[metric].get(frac, {}) for metric in METRIC_MAPPING.keys())
                if not has_data:
                    continue

                record = {'label_fraction': frac, 'model': model.replace('Supervised_', '')}
                for metric_key in METRIC_MAPPING.keys():
                    scores = results_dict.get(metric_key, {}).get(frac, {}).get(model, [])
                    mean, (lower, upper) = bootstrap_ci(scores)
                    record[f'{metric_key}_mean'] = mean
                    record[f'{metric_key}_ci_lower'] = lower
                    record[f'{metric_key}_ci_upper'] = upper
                    record[f'{metric_key}_ci_width'] = upper - lower if not np.isnan(upper) else np.nan
                records.append(record)
        return pd.DataFrame(records)
    
    supervised_df = create_df(supervised_results, SUPERVISED_MODELS)
    supervised_df.to_csv(results_dir / 'supervised_ci.csv', index=False)
    
    ssl_linear_df = create_df(ssl_linear_results, SSL_MODELS)
    ssl_linear_df.to_csv(results_dir / 'ssl_linear_ci.csv', index=False)
    
    ssl_mlp_df = create_df(ssl_mlp_results, SSL_MODELS)
    ssl_mlp_df.to_csv(results_dir / 'ssl_mlp_ci.csv', index=False)
    
    print("\nResults for all metrics saved to:")
    print(f"- {results_dir}/supervised_ci.csv")
    print(f"- {results_dir}/ssl_linear_ci.csv")
    print(f"- {results_dir}/ssl_mlp_ci.csv")

def get_scores_by_model(root: Path, model_list: list, prefix_filter: str = ""):
    """
    Generic function to collect scores for specified metrics for a list of models.
    Can filter subdirectories by a prefix (e.g., 'LinearClassifier').
    """
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model_name in model_list:
        model_dir = root / model_name
        if not model_dir.is_dir():
            continue

        for sub in model_dir.iterdir():
            if not sub.is_dir() or (prefix_filter and not sub.name.startswith(prefix_filter)):
                continue
            
            m = pattern_label.match(sub.name)
            if not m:
                continue
            
            frac = float(m.group(1))
            csv_path = sub / "individual_runs.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            for metric_key, metric_cols in METRIC_MAPPING.items():
                actual_col = next((col for col in metric_cols if col in df.columns), None)
                if actual_col:
                    results[metric_key][frac][model_name].extend(df[actual_col].dropna().values)
    return results

def print_ci_results(results_dict, model_list, title):
    """Helper function to print CI results for all metrics."""
    for metric_key in METRIC_MAPPING.keys():
        print(f"\n--- {title} - METRIC: {metric_key.upper()} ---")
        print("=" * 70)
        metric_results = results_dict.get(metric_key, {})
        if not metric_results:
            print("No data found for this metric.")
            continue
        
        for frac in sorted(metric_results.keys()):
            print(f"\nLabel Fraction = {frac:.2f}")
            print("-" * 40)
            for model in model_list:
                scores = metric_results[frac].get(model, [])
                if not scores:
                    continue
                mean, (lower, upper) = bootstrap_ci(scores)
                ci_width = upper - lower
                print(f"{model.replace('Supervised_', ''):<12} → "
                      f"Mean: {mean:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}] "
                      f"(±{ci_width/2:.3f})")

# Main Execution
# 1. Collect scores for all model types and all metrics
supervised_results = get_scores_by_model(ROOT, SUPERVISED_MODELS)
ssl_linear_results = get_scores_by_model(ROOT, SSL_MODELS, prefix_filter="LinearClassifier")
ssl_mlp_results = get_scores_by_model(ROOT, SSL_MODELS, prefix_filter="MLPClassifier")

# 2. Print all results to the console
print_ci_results(supervised_results, SUPERVISED_MODELS, "Supervised Models")
print_ci_results(ssl_linear_results, SSL_MODELS, "SSL Models with Linear Classifier")
print_ci_results(ssl_mlp_results, SSL_MODELS, "SSL Models with MLP Classifier")

# 3. Save all results to CSV files
save_results_to_csv(supervised_results, ssl_linear_results, ssl_mlp_results)