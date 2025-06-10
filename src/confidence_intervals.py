import numpy as np
import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

# Bootstrapped 95% CI function
def bootstrap_ci(data, n_boot=10000, ci=95):
    boot_means = []
    data = np.array(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), (lower, upper)

# Folder config
ROOT = Path("/Users/carlitos/Desktop/ECG-Project/results")
SSL_MODELS = {"TS2Vec", "SoftTS2Vec", "TSTCC", "SoftTSTCC", "SimCLR"}
SUPERVISED_MODELS = ["Supervised_CNN", "Supervised_TCN", "Supervised_Transformer"]

# SUPERVISED VS SSL LINEAR VS SSL MLP
# Aggregate F1 scores by group and label fraction
def scores_by_fraction(root: Path):
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(list))

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        model_type = model_dir.name

        if model_type.startswith("Supervised"):
            group = "Supervised"
        elif model_type in SSL_MODELS:
            group = "SSL"
        else:
            continue

        for sub in model_dir.iterdir():
            if not sub.is_dir():
                continue
            m = pattern_label.match(sub.name)
            if not m:
                continue
            frac = float(m.group(1))
            clf_type = sub.name.split("_")[0]

            if group == "Supervised":
                cat = "Supervised"
            elif clf_type == "LinearClassifier":
                cat = "SSL_Linear"
            elif clf_type == "MLPClassifier":
                cat = "SSL_MLP"
            else:
                continue

            csv_path = sub / "individual_runs.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            results[frac][cat].extend(df["test_f1"].values)

    return results

results = scores_by_fraction(ROOT)

print("\nBootstrapped 95% CI for F1 scores per label fraction:")
print("-" * 70)
for frac in sorted(results.keys()):
    print(f"\nLabel Fraction = {frac:.2f}")
    for group in ["Supervised", "SSL_Linear", "SSL_MLP"]:
        scores = results[frac].get(group, [])
        if not scores:
            continue
        mean, (lower, upper) = bootstrap_ci(scores)
        print(f"{group:12} → Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
        
# SUPERVISED MODELS F1 SCORES BY LABEL FRACTION
def scores_by_supervised_model(root: Path):
    """Collect F1 scores for each supervised model by label fraction."""
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(list))

    for model_name in SUPERVISED_MODELS:
        model_dir = root / model_name
        if not model_dir.is_dir():
            continue

        for sub in model_dir.iterdir():
            if not sub.is_dir():
                continue
            m = pattern_label.match(sub.name)
            if not m:
                continue
            
            frac = float(m.group(1))
            csv_path = sub / "individual_runs.csv"
            
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            results[frac][model_name].extend(df["test_f1"].values)

    return results

# Get results for supervised models
supervised_results = scores_by_supervised_model(ROOT)

# Print confidence intervals
print("\nBootstrapped 95% CI for Supervised Models F1 scores:")
print("=" * 70)

for frac in sorted(supervised_results.keys()):
    print(f"\nLabel Fraction = {frac:.2f}")
    print("-" * 40)
    
    for model in SUPERVISED_MODELS:
        scores = supervised_results[frac].get(model, [])
        if not scores:
            continue
            
        mean, (lower, upper) = bootstrap_ci(scores)
        ci_width = upper - lower
        
        print(f"{model.replace('Supervised_', ''):12} → "
              f"Mean: {mean:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}] "
              f"(±{ci_width/2:.3f})")

# SSL MODELS WITH LINEAR CLASSIFIER F1 SCORES BY LABEL FRACTION
def scores_by_ssl_linear(root: Path):
    """Collect F1 scores for each SSL model with linear classifier by label fraction."""
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(list))

    for model_name in SSL_MODELS:
        model_dir = root / model_name
        if not model_dir.is_dir():
            continue

        # Look for LinearClassifier results
        for sub in model_dir.iterdir():
            if not sub.is_dir() or not sub.name.startswith("LinearClassifier"):
                continue
                
            m = pattern_label.match(sub.name)
            if not m:
                continue
            
            frac = float(m.group(1))
            csv_path = sub / "individual_runs.csv"
            
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            results[frac][model_name].extend(df["test_f1"].values)

    return results

# Get results for SSL Linear models
ssl_linear_results = scores_by_ssl_linear(ROOT)

# Print confidence intervals
print("\nBootstrapped 95% CI for SSL Models with Linear Classifier:")
print("=" * 70)

for frac in sorted(ssl_linear_results.keys()):
    print(f"\nLabel Fraction = {frac:.2f}")
    print("-" * 40)
    
    for model in SSL_MODELS:
        scores = ssl_linear_results[frac].get(model, [])
        if not scores:
            continue
            
        mean, (lower, upper) = bootstrap_ci(scores)
        ci_width = upper - lower
        
        print(f"{model:12} → "
              f"Mean: {mean:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}] "
              f"(±{ci_width/2:.3f})")

# SSL MODELS WITH MLP CLASSIFIER F1 SCORES BY LABEL FRACTION
def scores_by_ssl_mlp(root: Path):
    """Collect F1 scores for each SSL model with MLP classifier by label fraction."""
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(list))

    for model_name in SSL_MODELS:
        model_dir = root / model_name
        if not model_dir.is_dir():
            continue

        # Look for MLPClassifier results
        for sub in model_dir.iterdir():
            if not sub.is_dir() or not sub.name.startswith("MLPClassifier"):
                continue
                
            m = pattern_label.match(sub.name)
            if not m:
                continue
            
            frac = float(m.group(1))
            csv_path = sub / "individual_runs.csv"
            
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            results[frac][model_name].extend(df["test_f1"].values)

    return results

# Get results for SSL MLP models
ssl_mlp_results = scores_by_ssl_mlp(ROOT)

# Print confidence intervals
print("\nBootstrapped 95% CI for SSL Models with MLP Classifier:")
print("=" * 70)

for frac in sorted(ssl_mlp_results.keys()):
    print(f"\nLabel Fraction = {frac:.2f}")
    print("-" * 40)
    
    for model in SSL_MODELS:
        scores = ssl_mlp_results[frac].get(model, [])
        if not scores:
            continue
            
        mean, (lower, upper) = bootstrap_ci(scores)
        ci_width = upper - lower
        
        print(f"{model:12} → "
              f"Mean: {mean:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}] "
              f"(±{ci_width/2:.3f})")