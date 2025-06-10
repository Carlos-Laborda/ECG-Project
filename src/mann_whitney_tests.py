"""
Compare F1 scores of various model groups across label fractions.

Performs Welch’s t-test and checks for normality using Shapiro–Wilk.
Expected folder structure:
└── /Users/carlitos/Desktop/ECG-Project/results/
    ├── Supervised_CNN/Supervised_0.1/individual_runs.csv
    ├── TS2Vec/LinearClassifier_0.1/individual_runs.csv
    ├── TS2Vec/MLPClassifier_0.1/individual_runs.csv
    └── ...

"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

# ---------- USER CONFIG ----------
ROOT = Path("/Users/carlitos/Desktop/ECG-Project/results")
NORM_TEST_ALPHA = 0.05
# ----------------------------------

SSL_MODELS = {
    "TS2Vec", 
    "SoftTS2Vec", 
    "TSTCC", 
    "SoftTSTCC", 
    #"SimCLR"
    }

SUPERVISED_MODELS = [
    "Supervised_CNN", 
    "Supervised_TCN", 
    #"Supervised_Transformer"
    ]

TRANSFORMER_MODEL = "Supervised_Transformer"

def find_runs_by_comparison(root: Path):
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    data = defaultdict(lambda: defaultdict(list))

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        model_type = model_dir.name

        if model_type == TRANSFORMER_MODEL:
            group = "Transformer"
        elif model_type.startswith("Supervised"):
            if model_type not in SUPERVISED_MODELS:
                continue
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
            label_frac = float(m.group(1))
            clf_type = sub.name.split("_")[0]

            if group == "Transformer":
                cat = "Transformer"
            elif group == "Supervised":
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
            for f1 in df["test_f1"].values:
                data[label_frac][cat].append(f1)
            
            # # Check for both possible AUC metric names
            # if "test_auc_roc" in df.columns:
            #     auc_values = df["test_auc_roc"].values
            # elif "test_auroc" in df.columns:
            #     auc_values = df["test_auroc"].values
            # else:
            #     continue  # Skip if neither metric is found
                
            # for auc in auc_values:
            #     data[label_frac][cat].append(auc)

    return data

def normality_report(values):
    if len(values) < 3:
        return False, "n/a"
    stat, p = shapiro(values)
    return p > NORM_TEST_ALPHA, f"{p:.4f}"
        
def cliffs_delta(a, b):
    """Compute Cliff's Delta for two independent samples."""
    n1, n2 = len(a), len(b)
    more = sum(x > y for x in a for y in b)
    less = sum(x < y for x in a for y in b)
    delta = (more - less) / (n1 * n2)
    return delta

def cliffs_magnitude(d):
    d_abs = abs(d)
    if d_abs < 0.147:
        return "negligible"
    elif d_abs < 0.33:
        return "small"
    elif d_abs < 0.474:
        return "medium"
    else:
        return "large"

def mannwhitney_comparison(results, group1, group2):
    print(f"\nMann–Whitney U test with Cliff's Delta: {group1} vs {group2}")
    print("-" * 95)
    print(f"{'Frac':>5} | N_1  N_2 | U-stat   p-value   sig | Cliff’s d   Magnitude")
    print("-" * 95)
    for frac in sorted(results):
        v1 = results[frac].get(group1, [])
        v2 = results[frac].get(group2, [])
        if len(v1) < 2 or len(v2) < 2:
            continue
        u_stat, p_val = mannwhitneyu(v1, v2, alternative='two-sided')
        sig = "*" if p_val < 0.05 else ""
        d = cliffs_delta(v1, v2)
        mag = cliffs_magnitude(d)
        print(f"{frac:5.2f} | {len(v1):^4} {len(v2):^4} | {u_stat:7.3f}  {p_val:8.4f}  {sig} | {d:10.3f}  {mag:>9}")


all_results = find_runs_by_comparison(ROOT)

# Original comparisons
print("\n=== Original Comparisons ===")
mannwhitney_comparison(all_results, "Supervised", "SSL_Linear")
mannwhitney_comparison(all_results, "SSL_Linear", "SSL_MLP")
mannwhitney_comparison(all_results, "Supervised", "SSL_MLP")

# New comparisons with Transformer
print("\n=== Transformer Comparisons ===")
mannwhitney_comparison(all_results, "Transformer", "Supervised")
mannwhitney_comparison(all_results, "Transformer", "SSL_Linear")
mannwhitney_comparison(all_results, "Transformer", "SSL_MLP")


# COMPARISON OF INDIVIDUAL SSL MODELS
SSL_MODELS = {
    "TS2Vec", 
    "SoftTS2Vec", 
    "TSTCC", 
    "SoftTSTCC", 
    "SimCLR"
    }
# Load scores for individual SSL models
def get_ssl_model_scores(root: Path):
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model_dir in root.iterdir():
        if not model_dir.is_dir() or model_dir.name not in SSL_MODELS:
            continue
        model_name = model_dir.name

        for sub in model_dir.iterdir():
            if not sub.is_dir():
                continue
            m = pattern_label.match(sub.name)
            if not m:
                continue
            frac = float(m.group(1))
            clf_type = sub.name.split("_")[0]
            clf = "Linear" if clf_type == "LinearClassifier" else "MLP" if clf_type == "MLPClassifier" else None
            if clf is None:
                continue

            csv_path = sub / "individual_runs.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            results[frac][clf][model_name].extend(df["test_f1"].values)

    return results

# Perform all pairwise comparisons
def compare_ssl_models(results):
    for frac in sorted(results.keys()):
        for clf in ["Linear", "MLP"]:
            print(f"\nLabel Fraction = {frac:.2f} | Classifier = {clf}")
            models = list(results[frac][clf].keys())
            for m1, m2 in combinations(models, 2):
                v1 = results[frac][clf][m1]
                v2 = results[frac][clf][m2]
                if len(v1) < 2 or len(v2) < 2:
                    continue
                u_stat, p_val = mannwhitneyu(v1, v2, alternative="two-sided")
                d = cliffs_delta(v1, v2)
                mag = cliffs_magnitude(d)
                sig = "*" if p_val < 0.05 else ""
                print(f"{m1:12} vs {m2:12} | p = {p_val:.4f} | δ = {d:.3f} ({mag}) {sig}")


ssl_results = get_ssl_model_scores(ROOT)
compare_ssl_models(ssl_results)


# COMPARISON OF SUPERVISED MODELS
SUPERVISED_MODELS = [
    "Supervised_CNN", 
    "Supervised_TCN", 
    "Supervised_Transformer"
    ]
# Load F1 scores
def get_supervised_scores(root: Path):
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(list))

    for model_dir in root.iterdir():
        if not model_dir.is_dir() or model_dir.name not in SUPERVISED_MODELS:
            continue
        model_name = model_dir.name.replace("Supervised_", "")

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

# Pairwise comparisons
def compare_supervised_models(results):
    for frac in sorted(results.keys()):
        print(f"\nLabel Fraction = {frac:.2f}")
        models = list(results[frac].keys())
        for m1, m2 in combinations(models, 2):
            v1 = results[frac][m1]
            v2 = results[frac][m2]
            if len(v1) < 2 or len(v2) < 2:
                continue

            u_stat, p_val = mannwhitneyu(v1, v2, alternative="two-sided")
            d = cliffs_delta(v1, v2)
            mag = cliffs_magnitude(d)
            sig = "*" if p_val < 0.05 else ""
            print(f"{m1:10} vs {m2:10} | p = {p_val:.4f} | δ = {d:.3f} ({mag}) {sig}")


sup_results = get_supervised_scores(ROOT)
compare_supervised_models(sup_results)