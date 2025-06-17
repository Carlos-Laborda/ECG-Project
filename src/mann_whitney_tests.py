"""
Compare F1 scores of various model groups across label fractions.

Expected folder structure:
└── /Users/carlitos/Desktop/ECG-Project/results/
    ├── Supervised_CNN/Supervised_0.1/individual_runs.csv
    ├── TS2Vec/LinearClassifier_0.1/individual_runs.csv
    ├── TS2Vec/MLPClassifier_0.1/individual_runs.csv
    └── ...

"""

import os, re, json
import pandas as pd
import numpy as np, pathlib
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from scipy.stats import shapiro, wilcoxon, mannwhitneyu
import scipy.stats as stats

# configs
ROOT = Path("/Users/carlitos/Desktop/ECG-Project/results")
SIGNIF_DIR = ROOT / "significance"        
SIGNIF_DIR.mkdir(exist_ok=True, parents=True)
ALPHA = 0.05

SSL_MODELS_ALL = {
    "TS2Vec", 
    "SoftTS2Vec", 
    "TSTCC", 
    "SoftTSTCC", 
    "SimCLR"
    }

SUPERVISED_MODELS_ALL = [
    "Supervised_CNN", 
    "Supervised_TCN", 
    "Supervised_Transformer"
    ]

SSL_MODELS = {
    "TS2Vec", 
    "SoftTS2Vec", 
    "TSTCC", 
    "SoftTSTCC", 
    }

SUPERVISED_MODELS = [
    "Supervised_CNN", 
    "Supervised_TCN", 
    ]

TRANSFORMER_MODEL = "Supervised_Transformer"


# Load F1 scores for individual supervised models
def get_supervised_scores(root: Path):
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(list))

    for model_dir in root.iterdir():
        if not model_dir.is_dir() or model_dir.name not in SUPERVISED_MODELS_ALL:
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

# Load scores for individual SSL models
def get_ssl_model_scores(root: Path):
    """Load individual SSL model results."""
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(list))  # Changed structure

    for model_dir in root.iterdir():
        if not model_dir.is_dir() or model_dir.name not in SSL_MODELS_ALL:
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
            
            # Create combined model name with classifier type
            model_clf = f"{model_name}_{clf_type}"
            
            csv_path = sub / "individual_runs.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            results[frac][model_clf].extend(df["test_f1"].values)

    return results

# Load f1 scores for the curated supervised and ssl models
def get_supervised_and_ssl_scores(root: Path):
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

# Helper functions
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

def build_sig_dict(results: dict, models: list[str], alpha: float = ALPHA):
    """
    Build a {frac: {(m1,m2): bool, …}, …} dictionary with Mann–Whitney
    significance flags (True ⇢ p < alpha).
    """
    sig = defaultdict(dict)
    for frac in results:
        for m1, m2 in combinations(models, 2):
            v1, v2 = results[frac].get(m1, []), results[frac].get(m2, [])
            if len(v1) < 2 or len(v2) < 2:
                sig[frac][(m1, m2)] = False
                continue
            _, p = mannwhitneyu(v1, v2, alternative="two-sided")
            sig[frac][(m1, m2)] = p < alpha
    return sig

def save_sig_to_json(sig_dict: dict, out_path: Path):
    """
    Store the dict in JSON. Tuple keys become "modelA|modelB" strings,
    and numpy.bool_ → native bool.
    """
    serialisable = {
        str(frac): {
            f"{k[0]}|{k[1]}": bool(v)   
            for k, v in inner.items()
        }
        for frac, inner in sig_dict.items()
    }
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[saved] {out_path.relative_to(ROOT)}")


# -----------------------------------------------------
# RUN ALL COMPARISONS
# -----------------------------------------------------
# 1) Individual Supervised
sup_results = get_supervised_scores(ROOT)
sup_models  = sorted({m for frac in sup_results for m in sup_results[frac]})
sup_sig     = build_sig_dict(sup_results, sup_models)
save_sig_to_json(sup_sig, SIGNIF_DIR / "supervised.json")

print("\n=== Individual Supervised Model Comparisons ===")
for m1, m2 in combinations(sup_models, 2):
    mannwhitney_comparison(sup_results, m1, m2)

# 2) Individual SSL
ssl_results = get_ssl_model_scores(ROOT)
linear_models = sorted({m for frac in ssl_results for m in ssl_results[frac] if "LinearClassifier" in m})
mlp_models    = sorted({m for frac in ssl_results for m in ssl_results[frac] if "MLPClassifier" in m})

ssl_lin_sig = build_sig_dict(ssl_results, linear_models)
save_sig_to_json(ssl_lin_sig, SIGNIF_DIR / "ssl_linear.json")
print("\n=== Individual SSL: Linear Classifier Comparisons ===")
for m1, m2 in combinations(linear_models, 2):
    mannwhitney_comparison(ssl_results, m1, m2)

ssl_mlp_sig = build_sig_dict(ssl_results, mlp_models)
save_sig_to_json(ssl_mlp_sig, SIGNIF_DIR / "ssl_mlp.json")
print("\n=== Individual SSL: MLP Classifier Comparisons ===")
for m1, m2 in combinations(mlp_models, 2):
    mannwhitney_comparison(ssl_results, m1, m2)

# 3) Group-level
all_results = get_supervised_and_ssl_scores(ROOT)
group_names = sorted({g for frac in all_results for g in all_results[frac]})
group_sig   = build_sig_dict(all_results, group_names)
save_sig_to_json(group_sig, SIGNIF_DIR / "groups.json")

print("\n=== Group Comparisons ===")
mannwhitney_comparison(all_results, "Supervised", "SSL_Linear")
mannwhitney_comparison(all_results, "SSL_Linear","SSL_MLP")
mannwhitney_comparison(all_results, "Supervised", "SSL_MLP")

print("\n=== Transformer vs Other Groups ===")
mannwhitney_comparison(all_results, "Transformer", "Supervised")
mannwhitney_comparison(all_results, "Transformer", "SSL_Linear")
mannwhitney_comparison(all_results, "Transformer", "SSL_MLP")

# # 4) Full data against partial data
# ROOT = pathlib.Path("/Users/carlitos/Desktop/ECG-Project/results")
# FRACS = [0.01, 0.05, 0.10, 0.50]     
# SSL   = ["TS2Vec","SoftTS2Vec","TSTCC","SoftTSTCC"]
# SUP   = ["Supervised_CNN","Supervised_TCN","Supervised_Transformer"]

# def load_f1_df(folder: pathlib.Path) -> pd.DataFrame:
#     """Return df with columns [seed, f1] for one model × fraction."""
#     df = pd.read_csv(folder / "individual_runs.csv")
#     if 'seed' in df.columns:
#         key = 'seed'
#     elif 'run_id' in df.columns:
#         key = 'run_id'
#     else:
#         key = df.columns[0]  # fallback
#     return df[[key, 'test_f1']].rename(columns={key: 'seed', 'test_f1': 'f1'})

# def paired_diffs(model_root: pathlib.Path, frac_values_list, label, classifier=""):
#     """Yield rows dict(model, frac, Δ_mean, Δ_std, p, r)."""
#     full_df_path = model_root / f"{classifier}1.0"
#     full_df = load_f1_df(full_df_path).set_index('seed')
#     rows = []
#     for f in frac_values_list:
#         part_df_path = model_root / f"{classifier}{f}"
#         part_df = load_f1_df(part_df_path)
#         merged = part_df.set_index('seed').join(full_df, lsuffix='_part', rsuffix='_full', how='inner')
#         if len(merged) < 3:
#             print(f"Warning: Not enough matched samples (<3) for {label} at fraction {f}. Found {len(merged)}. Skipping Wilcoxon.")
#             continue
#         diff = merged['f1_part'] - merged['f1_full']
#         stat, p = stats.wilcoxon(diff, alternative='less')  # f1_partial < f1_full
#         # Compute effect size: r = Z / sqrt(N), using Z approximation from T-statistic
#         z = stats.norm.ppf(p) if p > 0 else -5  # cap extreme case
#         r = z / (len(diff) ** 0.5)
#         rows.append(dict(model=label, frac=f,
#                          delta_mean   = diff.mean(),
#                          delta_std    = diff.std(ddof=1),
#                          p_wilcoxon   = p,
#                          effect_size_r = r))
#     return rows

# all_rows = []

# # SSL – Linear / MLP 
# for m in SSL: # m is "TS2Vec", "SoftTS2Vec", etc.
#     model_specific_root = ROOT / m # e.g., ROOT / "TS2Vec"
#     all_rows.extend(paired_diffs(model_specific_root, FRACS, f"{m}_Linear",  "LinearClassifier_"))
#     all_rows.extend(paired_diffs(model_specific_root, FRACS, f"{m}_MLP",     "MLPClassifier_"))

# # Supervised
# for m in SUP: # m is "Supervised_CNN", "Supervised_TCN", "Supervised_Transformer"
#     model_specific_root = ROOT / m # e.g., ROOT / "Supervised_CNN"
#     # For supervised models, the subfolder prefix is "Supervised_"
#     # e.g., Supervised_CNN/Supervised_0.1/individual_runs.csv
#     classifier_prefix_for_supervised = "Supervised_"
#     table_label = m.replace("Supervised_","") # "CNN", "TCN", "Transformer"
#     all_rows.extend(paired_diffs(model_specific_root, FRACS, table_label, classifier_prefix_for_supervised))

# results = (pd.DataFrame(all_rows)
#              .sort_values(["model","frac"])
#              .reset_index(drop=True))

# # nice round-up
# pd.set_option('display.float_format', '{:.3f}'.format)
# print(results)
