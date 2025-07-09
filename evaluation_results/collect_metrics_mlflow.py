import argparse
import os
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from collections import defaultdict

# ──────────────
# CLI arguments
# ──────────────
parser = argparse.ArgumentParser(description="Collect and aggregate test metrics from MLflow.")
parser.add_argument("--runs", nargs='+', required=True, help="One or more run names from MLflow UI.")
parser.add_argument("--experiment", required=True, help="MLflow experiment name.")
parser.add_argument("--tracking_uri", default="http://fs0.das6.cs.vu.nl:5005", help="MLflow tracking URI.")
parser.add_argument("--save_dir", default="results", help="Base directory to save CSV results.")
args = parser.parse_args()

# Setup
mlflow.set_tracking_uri(args.tracking_uri)
client = MlflowClient()
exp = client.get_experiment_by_name(args.experiment)
if exp is None:
    raise SystemExit(f"Experiment '{args.experiment}' not found.")
exp_id = exp.experiment_id

# Fetch all run results
all_rows = []
for run_name in args.runs:
    query = f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'"
    runs = mlflow.search_runs([exp_id], filter_string=query)
    if runs.empty:
        print(f"No run named '{run_name}' found in experiment '{args.experiment}'. Skipping.")
        continue

    run_id = runs.iloc[0]["run_id"]
    run_info = client.get_run(run_id)

    metrics = run_info.data.metrics
    params = run_info.data.params

    row = {
        "run_name": run_name,
        "run_id": run_id,
        "classifier_model": params.get("classifier_model", "Supervised"),
        "model_name": params.get("model_name", "NA"),
        "seed": params.get("seed", "NA"),
        "label_fraction": params.get("label_fraction", "1.0"),
    }
    row.update({k: round(v, 2) for k, v in metrics.items()})
    all_rows.append(row)

if not all_rows:
    raise SystemExit("No valid runs found. Exiting.")


# Save individual and aggregated metrics
df = pd.DataFrame(all_rows)
test_metrics = [col for col in df.columns if col.startswith("test_")]
# Round 
df[test_metrics] = (df[test_metrics] * 100).round(2)

agg = df[test_metrics].agg(["mean", "std"]).round(2).T
agg.columns = ["mean", "std"]
agg.reset_index(inplace=True)
agg.rename(columns={"index": "metric"}, inplace=True)

classifier_model = df["classifier_model"].iloc[0]
label_fraction = df["label_fraction"].iloc[0]
save_path = os.path.join(args.save_dir, args.experiment, f"{classifier_model}_{label_fraction}")
os.makedirs(save_path, exist_ok=True)

# Save individual + aggregated CSVs
df.to_csv(os.path.join(save_path, "individual_runs.csv"), index=False)
agg.to_csv(os.path.join(save_path, "aggregated_metrics.csv"), index=False)

print("\nAggregated Test Metrics:")
print(agg)

print(f"\nSaved in: {save_path}")

#example usage:
"""
python collect_metrics_mlflow.py \
--runs 1745389226440130 1745389226456781 1745389226459001 \
--experiment Supervised_TCN
"""
