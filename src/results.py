import argparse
import os
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser(description="Collect test metrics from MLflow")
parser.add_argument("--run", required=True, help="Run name in MLflow UI")
parser.add_argument("--experiment", required=True, help="MLflow experiment name")
parser.add_argument("--tracking_uri", default="http://fs0.das6.cs.vu.nl:5005", help="MLflow tracking URI")
parser.add_argument("--save_dir", default="results", help="Where to save CSVs")
args = parser.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
client = MlflowClient()

exp = client.get_experiment_by_name(args.experiment)
if exp is None:
    raise SystemExit(f"Experiment '{args.experiment}' not found.")
exp_id = exp.experiment_id

# Find run by run-name
query = f"tags.mlflow.runName = '{args.run}' and attributes.status = 'FINISHED'"
runs = mlflow.search_runs([exp_id], filter_string=query)
if runs.empty:
    raise SystemExit(f"No run named '{args.run}' in experiment '{args.experiment}'")
run_id = runs.iloc[0]["run_id"]
run_info = client.get_run(run_id)

# Extract metrics + parameters
metrics = run_info.data.metrics
params = run_info.data.params
tag_seed = params.get("seed", "NA")
tag_model = params.get("model_name", "NA")
tag_label_fraction = params.get("label_fraction", "NA")

row = {
    "run_name": args.run,
    "run_id": run_id,
    "model_name": tag_model,
    "seed": tag_seed,
    "label_fraction": tag_label_fraction,
}
row.update({k: round(v, 4) for k, v in metrics.items()})

# Save 
experiment_dir = os.path.join(args.save_dir, args.experiment)
os.makedirs(experiment_dir, exist_ok=True)
csv_path = os.path.join(experiment_dir, f"metrics_{args.run}.csv")
pd.DataFrame([row]).to_csv(csv_path, index=False)

print("\nFinal Test Metrics:")
print(pd.DataFrame([row]).T)

print(f"\nSaved: {csv_path}")

"""
Usage
-----
python collect_metrics_mlflow.py \
    --run 1745389226440130 \
    --experiment Default
"""