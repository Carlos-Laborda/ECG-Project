import argparse
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Collect test metrics from MLflow")
parser.add_argument("--run", required=True,
                    help="Run name in MLflow UI (we match tags.mlflow.runName)")
parser.add_argument("--experiment", required=True,
                    help="MLflow experiment name")
parser.add_argument("--tracking_uri", default="http://fs0.das6.cs.vu.nl:5005",
                    help="MLflow tracking URI (default: DAS-6 server)")

args = parser.parse_args()

# ---------- Connect ----------
mlflow.set_tracking_uri(args.tracking_uri)
client = MlflowClient()

exp = client.get_experiment_by_name(args.experiment)
if exp is None:
    raise SystemExit(f" Experiment '{args.experiment}' not found on server {args.tracking_uri}")
exp_id = exp.experiment_id

# ---------- Find run by run-name ----------
query = f"tags.mlflow.runName = '{args.run}' and attributes.status = 'FINISHED'"
runs = mlflow.search_runs(experiment_ids=[exp_id], filter_string=query)

if runs.empty:
    raise SystemExit(f"No finished MLflow run with name '{args.run}' in experiment '{args.experiment}'")

if len(runs) > 1:
    print(f" Warning: multiple runs matched; taking the first one (run_id={runs.iloc[0]['run_id']})")

run_id = runs.iloc[0]["run_id"]
mlflow_run = client.get_run(run_id)

# ---------- Collect metrics ----------
metrics = mlflow_run.data.metrics
if not metrics:
    raise SystemExit("Run has no recorded metrics.")

df = pd.DataFrame([metrics]).T
df.columns = ["Value"]
df["Value"] = df["Value"].astype(float).round(4)

print("\n Final Test / Aggregate Metrics")
print(df)

# ---------- Save to CSV ----------
csv_path = f"aggregated_metrics_{args.run}.csv"
df.to_csv(csv_path)
print(f"\n Saved metrics to {csv_path}")

"""
Usage
-----
python collect_metrics_mlflow.py \
    --run 1745389226440130 \
    --experiment ecg_training_ts2vec_soft
"""