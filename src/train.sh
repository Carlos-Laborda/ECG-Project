#!/bin/bash
#SBATCH --job-name=ECG_training         # Job name
#SBATCH --time=00:15:00                 # Max run time (HH:MM:SS)
#SBATCH -N 1                           # Number of nodes
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=16              # CPU cores per task
#SBATCH --output=ecg_train.out         # Standard output log
#SBATCH --error=ecg_train.err          # Standard error log

# --- Load necessary modules ---
module add cuda12.3/toolkit/12.3

# --- Source the Miniconda initialization script from scratch ---
source /var/scratch/cla224/miniconda3/etc/profile.d/conda.sh

# --- Activate your conda environment ---
source activate /var/scratch/cla224/ECG_env

# --- Set the MLflow tracking URI to HTTP on port 5005 ---
export MLFLOW_TRACKING_URI=http://fs0.das6.cs.vu.nl:5005
echo "MLFLOW_TRACKING_URI is set to: $MLFLOW_TRACKING_URI"

# --- Debug: Verify Python and pip installations ---
echo "Python version: $(python --version)"

# --- Run your training script ---
python torch_pipeline.py run \
  --model_type "1DCNN_improved" \
  --model_description "1DCNN_improved with increased batch size 32" \
  --num_epochs 5
