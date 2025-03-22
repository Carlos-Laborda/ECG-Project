#!/bin/bash
#SBATCH --job-name=ECG             # Job name
#SBATCH --time=00:30:00            # Max run time (HH:MM:SS)
#SBATCH -N 1                       # Number of nodes
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=24         # CPU cores per task
#SBATCH --output=ecg_train.out     # Standard output log
#SBATCH --error=ecg_train.err      # Standard error log

# --- Load CUDA module ---
module add cuda12.3/toolkit/12.3

# --- Set CUDA device visibility (necessary for GPU selection) ---
export CUDA_VISIBLE_DEVICES=0

# --- Initialize Conda ---
source activate /var/scratch/cla224/ECG_env

# --- Set MLflow tracking server ---
export MLFLOW_TRACKING_URI=http://fs0.das6.cs.vu.nl:5005

# --- Run training script ---
python torch_pipeline.py run \
  --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
  --segmented_data_path "../../../../var/scratch/cla224/ECG-Project/data/ecg_data_segmented.h5" \
  --cleaned_data_path "../../../../var/scratch/cla224/ECG-Project/data/ecg_data_cleaned.h5" \
  --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5"