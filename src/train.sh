#!/bin/bash
#SBATCH --job-name=ECG             # Job name
#SBATCH --time=24:00:00            # Max run time (HH:MM:SS)
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

# --- Run Fully Supervised Model script ---
# python torch_pipeline.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --patience 20 \
#   --lr 0.00001 \
#   --batch_size 32 \
#   --num_epochs 20

# --- Run Self Supervised Model script ---
python tstcc_train.py run \
  --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
  --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
  --tcc_epochs 3 \
  --label_fraction 1.0 \
  --cc_temperature 0.07 \
  --tc_timesteps 70 \