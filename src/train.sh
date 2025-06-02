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
# echo "Launching training with seed: $1"
python supervised_training.py run \
  --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
  --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
  --model_type "transformer" \
  --seed $1 \
  --lr 1e-5 \
  --patience 20 \
  # --scheduler_factor 0.1 \

  

# echo "Launching training with seed: $1"
# python tstcc_train.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --tcc_epochs 40 \
#   --seed $1
#   --label_fraction 1.0 \
#   --cc_temperature 0.07 \
#   --tc_timesteps 70 \
#   --tc_hidden_dim 128 \
#   --tau_inst 10 \
#   --lambda_aux 0.2 \
#   --tau_temp 1 \
  
# echo "Launching training with seed: $1"
# python simclr_train.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --epochs 100 \
#   --label_fraction 1.0 \
#   --seed $1

# Transfer learning
# python ppg_ts2vecsoft_train.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --ecg_window_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --ppg_window_path "../../../../var/scratch/cla224/ECG-Project/data/ppg_windows.h5" \

# echo "Launching training with seed: $1"
# python ts2vec_train.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --ts2vec_epochs 5 \
#   --seed $1