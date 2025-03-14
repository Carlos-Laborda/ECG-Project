#!/bin/bash
#
#SBATCH --job-name=ECG_training         # Job name
#SBATCH --time=00:15:00                 # Max run time (HH:MM:SS)
#SBATCH -N 1                           # Number of nodes
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=16              # CPU cores per task
#SBATCH --output=ecg_train.out         # Standard output log
#SBATCH --error=ecg_train.err          # Standard error log

# --- Load necessary modules ---
module add cuda12.3/toolkit/12.3

# --- Source local Miniconda initialization script ---
source ~/miniconda3/etc/profile.d/conda.sh   

# --- Activate environment ---
conda activate ecg_env               

pip install torch
pip install torchvision 
pip install numpy
pip install mlflow
pip install metaflow
pip install h5py
pip install scikit-learn

# --- Debug: Verify Python and installed packages ---
echo "Python version: $(python --version)"
python -m pip list


# --- Run your training script ---
python torch_pipeline.py run \
  --model_type "1DCNN_improved" \
  --model_description "1DCNN_improved with increased batch size 32" \
  --num_epochs 5
