#!/bin/bash
#SBATCH --job-name=ECG_training         # Job name
#SBATCH --time=00:30:00                 # Max run time (HH:MM:SS)
#SBATCH -N 1                           # Number of nodes
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=16              # CPU cores per task
#SBATCH --output=ecg_train.out         # Standard output log
#SBATCH --error=ecg_train.err          # Standard error log

# --- Load necessary modules ---
module add py3-numpy/1.19.5
module add cuda12.3/toolkit/12.3

# --- Use scratch directory for virtual environment ---
mkdir -p /var/scratch/cla224
python3 -m venv /var/scratch/cla224/ecg_venv
source /var/scratch/cla224/ecg_venv/bin/activate

# --- Upgrade pip and install minimal required packages (no-cache to save space) ---
pip install --upgrade pip
pip install torch torchvision mlflow metaflow h5py scikit-learn --no-cache-dir

# --- Debug: Verify Python and pip installations ---
echo "Python version: $(python --version)"
python -m pip list

# --- Run your training script ---
python torch_pipeline.py run \
  --model_type "1DCNN_improved" \
  --model_description "1DCNN_improved with increased batch size 32" \
  --num_epochs 5

# --- (Optional) Clean up the virtual environment from scratch ---
rm -rf /var/scratch/cla224/ecg_venv
