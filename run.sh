#!/bin/bash
#SBATCH --job-name=custom-transformer
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --gres=gpu:1

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load modules if needed
# Uncomment and adjust based on available modules on your cluster
# module load cuda/11.8
# module load python/3.12

# Verify CUDA is available (optional, for debugging)
# nvidia-smi

# Activate virtual environment if needed (uncomment if using conda/venv)
# source activate your_env_name

# Run the training script
uv run main.py

# Print completion time
echo "End Time: $(date)"
echo "Job completed successfully"