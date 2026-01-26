#!/usr/bin/env bash

#SBATCH -A NAISS2025-5-98              # Account/project
#SBATCH -p alvis                       # Partition/queue
#SBATCH --cpus-per-task=8
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-20:00:00                  # Walltime
#SBATCH --job-name=G_-5dB
#SBATCH --output=G_-5dB_job.log
#SBATCH --error=G_-5dB_job.err

# Load CUDA module (MUST be before running apptainer)
module load CUDA/11.8.0

module list 2>&1 || true
which python || true
which apptainer || true
nvcc --version || true

echo "===== GPU (HOST) ====="
nvidia-smi || true
nvidia-smi -L || true

# Log GPU usage every 60 seconds in background
(while true; do nvidia-smi >> gpu_monitor_G_-5dB.log 2>&1; sleep 60; done) &

# Container and paths
CONTAINER=/cephyr/users/zhiping/Alvis/aucum/buiild/aucum.sif
DATA_PATH=/mimer/NOBACKUP/groups/ulio_inverse/zhiping/data/SpeechCommands

# ==============================================
# Gaussian -5dB Noise Experiment (15 epochs)
# ==============================================

# Step 1: Standard training with -5dB noise (generates checkpoints for scoring)
echo "=== Step 1: Standard Training with Gaussian -5dB ==="
#apptainer exec --nv \
#    --env LD_LIBRARY_PATH=/apps/Common/software/CUDA/11.8.0/lib64:\$LD_LIBRARY_PATH \
#    $CONTAINER \
#    aucurriculum train -cn Gaussian_-5dB device=cuda ++dataset.path=$DATA_PATH

# Step 2: Compute CumAcc difficulty scores
echo "=== Step 2: CumAcc Scoring ==="
apptainer exec --nv \
    --env LD_LIBRARY_PATH=/apps/Common/software/CUDA/11.8.0/lib64:\$LD_LIBRARY_PATH \
    $CONTAINER \
    aucurriculum curriculum -cn Gaussian_-5dB_scoring device=cuda ++dataset.path=$DATA_PATH

# Step 3: Curriculum training (easy -> hard)
echo "=== Step 3: Curriculum Training ==="
apptainer exec --nv \
    --env LD_LIBRARY_PATH=/apps/Common/software/CUDA/11.8.0/lib64:\$LD_LIBRARY_PATH \
    $CONTAINER \
    aucurriculum train -cn Gaussian_-5dB_curriculum device=cuda ++dataset.path=$DATA_PATH

# Step 4: AntiCurriculum training (hard -> easy)
echo "=== Step 4: AntiCurriculum Training ==="
apptainer exec --nv \
    --env LD_LIBRARY_PATH=/apps/Common/software/CUDA/11.8.0/lib64:\$LD_LIBRARY_PATH \
    $CONTAINER \
    aucurriculum train -cn Gaussian_-5dB_anticurriculum device=cuda ++dataset.path=$DATA_PATH

echo "=== All steps completed ==="
echo "Job finished."
