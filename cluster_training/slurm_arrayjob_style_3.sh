#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)

# ====================
# Options for sbatch
# ====================

#SBATCH --job-name=folding_model_training_style_3
#SBATCH -o /home/$USER/slogs/sl_%A.out
#SBATCH -e /home/$USER/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:nvidia_rtx_a6000  # use 1 GPU
#SBATCH --nodelist="landonia11"
#SBATCH --mem=32000  # memory in Mb
#SBATCH --partition=Teaching
#SBATCH -t 1-00:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=8

export PATH="$HOME/.local/bin:$PATH"
set -e # fail fast 
echo "Initialising environment"
rm -rf .venv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r Dissertation-Robotics/cluster_training/requirements.txt
uv pip install lerobot

echo "Environment initialised and sourced!"

# =====================
# Logging information
# =====================
# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

## create data in pwd base on the util script
mkdir -p data

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=/disk/scratch/${USER}
export DATA_HOME=${PWD}/data
export DATA_SCRATCH=${SCRATCH_HOME}/practical/data
mkdir -p ${SCRATCH_HOME}/practical/data
rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}

# ====================
# Run training. Here we use src/gpu.py
# ====================
echo "Creating directory to save model weights"
export OUTPUT_DIR=${SCRATCH_HOME}/practical/example
mkdir -p ${OUTPUT_DIR}

# Training
uv run Dissertation-Robotics/src/lerobot/scripts/lerobot_train.py \
        --dataset.repo_id="the-sam-uel/bi-so101-fold-horizontal-style-3"  \
        --batch_size=64 \
        --steps=20000  \
        --job_name="bi_so101_folding_training_style_3" \
         --policy.device="cuda" \
        --policy.type=smolvla \
        --wandb.enable="false" \
        --policy.repo_id="the-sam-uel/folding-style-3"

# ====================
# RSYNC data from /disk/scratch/ to /home/. This moves everything we want back onto the distributed file system
# ====================
OUTPUT_HOME=${PWD}/exps/style-3
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

# ====================
# Finally we cleanup after ourselves by deleting what we created on /disk/scratch/
# ====================
rm -rf ${OUTPUT_DIR}

echo "Job ${SLURM_JOB_ID} is done!"