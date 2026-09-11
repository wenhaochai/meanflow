#!/bin/bash
# Resumable training segment. Usage (via submit_chain.sh):
#   sbatch --job-name=NAME --gres=gpu:N ... scripts/sbatch_train.sh CONFIG JOBNAME
# The workdir is fixed (no timestamp) so chained segments auto-resume from checkpoint_*.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%x_%j.out
set -eo pipefail
cd /path/to/work/project/meanflow_private
source scripts/env.sh
MODEL_CONFIG=$1
JOBNAME=${2:-$MODEL_CONFIG}
export LOG_DIR=/path/to/work/project/meanflow_private/log/$USER/$JOBNAME
mkdir -p $LOG_DIR
echo "[$(date)] job $SLURM_JOB_ID config=$MODEL_CONFIG workdir=$LOG_DIR node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
python3 main.py \
    --workdir=$LOG_DIR \
    --config=configs/load_config.py:run_${MODEL_CONFIG} \
    2>&1 | tee -a $LOG_DIR/output.log
