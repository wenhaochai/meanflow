#!/bin/bash
#SBATCH --job-name=batch_eval
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu80
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=1-22%4

JOB_DIRS=(
    "loopb4p0s2l6e10_le_20250927_132934"
    "loopb4p10s2l6e0_le_20250926_204444"
    "loopb4p1s2l6e9_le_20250927_124703"
    "loopb4p2s2l6e8_le_20250927_103520"
    "loopb4p3s2l6e7_le_20250927_060849"
    "loopb4p4s2l6e6_le_20250927_052448"
    "loopb4p4s4l16e4_le_0.1drop_20250926_013932"
    "loopb4p4s4l20e4_le_0.1drop_20250926_044745"
    "loopb4p5s2l16e5_le_d05_20250926_130055"
    "loopb4p5s2l16e5_le_d20_20250926_130055"
    "loopb4p5s2l16e5_le_d30_20250926_130055"
    "loopb4p5s2l16e5_le_d40_20250926_170632"
    "loopb4p5s2l6e5_le_20250927_044820"
    "loopb4p5s2l6e5_le_20250927_141047"
    "loopb4p5s2l6e5_le_d05_20250926_093137"
    "loopb4p5s2l6e5_le_d20_20250926_093207"
    "loopb4p5s2l6e5_le_d30_20250926_093207"
    "loopb4p5s2l6e5_le_d40_20250926_093307"
    "loopb4p6s2l6e4_le_20250926_204444"
    "loopb4p7s2l6e3_le_20250926_220651"
    "loopb4p8s2l6e2_le_20250927_023447"
    "loopb4p9s2l6e1_le_20250927_044640"
)

# Get the job directory for this array task
JOB_DIR=${JOB_DIRS[$SLURM_ARRAY_TASK_ID-1]}

echo "Array task $SLURM_ARRAY_TASK_ID: Evaluating $JOB_DIR"

# Change to project directory
cd /path/to/work/project/meanflow

# Load environment
module load anaconda3/2024.10
conda activate jax

# Run evaluation
bash scripts/launch_eval.sh "$JOB_DIR"