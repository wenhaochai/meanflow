#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=your_partition
#SBATCH --time=01:00:00
#SBATCH --qos=your_qos
#SBATCH --constraint=gpu80
#SBATCH --account=your_account
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# NFE Sweep Evaluation Script
# Usage: sbatch scripts/eval_nfe.sh MODEL_CONFIG [MIN_NFE] [MAX_NFE] [JOB_NAME]
# Example: sbatch scripts/eval_nfe.sh b4 1 10 b4_nfe_sweep

module load anaconda3/2024.10
conda activate jax

export LOG_DIR="/path/to/work/project/meanflow/log_eval"

MODEL_CONFIG=${1:-loopb4_eval}
JOB_NAME=${2:-$MODEL_CONFIG}
MIN_NFE=${3:-1}
MAX_NFE=${4:-10}

export now=`date '+%Y%m%d_%H%M%S'`
export JOBNAME=${JOB_NAME}_${now}
export LOG_DIR=$LOG_DIR/$USER/$JOBNAME

mkdir -p ${LOG_DIR}
chmod 777 -R ${LOG_DIR}

echo "NFE Sweep Evaluation"
echo "===================="
echo "Model config: $MODEL_CONFIG"
echo "NFE range: $MIN_NFE to $MAX_NFE"
echo "Output directory: $LOG_DIR"
echo ""

# Loop through different NFE values
for nfe in $(seq $MIN_NFE $MAX_NFE); do
    echo "============================================"
    echo "Evaluating with NFE = $nfe"
    echo "============================================"

    # Create temporary config file with modified num_steps
    TEMP_CONFIG="configs/temp_eval_nfe${nfe}_$(date +%s).yml"
    cp "configs/run_${MODEL_CONFIG}.yml" "$TEMP_CONFIG"

    # Add sampling section with num_steps at the end of the file
    cat >> "$TEMP_CONFIG" << EOF

sampling:
    num_steps: $nfe
EOF

    # Create NFE-specific output directory
    NFE_LOG_DIR="${LOG_DIR}/nfe_${nfe}"
    mkdir -p ${NFE_LOG_DIR}

    # Run evaluation with modified config
    python3 main.py \
        --workdir=${NFE_LOG_DIR} \
        --config=configs/load_config.py:$(basename $TEMP_CONFIG .yml) \
        2>&1 | tee -a $NFE_LOG_DIR/output.log

    # Clean up temporary config
    rm -f "$TEMP_CONFIG"

    echo ""
done

echo "NFE sweep evaluation completed!"
echo "Results saved to: $LOG_DIR"
