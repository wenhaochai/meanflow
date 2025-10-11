#!/bin/bash

# Auto-evaluation script - detects config and checkpoint from job directory
# Usage: bash scripts/launch_eval.sh JOB_DIR [JOB_NAME]
# Example: bash scripts/launch_eval.sh b4_240ep

module load anaconda3/2024.10
conda activate jax

export LOG_DIR="/path/to/work/project/meanflow/log_eval"

auto_detect_config() {
    [[ ! -d "$1" ]] && { echo "Error: $1 not found"; exit 1; }

    local job_name=$(basename "$1")
    # Remove timestamp suffix (e.g., loopb4p0s2l6e0_20250927_123456 -> loopb4p0s2l6e0)
    local config=$(echo "$job_name" | sed 's/_[0-9]\{8\}_[0-9]\{6\}$//')
    echo "$config"
}

# Find latest checkpoint
find_latest_checkpoint() {
    find "$1" -maxdepth 1 -type d -name "checkpoint_*" | sort -V | tail -1
}

# Main
[[ -z "$1" ]] && { echo "Usage: bash scripts/launch_eval.sh JOB_DIR [JOB_NAME]"; exit 1; }

JOB_PATTERN="$1"
LOG_BASE="/path/to/work/project/meanflow/log/user"

# Find longest matching directory
JOB_DIR=$(ls "$LOG_BASE" | grep "^${JOB_PATTERN}" | sort -r | head -1)
[[ -z "$JOB_DIR" ]] && { echo "Error: No directory found matching '$JOB_PATTERN'"; exit 1; }

LOG_PATH="$LOG_BASE/$JOB_DIR"

MODEL_CONFIG=$(auto_detect_config "$LOG_PATH")
CHECKPOINT_PATH=$(find_latest_checkpoint "$LOG_PATH")
JOB_NAME="${2:-${JOB_DIR}}"

export JOBNAME="${JOB_NAME}_$(date '+%Y%m%d_%H%M%S')"
export LOG_DIR="$LOG_DIR/$USER/$JOBNAME"
mkdir -p "$LOG_DIR"

echo "Evaluating: $JOB_DIR -> $MODEL_CONFIG"
[[ -n "$CHECKPOINT_PATH" ]] && echo "Checkpoint: $CHECKPOINT_PATH" || echo "Warning: No checkpoints found"

# Create temp config with checkpoint path
TEMP_NAME="temp_eval_$(date +%s)"
TEMP_CONFIG="configs/${TEMP_NAME}.yml"
cp "configs/run_$MODEL_CONFIG.yml" "$TEMP_CONFIG"

# Remove existing load_from and eval_only lines, then add new ones
sed -i "/^load_from:/d" "$TEMP_CONFIG"
sed -i "/^eval_only:/d" "$TEMP_CONFIG"
# Ensure file ends with newline, then add new config lines
echo "" >> "$TEMP_CONFIG"
echo "load_from: \"$CHECKPOINT_PATH\"" >> "$TEMP_CONFIG"
echo "eval_only: true" >> "$TEMP_CONFIG"

python3 eval.py \
    --workdir="$LOG_DIR" \
    --config="configs/load_config.py:$TEMP_NAME" \
    2>&1 | tee -a $LOG_DIR/output.log

rm -f "$TEMP_CONFIG"