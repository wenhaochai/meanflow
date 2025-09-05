#!/bin/bash

# Launch script for eval.py - LoopedDiT FID evaluation across multiple loop counts
# Usage: bash scripts/launch_eval.sh [MODEL_CONFIG] [JOB_NAME (leave empty for default)]
# Example: bash scripts/launch_eval.sh loopb4p0s2l6e0
# Note: Loop counts should be specified in config file under evaluation.loop_list

# suggest use absolute path
export LOG_DIR="/path/to/work/project/meanflow/eval_log"

MODEL_CONFIG=${1:-loopb4p0s2l6e0}
JOB_NAME=${2:-eval_${MODEL_CONFIG}}

export now=`date '+%Y%m%d_%H%M%S'`
export JOBNAME=${JOB_NAME}_${now}
export LOG_DIR=$LOG_DIR/$USER/$JOBNAME

mkdir -p ${LOG_DIR}
chmod 777 -R ${LOG_DIR}

python3 eval.py \
    --workdir=${LOG_DIR} \
    --config=configs/load_config.py:run_${MODEL_CONFIG} \
    2>&1 | tee -a $LOG_DIR/output.log