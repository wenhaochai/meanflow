#!/bin/bash

# Note: You should update the dataset.root and fid.cache_ref in configs/run_*_config.yml

# suggest use absolute path
export LOG_DIR="/path/to/work/project/meanflow/log_test"

MODEL_CONFIG=${1:-loopb4moe8c1p5s2l6e5}
JOB_NAME=${2:-${MODEL_CONFIG}}

export now=`date '+%Y%m%d_%H%M%S'`
export JOBNAME=${JOB_NAME}_${now}
export LOG_DIR=$LOG_DIR/$USER/$JOBNAME

mkdir -p ${LOG_DIR}
chmod 777 -R ${LOG_DIR}

python3 main.py \
    --workdir=${LOG_DIR} \
    --config=configs/load_config.py:run_${MODEL_CONFIG} \
    2>&1 | tee -a $LOG_DIR/output.log
