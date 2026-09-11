#!/bin/bash
# Submit a dependency chain of resumable segments.
# Usage: bash scripts/submit_chain.sh CONFIG NSEG QOS [GPUS=4] [TIME=auto] [JOBNAME=CONFIG]
#   QOS: cluster_queue-short (24h) | cluster_queue-medium (72h)
set -e
CONFIG=$1; NSEG=${2:-1}; QOS=${3:-cluster_queue-short}; GPUS=${4:-4}; TIME=${5:-auto}; JOBNAME=${6:-$CONFIG}
case $QOS in
  cluster_queue-short)  PART=cluster_queue-c; [ "$TIME" = auto ] && TIME=23:59:00 ;;
  cluster_queue-medium) PART=cluster_queue-c; [ "$TIME" = auto ] && TIME=71:59:00 ;;
  *) echo "unknown qos $QOS"; exit 1 ;;
esac
cd /path/to/work/project/meanflow_private
DEP=""; IDS=()
for i in $(seq 1 $NSEG); do
  JID=$(sbatch --parsable $DEP --job-name=$JOBNAME --account=your_account --partition=$PART --qos=$QOS \
        --gres=gpu:$GPUS --mem-per-gpu=64G --constraint=gpu80 --time=$TIME \
        scripts/sbatch_train.sh $CONFIG $JOBNAME)
  IDS+=($JID); DEP="--dependency=afterany:$JID"
done
echo "$CONFIG $JOBNAME qos=$QOS gpus=$GPUS segments: ${IDS[*]}"
