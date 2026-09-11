#!/bin/bash
#SBATCH --job-name=mf-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --account=your_account
#SBATCH --partition=your_partition
#SBATCH --qos=your_qos
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/%x_%j.out
set -eo pipefail
cd /path/to/work/project/meanflow_private
source scripts/env.sh
python -c "import jax; print('devices', jax.devices())"
export LOG_DIR=/path/to/work/project/meanflow_private/log_test/$USER/smoke_$SLURM_JOB_ID
mkdir -p $LOG_DIR
python3 main.py --workdir=$LOG_DIR --config=configs/load_config.py:run_smoke 2>&1 | tee $LOG_DIR/output.log
echo "=== second launch: auto-resume should restore checkpoint_2 and exit immediately ==="
python3 main.py --workdir=$LOG_DIR --config=configs/load_config.py:run_smoke 2>&1 | grep -i "auto_resume\|epoch\|Error" | head -20
# dense depth-24 path, 1 epoch
sed -e 's/cls: LoopedDiT_B_4/cls: DiT_B_4_D24/' -e '/n_prefix\|n_shared\|n_loops\|n_suffix\|shared_mlp_ratio/d' -e 's/num_epochs: 2/num_epochs: 1/' configs/run_smoke.yml > configs/run_smoke_d24.yml
python3 main.py --workdir=${LOG_DIR}_d24 --config=configs/load_config.py:run_smoke_d24 2>&1 | grep -i "fid\|params\|Error\|epoch" | tail -15
