#!/bin/bash
#SBATCH --job-name=mf-prep-latent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --account=your_account
#SBATCH --partition=your_partition
#SBATCH --qos=your_qos
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
set -eo pipefail
cd /path/to/work/project/meanflow_private
source scripts/env.sh
OUT=/path/to/work/data/meanflow
mkdir -p $OUT/imagenet_train_symlinks
# ImageFolder root with class dirs only (the shared dir also holds sibling <wnid>.tar archives)
for d in /path/to/work/ilsvrc_2012_classification_localization/train/*/; do
  n=$(basename $d); [ -e $OUT/imagenet_train_symlinks/train/$n ] || { mkdir -p $OUT/imagenet_train_symlinks/train; ln -s $d $OUT/imagenet_train_symlinks/train/$n; }
done
python3 prepare_dataset.py \
    --imagenet_root=$OUT/imagenet_train_symlinks \
    --output_dir=$OUT \
    --batch_size=256 --vae_type=mse --image_size=256 \
    --compute_latent=True --compute_fid=False --overwrite=False
ls -la $OUT
