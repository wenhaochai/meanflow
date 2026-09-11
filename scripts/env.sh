# Source inside SLURM jobs / login shells: dedicated env + offline asset locations.
module load anaconda3/2024.10
source activate /path/to/work/envs/jax 2>/dev/null || conda activate /path/to/work/envs/jax
export HF_HOME=/path/to/work/cache/huggingface
export HF_HUB_OFFLINE=1
export JAX_FID_CKPT_DIR=/path/to/work/data/meanflow   # holds jax_fid/inception_v3_weights_fid.pickle
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
