"""
Evaluate FID across multiple LoopedDiT loop counts.

Usage example:

  python eval.py \
    --workdir=/path/to/work/project/meanflow/log/eval \
    --config=configs/load_config.py:run_loopb4p0s2l6e0

Set loop values in your config YAML under `evaluation.loop_list`,
and set `load_from` to a checkpoint dir.

Notes:
- Reuses the train.train_and_evaluate eval-only path and FID evaluator.
- For each loop value, creates a subfolder under --workdir (e.g., loop_6/).
- Requires the checkpoint path in config.load_from and a LoopedDiT model config.
"""

import os
from copy import deepcopy
import re

from absl import app, flags
from ml_collections import config_flags

import jax

import train
from utils import logging_util
from utils.distributed_util import auto_init_distributed, print_distributed_setup


# Auto-init distributed and adjust logging (match main.py behavior)
auto_init_distributed()
logging_util.supress_checkpt_info()

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'Base directory to store eval logs per loop.')

config_flags.DEFINE_config_file(
    'config',
    None,
    'Path-like string to config',
    lock_config=True,
)


def _prepare_eval_config(base_cfg, n_loops):
  cfg = deepcopy(base_cfg)
  # Ensure evaluation-only and FID-on-training path is active
  cfg.eval_only = True
  if hasattr(cfg, 'fid') and hasattr(cfg.fid, 'on_training'):
    cfg.fid.on_training = True
  # Mark WandB job type and add a tag for clarity (best-effort)
  if hasattr(cfg, 'wandb'):
    try:
      cfg.wandb.job_type = 'eval'
      tags = list(cfg.wandb.tags)
      tags.append(f'loop_{n_loops}')
      cfg.wandb.tags = tags
    except Exception:
      pass
    
  # Set loop count for LoopedDiT
  cfg.model.n_loops = int(n_loops)
  # TODO: NOT HARD CODE loop_emb_max and loop_emb_min here
  cfg.model.loop_emb_max = 1.0
  cfg.model.loop_emb_min = 0.0
  
  return cfg


def extract_fid_from_logs(workdir):
  """Extract FID score from output.log in the parent directory."""
  parent_dir = os.path.dirname(workdir)
  log_file = os.path.join(parent_dir, 'output.log')
  
  if os.path.exists(log_file):
    try:
      with open(log_file, 'r') as f:
        content = f.read()
      
      # Look for the most recent FID pattern: "FID w/ EMA at {samples} samples: {score}"
      pattern = r'FID w/ EMA at \d+ samples: ([\d.]+)'
      matches = re.findall(pattern, content)
      if matches:
        # Return the last (most recent) FID score
        return float(matches[-1])
    except Exception:
      pass
  
  return None


def generate_results_summary(base_workdir, loop_list, results, base_cfg, config_path=None):
  """Generate and save a comprehensive results summary."""
  if jax.process_index() != 0:
    return
  
  # Create summary content
  summary_lines = []
  summary_lines.append("=" * 60)
  summary_lines.append("LoopedDiT Evaluation Results Summary")
  summary_lines.append("=" * 60)
  summary_lines.append("")
  
  # Add basic configuration information
  summary_lines.append("Configuration:")
  summary_lines.append(f"Config Path:     {config_path or 'N/A'}")
  summary_lines.append(f"Checkpoint Path: {getattr(base_cfg, 'load_from', 'N/A')}")
  summary_lines.append("")

  
  summary_lines.append("Detailed Results:")
  summary_lines.append("-" * 30)
  for loop_count, fid_score in results.items():
    if fid_score is not None:
      summary_lines.append(f"Loop {loop_count:2d}: FID = {fid_score:7.3f}")
    else:
      summary_lines.append(f"Loop {loop_count:2d}: FID = ERROR/MISSING")
  
  # Add detailed config contents at the end
  summary_lines.append("")
  summary_lines.append("Configuration Details:")
  summary_lines.append("-" * 30)
  summary_lines.append("")
  config_str = str(base_cfg)
  summary_lines.append(config_str)
  summary_lines.append("=" * 60)
  
  # Print to console
  print("\n" + "\n".join(summary_lines))
  
  # Save to file
  summary_file = os.path.join(base_workdir, 'evaluation_summary.txt')
  with open(summary_file, 'w') as f:
    f.write("\n".join(summary_lines))
  
  print(f"\n📁 Results saved to: {summary_file}")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Show distributed setup on startup
  print_distributed_setup()

  if FLAGS.config is None or FLAGS.workdir is None:
    raise app.UsageError('--config and --workdir are required')

  base_cfg = FLAGS.config
  # Get the actual config file path from FLAGS
  import sys
  config_path = None
  for arg in sys.argv:
    if arg.startswith('--config='):
      config_path = arg.split('=', 1)[1]
      break
  config_path = config_path or 'N/A'

  # Validate checkpoint
  if not hasattr(base_cfg, 'load_from') or base_cfg.load_from is None:
    raise app.UsageError('Config must set load_from to a checkpoint directory for evaluation.')

  # Get loop list from config
  if not hasattr(base_cfg, 'evaluation') or not hasattr(base_cfg.evaluation, 'loop_list'):
    raise app.UsageError('Config must set evaluation.loop_list for LoopedDiT evaluation.')
  loop_list = base_cfg.evaluation.loop_list

  # Base workdir
  base_workdir = FLAGS.workdir
  os.makedirs(base_workdir, exist_ok=True)

  # Store results for summary
  results = {}

  # Iterate and evaluate FID for each loop count
  for n_loops in loop_list:
    loop_workdir = os.path.join(base_workdir, f'loop_{n_loops}')
    os.makedirs(loop_workdir, exist_ok=True)

    cfg = _prepare_eval_config(base_cfg, n_loops)

    # Reuse training pipeline in eval-only mode to run FID once
    train.train_and_evaluate(cfg, loop_workdir)

    # Extract FID result after evaluation
    fid_score = extract_fid_from_logs(loop_workdir)
    results[n_loops] = fid_score
    print(f'Loop {n_loops}: FID = {fid_score}')

  # Generate comprehensive results summary
  generate_results_summary(base_workdir, loop_list, results, base_cfg, config_path)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
