import jax
import os
from flax.training import checkpoints

from utils.logging_util import log_for_0


def restore_checkpoint(state, workdir):
  if not os.path.exists(workdir):
    raise FileNotFoundError(f'Checkpoint not found: {workdir}')
  state = checkpoints.restore_checkpoint(workdir, state)
  return state


def save_checkpoint(state, workdir, epoch=None, keep=3):
  state = jax.device_get(jax.tree.map(lambda x: x[0], state))
  if epoch is not None:
    log_for_0('Saving checkpoint epoch %d.', epoch)
    checkpoints.save_checkpoint_multiprocess(workdir, state, epoch, keep=keep)
    log_for_0('Checkpoint epoch %d saved.', epoch)
  else:
    # Fallback to step for backward compatibility
    step = int(state.step)
    log_for_0('Saving checkpoint step %d.', step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=keep)
    log_for_0('Checkpoint step %d saved.', step)