""" Main file for running the ImageNet experiments.
"""
import jax
from absl import app, flags
from ml_collections import config_flags

import train
from utils import logging_util
from utils.logging_util import log_for_0
from utils.distributed_util import auto_init_distributed, print_distributed_setup

# Auto-initialize distributed mode
auto_init_distributed()

logging_util.supress_checkpt_info()

import warnings

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_bool('debug', False, 'Debugging mode.')

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Print distributed setup information
  print_distributed_setup()
  log_for_0('FLAGS.config: \n{}'.format(FLAGS.config))

  train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
