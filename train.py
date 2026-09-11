"""ImageNet DiT example.

This script trains a DiT on the ImageNet dataset.
The data is loaded using pytorch dataset.
"""
from copy import deepcopy
from functools import partial
from typing import Any
import glob
import os

import jax
import jax.numpy as jnp
import ml_collections
import optax
from clu import metric_writers
from flax import jax_utils
from flax.training import common_utils, train_state
from jax import lax, random
from optax._src.alias import *

import utils.input_pipeline as input_pipeline
from meanflow import MeanFlow, generate
from utils import fid_util, sample_util
from utils.ckpt_util import restore_checkpoint, save_checkpoint
from utils.ema_util import ema_schedules, update_ema
from utils.info_util import print_params
from utils.logging_util import Timer, log_for_0
from utils.vae_util import LatentManager
from utils.vis_util import make_grid_visualization
from utils.cache_util import setup_compilation_cache

# WandB import with optional availability
try:
  import wandb
  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False
  log_for_0("WandB not available, skipping wandb logging")

#######################################################
# WandB Integration
#######################################################

def init_wandb(config, workdir):
  """Initialize WandB logging if enabled and available."""
  if not config.wandb.enable or not WANDB_AVAILABLE:
    log_for_0("WandB logging disabled or not available")
    return None
    
  if jax.process_index() != 0:
    return None
  
  job_name = os.path.basename(workdir)
  os.makedirs(config.wandb.root, exist_ok=True)
  
  wandb_config = {
    "model": dict(config.model),
    "dataset": dict(config.dataset), 
    "training": dict(config.training),
    "sampling": dict(config.sampling),
    "method": dict(config.method),
    "fid": dict(config.fid),
    "devices": jax.device_count(),
    "local_devices": jax.local_device_count(),
  }
  
  wandb.init(
    project=config.wandb.project,
    entity=config.wandb.entity,
    name=job_name,
    tags=list(config.wandb.tags),
    notes=config.wandb.notes,
    group=config.wandb.group,
    job_type=config.wandb.job_type,
    mode=config.wandb.mode,
    save_code=config.wandb.save_code,
    config=wandb_config,
    dir=config.wandb.root,
  )
  
  log_for_0(f"WandB initialized: {wandb.run.name} ({wandb.run.id})")
  return wandb.run

# Direct wandb.log() calls are now used throughout the code instead of this wrapper function

#######################################################
# Initialize
#######################################################

def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 4)
  x = jnp.ones(input_shape)
  t = jnp.ones((1,), dtype=int)
  y = jnp.ones((1,), dtype=int)

  @jax.jit
  def init(*args):
    return model.init(*args)

  variables = init({'params': key}, x, t, y)

  param_count = sum(x.size for x in jax.tree.leaves(variables['params']))
  log_for_0("Total trainable parameters: " + str(param_count))
  return variables, variables['params']


class TrainState(train_state.TrainState):
  ema_params: Any


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, lr_value
):
  """
  Create initial training state.
  ---
  apply_fn: output a dict, with key 'loss', 'mse'
  """

  rng, rng_init = random.split(rng)
  
  _, params = initialized(rng_init, image_size, model)
  ema_params = deepcopy(params)
  ema_params = update_ema(ema_params, params, 0)
  # print_params(params['net'])
  tx = optax.adamw(
      learning_rate=lr_value,
      weight_decay=0,
      b2=config.training.adam_b2,
  )
  state = TrainState.create(
      apply_fn=partial(model.apply, method=model.forward),
      params=params,
      ema_params=ema_params,
      tx=tx,
  )
  return state

#######################################################
# Train Step
#######################################################

def compute_metrics(dict_losses):
  metrics = dict_losses.copy()
  metrics = lax.all_gather(metrics, axis_name='batch')
  metrics = jax.tree.map(lambda x: x.flatten(), metrics)  # (batch_size,)
  return metrics


def train_step_with_vae(state, batch, rng_init, config, lr, ema_fn, latent_mnger):
  """
  Perform a single training step.
  """
  
  rng_step = random.fold_in(rng_init, state.step)
  rng_base = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))

  cached = batch['image'] # [B, H, W, C]
  rng_base, rng_vae = random.split(rng_base)
  images = latent_mnger.cached_encode(cached, rng_vae) # [B, H, W, C] sample latent

  labels = batch['label']

  def loss_fn(params):
    """loss function used for training."""
    variables = {
        "params": params,
    }
    outputs = state.apply_fn(
      variables,
      imgs=images,
      labels=labels,
      rngs=dict(gen=rng_base,),
    )
    loss, dict_losses = outputs
    return loss, (dict_losses,)
  
  # Forward and backward pass
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  grads = lax.pmean(grads, axis_name='batch')

  grad_global_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree.leaves(grads)]))

  dict_losses, = aux[1]
  metrics = compute_metrics(dict_losses)
  metrics["lr"] = lr
  metrics["grad_norm"] = grad_global_norm

  # Parameter update
  new_state = state.apply_gradients(
    grads=grads,
  )

  # EMA update
  ema_value = ema_fn(state.step)
  new_ema = update_ema(new_state.ema_params, new_state.params, ema_value)
  new_state = new_state.replace(ema_params=new_ema)

  return new_state, metrics

#######################################################
# Sampling and Metrics
#######################################################

def sample_step(variable, sample_idx, model, rng_init, device_batch_size, config):
  """
  sample_idx: each random sampled image corrresponds to a seed
  """
  rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
  images = generate(variable, model, rng_sample, n_sample=device_batch_size, config=config)
  images = images.transpose(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
  return images


def run_p_sample_step(p_sample_step, state, sample_idx, latent_manager, ema=True):
  variable = {"params": state.ema_params if ema else state.params}
  latent = p_sample_step(variable, sample_idx=sample_idx)
  latent = latent.reshape(-1, *latent.shape[2:])

  # Decode
  samples = latent_manager.decode(latent)
  assert not jnp.any(jnp.isnan(samples)), f"There is nan in decoded samples! Latent range: {latent.min()}, {latent.max()}. nan in latent: {jnp.any(jnp.isnan(latent))}"

  samples = samples.transpose(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
  samples = 127.5 * samples + 128.0
  samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)

  jax.random.normal(random.key(0), ()).block_until_ready() # dist sync
  return samples


def get_fid_evaluator(workdir, config, writer, p_sample_step, latent_manager):
  inception_net = fid_util.build_jax_inception(batch_size=config.fid.stats_batch_size)
  stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)
  run_p_sample_step_inner = partial(run_p_sample_step, latent_manager=latent_manager)

  def evaluator(state, epoch):
    log_for_0('Eval fid at epoch: {}'.format(epoch))

    samples_all = sample_util.generate_fid_samples(
          state, workdir, config, p_sample_step, run_p_sample_step_inner
    )

    mu, sigma = fid_util.compute_stats(samples_all, inception_net, batch_size=config.fid.stats_batch_size)
    fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])

    log_for_0(f'FID w/ EMA at {samples_all.shape[0]} samples: {fid_score}')

    writer.write_scalars(epoch+1, {'FID_ema': fid_score})
    writer.flush()

    # Log to WandB
    if WANDB_AVAILABLE and wandb.run is not None and jax.process_index() == 0:
      wandb.log({
        'fid_quality/FID_ema': fid_score,
      }, step=epoch+1, commit=True)

  return evaluator

#######################################################
# Main
#######################################################

def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> TrainState:

  # Initialize WandB first
  init_wandb(config, workdir)

  # Set JAX compilation cache with optimized settings
  setup_compilation_cache(config)

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  )

  rng = random.key(config.training.seed)
  image_size = config.dataset.image_size

  if config.training.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.training.batch_size // jax.process_count()
  
  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  # Skip data loading for evaluation-only mode
  if config.eval_only:
    steps_per_epoch = 1 # avoid bug
  else:
    train_loader, steps_per_epoch = input_pipeline.create_split(
      config.dataset,
      local_batch_size,
      split='train',
    )

  model_config = config.model.to_dict()
  model_str = model_config.pop('cls')

  model = MeanFlow(
    model_str=model_str,
    model_config=model_config,
    **config.sampling,
    **config.method,
  )

  base_lr = config.training.learning_rate
  state = create_train_state(rng, config, model, image_size, lr_value=base_lr)
  if config.load_from is not None:
    state = restore_checkpoint(state, config.load_from)
  elif config.get('auto_resume', False) and glob.glob(os.path.join(workdir, 'checkpoint_*')):
    log_for_0('auto_resume: restoring latest checkpoint from %s', workdir)
    state = restore_checkpoint(state, workdir)

  step_offset = int(state.step)
  epoch_offset = step_offset // steps_per_epoch

  state = jax_utils.replicate(state)
  ema_fn = ema_schedules(config)

  latent_manager = LatentManager(config.dataset.vae, config.fid.device_batch_size, image_size)

  # Only compile training step if not in eval-only mode
  if not config.eval_only:
    p_train_step = jax.pmap(
        partial(
          train_step_with_vae,
          rng_init=rng,
          config=config,
          lr=base_lr,
          ema_fn=ema_fn,
          latent_mnger=latent_manager,
        ),
        axis_name='batch',
    )
    train_metrics = []

  vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())

  p_sample_step = jax.pmap(
    partial(sample_step,
            model=model,
            rng_init=random.PRNGKey(config.sampling.seed),
            device_batch_size=config.fid.device_batch_size,
            config=config,
    ),
    axis_name='batch'
  )

  if config.fid.on_training:
    fid_evaluator = get_fid_evaluator(workdir, config, writer, p_sample_step, latent_manager)

  if config.eval_only:
    fid_evaluator(state, epoch_offset)
    return state
  for epoch in range(epoch_offset, config.training.num_epochs):
    if jax.process_count() > 1:
      train_loader.sampler.set_epoch(epoch)
    log_for_0('epoch {}...'.format(epoch))

    # Sampling
    if (epoch+1) % config.training.sample_per_epoch == 0 and config.training.get('sample_on_training', True):
      vis_sample = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager)
      vis_sample = make_grid_visualization(vis_sample, grid=4)
      writer.write_images(epoch+1, {'vis_sample': vis_sample})
      writer.flush()

    # Training
    timer = Timer()
    timer.reset()

    for n_batch, batch in enumerate(train_loader):
      step = epoch * steps_per_epoch + n_batch
      batch = input_pipeline.prepare_batch_data(batch)
      state, metrics = p_train_step(state, batch)

      if epoch == epoch_offset and n_batch == 0:
        timer.elapse_with_reset()

      # Metrics
      train_metrics.append(metrics)
      if (step+1) % config.training.log_per_step == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = jax.tree.map(lambda x: float(x.mean()), train_metrics)
        summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset()
        summary["ep"] = epoch
        writer.write_scalars(step + 1, summary)

        # Log to WandB
        if WANDB_AVAILABLE and wandb.run is not None and jax.process_index() == 0:
          wandb_metrics = {
            'quality/v_loss': summary.get('v_loss', 0),
            'training/loss': summary.get('loss', 0),
            'training/lr': summary.get('lr', 0),
            'training/steps_per_second': summary.get('steps_per_second', 0),
            'training/grad_norm': summary.get('grad_norm', 0),
          }
          wandb.log(wandb_metrics, step=step+1, commit=True)

        train_metrics = []

    # Save Checkpoint
    if (
      (epoch+1) % config.training.checkpoint_per_epoch == 0
      or (epoch+1) == config.training.num_epochs
    ):
      save_checkpoint(state, workdir, epoch=epoch+1, keep=config.training.checkpoint_keep)

    # FID
    if (
        (epoch+1) % config.training.fid_per_epoch == 0
        or (epoch+1) == config.training.num_epochs
      ) and config.fid.on_training:
      fid_evaluator(state, epoch)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()

  # Finish WandB run
  if WANDB_AVAILABLE and wandb.run is not None and jax.process_index() == 0:
    wandb.finish()

  return state
