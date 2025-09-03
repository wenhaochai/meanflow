"""ImageNet DiT example.

This script trains a DiT on the ImageNet dataset.
The data is loaded using pytorch dataset.
"""
from copy import deepcopy
from functools import partial
from typing import Any
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
from utils.cache_util import setup_compilation_cache, log_cache_to_wandb

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
    
  log_for_0("Initializing WandB...")
  
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

  log_for_0('Initializing params...')
  variables = init({'params': key}, x, t, y)
  log_for_0('Initializing params done.')

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
  print_params(params['net'])
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
  # Note: Detailed timing within JAX functions may not be accurate due to async execution
  # For more accurate timing, consider using JAX profiler or timing outside pmap
  
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

  # Minimal gradient monitoring (no clipping): global norm
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
    import time
    fid_eval_start_time = time.time()
    log_for_0('Eval fid at epoch: {}'.format(epoch))

    sample_gen_start_time = time.time()
    samples_all = sample_util.generate_fid_samples(
          state, workdir, config, p_sample_step, run_p_sample_step_inner
    )
    sample_gen_time = time.time() - sample_gen_start_time
    log_for_0(f'FID sample generation completed in {sample_gen_time:.2f}s')
    
    stats_start_time = time.time()
    mu, sigma = fid_util.compute_stats(samples_all, inception_net, batch_size=config.fid.stats_batch_size)
    fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
    stats_time = time.time() - stats_start_time
    log_for_0(f'FID stats computation completed in {stats_time:.2f}s')
    
    log_for_0(f'FID w/ EMA at {samples_all.shape[0]} samples: {fid_score}')
    
    writer.write_scalars(epoch+1, {'FID_ema': fid_score})
    writer.flush()
    
    fid_eval_total_time = time.time() - fid_eval_start_time
    log_for_0(f'Total FID evaluation time: {fid_eval_total_time:.2f}s')
    
    # Log to WandB - ensure we have access to global variables
    if WANDB_AVAILABLE and wandb.run is not None and jax.process_index() == 0:
      wandb.log({
        'fid_quality/FID_ema': fid_score,
        'fid_timing/fid_eval_time': fid_eval_total_time,
        'fid_timing/fid_sample_gen_time': sample_gen_time,
        'fid_timing/fid_stats_time': stats_time,
      }, step=epoch+1, commit=True)
      
  return evaluator

#######################################################
# Main
#######################################################

def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> TrainState:
  import time
  overall_start_time = time.time()
  
  ########### Initialize ###########
  init_start_time = time.time()
  log_for_0('=== Starting initialization phase ===')
  
  # Initialize WandB first
  wandb_run = init_wandb(config, workdir)
  
  # Set JAX compilation cache with optimized settings
  cache_dir, cache_stats = setup_compilation_cache(config)
  
  # Log cache info to WandB (now that wandb is initialized)
  if cache_stats:
    log_cache_to_wandb(cache_stats)
  
  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  )

  rng = random.key(config.training.seed)
  image_size = config.dataset.image_size

  log_for_0('config.training.batch_size: {}'.format(config.training.batch_size))

  if config.training.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.training.batch_size // jax.process_count()
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))

  init_time = time.time() - init_start_time
  log_for_0(f'=== Initialization phase completed in {init_time:.2f}s ===')
  
  ########### Create DataLoaders ###########
  dataloader_start_time = time.time()
  log_for_0('=== Starting data loader creation ===')
  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  train_loader, steps_per_epoch = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='train',
  )
  log_for_0('Steps per Epoch: {}'.format(steps_per_epoch))
  dataloader_time = time.time() - dataloader_start_time
  log_for_0(f'=== Data loader creation completed in {dataloader_time:.2f}s ===')

  ########### Create Model ###########
  model_start_time = time.time()
  log_for_0('=== Starting model creation ===')
  model_config = config.model.to_dict()
  model_str = model_config.pop('cls')

  model = MeanFlow(
    model_str=model_str,
    model_config=model_config,
    **config.sampling,
    **config.method,
  )
  model_time = time.time() - model_start_time
  log_for_0(f'=== Model creation completed in {model_time:.2f}s ===')

  ########### Create Train State ###########
  train_state_start_time = time.time()
  log_for_0('=== Starting train state creation ===')
  base_lr = config.training.learning_rate
  state = create_train_state(rng, config, model, image_size, lr_value=base_lr)
  if config.load_from is not None:
    checkpoint_start_time = time.time()
    log_for_0('Loading checkpoint...')
    state = restore_checkpoint(state, config.load_from)
    checkpoint_time = time.time() - checkpoint_start_time
    log_for_0(f'Checkpoint loaded in {checkpoint_time:.2f}s')
  
  step_offset = int(state.step)
  epoch_offset = step_offset // steps_per_epoch
  
  state = jax_utils.replicate(state)
  ema_fn = ema_schedules(config)

  latent_manager = LatentManager(config.dataset.vae, config.fid.device_batch_size, image_size)
  train_state_time = time.time() - train_state_start_time
  log_for_0(f'=== Train state creation completed in {train_state_time:.2f}s ===')

  compilation_setup_start_time = time.time()
  log_for_0('=== Starting compilation setup ===')
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
  log_for_0('Initial compilation, this might take some minutes...')

  vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())

  ########### Sampling ###########
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
  
  compilation_setup_time = time.time() - compilation_setup_start_time
  log_for_0(f'=== Compilation setup completed in {compilation_setup_time:.2f}s ===')

  if config.eval_only:
    eval_start_time = time.time()
    log_for_0('=== Starting evaluation only mode ===')
    fid_evaluator(state, epoch_offset)
    eval_time = time.time() - eval_start_time
    log_for_0(f'=== Evaluation completed in {eval_time:.2f}s ===')
    return state

  setup_time = time.time() - overall_start_time
  log_for_0(f'=== Total setup time: {setup_time:.2f}s ===')

  ########### Training Loop ###########
  training_start_time = time.time()
  log_for_0('=== Starting training loop ===')
  for epoch in range(epoch_offset, config.training.num_epochs):
    epoch_start_time = time.time()

    if jax.process_count() > 1:
      train_loader.sampler.set_epoch(epoch)
    log_for_0('epoch {}...'.format(epoch))
    
    ########### Sampling ###########
    if (epoch+1) % config.training.sample_per_epoch == 0 and config.training.get('sample_on_training', True):
      sampling_start_time = time.time()
      log_for_0(f'Samples at epoch {epoch}...')
      vis_sample = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager)
      vis_sample = make_grid_visualization(vis_sample, grid=4)
      writer.write_images(epoch+1, {'vis_sample': vis_sample})
      writer.flush()
      
      # Log sample images to WandB
      if WANDB_AVAILABLE and wandb.run is not None and jax.process_index() == 0:
        # Convert to wandb Image format
        # vis_sample is expected to be numpy array with shape (H, W, C) and values in [0, 1]
        wandb_image = wandb.Image(vis_sample, caption=f"Generated samples at epoch {epoch}")
        wandb.log({"samples/generated": wandb_image}, commit=True)
      
      sampling_time = time.time() - sampling_start_time
      log_for_0(f'Sampling at epoch {epoch} completed in {sampling_time:.2f}s')

    ########### Train ###########
    epoch_train_start_time = time.time()
    timer = Timer()
    log_for_0('epoch {}...'.format(epoch))
    timer.reset()
    
    data_load_times = []
    data_prep_times = []
    train_step_times = []
    
    data_load_start = time.time()
    for n_batch, batch in enumerate(train_loader):
      data_load_time = time.time() - data_load_start
      data_load_times.append(data_load_time)
      
      
      step = epoch * steps_per_epoch + n_batch

      data_prep_start = time.time()
      batch = input_pipeline.prepare_batch_data(batch) # the batch contains latent, both mean and var.
      data_prep_time = time.time() - data_prep_start
      data_prep_times.append(data_prep_time)
      
      train_step_start = time.time()
      state, metrics = p_train_step(state, batch)
      train_step_time = time.time() - train_step_start
      train_step_times.append(train_step_time)
      
      if epoch == epoch_offset and n_batch == 0:
        log_for_0('Initial compilation completed. Reset timer.')
        compilation_time = timer.elapse_with_reset()
        log_for_0('p_train_step compiled in {:.2f}s'.format(compilation_time))
      
      data_load_start = time.time()

      ########### Metrics ###########
      train_metrics.append(metrics)
      if (step+1) % config.training.log_per_step == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = jax.tree.map(lambda x: float(x.mean()), train_metrics)
        summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset() 
        summary["ep"] = epoch
        writer.write_scalars(step + 1, summary)

        # Log timing averages over the last log_per_step steps
        num_recent = min(config.training.log_per_step, len(data_load_times))
        avg_data_load = sum(data_load_times[-num_recent:]) / num_recent if num_recent > 0 else 0
        avg_data_prep = sum(data_prep_times[-num_recent:]) / num_recent if num_recent > 0 else 0
        avg_train_step = sum(train_step_times[-num_recent:]) / num_recent if num_recent > 0 else 0

        # Log to WandB with organized groups
        if WANDB_AVAILABLE and wandb.run is not None and jax.process_index() == 0:
          wandb_metrics = {
            # Model quality metrics
            'quality/v_loss': summary.get('v_loss', 0),
            
            # Training metrics
            'training/loss': summary.get('loss', 0),
            'training/lr': summary.get('lr', 0),
            'training/steps_per_second': summary.get('steps_per_second', 0),
            'training/grad_norm': summary.get('grad_norm', 0),
            
            # Timing metrics
            'timing/data_load_time': avg_data_load,
            'timing/data_prep_time': avg_data_prep, 
            'timing/train_step_time': avg_train_step,
          }
          wandb.log(wandb_metrics, step=step+1, commit=True)

        train_metrics = []

    epoch_train_time = time.time() - epoch_train_start_time
    log_for_0(f'Epoch {epoch} training completed in {epoch_train_time:.2f}s')

    ########### Save Checkpoint ###########
    if (
      (epoch+1) % config.training.checkpoint_per_epoch == 0
      or (epoch+1) == config.training.num_epochs
    ):
      checkpoint_start_time = time.time()
      save_checkpoint(state, workdir, epoch=epoch+1)
      checkpoint_time = time.time() - checkpoint_start_time
      log_for_0(f'Checkpoint saved in {checkpoint_time:.2f}s')

    ########### FID ###########
    if (
        (epoch+1) % config.training.fid_per_epoch == 0
        or (epoch+1) == config.training.num_epochs
      ) and config.fid.on_training:
      fid_start_time = time.time()
      log_for_0(f'Starting FID evaluation at epoch {epoch}...')
      fid_evaluator(state, epoch)
      fid_time = time.time() - fid_start_time
      log_for_0(f'FID evaluation completed in {fid_time:.2f}s')
    
    epoch_total_time = time.time() - epoch_start_time
    log_for_0(f'=== Epoch {epoch} total time: {epoch_total_time:.2f}s ===')

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  
  total_training_time = time.time() - training_start_time
  total_overall_time = time.time() - overall_start_time
  log_for_0(f'=== Total training time: {total_training_time:.2f}s ===')
  log_for_0(f'=== Total overall time: {total_overall_time:.2f}s ===')
  
  # Log final timing summary to WandB
  if WANDB_AVAILABLE and wandb.run is not None and jax.process_index() == 0:
    wandb.log({
      'timing/final_total_training_time': total_training_time,
      'timing/final_total_overall_time': total_overall_time,
    }, commit=True)
  
  # Finish WandB run
  if WANDB_AVAILABLE and wandb.run is not None and jax.process_index() == 0:
    log_for_0("Finishing WandB run...")
    # Force final flush before finishing
    wandb.run.log({}, commit=True)
    wandb.finish()
  
  return state
