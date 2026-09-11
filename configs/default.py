"""Default Hyperparameter configuration."""


import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # ------------------------------------------------------------
  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  
  dataset.name = 'imgnet_latent'
  dataset.root = '/path/to/work/data/meanflow/train_merged.pt'

  dataset.num_workers = 0
  dataset.prefetch_factor = 2
  dataset.pin_memory = False
  dataset.cache = False
  dataset.shuffle_seed = 0   # DistributedSampler seed (data order); training.seed covers init/noise

  dataset.image_size = 32 
  dataset.image_channels = 4 
  dataset.num_classes = 1000
  dataset.vae = 'mse'

  # ------------------------------------------------------------
  # Training
  config.training = training = ml_collections.ConfigDict()

  training.learning_rate = 0.0001
  training.batch_size = 256

  training.num_epochs = 80
  training.log_per_step = 100
  training.sample_on_training = True
  training.sample_per_epoch = 1
  training.checkpoint_per_epoch = 10
  training.checkpoint_keep = 999
  training.fid_per_epoch = 10
  training.half_precision = False # Not supported
  
  training.seed = 42

  training.adam_b2 = 0.95
  training.ema_val = 0.9999
  training.ema_type = 'const'

  # ------------------------------------------------------------
  # MeanFlow
  config.method = method = ml_collections.ConfigDict()
  
  # Noise Distribution
  method.noise_dist = 'logit_normal'
  method.P_mean = -0.4
  method.P_std = 1.0

  # Loss
  method.data_proportion = 0.75
  method.class_dropout_prob = 0.1

  # Guidance
  method.guidance_eq = 'cfg'
  method.omega = 1.0
  method.kappa = 0.5

  # Time Interval
  method.t_start = 0.0
  method.t_end = 1.0

  # Training Dynamics
  method.norm_p = 1.0
  method.norm_eps = 0.01

  # MoE Auxiliary Loss (for load balancing)
  method.aux_loss_weight = 0.0

  # ------------------------------------------------------------
  # model
  config.model = model = ml_collections.ConfigDict()
  model.cls = 'LoopedDiT_B_4'
  model.n_prefix = 2
  model.n_shared = 2
  model.n_loops = 4
  model.n_suffix = 2
  # model.num_experts = 4
  # model.num_active_experts = 1
  # model.num_shared_experts = 0
  model.loop_emb_max = 1.0
  model.loop_emb_min = 0.0

  # ------------------------------------------------------------
  # Sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.seed = 0
  sampling.num_steps = 1
  sampling.schedule = 'default'
  sampling.sampling_timesteps = None
  sampling.num_classes = dataset.num_classes

  # ------------------------------------------------------------
  # FID
  config.fid = fid = ml_collections.ConfigDict()
  fid.on_training = True
  fid.num_samples = 50000
  fid.device_batch_size = 16
  fid.stats_batch_size = 200
  fid.cache_ref = '/path/to/work/data/meanflow/imgnet256_adm.npz'

  # ------------------------------------------------------------
  # WandB (Weights & Biases) Configuration
  config.wandb = wandb = ml_collections.ConfigDict()
  wandb.enable = False                    # Enable/disable wandb logging
  wandb.root = '/path/to/work/project/meanflow_private'
  wandb.project = 'meanflow'             # WandB project name
  wandb.entity = 'reself'                # WandB entity (username/team), None for default
  wandb.name = None                      # Run name, None for auto-generated
  wandb.tags = []                        # List of tags for the run
  wandb.notes = ''                       # Notes for the run
  wandb.group = 'baseline'               # Group name for organizing runs
  wandb.job_type = 'train'               # Job type (train, eval, etc.)
  wandb.mode = 'offline'                 # 'online', 'offline', or 'disabled'
  wandb.save_code = False                # Save code to wandb
  
  # Loop Evaluation Configuration
  config.evaluation = evaluation = ml_collections.ConfigDict()
  evaluation.loop_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 32]    # List of loop counts to evaluate (for LoopedDiT)
  evaluation.stepsize_preserving = True

  # ------------------------------------------------------------
  # System Configuration
  config.system = system = ml_collections.ConfigDict()
  system.set_compilation_cache = True  # Whether to set JAX compilation cache
  system.jax_compilation_cache_root = '/path/to/work/cache/jax_cache'  # JAX compilation cache root directory
  system.cache_min_compile_time = 0.1  # Minimum compilation time (seconds) to cache
  system.cache_max_size_gb = 5  # Maximum cache size in GB
  system.cache_debug = False  # Enable cache debugging and miss explanation

  # others
  config.load_from = None
  config.eval_only = False
  config.auto_resume = True  # resume from the latest checkpoint_* in workdir when load_from is None

  return config

def enforce_relations(config):
  config.sampling.num_classes = config.dataset.num_classes

def metrics():
  return [
      'train_loss',
      'eval_loss',
      'train_accuracy',
      'eval_accuracy',
      'steps_per_second',
      'train_learning_rate',
  ]
