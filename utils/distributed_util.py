"""Utilities for JAX distributed training setup."""

import os
import jax
from utils.logging_util import log_for_0


def auto_init_distributed():
    """Automatically initialize JAX distributed mode when appropriate.
    
    Detection logic:
    1. Manual override via environment variables
    2. SLURM multi-node detection
    3. SLURM multi-task detection
    4. Default to single-node mode
    
    Environment variables:
    - JAX_FORCE_DISTRIBUTED=true: Force distributed initialization
    - JAX_DISABLE_DISTRIBUTED=true: Disable distributed initialization
    """
    
    # Method 1: Check SLURM environment for multi-node jobs
    slurm_nnodes = int(os.environ.get('SLURM_NNODES', '1'))
    slurm_ntasks = int(os.environ.get('SLURM_NTASKS', '1'))
    
    # Method 2: Check if running in distributed environment
    # These env vars are typically set by SLURM for multi-node jobs
    slurm_procid = os.environ.get('SLURM_PROCID')
    slurm_localid = os.environ.get('SLURM_LOCALID')
    
    # Method 3: Manual override via environment variable
    force_distributed = os.environ.get('JAX_FORCE_DISTRIBUTED', 'false').lower() == 'true'
    disable_distributed = os.environ.get('JAX_DISABLE_DISTRIBUTED', 'false').lower() == 'true'
    
    should_init_distributed = False
    reason = "single-node mode (default)"
    
    # Decision logic
    if disable_distributed:
        should_init_distributed = False
        reason = "manually disabled via JAX_DISABLE_DISTRIBUTED"
    elif force_distributed:
        should_init_distributed = True
        reason = "manually forced via JAX_FORCE_DISTRIBUTED"
    elif slurm_nnodes > 1:
        should_init_distributed = True
        reason = f"multi-node SLURM job ({slurm_nnodes} nodes)"
    elif slurm_ntasks > 1 and slurm_procid is not None:
        should_init_distributed = True
        reason = f"multi-task SLURM job ({slurm_ntasks} tasks)"
    
    # Initialize if needed
    if should_init_distributed:
        try:
            jax.distributed.initialize()
            log_for_0(f"✅ JAX distributed initialized: {reason}")
            
            # Log distributed setup info
            log_for_0(f"   Process {jax.process_index()}/{jax.process_count()}")
            log_for_0(f"   Local devices: {len(jax.local_devices())}")
            log_for_0(f"   Total devices: {jax.device_count()}")
            
        except Exception as e:
            log_for_0(f"⚠️  JAX distributed initialization failed: {e}")
            log_for_0(f"   Falling back to single-node mode")
    else:
        log_for_0(f"ℹ️  JAX distributed not initialized: {reason}")
        log_for_0(f"   Local devices: {len(jax.local_devices())}")


def get_distributed_info():
    """Get information about the current distributed setup.
    
    Returns:
        dict: Dictionary containing distributed setup information
    """
    return {
        'process_index': jax.process_index(),
        'process_count': jax.process_count(),
        'local_device_count': jax.local_device_count(),
        'total_device_count': jax.device_count(),
        'local_devices': [str(d) for d in jax.local_devices()],
        'is_distributed': jax.process_count() > 1,
        'slurm_info': {
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'nnodes': os.environ.get('SLURM_NNODES'),
            'ntasks': os.environ.get('SLURM_NTASKS'),
            'procid': os.environ.get('SLURM_PROCID'),
            'localid': os.environ.get('SLURM_LOCALID'),
            'nodeid': os.environ.get('SLURM_NODEID'),
        }
    }


def print_distributed_setup():
    """Print detailed information about the distributed setup."""
    info = get_distributed_info()
    
    log_for_0("=== JAX Distributed Setup ===")
    log_for_0(f"Process: {info['process_index']}/{info['process_count']}")
    log_for_0(f"Local devices: {info['local_device_count']}")
    log_for_0(f"Total devices: {info['total_device_count']}")
    log_for_0(f"Is distributed: {info['is_distributed']}")
    
    if info['slurm_info']['job_id']:
        log_for_0("SLURM environment:")
        for key, value in info['slurm_info'].items():
            if value is not None:
                log_for_0(f"  {key.upper()}: {value}")
    
    log_for_0("Local devices details:")
    for i, device in enumerate(info['local_devices']):
        log_for_0(f"  Device {i}: {device}")
    log_for_0("=============================")


def validate_distributed_config(config):
    """Validate configuration for distributed training.
    
    Args:
        config: Training configuration object
        
    Raises:
        ValueError: If configuration is incompatible with distributed setup
    """
    info = get_distributed_info()
    
    # Check batch size divisibility
    if config.training.batch_size % info['process_count'] != 0:
        raise ValueError(
            f"Batch size ({config.training.batch_size}) must be divisible by "
            f"process count ({info['process_count']})"
        )
    
    # Check local batch size vs local devices
    local_batch_size = config.training.batch_size // info['process_count']
    if local_batch_size % info['local_device_count'] != 0:
        raise ValueError(
            f"Local batch size ({local_batch_size}) must be divisible by "
            f"local device count ({info['local_device_count']})"
        )
    
    log_for_0(f"✅ Distributed config validation passed")
    log_for_0(f"   Global batch size: {config.training.batch_size}")
    log_for_0(f"   Local batch size: {local_batch_size}")
    log_for_0(f"   Per-device batch size: {local_batch_size // info['local_device_count']}")