"""Utilities for JAX compilation cache management."""

import os
import jax
from utils.logging_util import log_for_0

# WandB import with optional availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_cache_stats(cache_dir):
    """Get JAX compilation cache statistics.
    
    Args:
        cache_dir (str): Path to the cache directory
        
    Returns:
        dict: Dictionary containing cache statistics
            - total_size_mb (float): Total size in MB
            - num_files (int): Number of cache files
    """
    if not os.path.exists(cache_dir):
        return {'total_size_mb': 0, 'num_files': 0}
    
    total_size = 0
    num_files = 0
    
    for dirpath, _, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
                num_files += 1
            except (OSError, IOError):
                continue  # Skip files that can't be accessed
    
    return {
        'total_size_mb': total_size / (1024 * 1024),
        'num_files': num_files
    }


def setup_compilation_cache(config):
    """Setup JAX compilation cache with optimized settings.
    
    Args:
        config: Configuration object containing system settings
        
    Returns:
        tuple: (cache_dir, cache_stats) if cache is enabled, (None, None) otherwise
    """
    if not config.system.set_compilation_cache:
        log_for_0("JAX compilation cache disabled")
        return None, None
        
    # Create detailed cache directory structure for better organization
    model_cls = config.model.cls
    batch_size = config.training.batch_size
    image_size = config.dataset.image_size
    num_devices = jax.device_count()
    
    # Generate architecture-specific suffix for looped models
    arch_suffix = ""
    if "LoopedDiT" in model_cls and hasattr(config.model, 'n_prefix'):
        # For segmented looped models: include prefix/shared/loops/suffix info
        n_prefix = getattr(config.model, 'n_prefix', 0)
        n_shared = getattr(config.model, 'n_shared', 0) 
        n_loops = getattr(config.model, 'n_loops', 0)
        n_suffix = getattr(config.model, 'n_suffix', 0)
        use_loop_emb = getattr(config.model, 'use_loop_emb', False)
        arch_suffix = (
            f"_p{n_prefix}s{n_shared}l{n_loops}e{n_suffix}"
            f"_le{int(use_loop_emb)}"
        )
        log_for_0(
            f"Looped model detected: {n_prefix} prefix + {n_shared} shared×{n_loops} loops + {n_suffix} suffix "
            f"(loop_emb={use_loop_emb})"
        )
    
    cache_dir = os.path.join(
        config.system.jax_compilation_cache_root, 
        f"{model_cls}{arch_suffix}_bs_{batch_size}_img_{image_size}_gpu_{num_devices}"
    )
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use the modern JAX config API instead of environment variables
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    
    # Optimize cache settings for better performance
    min_compile_time = config.system.cache_min_compile_time
    max_size_bytes = int(config.system.cache_max_size_gb * 1024**3)
    cache_debug = config.system.cache_debug
    
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 1)     # Cache all entry sizes
    jax.config.update("jax_compilation_cache_max_size", max_size_bytes)
    
    # Enable cache debugging for better monitoring
    if cache_debug:
        jax.config.update("jax_explain_cache_misses", True)
    
    log_for_0(f'JAX compilation cache configured:')
    log_for_0(f'  Directory: {cache_dir}')
    log_for_0(f'  Min compile time: {min_compile_time}s')
    log_for_0(f'  Max cache size: {config.system.cache_max_size_gb}GB')
    log_for_0(f'  Cache miss explanation: {cache_debug}')
    
    # Get initial cache statistics (only on main process to avoid I/O contention)
    cache_stats = None
    if jax.process_index() == 0:
        cache_stats = get_cache_stats(cache_dir)
        log_for_0(f'  Current cache: {cache_stats["total_size_mb"]:.1f}MB, {cache_stats["num_files"]} files')
    
    return cache_dir, cache_stats

def clear_cache(cache_dir):
    """Clear compilation cache directory.
    
    Args:
        cache_dir (str): Path to the cache directory to clear
    """
    if not os.path.exists(cache_dir):
        log_for_0(f"Cache directory does not exist: {cache_dir}")
        return
        
    try:
        import shutil
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        log_for_0(f"✅ Cache cleared: {cache_dir}")
    except Exception as e:
        log_for_0(f"❌ Failed to clear cache: {e}")


def get_cache_info(config):
    """Get comprehensive cache information.
    
    Args:
        config: Configuration object
        
    Returns:
        dict: Comprehensive cache information
    """
    if not config.system.set_compilation_cache:
        return {'enabled': False}
        
    model_cls = config.model.cls
    batch_size = config.training.batch_size
    image_size = config.dataset.image_size
    num_devices = jax.device_count()
    
    # Generate architecture-specific suffix for looped models
    arch_suffix = ""
    if "LoopedDiT" in model_cls and hasattr(config.model, 'n_prefix'):
        # For segmented looped models: include prefix/shared/loops/suffix info
        n_prefix = getattr(config.model, 'n_prefix', 0)
        n_shared = getattr(config.model, 'n_shared', 0) 
        n_loops = getattr(config.model, 'n_loops', 0)
        n_suffix = getattr(config.model, 'n_suffix', 0)
        use_loop_emb = getattr(config.model, 'use_loop_emb', True)
        arch_suffix = (
            f"_p{n_prefix}s{n_shared}l{n_loops}e{n_suffix}"
            f"_le{int(use_loop_emb)}"
        )
    
    cache_dir = os.path.join(
        config.system.jax_compilation_cache_root, 
        f"{model_cls}{arch_suffix}_bs_{batch_size}_img_{image_size}_gpu_{num_devices}"
    )
    
    cache_stats = get_cache_stats(cache_dir)
    
    return {
        'enabled': True,
        'cache_dir': cache_dir,
        'stats': cache_stats,
        'config': {
            'min_compile_time': config.system.cache_min_compile_time,
            'max_size_gb': config.system.cache_max_size_gb,
            'debug_enabled': config.system.cache_debug,
        },
        'jax_config': {
            'cache_dir': jax.config.jax_compilation_cache_dir if hasattr(jax.config, 'jax_compilation_cache_dir') else None,
        }
    }
