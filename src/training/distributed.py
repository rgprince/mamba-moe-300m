"""
TPU/GPU distributed training utilities
"""

import jax
import jax.numpy as jnp
from flax import jax_utils
from typing import Any, Callable
import numpy as np


def get_device_count() -> int:
    """Get number of available devices (TPU cores or GPUs)"""
    return jax.device_count()


def get_local_device_count() -> int:
    """Get number of local devices"""
    return jax.local_device_count()


def replicate_on_devices(pytree: Any) -> Any:
    """
    Replicate pytree across all devices
    
    Args:
        pytree: Any JAX pytree (params, state, etc.)
    
    Returns:
        Replicated pytree with leading device dimension
    """
    return jax_utils.replicate(pytree)


def unreplicate_from_devices(pytree: Any) -> Any:
    """
    Get first copy from replicated pytree
    
    Args:
        pytree: Replicated pytree
    
    Returns:
        Single copy (from device 0)
    """
    return jax_utils.unreplicate(pytree)


def shard_batch(batch: dict, num_devices: int) -> dict:
    """
    Shard batch across devices
    
    Args:
        batch: Dict with arrays [batch_size, ...]
        num_devices: Number of devices
    
    Returns:
        Dict with arrays [num_devices, batch_per_device, ...]
    """
    def _shard_array(arr):
        """Shard single array"""
        batch_size = arr.shape[0]
        per_device = batch_size // num_devices
        
        # Reshape to [num_devices, per_device, ...]
        return arr.reshape(num_devices, per_device, *arr.shape[1:])
    
    return jax.tree_map(_shard_array, batch)


def create_distributed_train_step(train_step_fn: Callable) -> Callable:
    """
    Create distributed training step using pmap
    
    Args:
        train_step_fn: Single-device training step
    
    Returns:
        Multi-device training step
    """
    # Use pmap for data parallelism
    pmap_train_step = jax.pmap(
        train_step_fn,
        axis_name='batch',  # For gradient synchronization
        donate_argnums=(0,)  # Donate state to avoid copies
    )
    
    return pmap_train_step


def synchronize_gradients(grads: Any, axis_name: str = 'batch') -> Any:
    """
    Average gradients across devices
    
    Args:
        grads: Gradient pytree
        axis_name: pmap axis name
    
    Returns:
        Averaged gradients
    """
    return jax.lax.pmean(grads, axis_name=axis_name)


class DistributedTrainer:
    """
    Wrapper for distributed training on TPUs/GPUs
    
    Handles:
    - State replication
    - Batch sharding
    - Gradient synchronization
    """
    
    def __init__(self, train_step_fn: Callable):
        """
        Initialize distributed trainer
        
        Args:
            train_step_fn: Single-device training step
        """
        self.num_devices = get_device_count()
        self.train_step_pmap = create_distributed_train_step(train_step_fn)
        
        print(f"Distributed trainer initialized:")
        print(f"  Devices: {self.num_devices}")
        print(f"  Local devices: {get_local_device_count()}")
    
    def replicate_state(self, state: Any) -> Any:
        """Replicate state across devices"""
        return replicate_on_devices(state)
    
    def shard_batch(self, batch: dict) -> dict:
        """Shard batch for all devices"""
        return shard_batch(batch, self.num_devices)
    
    def train_step(self, state: Any, batch: dict) -> tuple:
        """
        Execute distributed training step
        
        Args:
            state: Replicated training state
            batch: Non-sharded batch
        
        Returns:
            (new_state, metrics)
        """
        # Shard batch
        sharded_batch = self.shard_batch(batch)
        
        # Run pmap step
        new_state, metrics = self.train_step_pmap(state, sharded_batch)
        
        # Average metrics across devices
        metrics = jax.tree_map(lambda x: jnp.mean(x), metrics)
        
        return new_state, metrics
    
    def unreplicate_state(self, state: Any) -> Any:
        """Get state from first device"""
        return unreplicate_from_devices(state)


def create_data_parallel_trainer(
    model,
    optimizer,
    learning_rate_fn,
    params,
    rng
):
    """
    Create complete distributed training setup
    
    Args:
        model: Flax model
        optimizer: Optax optimizer
        learning_rate_fn: LR schedule
        params: Model parameters
        rng: Random key
    
    Returns:
        (distributed_trainer, replicated_state)
    """
    from .train_step import create_train_step, create_train_state
    
    # Create training step
    train_step = create_train_step(model, learning_rate_fn)
    
    # Create initial state
    state = create_train_state(model, params, optimizer, learning_rate_fn, rng)
    
    # Create distributed trainer
    trainer = DistributedTrainer(train_step)
    
    # Replicate state
    replicated_state = trainer.replicate_state(state)
    
    return trainer, replicated_state
