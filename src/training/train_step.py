"""
Training step and state management
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from typing import Any, Callable, Dict, Tuple
from functools import partial


class TrainState(train_state.TrainState):
    """
    Extended training state with additional metrics
    
    Adds:
    - Learning rate tracking
    - Step counting
    - Dropout RNG
    """
    dropout_rng: jax.random.PRNGKey
    
    def apply_gradients(self, *, grads, **kwargs):
        """Apply gradients and update dropout RNG"""
        # Split RNG for next step
        dropout_rng, new_dropout_rng = jax.random.split(self.dropout_rng)
        
        # Update state
        new_state = super().apply_gradients(grads=grads, **kwargs)
        
        return new_state.replace(dropout_rng=new_dropout_rng)


def create_train_step(model, learning_rate_fn):
    """
    Create a JIT-compiled training step function
    
    Args:
        model: Flax model
        learning_rate_fn: Learning rate schedule function
    
    Returns:
        train_step function
    """
    
    @jax.jit
    def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]) -> Tuple[TrainState, Dict[str, float]]:
        """
        Single training step
        
        Args:
            state: Current training state
            batch: Dict with 'input_ids' and 'labels'
        
        Returns:
            (new_state, metrics dict)
        """
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
        
        def loss_fn(params):
            """Compute loss and auxiliary outputs"""
            # Forward pass
            logits, aux = model.apply(
                {'params': params},
                batch['input_ids'],
                deterministic=False,
                return_aux=True,
                rngs={'dropout': dropout_rng}
            )
            
            # Main loss: cross-entropy
            # Flatten for loss computation
            logits_flat = logits.reshape(-1, logits.shape[-1])
            labels_flat = batch['labels'].reshape(-1)
            
            # Cross-entropy (ignoring padding)
            ce_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits_flat,
                labels_flat
            )
            
            # Mask padding tokens
            mask = (labels_flat != 0)  # Assuming pad_id = 0
            ce_loss = jnp.sum(ce_loss * mask) / jnp.sum(mask)
            
            # Auxiliary losses (MoE load balancing, etc.)
            aux_losses = aux.get('aux_losses', {})
            aux_loss_total = sum(aux_losses.values()) if aux_losses else 0.0
            
            # Total loss
            total_loss = ce_loss + 0.01 * aux_loss_total
            
            # Return loss and metrics
            metrics = {
                'loss': total_loss,
                'ce_loss': ce_loss,
                'aux_loss': aux_loss_total,
                'perplexity': jnp.exp(ce_loss)
            }
            
            return total_loss, metrics
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        
        # Compute gradient norm
        grad_norm = optax.global_norm(grads)
        metrics['grad_norm'] = grad_norm
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        
        # Add learning rate to metrics
        metrics['learning_rate'] = learning_rate_fn(state.step)
        
        return new_state, metrics
    
    return train_step


def create_train_state(
    model,
    params,
    optimizer,
    learning_rate_fn,
    rng: jax.random.PRNGKey
) -> TrainState:
    """
    Create initial training state
    
    Args:
        model: Flax model
        params: Model parameters
        optimizer: Optax optimizer
        learning_rate_fn: Learning rate schedule
        rng: Random key
    
    Returns:
        TrainState instance
    """
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=rng
    )
