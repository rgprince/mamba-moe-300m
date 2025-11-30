"""
Optimizer and learning rate scheduling
"""

import optax
from typing import Callable, Optional


def create_learning_rate_schedule(
    warmup_steps: int = 2000,
    max_learning_rate: float = 3e-4,
    total_steps: int = 100000,
    min_learning_rate: Optional[float] = None,
    schedule_type: str = "cosine"
) -> Callable:
    """
    Create learning rate schedule with warmup
    
    Args:
        warmup_steps: Number of warmup steps
        max_learning_rate: Peak learning rate
        total_steps: Total training steps
        min_learning_rate: Minimum LR (default: max_lr * 0.1)
        schedule_type: "cosine" or "linear"
    
    Returns:
        Learning rate schedule function
    """
    if min_learning_rate is None:
        min_learning_rate = max_learning_rate * 0.1
    
    # Warmup schedule
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=max_learning_rate,
        transition_steps=warmup_steps
    )
    
    # Decay schedule
    if schedule_type == "cosine":
        decay = optax.cosine_decay_schedule(
            init_value=max_learning_rate,
            decay_steps=total_steps - warmup_steps,
            alpha=min_learning_rate / max_learning_rate
        )
    else:  # linear
        decay = optax.linear_schedule(
            init_value=max_learning_rate,
            end_value=min_learning_rate,
            transition_steps=total_steps - warmup_steps
        )
    
    # Combine schedules
    schedule = optax.join_schedules(
        schedules=[warmup, decay],
        boundaries=[warmup_steps]
    )
    
    return schedule


def create_optimizer(
    learning_rate_fn: Callable,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1
) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with gradient clipping
    
    Args:
        learning_rate_fn: Learning rate schedule
        weight_decay: Weight decay coefficient
        beta1: Adam beta1
        beta2: Adam beta2
        epsilon: Adam epsilon
        max_grad_norm: Max gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
    
    Returns:
        Optax optimizer
    """
    # Build optimizer chain
    optimizer = optax.chain(
        # Gradient clipping
        optax.clip_by_global_norm(max_grad_norm),
        
        # Gradient accumulation (if needed)
        optax.MultiSteps(
            optax.adamw(
                learning_rate=learning_rate_fn,
                b1=beta1,
                b2=beta2,
                eps=epsilon,
                weight_decay=weight_decay
            ),
            every_k_schedule=gradient_accumulation_steps
        ) if gradient_accumulation_steps > 1 else optax.adamw(
            learning_rate=learning_rate_fn,
            b1=beta1,
            b2=beta2,
            eps=epsilon,
            weight_decay=weight_decay
        )
    )
    
    return optimizer


def create_optimizer_from_config(config: dict, total_steps: int) -> tuple:
    """
    Create optimizer and schedule from config
    
    Args:
        config: Training config dict
        total_steps: Total training steps
    
    Returns:
        (optimizer, learning_rate_fn)
    """
    # Create schedule
    lr_fn = create_learning_rate_schedule(
        warmup_steps=config.get('warmup_steps', 2000),
        max_learning_rate=config.get('learning_rate', 3e-4),
        total_steps=total_steps,
        min_learning_rate=config.get('min_learning_rate'),
        schedule_type=config.get('schedule_type', 'cosine')
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        learning_rate_fn=lr_fn,
        weight_decay=config.get('weight_decay', 0.1),
        beta1=config.get('beta1', 0.9),
        beta2=config.get('beta2', 0.95),
        epsilon=config.get('epsilon', 1e-8),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1)
    )
    
    return optimizer, lr_fn
