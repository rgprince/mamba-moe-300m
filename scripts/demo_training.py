#!/usr/bin/env python3
"""
Simple training demo for Mamba-MoE 300M

Demonstrates the full training pipeline with dummy data.
Run on Colab TPU to test distributed training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random
import time

print("=" * 60)
print("Mamba-MoE 300M - Training Demo")
print("=" * 60)

# Imports
from src.model import create_model_from_config, ModelConfig
from src.training import (
    create_train_step,
    create_train_state,
    create_learning_rate_schedule,
    create_optimizer,
    ConsoleLogger
)
from src.training.distributed import DistributedTrainer

# Configuration
print("\n[1/6] Configuration")
DEMO_STEPS = 10
BATCH_SIZE = 4  # Small for demo
SEQ_LEN = 512   # Short sequences for speed
LEARNING_RATE = 3e-4
VOCAB_SIZE = 32000

print(f"  Steps: {DEMO_STEPS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Devices: {jax.device_count()}")

# Load model
print("\n[2/6] Loading model")
config_path = 'configs/model_config.yaml'
model = create_model_from_config(config_path)
model_config = ModelConfig.from_yaml(config_path)

print(f"  ✓ Model: {model_config.name}")
print(f"  ✓ Layers: {model_config.num_layers}")
print(f"  ✓ Hidden dim: {model_config.hidden_dim}")

# Initialize model
print("\n[3/6] Initializing parameters")
rng = random.PRNGKey(42)
rng, init_rng, dropout_rng = random.split(rng, 3)

# Dummy input for initialization
dummy_input = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
variables = model.init(init_rng, dummy_input, deterministic=True)
params = variables['params']

# Count parameters
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"  ✓ Parameters: {param_count / 1e6:.1f}M")

# Create optimizer
print("\n[4/6] Setting up optimizer")
lr_schedule = create_learning_rate_schedule(
    warmup_steps=100,
    max_learning_rate=LEARNING_RATE,
    total_steps=1000,
    schedule_type='cosine'
)

optimizer = create_optimizer(
    learning_rate_fn=lr_schedule,
    weight_decay=0.1,
    max_grad_norm=1.0
)

print(f"  ✓ Optimizer: AdamW")
print(f"  ✓ LR: {LEARNING_RATE}")
print(f"  ✓ Schedule: warmup + cosine decay")

# Create training state
print("\n[5/6] Creating training state")
state = create_train_state(model, params, optimizer, lr_schedule, dropout_rng)
print(f"  ✓ State initialized")

# Setup distributed training
train_step_fn = create_train_step(model, lr_schedule)
trainer = DistributedTrainer(train_step_fn)
replicated_state = trainer.replicate_state(state)
print(f"  ✓ Distributed setup: {jax.device_count()} devices")

# Training loop
print("\n[6/6] Training loop (dummy data)")
print("-" * 60)

logger = ConsoleLogger(log_interval=1)  # Log every step for demo

for step in range(DEMO_STEPS):
    # Create dummy batch
    rng, data_rng = random.split(rng)
    batch = {
        'input_ids': random.randint(data_rng, (BATCH_SIZE, SEQ_LEN), 0, VOCAB_SIZE),
        'labels': random.randint(data_rng, (BATCH_SIZE, SEQ_LEN), 0, VOCAB_SIZE)
    }
    
    # Training step
    start_time = time.time()
    replicated_state, metrics = trainer.train_step(replicated_state, batch)
    step_time = time.time() - start_time
    
    # Add timing info
    metrics['step_time_ms'] = step_time * 1000
    metrics['tokens_per_sec'] = (BATCH_SIZE * SEQ_LEN) / step_time
    
    # Log
    logger.log(metrics, step)

print("-" * 60)

# Get final state
final_state = trainer.unreplicate_state(replicated_state)

# Summary
print("\n" + "=" * 60)
print("✅ TRAINING DEMO COMPLETE!")
print("=" * 60)

print("\nSummary:")
print(f"  ✓ Ran {DEMO_STEPS} training steps")
print(f"  ✓ Model: {param_count / 1e6:.1f}M parameters")
print(f"  ✓ Devices: {jax.device_count()}")
print(f"  ✓ Final step: {final_state.step}")

print("\nWhat was tested:")
print("  [✓] Model forward pass")
print("  [✓] Loss computation")
print("  [✓] Gradient computation")
print("  [✓] Parameter updates")
print("  [✓] Distributed training (pmap)")
print("  [✓] Metrics tracking")

print("\nNext steps:")
print("  1. Prepare real training data (JSONL)")
print("  2. Train/download tokenizer")
print("  3. Run full training with scripts/train.py")

print("\n" + "=" * 60)
