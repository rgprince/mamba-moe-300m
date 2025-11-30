"""
Main training script for Mamba-MoE 300M

Run with:
    python scripts/train.py --config configs/stage1_pretrain.yaml
"""

import sys
import os
from pathlib import Path
import argparse
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random

# Model imports
from src.model import create_model_from_config, ModelConfig

# Data imports  
from src.data import SPTokenizer
from src.data.loader import StreamingDataLoader

# Training imports
from src.training import (
    create_train_step,
    create_train_state,
    create_optimizer_from_config,
    CheckpointManager,
    ConsoleLogger
)
from src.training.distributed import create_data_parallel_trainer


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training loop"""
    parser = argparse.ArgumentParser(description="Train Mamba-MoE 300M")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/', help='Checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    print("=" * 60)
    print("Mamba-MoE 300M Training")
    print("=" * 60)
    
    config = load_config(args.config)
    print(f"\n✓ Config loaded from {args.config}")
    
    # Initialize model
    model_config_path = config['model']['config_path']
    model = create_model_from_config(model_config_path)
    model_config = ModelConfig.from_yaml(model_config_path)
    
    print(f"✓ Model loaded: {model_config.name}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    
    # Initialize tokenizer (placeholder - needs actual tokenizer file)
    print("\n[INFO] Tokenizer setup required")
    print("  Download or train a SentencePiece tokenizer")
    print("  Update config with tokenizer path")
    
    # Training parameters
    training_config = config.get('training', {})
    total_steps = training_config.get('total_steps', 100000)
    batch_size = training_config.get('batch_size', 32)
    
    print(f"\n✓ Training config:")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Devices: {jax.device_count()}")
    
    # Create optimizer
    optimizer, lr_fn = create_optimizer_from_config(training_config, total_steps)
    print(f"✓ Optimizer created: AdamW")
    
    # Initialize RNG
    rng = random.PRNGKey(training_config.get('seed', 42))
    
    # Dummy initialization (replace with real data)
    print(f"\n[INFO] Model initialization")
    dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
    params = model.init(rng, dummy_input, deterministic=True)
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"✓ Parameters: {param_count / 1e6:.1f}M")
    
    # Create training state
    rng, dropout_rng = random.split(rng)
    state = create_train_state(model, params['params'], optimizer, lr_fn, dropout_rng)
    
    # Setup distributed training
    trainer, replicated_state = create_data_parallel_trainer(
        model, optimizer, lr_fn, params['params'], dropout_rng
    )
    
    # Setup checkpointing
    ckpt_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        max_to_keep=3,
        save_interval_steps=1000
    )
    
    # Resume if requested
    if args.resume:
        state = ckpt_manager.restore(state)
        replicated_state = trainer.replicate_state(state)
    
    # Setup logging
    logger = ConsoleLogger(log_interval=100)
    
    print("\n" + "=" * 60)
    print("✅ Training setup complete!")
    print("=" * 60)
    print("\n[NEXT STEPS]")
    print("1. Prepare training data (JSONL format)")
    print("2. Train or download tokenizer")
    print("3. Update config with data/tokenizer paths")
    print("4. Run: python scripts/train.py --config configs/stage1_pretrain.yaml")
    
    print("\n[NOTE] This is a template training script")
    print("Actual training requires:")
    print("  - Data pipeline setup")
    print("  - Tokenizer initialization")
    print("  - Training loop implementation")
    

if __name__ == "__main__":
    main()
