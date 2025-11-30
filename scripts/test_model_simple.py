#!/usr/bin/env python3
"""
Simplified test script that works regardless of path setup
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Mamba-MoE 300M - Simple Test")
print("=" * 60)

# Test 1: Import JAX
print("\n[1/5] Testing JAX...")
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    print(f"✓ JAX {jax.__version__} imported successfully")
except ImportError as e:
    print(f"✗ JAX import failed: {e}")
    print("Run: pip install jax[tpu] flax optax")
    sys.exit(1)

# Test 2: Import Flax
print("\n[2/5] Testing Flax...")
try:
    import flax.linen as nn
    print(f"✓ Flax imported successfully")
except ImportError as e:
    print(f"✗ Flax import failed: {e}")
    sys.exit(1)

# Test 3: Import model package
print("\n[3/5] Testing model imports...")
try:
    from src.model import (
        ModelConfig,
        count_parameters,
        create_model_from_config
    )
    print("✓ Model package imported successfully")
except ImportError as e:
    print(f"✗ Model import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load config
print("\n[4/5] Loading model config...")
try:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs',
        'model_config.yaml'
    )
    config = ModelConfig.from_yaml(config_path)
    print(f"✓ Config loaded: {config.name}")
    print(f"  - Layers: {config.num_layers}")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Estimated params: {count_parameters(config) / 1e6:.1f}M")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create and test model
print("\n[5/5] Testing model forward pass...")
try:
    # Create model
    model = create_model_from_config(config_path)
    print("✓ Model created")
    
    # Create dummy input
    batch_size = 2
    seq_len = 64
    rng = random.PRNGKey(42)
    input_ids = random.randint(
        rng, 
        (batch_size, seq_len), 
        0, 
        config.embedding.vocab_size
    )
    print(f"✓ Input created: {input_ids.shape}")
    
    # Initialize model
    print("  Initializing parameters...")
    variables = model.init(random.PRNGKey(0), input_ids, deterministic=True)
    
    # Count actual parameters
    params = variables['params']
    actual_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"✓ Model initialized: {actual_params / 1e6:.1f}M parameters")
    
    # Forward pass
    print("  Running forward pass...")
    logits = model.apply(variables, input_ids, deterministic=True)
    print(f"✓ Forward pass complete")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {config.embedding.vocab_size})")
    
    # Verify shape
    assert logits.shape == (batch_size, seq_len, config.embedding.vocab_size)
    print("✓ Output shape correct!")
    
except Exception as e:
    print(f"\n✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nNext steps:")
print("1. Model architecture is working")
print("2. Ready for training infrastructure")
print("3. Continue with Phase 2: Data pipeline & training loop")
