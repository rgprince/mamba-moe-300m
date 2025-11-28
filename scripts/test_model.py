#!/usr/bin/env python3
"""
Test basic model instantiation and forward pass
"""

import sys
import jax
import jax.numpy as jnp
from jax import random

# Add src to path
sys.path.insert(0, '/home/prince/my/mamba-moe-300m')

from src.model import create_model_from_config, count_parameters


def test_model():
    """Test model creation and forward pass"""
    
    print("=" * 60)
    print("Mamba-MoE Model Test")
    print("=" * 60)
    
    # Load config
    config_path = '/home/prince/my/mamba-moe-300m/configs/model_config.yaml'
    print(f"\n✓ Loading config from: {config_path}")
    
    try:
        model = create_model_from_config(config_path)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Estimate parameter count
    from src.model.config import ModelConfig
    config = ModelConfig.from_yaml(config_path)
    param_count = count_parameters(config)
    print(f"✓ Estimated parameters: {param_count / 1e6:.1f}M")
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 128
        
        rng = random.PRNGKey(0)
        input_ids = random.randint(rng, (batch_size, seq_len), 0, config.embedding.vocab_size)
        
        print(f"\n✓ Input shape: {input_ids.shape}")
        
        # Initialize model
        variables = model.init(random.PRNGKey(1), input_ids, deterministic=True)
        print(f"✓ Model initialized")
        
        # Count actual parameters
        params = variables['params']
        actual_param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"✓ Actual parameters: {actual_param_count / 1e6:.1f}M")
        
        # Forward pass
        logits = model.apply(variables, input_ids, deterministic=True)
        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Expected: ({batch_size}, {seq_len}, {config.embedding.vocab_size})")
        
        # Verify output shape
        assert logits.shape == (batch_size, seq_len, config.embedding.vocab_size)
        print("\n✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
