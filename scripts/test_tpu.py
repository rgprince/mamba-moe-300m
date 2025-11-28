#!/usr/bin/env python3
"""
Test TPU connection and basic JAX functionality
"""

import sys
try:
    import jax
    import jax.numpy as jnp
    from jax import random
except ImportError:
    print("❌ JAX not installed. Please install: pip install jax[tpu]")
    sys.exit(1)


def test_tpu_connection():
    """Test TPU availability and basic operations"""
    
    print("=" * 60)
    print("TPU Connection Test")
    print("=" * 60)
    
    # Check JAX version
    print(f"\n✓ JAX version: {jax.__version__}")
    
    # Check devices
    devices = jax.devices()
    print(f"\n✓ Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  [{i}] {device}")
    
    # Check for TPU
    tpu_devices = [d for d in devices if 'tpu' in str(d).lower()]
    if tpu_devices:
        print(f"\n✅ TPU detected! {len(tpu_devices)} TPU cores available")
    else:
        print(f"\n⚠️  No TPU detected. Running on: {devices[0]}")
        print("   If you expected TPU, check your runtime settings.")
    
    # Test basic computation
    print("\n" + "=" * 60)
    print("Testing Basic Computation")
    print("=" * 60)
    
    try:
        # Simple matrix multiplication
        key = random.PRNGKey(0)
        x = random.normal(key, (1000, 1000))
        y = jnp.dot(x, x.T)
        y.block_until_ready()  # Wait for computation
        
        print("✓ Matrix multiplication (1000x1000): SUCCESS")
        
        # Test sharding across devices
        if len(devices) > 1:
            from jax.experimental.pjit import pjit
            from jax.sharding import Mesh, PartitionSpec as P
            
            print(f"✓ Multi-device sharding available ({len(devices)} devices)")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during computation: {e}")
        return False


if __name__ == "__main__":
    success = test_tpu_connection()
    sys.exit(0 if success else 1)
