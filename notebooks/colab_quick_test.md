# Colab Quick Test

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rgprince/mamba-moe-300m/blob/main/notebooks/colab_quick_test.md)

## Quick Setup and Test on Colab TPU

### 1. Clone and Install
```python
# Clone repo
!git clone https://github.com/rgprince/mamba-moe-300m.git
%cd mamba-moe-300m

# Install dependencies
!pip install -q jax[tpu] flax optax chex einops pyyaml pydantic sentencepiece
```

### 2. Test TPU
```python
!python scripts/test_tpu.py
```

### 3. Test Model
```python
!python scripts/test_model.py
```

### 4. Manual Forward Pass
```python
import jax
import jax.numpy as jnp
from jax import random
import sys

sys.path.insert(0, '.')
from src.model import create_model_from_config, ModelConfig, count_parameters

# Load model
print("Loading model...")
model = create_model_from_config('configs/model_config.yaml')
config = ModelConfig.from_yaml('configs/model_config.yaml')

print(f"Estimated parameters: {count_parameters(config) / 1e6:.1f}M")

# Create input
batch_size, seq_len = 4, 256
rng = random.PRNGKey(42)
input_ids = random.randint(rng, (batch_size, seq_len), 0, config.embedding.vocab_size)

# Initialize
variables = model.init(random.PRNGKey(0), input_ids, deterministic=True)

# Forward pass
logits, aux = model.apply(variables, input_ids, deterministic=True, return_aux=True)

print(f"✅ Output shape: {logits.shape}")
print(f"✅ Memory shape: {aux['memory'].shape}")
```

### 5. Benchmark
```python
import time

# Warmup
for _ in range(3):
    _ = model.apply(variables, input_ids, deterministic=True)

# Benchmark
num_runs = 10
start = time.time()
for _ in range(num_runs):
    logits = model.apply(variables, input_ids, deterministic=True)
    logits.block_until_ready()
end = time.time()

avg_time = (end - start) / num_runs
tokens_per_sec = (batch_size * seq_len) / avg_time

print(f"Average forward pass: {avg_time*1000:.2f}ms")
print(f"Throughput: {tokens_per_sec:.0f} tokens/sec")
```

## Expected Results

- ✅ Model instantiates (~290-310M parameters)
- ✅ Forward pass completes
- ✅ Output shape: `[batch, seq_len, 32000]`
- ✅ TPU throughput: >>10k tokens/sec

## Next Steps

If all tests pass, the architecture is working! Next:
1. Implement training loop
2. Prepare data pipeline
3. Train the model!
