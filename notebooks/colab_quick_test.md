# Colab Quick Test

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rgprince/mamba-moe-300m/blob/main/notebooks/colab_quick_test.md)

## ⚡ Quick Setup and Test on Colab TPU

### Important: Make sure you're using a fresh Colab session!

### Step 1: Setup Runtime
```
Runtime → Change runtime type → TPU
```

### Step 2: Clone and Install (Copy-paste this entire block)
```python
# Clean start - remove any old clones
!rm -rf mamba-moe-300m

# Clone fresh
!git clone https://github.com/rgprince/mamba-moe-300m.git
%cd mamba-moe-300m

# Install dependencies
!pip install -q jax[tpu] flax optax chex einops pyyaml pydantic

print("✅ Setup complete!")
```

### Step 3: Run Simple Test
```python
!python3 scripts/test_model_simple.py
```

### Expected Output:
```
============================================================
Mamba-MoE 300M - Simple Test
============================================================

[1/5] Testing JAX...
✓ JAX 0.4.x imported successfully

[2/5] Testing Flax...
✓ Flax imported successfully

[3/5] Testing model imports...
✓ Model package imported successfully

[4/5] Loading model config...
✓ Config loaded: mamba-moe-300m-v1
  - Layers: 24
  - Hidden dim: 1024
  - Estimated params: XXX.XM

[5/5] Testing model forward pass...
✓ Model created
✓ Input created: (2, 64)
  Initializing parameters...
✓ Model initialized: XXX.XM parameters
  Running forward pass...
✓ Forward pass complete
  Output shape: (2, 64, 32000)
  Expected: (2, 64, 32000)
✓ Output shape correct!

============================================================
✅ ALL TESTS PASSED!
============================================================
```

---

## 🐛 Troubleshooting

### Error: "No module named 'src'"
**Solution**: Make sure you ran `%cd mamba-moe-300m` after cloning

### Error: "ImportError: cannot import name 'training'"
**Solution**: You have an old version. Run:
```python
!rm -rf mamba-moe-300m
!git clone https://github.com/rgprince/mamba-moe-300m.git
```

### Error: "No module named 'jax'"
**Solution**: Install dependencies:
```python
!pip install jax[tpu] flax optax
```

---

## 📊 What This Tests

- ✅ JAX/Flax imports work
- ✅ Model package loads without circular imports
- ✅ Config parsing works
- ✅ Model instantiation succeeds
- ✅ Forward pass completes
- ✅ Output shape is correct
- ✅ All ~300M parameters initialized

---

## 🚀 Next Steps

If all tests pass:
1. **Architecture is validated!** ✅
2. Next: Implement training loop (Phase 2)
3. Then: Prepare data pipeline
4. Finally: Train the model!

**Repo**: https://github.com/rgprince/mamba-moe-300m
