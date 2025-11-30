# Mamba-MoE 300M - Kaggle Training Guide

## Quick Setup for Kaggle

### Step 1: Create Kaggle Notebook
1. Go to https://www.kaggle.com/code
2. Create new notebook
3. **Accelerator**: GPU P100 or TPU v3-8
4. **Internet**: ON

### Step 2: Clone Repository
```python
!git clone https://github.com/rgprince/mamba-moe-300m.git
%cd mamba-moe-300m
```

### Step 3: Install Dependencies
```python
!pip install -q jax[tpu] flax optax chex einops pyyaml pydantic datasets sentencepiece
```

### Step 4: Test Model (Optional)
```python
# Quick architecture test
!python3 scripts/test_model_simple.py
```

### Step 5: Run Real Training Test
```python
# This will:
# - Download WikiText-2 dataset
# - Tokenize with character-level
# - Run 20 training steps
# - Show loss/perplexity metrics

!python3 scripts/test_real_training.py
```

**Expected compilation time:**
- GPU P100: ~3-5 minutes (first step only)
- TPU v3-8: ~5-10 minutes (first step only)
- After compilation: ~100-200ms per step

### Step 6: Check Output
You should see:
```
[7/7] Training on real data
Compiling (this takes 2-3 min on first step)...
✓ Compilation done (180s)
------------------------------------------------------------
Step   0 | loss=10.3724 ce=10.3723 ppl=32000.45 lr=0.000015 | 1800ms
Step   1 | loss=10.3701 ce=10.3699 ppl=31850.23 lr=0.000030 | 120ms
Step   2 | loss=10.3678 ce=10.3676 ppl=31701.12 lr=0.000045 | 115ms
...
Step  19 | loss=10.2845 ce=10.2843 ppl=29345.67 lr=0.000285 | 110ms

✅ REAL DATA TRAINING COMPLETE!
```

---

## Model Details

**Architecture:**
- **Name**: mamba-moe-300m-v1
- **Parameters**: ~509M
- **Layers**: 24
- **Hidden dim**: 1024
- **Vocab**: 32,000 (SentencePiece)

**Components:**
- Mamba 2 SSM (all layers)
- Soft MoE (6 layers: 3, 7, 11, 15, 19, 23)
- Differential Attention (4 layers: 5, 11, 17, 23)
- Hierarchical Memory (4-tier)

**Training:**
- Optimizer: AdamW
- Learning rate: 3e-4 (with warmup + cosine decay)
- Batch size: Adjust based on GPU memory
- Sequence length: Up to 8192

---

## Memory Requirements

| Hardware | Batch Size | Seq Length | Status |
|----------|------------|------------|--------|
| Colab Free TPU | 1 | 128 | ⚠️ Slow compilation |
| Kaggle GPU P100 | 2-4 | 512 | ✅ Good |
| Kaggle TPU v3-8 | 8-16 | 1024 | ✅ Best |
| GCP TPU v3-8 | 16-32 | 2048 | ✅ Optimal |

---

## Troubleshooting

### Issue: Compilation hangs
**Solution**: Use Kaggle instead of Colab free tier

### Issue: Out of memory
**Solution**: Reduce batch size or sequence length in script:
```python
BATCH_SIZE = 1  # Reduce this
SEQ_LEN = 128   # Reduce this
```

### Issue: JAX not found
**Solution**: 
```python
!pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

## Full Training (Large Dataset)

For actual training on large datasets:

1. **Prepare data** in JSONL format:
```json
{"text": "Your training text here..."}
{"text": "More training text..."}
```

2. **Train tokenizer**:
```python
from src.data import SPTokenizer
tokenizer = SPTokenizer.train(
    input_files=['data.txt'],
    vocab_size=32000,
    model_prefix='tokenizer'
)
```

3. **Update config** (`configs/stage1_pretrain.yaml`):
```yaml
data:
  train_path: 'data/train.jsonl'
  tokenizer_path: 'tokenizer.model'

training:
  total_steps: 100000
  batch_size: 16
  learning_rate: 3e-4
```

4. **Run training**:
```python
!python3 scripts/train.py --config configs/stage1_pretrain.yaml
```

---

## Repository Structure

```
mamba-moe-300m/
├── configs/
│   ├── model_config.yaml          # Full 509M model
│   ├── model_config_tiny.yaml     # Tiny model for testing
│   └── stage1_pretrain.yaml       # Training config
├── src/
│   ├── model/                     # Model architecture (1,774 lines)
│   ├── data/                      # Data pipeline (482 lines)
│   └── training/                  # Training infrastructure (945 lines)
├── scripts/
│   ├── test_model_simple.py       # Test architecture
│   ├── test_real_training.py      # Test with real data
│   ├── test_phase2.py             # Test all modules
│   └── train.py                   # Main training script
└── notebooks/
    └── colab_phase2_test.md       # Colab guide
```

---

## GitHub

**Repository**: https://github.com/rgprince/mamba-moe-300m

**Latest code includes:**
- ✅ Full 509M architecture
- ✅ Complete training pipeline
- ✅ Data loading with HuggingFace datasets
- ✅ TPU/GPU distribution
- ✅ Checkpointing & logging
- ✅ Real data training test

---

## Support

For issues, check:
1. GitHub Issues
2. Code comments in source files
3. This guide

**Good luck with training!** 🚀
