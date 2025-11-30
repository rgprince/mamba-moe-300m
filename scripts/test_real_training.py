#!/usr/bin/env python3
"""
End-to-end training test with real data

Downloads WikiText-2 dataset and runs actual training.
Small dataset (~4MB) perfect for testing the full pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random
import time
import json
import urllib.request
import zipfile
import os

print("=" * 60)
print("Mamba-MoE 300M - Real Data Training Test")
print("=" * 60)

# Configuration
BATCH_SIZE = 1
SEQ_LEN = 128
LEARNING_RATE = 3e-4
TOTAL_STEPS = 20  # Just 20 steps to verify it works
VOCAB_SIZE = 256  # Character-level (simple, no tokenizer needed)

print(f"\n[1/7] Configuration")
print(f"  Dataset: WikiText-2")
print(f"  Steps: {TOTAL_STEPS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Vocab: {VOCAB_SIZE} (character-level)")

# Download WikiText-2
print(f"\n[2/7] Downloading dataset")
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

wikitext_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
zip_path = data_dir / "wikitext-2.zip"
extract_dir = data_dir / "wikitext-2"

if not extract_dir.exists():
    print(f"  Downloading from {wikitext_url}")
    urllib.request.urlretrieve(wikitext_url, zip_path)
    print(f"  ✓ Downloaded to {zip_path}")
    
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"  ✓ Extracted to {extract_dir}")
else:
    print(f"  ✓ Dataset already exists at {extract_dir}")

# Read training data
train_file = extract_dir / "wikitext-2-raw" / "wiki.train.raw"
with open(train_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"  ✓ Loaded {len(text):,} characters")
print(f"  Sample: {text[:100]}")

# Character-level tokenization (simple, no SentencePiece needed)
print(f"\n[3/7] Creating tokenizer")
# Build vocab from data
chars = sorted(list(set(text)))
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}
actual_vocab_size = len(chars)

print(f"  ✓ Vocabulary size: {actual_vocab_size}")
print(f"  Example chars: {chars[:20]}")

# Tokenize entire dataset
token_ids = [char_to_id.get(ch, 0) for ch in text]
print(f"  ✓ Tokenized: {len(token_ids):,} tokens")

# Create batches
print(f"\n[4/7] Creating batches")
batches = []
for i in range(0, len(token_ids) - SEQ_LEN - 1, SEQ_LEN):
    if len(batches) >= TOTAL_STEPS * BATCH_SIZE:
        break
    
    input_ids = token_ids[i:i + SEQ_LEN]
    labels = token_ids[i + 1:i + SEQ_LEN + 1]
    
    if len(input_ids) == SEQ_LEN and len(labels) == SEQ_LEN:
        batches.append({
            'input_ids': jnp.array([input_ids], dtype=jnp.int32),
            'labels': jnp.array([labels], dtype=jnp.int32)
        })

print(f"  ✓ Created {len(batches)} batches")

# Load model
print(f"\n[5/7] Loading model")
from src.model import create_model_from_config, ModelConfig
from src.training import (
    create_train_step,
    create_train_state,
    create_learning_rate_schedule,
    create_optimizer
)

config_path = 'configs/model_config.yaml'
model = create_model_from_config(config_path)
model_config = ModelConfig.from_yaml(config_path)

print(f"  ✓ Model: {model_config.name}")
print(f"  ✓ Layers: {model_config.num_layers}")

# Initialize model
print(f"\n[6/7] Initializing model")
rng = random.PRNGKey(42)
rng, init_rng, dropout_rng = random.split(rng, 3)

dummy_input = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
variables = model.init(init_rng, dummy_input, deterministic=True)
params = variables['params']

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"  ✓ Parameters: {param_count / 1e6:.1f}M")

# Create optimizer
lr_schedule = create_learning_rate_schedule(
    warmup_steps=10,
    max_learning_rate=LEARNING_RATE,
    total_steps=TOTAL_STEPS,
    schedule_type='cosine'
)

optimizer = create_optimizer(
    learning_rate_fn=lr_schedule,
    weight_decay=0.1,
    max_grad_norm=1.0
)

# Create training state
state = create_train_state(model, params, optimizer, lr_schedule, dropout_rng)

# Create training step
train_step = create_train_step(model, lr_schedule)
train_step = jax.jit(train_step)  # JIT compile

print(f"  ✓ Training state created")

# Training loop
print(f"\n[7/7] Training on real data")
print("-" * 60)
print("Compiling (this takes 2-3 min on first step)...")

start_time = time.time()

for step in range(min(TOTAL_STEPS, len(batches))):
    batch = batches[step]
    
    step_start = time.time()
    state, metrics = train_step(state, batch)
    step_time = time.time() - step_start
    
    # Print metrics
    if step == 0:
        print(f"✓ Compilation done ({step_time:.1f}s)")
        print("-" * 60)
    
    loss = float(metrics['loss'])
    ce_loss = float(metrics['ce_loss'])
    ppl = float(metrics['perplexity'])
    lr = float(metrics['learning_rate'])
    
    print(f"Step {step:3d} | loss={loss:.4f} ce={ce_loss:.4f} ppl={ppl:.2f} lr={lr:.6f} | {step_time*1000:.0f}ms")

total_time = time.time() - start_time

print("-" * 60)
print("\n" + "=" * 60)
print("✅ REAL DATA TRAINING COMPLETE!")
print("=" * 60)

print(f"\nResults:")
print(f"  ✓ Trained on WikiText-2")
print(f"  ✓ Ran {TOTAL_STEPS} steps")
print(f"  ✓ Total time: {total_time:.1f}s")
print(f"  ✓ Avg step time: {total_time/TOTAL_STEPS:.2f}s")
print(f"  ✓ Final loss: {loss:.4f}")
print(f"  ✓ Final perplexity: {ppl:.2f}")

print(f"\nWhat was tested:")
print("  [✓] Real text data (WikiText-2)")
print("  [✓] Character tokenization")
print("  [✓] Batch creation")
print("  [✓] Model forward pass")
print("  [✓] Loss computation")
print("  [✓] Gradient updates")
print("  [✓] Learning rate schedule")
print("  [✓] Full training loop")

print(f"\n🎉 EVERYTHING WORKS!")
print("=" * 60)
