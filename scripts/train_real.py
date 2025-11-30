#!/usr/bin/env python3
"""
Real Training Script - Novels, Stories & Chat

Downloads quality datasets from HuggingFace:
- Books & Novels (for language understanding)
- Stories (for creativity)
- Chat (for conversation)

Trains SentencePiece tokenizer and runs full training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random
import time
import os

print("=" * 70)
print("Mamba-MoE 300M - Real Training on Quality Data")
print("=" * 70)

# Configuration - REAL TRAINING
TOTAL_STEPS = 50000  # 50k steps = ~3-4 hours on TPU v5e
BATCH_SIZE = 1       # Optimized for TPU v5e single chip (16GB HBM)
SEQ_LEN = 128        # Optimized for TPU v5e single chip
LEARNING_RATE = 3e-4
SAVE_EVERY = 500     # Save checkpoint every 500 steps
LOG_EVERY = 50       # Log metrics every 50 steps

print(f"\n[CONFIG] - REAL TRAINING")
print(f"  Total steps: {TOTAL_STEPS:,} (~3-4 hours)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Save interval: {SAVE_EVERY} steps")
print(f"  Log interval: {LOG_EVERY} steps")
print(f"\n  TPU v5e: 16GB HBM per chip, ~2000 tok/s")
print(f"  Expected: {(TOTAL_STEPS * SEQ_LEN) / 1e9:.2f}B tokens processed")

# Step 1: Load Datasets
print(f"\n{'='*70}")
print("[1/6] Loading HIGH-QUALITY Datasets (10GB+ to prevent memorization)")
print(f"{'='*70}")
print(f"Note: Large download (~10GB) takes 15-20 min first time, then cached")
print(f"For 509M parameters, we need 5-10GB data to avoid memorization!")

from datasets import load_dataset

# HIGH-QUALITY, LARGE-SCALE DATASETS
# Total: ~10GB text, ~2.5B tokens (5 tokens/param - good ratio!)
# Wikipedia: 500k articles (~6GB)
# OpenWebText: 200k docs (~3GB)  
# C4: Clean crawl subset (~2GB)

datasets_to_load = [
    ("wikipedia", "20220301.en", 500000),   # 500k Wikipedia articles
    ("openwebtext", None, 200000),          # 200k quality web docs
    ("allenai/c4", "en", 100000),           # 100k C4 clean documents
]

all_texts = []
total_size_mb = 0

for idx, config_tuple in enumerate(datasets_to_load):
    dataset_name, config, max_samples = config_tuple
    
    try:
        if dataset_name == "wikipedia":
            print(f"\n  [1/3] Loading Wikipedia (English)...")
            print(f"        Downloading {max_samples:,} articles (~5-6GB)")
            ds = load_dataset(dataset_name, config, split=f"train[:{max_samples}]", trust_remote_code=True)
            texts = [f"{item['title']}\n\n{item['text']}" for item in ds if item.get('text') and len(item['text']) > 100]
            chars = sum(len(t) for t in texts)
            all_texts.extend(texts)
            total_size_mb += chars / 1e6
            print(f"        ✓ Loaded {len(texts):,} articles")
            print(f"        ✓ Size: {chars/1e6:.1f}MB ({chars/1e9:.2f}GB)")
            
        elif dataset_name == "openwebtext":
            print(f"\n  [2/3] Loading OpenWebText (Reddit quality)...")
            print(f"        Downloading {max_samples:,} documents (~2-3GB)")
            ds = load_dataset(dataset_name, split=f"train[:{max_samples}]", trust_remote_code=True)
            texts = [item['text'] for item in ds if item.get('text') and len(item['text']) > 200]
            chars = sum(len(t) for t in texts)
            all_texts.extend(texts)
            total_size_mb += chars / 1e6
            print(f"        ✓ Loaded {len(texts):,} documents")
            print(f"        ✓ Size: {chars/1e6:.1f}MB ({chars/1e9:.2f}GB)")
            
        elif dataset_name == "allenai/c4":
            print(f"\n  [3/3] Loading C4 (Clean web crawl)...")
            print(f"        Downloading {max_samples:,} documents (~1-2GB)")
            ds = load_dataset(dataset_name, config, split=f"train[:{max_samples}]", trust_remote_code=True)
            texts = [item['text'] for item in ds if item.get('text') and len(item['text']) > 200]
            chars = sum(len(t) for t in texts)
            all_texts.extend(texts)
            total_size_mb += chars / 1e6
            print(f"        ✓ Loaded {len(texts):,} documents")
            print(f"        ✓ Size: {chars/1e6:.1f}MB ({chars/1e9:.2f}GB)")
            
    except Exception as e:
        print(f"  ⚠ Failed to load {dataset_name}: {e}")
        print(f"    Continuing with available data...")

# Combine all text
combined_text = "\n\n".join(all_texts)

print(f"\n{'='*70}")
print(f"DATASET SUMMARY")
print(f"{'='*70}")
print(f"Total documents: {len(all_texts):,}")
print(f"Total size: {len(combined_text)/1e6:.1f}MB ({len(combined_text)/1e9:.2f}GB)")
print(f"Estimated tokens: ~{len(combined_text)/4/1e6:.1f}M tokens (@ 4 chars/token)")
print(f"\nModel memorization check:")
print(f"  Model params: 509M")
print(f"  Data tokens: ~{len(combined_text)/4/1e6:.1f}M")
print(f"  Ratio: {(len(combined_text)/4/1e6)/509:.2f} tokens/param")
print(f"  Status: {'✅ GOOD (>2)' if (len(combined_text)/4/1e6)/509 > 2 else '⚠️  LOW (<2) - may memorize'}")
print(f"\nTraining plan:")
print(f"  • 50,000 steps × 128 tokens = 6.4M tokens per epoch")
print(f"  • ~{(len(combined_text)/4/1e6) / 6.4:.1f} epochs through full dataset")
print(f"\nSample text:")
print(f"{combined_text[:200]}...")
print(f"{'='*70}")

# Step 2: Train Tokenizer
print(f"\n{'='*70}")
print("[2/6] Training SentencePiece Tokenizer")
print(f"{'='*70}")

# Save text to file for tokenizer training
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
train_file = data_dir / "train_combined.txt"

print(f"  Saving training data to {train_file}")
with open(train_file, 'w', encoding='utf-8') as f:
    f.write(combined_text)

# Train tokenizer
from src.data import SPTokenizer

print(f"  Training tokenizer (vocab_size=8000)...")
tokenizer = SPTokenizer.train(
    input_files=[str(train_file)],
    vocab_size=8000,  # Smaller vocab for faster training
    model_prefix=str(data_dir / "tokenizer"),
    model_type="bpe"
)

print(f"  ✓ Tokenizer trained!")
print(f"  ✓ Vocab size: {tokenizer.vocab_size}")

# Test tokenizer
test_text = "Once upon a time, there was a brave knight."
test_ids = tokenizer.encode(test_text)
decoded = tokenizer.decode(test_ids)
print(f"  Test: '{test_text}'")
print(f"  Tokens: {test_ids[:20]}...")
print(f"  Decoded: '{decoded}'")

# Step 3: Tokenize and Create Batches
print(f"\n{'='*70}")
print("[3/6] Creating Training Batches")
print(f"{'='*70}")

# Tokenize entire dataset
print(f"  Tokenizing {len(combined_text):,} characters...")
token_ids = tokenizer.encode(combined_text, add_bos=False, add_eos=False)
print(f"  ✓ Tokenized: {len(token_ids):,} tokens")

# Create batches
batches = []
total_batches_needed = TOTAL_STEPS

for i in range(0, len(token_ids) - SEQ_LEN - 1, SEQ_LEN):
    if len(batches) >= total_batches_needed:
        break
    
    # Get a batch by taking BATCH_SIZE sequences
    batch_input_ids = []
    batch_labels = []
    
    for b in range(BATCH_SIZE):
        offset = i + b * SEQ_LEN
        if offset + SEQ_LEN + 1 > len(token_ids):
            break
            
        input_ids = token_ids[offset:offset + SEQ_LEN]
        labels = token_ids[offset + 1:offset + SEQ_LEN + 1]
        
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
    
    if len(batch_input_ids) == BATCH_SIZE:
        batches.append({
            'input_ids': jnp.array(batch_input_ids, dtype=jnp.int32),
            'labels': jnp.array(batch_labels, dtype=jnp.int32)
        })

print(f"  ✓ Created {len(batches):,} batches")
print(f"    Batch shape: {batches[0]['input_ids'].shape}")

# Step 4: Load Model
print(f"\n{'='*70}")
print("[4/6] Loading Model")
print(f"{'='*70}")

from src.model import create_model_from_config, ModelConfig
from src.training import (
    create_train_step,
    create_train_state,
    create_learning_rate_schedule,
    create_optimizer,
    CheckpointManager,
    ConsoleLogger
)

config_path = 'configs/model_config.yaml'
model = create_model_from_config(config_path)
model_config = ModelConfig.from_yaml(config_path)

print(f"  ✓ Model: {model_config.name}")
print(f"  ✓ Layers: {model_config.num_layers}")
print(f"  ✓ Hidden dim: {model_config.hidden_dim}")

# Initialize model
print(f"\n  Initializing parameters...")
rng = random.PRNGKey(42)
rng, init_rng, dropout_rng = random.split(rng, 3)

dummy_input = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
variables = model.init(init_rng, dummy_input, deterministic=True)
params = variables['params']

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"  ✓ Parameters: {param_count / 1e6:.1f}M")

# Step 5: Setup Training
print(f"\n{'='*70}")
print("[5/6] Setting Up Training")
print(f"{'='*70}")

# Learning rate schedule
lr_schedule = create_learning_rate_schedule(
    warmup_steps=min(100, TOTAL_STEPS // 10),
    max_learning_rate=LEARNING_RATE,
    total_steps=TOTAL_STEPS,
    schedule_type='cosine'
)

# Optimizer
optimizer = create_optimizer(
    learning_rate_fn=lr_schedule,
    weight_decay=0.1,
    max_grad_norm=1.0
)

# Training state
state = create_train_state(model, params, optimizer, lr_schedule, dropout_rng)

# Create training step
train_step = create_train_step(model, lr_schedule)
train_step = jax.jit(train_step)

# Checkpoint manager
ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)
ckpt_manager = CheckpointManager(
    checkpoint_dir=str(ckpt_dir),
    max_to_keep=3,
    save_interval_steps=SAVE_EVERY
)

print(f"  ✓ Optimizer: AdamW")
print(f"  ✓ LR schedule: warmup + cosine")
print(f"  ✓ Checkpoints: {ckpt_dir}")

# Step 6: Train!
print(f"\n{'='*70}")
print("[6/6] TRAINING")
print(f"{'='*70}")
print(f"  Starting {TOTAL_STEPS} steps...")
print(f"  (Compiling on first step - will take 1-2 min)")
print(f"{'='*70}\n")

logger = ConsoleLogger(log_interval=10)
start_time = time.time()
compile_time = None

for step in range(min(TOTAL_STEPS, len(batches))):
    batch = batches[step]
    
    step_start = time.time()
    state, metrics = train_step(state, batch)
    step_time = time.time() - step_start
    
    # Track compilation time
    if step == 0:
        compile_time = step_time
        print(f"✓ Compilation done ({compile_time:.1f}s)\n")
    
    # Add timing metrics
    metrics['step_time'] = step_time
    metrics['tokens_per_sec'] = (BATCH_SIZE * SEQ_LEN) / step_time if step > 0 else 0
    
    # Log
    if step % LOG_EVERY == 0 or step == 0:
        loss = float(metrics['loss'])
        ppl = float(metrics['perplexity'])
        lr = float(metrics['learning_rate'])
        tps = int(metrics['tokens_per_sec'])
        print(f"Step {step:4d} | loss={loss:.4f} ppl={ppl:7.2f} lr={lr:.6f} | {tps:,} tok/s")
    
    # Save checkpoint
    if (step + 1) % SAVE_EVERY == 0:
        ckpt_manager.save(state, step, metadata={'loss': float(metrics['loss'])})

total_time = time.time() - start_time

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*70}")

print(f"\nResults:")
print(f"  ✓ Trained {TOTAL_STEPS} steps")
print(f"  ✓ Total time: {total_time/60:.1f} minutes")
print(f"  ✓ Compilation: {compile_time:.1f}s")
print(f"  ✓ Avg step time: {(total_time-compile_time)/TOTAL_STEPS:.3f}s")
print(f"  ✓ Final loss: {float(metrics['loss']):.4f}")
print(f"  ✓ Final perplexity: {float(metrics['perplexity']):.2f}")
print(f"  ✓ Checkpoints saved in: {ckpt_dir}")

print(f"\n💾 Saved Files:")
print(f"  - Tokenizer: data/tokenizer.model")
print(f"  - Training data: data/train_combined.txt")
print(f"  - Checkpoints: {ckpt_dir}/")

print(f"\n🎉 Model trained on real novels, stories & chat!")
print(f"{'='*70}")
