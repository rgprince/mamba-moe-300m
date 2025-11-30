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

# Configuration
TOTAL_STEPS = 1000  # Train for 1000 steps (adjust based on time)
BATCH_SIZE = 4      # Increase for faster training
SEQ_LEN = 512       # Longer sequences for better context
LEARNING_RATE = 3e-4
SAVE_EVERY = 100    # Save checkpoint every 100 steps

print(f"\n[CONFIG]")
print(f"  Steps: {TOTAL_STEPS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Save interval: {SAVE_EVERY}")

# Step 1: Load Datasets
print(f"\n{'='*70}")
print("[1/6] Loading Datasets from HuggingFace")
print(f"{'='*70}")

from datasets import load_dataset, concatenate_datasets

datasets_to_load = [
    ("roneneldan/TinyStories", "Tiny Stories", 0.4),  # Creative stories
    ("HuggingFaceH4/ultrachat_200k", "Ultra Chat", 0.3),  # Conversations
    ("bookcorpus", "Book Corpus", 0.3),  # Books
]

all_texts = []

for dataset_name, display_name, weight in datasets_to_load:
    try:
        print(f"\n  Loading {display_name}...")
        
        if dataset_name == "HuggingFaceH4/ultrachat_200k":
            # Chat dataset - special handling
            ds = load_dataset(dataset_name, split="train_sft[:10000]")  # 10k samples
            texts = []
            for item in ds:
                # Combine messages into conversation
                if 'messages' in item:
                    conv = "\n".join([f"{m['role']}: {m['content']}" for m in item['messages']])
                    texts.append(conv)
            all_texts.extend(texts)
            print(f"  ✓ Loaded {len(texts):,} conversations")
            
        elif dataset_name == "roneneldan/TinyStories":
            # Stories dataset
            ds = load_dataset(dataset_name, split="train[:50000]")  # 50k stories
            texts = [item['text'] for item in ds if item.get('text')]
            all_texts.extend(texts)
            print(f"  ✓ Loaded {len(texts):,} stories")
            
        elif dataset_name == "bookcorpus":
            # Books dataset
            ds = load_dataset(dataset_name, split="train[:20000]")  # 20k book excerpts
            texts = [item['text'] for item in ds if item.get('text')]
            all_texts.extend(texts)
            print(f"  ✓ Loaded {len(texts):,} book excerpts")
            
    except Exception as e:
        print(f"  ⚠ Failed to load {display_name}: {e}")
        print(f"    Skipping...")

# Combine all text
combined_text = "\n\n".join(all_texts)
print(f"\n✓ Total text: {len(combined_text):,} characters")
print(f"  Sample: {combined_text[:200]}")

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
    if step % 10 == 0 or step == 0:
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
