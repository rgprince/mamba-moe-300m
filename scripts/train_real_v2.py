#!/usr/bin/env python3
"""
Real Training Script V2 - WORKING DATASETS (Streaming)

Uses MODERN datasets that actually work on Kaggle:
- FineWeb-Edu (10GB sample): Educational web content
- SlimPajama (5GB sample): Deduplicated high-quality text

STREAMING approach = Fast, no full download!
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
print("Mamba-MoE 300M - WORKING Real Training (Streaming)")
print("=" * 70)

# Configuration
TOTAL_STEPS = 50000
BATCH_SIZE = 1
SEQ_LEN = 128
LEARNING_RATE = 3e-4
SAVE_EVERY = 500
LOG_EVERY = 50

# DATA CONFIGURATION (adjust to control size)
MAX_SAMPLES = 2_500_000  # 2.5M samples = ~10GB text
TARGET_TOKENS = 2_500_000_000  # 2.5B tokens target

print(f"\n[CONFIG] - PRODUCTION TRAINING")
print(f"  Total steps: {TOTAL_STEPS:,} (~3-4 hours)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Target data: {MAX_SAMPLES:,} samples (~10GB, ~2.5B tokens)")
print(f"  Save interval: {SAVE_EVERY} steps")
print(f"  Log interval: {LOG_EVERY} steps")

# Step 1: Load Datasets via STREAMING
print(f"\n{'='*70}")
print("[1/6] Streaming HIGH-QUALITY Datasets")
print(f"{'='*70}")
print("Using STREAMING mode - no full download!")

from datasets import load_dataset
from tqdm import tqdm

# MODERN DATASETS (Parquet-based, no scripts!)
datasets_config = [
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "samples": 1_500_000,  # 1.5M samples (~6GB)
        "description": "FineWeb-Edu: Educational web content (CLEAN)"
    },
    {
        "name": "cerebras/SlimPajama-627B", 
        "split": "train",
        "samples": 1_000_000,  # 1M samples (~4GB)
        "description": "SlimPajama: Deduplicated high-quality text"
    }
]

all_texts = []
total_chars = 0
total_samples = 0

for ds_config in datasets_config:
    print(f"\n  Loading {ds_config['description']}...")
    print(f"  Streaming {ds_config['samples']:,} samples...")
    
    try:
        # STREAMING MODE - only downloads what we need!
        dataset = load_dataset(
            ds_config['name'],
            split=ds_config['split'],
            streaming=True,  # KEY: Streaming = fast!
            trust_remote_code=False
        )
        
        # Take only what we need
        samples_collected = 0
        dataset_chars = 0
        
        # Progress bar
        pbar = tqdm(total=ds_config['samples'], desc=f"  {ds_config['name'].split('/')[1][:20]}")
        
        for item in dataset:
            # Get text field (usually 'text' or 'content')
            text = item.get('text') or item.get('content') or ''
            
            if len(text) > 100:  # Filter very short texts
                all_texts.append(text)
                dataset_chars += len(text)
                samples_collected += 1
                pbar.update(1)
                
                if samples_collected >= ds_config['samples']:
                    break
        
        pbar.close()
        
        total_chars += dataset_chars
        total_samples += samples_collected
        
        print(f"  ✓ Collected {samples_collected:,} samples")
        print(f"  ✓ Size: {dataset_chars/1e6:.1f}MB ({dataset_chars/1e9:.2f}GB)")
        
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        print(f"  Continuing with other datasets...")

# Combine all text
combined_text = "\n\n".join(all_texts)

print(f"\n{'='*70}")
print(f"DATASET SUMMARY")
print(f"{'='*70}")
print(f"Total samples: {total_samples:,}")
print(f"Total size: {len(combined_text)/1e6:.1f}MB ({len(combined_text)/1e9:.2f}GB)")
print(f"Estimated tokens: ~{len(combined_text)/4/1e6:.1f}M tokens")
print(f"\nMemorization check:")
print(f"  Model params: 509M")
print(f"  Data tokens: ~{len(combined_text)/4/1e6:.1f}M")
print(f"  Ratio: {(len(combined_text)/4/1e6)/509:.2f} tokens/param")
if (len(combined_text)/4/1e6)/509 > 2:
    print(f"  Status: ✅ GOOD (>2) - No memorization!")
else:
    print(f"  Status: ⚠️ LOW (<2) - May memorize")
print(f"\nTraining plan:")
print(f"  • 50,000 steps × 128 tokens = 6.4M tokens per epoch")
print(f"  • ~{(len(combined_text)/4/1e6) / 6.4:.1f} epochs through dataset")
print(f"{'='*70}")

# Step 2: Train Tokenizer (on SAMPLE only to save RAM)
print(f"\n{'='*70}")
print("[2/6] Training SentencePiece Tokenizer")
print(f"{'='*70}")

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# CRITICAL: Only save a SAMPLE for tokenizer training (saves RAM!)
# 200MB is plenty for an 8K vocab tokenizer
TOKENIZER_SAMPLE_SIZE = 200_000_000  # 200MB
tokenizer_sample = combined_text[:TOKENIZER_SAMPLE_SIZE]

tokenizer_train_file = data_dir / "tokenizer_sample.txt"
print(f"  Saving tokenizer sample ({len(tokenizer_sample)/1e6:.1f}MB) to {tokenizer_train_file}")
with open(tokenizer_train_file, 'w', encoding='utf-8') as f:
    f.write(tokenizer_sample)

from src.data import SPTokenizer

print(f"  Training tokenizer (vocab_size=8000) on sample...")
print(f"  (This prevents RAM overflow - we only need a sample!)")
tokenizer = SPTokenizer.train(
    input_files=[str(tokenizer_train_file)],
    vocab_size=8000,
    model_prefix=str(data_dir / "tokenizer"),
    model_type="bpe",
    input_sentence_size=2_000_000  # Limit to 2M sentences max
)

print(f"  ✓ Tokenizer trained!")
print(f"  ✓ Vocab size: {tokenizer.vocab_size}")

# Step 3: Tokenize and Create Batches
print(f"\n{'='*70}")
print("[3/6] Creating Training Batches")
print(f"{'='*70}")

print(f"  Tokenizing FULL {len(combined_text):,} characters...")
print(f"  (Using all data for training, not just the tokenizer sample!)")
token_ids = tokenizer.encode(combined_text, add_bos=False, add_eos=False)
print(f"  ✓ Tokenized: {len(token_ids):,} tokens")

# Create batches
batches = []
total_batches_needed = TOTAL_STEPS

for i in range(0, len(token_ids) - SEQ_LEN - 1, SEQ_LEN):
    if len(batches) >= total_batches_needed:
        break
    
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

lr_schedule = create_learning_rate_schedule(
    warmup_steps=min(100, TOTAL_STEPS // 10),
    max_learning_rate=LEARNING_RATE,
    total_steps=TOTAL_STEPS,
    schedule_type='cosine'
)

optimizer = create_optimizer(
    learning_rate_fn=lr_schedule,
    weight_decay=0.1,
    max_grad_norm=1.0
)

state = create_train_state(model, params, optimizer, lr_schedule, dropout_rng)
train_step = create_train_step(model, lr_schedule)
train_step = jax.jit(train_step)

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
print(f"  Starting {TOTAL_STEPS:,} steps...")
print(f"  (Compiling on first step - takes ~1 min)")
print(f"{'='*70}\n")

logger = ConsoleLogger(log_interval=LOG_EVERY)
start_time = time.time()
compile_time = None

for step in range(min(TOTAL_STEPS, len(batches))):
    batch = batches[step]
    
    step_start = time.time()
    state, metrics = train_step(state, batch)
    step_time = time.time() - step_start
    
    if step == 0:
        compile_time = step_time
        print(f"✓ Compilation done ({compile_time:.1f}s)\n")
    
    metrics['step_time'] = step_time
    metrics['tokens_per_sec'] = (BATCH_SIZE * SEQ_LEN) / step_time if step > 0 else 0
    
    if step % LOG_EVERY == 0 or step == 0:
        loss = float(metrics['loss'])
        ppl = float(metrics['perplexity'])
        lr = float(metrics['learning_rate'])
        tps = int(metrics['tokens_per_sec'])
        print(f"Step {step:5d} | loss={loss:.4f} ppl={ppl:8.2f} lr={lr:.6f} | {tps:,} tok/s")
    
    if (step + 1) % SAVE_EVERY == 0:
        ckpt_manager.save(state, step, metadata={'loss': float(metrics['loss'])})

total_time = time.time() - start_time

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"\nResults:")
print(f"  ✓ Trained {TOTAL_STEPS:,} steps")
print(f"  ✓ Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"  ✓ Compilation: {compile_time:.1f}s")
print(f"  ✓ Avg step time: {(total_time-compile_time)/TOTAL_STEPS:.3f}s")
print(f"  ✓ Final loss: {float(metrics['loss']):.4f}")
print(f"  ✓ Final perplexity: {float(metrics['perplexity']):.2f}")
print(f"  ✓ Checkpoints saved in: {ckpt_dir}")

print(f"\n💾 Saved Files:")
print(f"  - Tokenizer: data/tokenizer.model")
print(f"  - Training data: data/train_combined.txt")
print(f"  - Checkpoints: {ckpt_dir}/")

print(f"\n🎉 Model trained on {len(combined_text)/1e9:.2f}GB of clean web data!")
print(f"{'='*70}")
