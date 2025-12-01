from src.data import SPTokenizer
from pathlib import Path
import sentencepiece as spm

print("="*70)
print("TRAINING TOKENIZER (RAM-SAFE)")
print("="*70)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Use MUCH SMALLER sample - 50MB is plenty for 8K vocab!
TOKENIZER_SAMPLE_SIZE = 50_000_000  # 50MB (reduced from 200MB!)
tokenizer_sample = combined_text[:TOKENIZER_SAMPLE_SIZE]

tokenizer_train_file = data_dir / "tokenizer_sample.txt"
print(f"Saving tokenizer sample ({len(tokenizer_sample)/1e6:.1f}MB)...")
with open(tokenizer_train_file, 'w', encoding='utf-8') as f:
    f.write(tokenizer_sample)

print(f"\nTraining tokenizer (vocab_size=8000)...")
print(f"(Ultra RAM-safe mode!)\n")

# Train with STRICT memory limits
train_args = [
    f"--input={tokenizer_train_file}",
    f"--model_prefix={data_dir / 'tokenizer'}",
    "--vocab_size=8000",
    "--model_type=bpe",
    "--character_coverage=0.9995",
    "--num_threads=4",  # Reduced threads
    "--bos_id=1",
    "--eos_id=2",
    "--pad_id=0",
    "--unk_id=3",
    "--byte_fallback=true",
    "--split_digits=true",
    "--normalization_rule_name=nmt_nfkc",
    # CRITICAL: These prevent RAM explosion
    "--input_sentence_size=500000",  # Max 500K sentences
    "--shuffle_input_sentence=true",  # Shuffle for better coverage
    "--max_sentence_length=4192",  # Skip very long lines
]

spm.SentencePieceTrainer.Train(" ".join(train_args))

# Load the trained tokenizer
tokenizer = SPTokenizer(model_path=str(data_dir / "tokenizer.model"))

print(f"\n✅ Tokenizer trained!")
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Files saved: data/tokenizer.model, data/tokenizer.vocab")
print(f"\n📥 DOWNLOAD: You can download tokenizer files from data/ folder")
