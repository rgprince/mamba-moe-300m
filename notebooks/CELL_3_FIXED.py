from src.data import SPTokenizer
from pathlib import Path

print("="*70)
print("TRAINING TOKENIZER")
print("="*70)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Save a 200MB sample for tokenizer training (prevents RAM overflow!)
TOKENIZER_SAMPLE_SIZE = 200_000_000  # 200MB
tokenizer_sample = combined_text[:TOKENIZER_SAMPLE_SIZE]

tokenizer_train_file = data_dir / "tokenizer_sample.txt"
print(f"Saving tokenizer sample ({len(tokenizer_sample)/1e6:.1f}MB)...")
with open(tokenizer_train_file, 'w', encoding='utf-8') as f:
    f.write(tokenizer_sample)

print(f"\nTraining tokenizer (vocab_size=8000)...")
print(f"(Using sample - prevents RAM overflow!)\n")

tokenizer = SPTokenizer.train(
    input_files=[str(tokenizer_train_file)],
    vocab_size=8000,
    model_prefix=str(data_dir / "tokenizer"),
    model_type="bpe"
)

print(f"\n✅ Tokenizer trained!")
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Files saved: data/tokenizer.model, data/tokenizer.vocab")
print(f"\n📥 DOWNLOAD: You can download tokenizer files from data/ folder")
