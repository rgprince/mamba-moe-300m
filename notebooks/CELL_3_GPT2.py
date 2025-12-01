# CELL 3 - USE GPT-2 TOKENIZER (NO RAM ISSUES!)
from transformers import GPT2TokenizerFast
from pathlib import Path

print("="*70)
print("LOADING GPT-2 TOKENIZER (PRODUCTION-READY)")
print("="*70)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

print("\nDownloading GPT-2 tokenizer...")
# Download GPT-2 tokenizer (50K vocab, BPE-based)
tokenizer_hf = GPT2TokenizerFast.from_pretrained("gpt2")

# Save it locally
tokenizer_save_path = data_dir / "gpt2_tokenizer"
tokenizer_hf.save_pretrained(str(tokenizer_save_path))
print(f"✓ Saved to {tokenizer_save_path}")

# Create wrapper that's compatible with our training code
class TokenizerWrapper:
    """Wraps HuggingFace tokenizer to match our SPTokenizer interface"""
    
    def __init__(self, hf_tokenizer):
        self.tokenizer = hf_tokenizer
        self.vocab_size = len(hf_tokenizer)
        self.bos_id = hf_tokenizer.bos_token_id or 50256  # GPT-2 end token
        self.eos_id = hf_tokenizer.eos_token_id or 50256
        self.pad_id = hf_tokenizer.pad_token_id or 50256
        self.unk_id = hf_tokenizer.unk_token_id or 50256
    
    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to token IDs"""
        ids = self.tokenizer.encode(text)
        
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

# Create wrapped tokenizer
tokenizer = TokenizerWrapper(tokenizer_hf)

print(f"\n✅ Tokenizer ready!")
print(f"   Type: GPT-2 BPE")
print(f"   Vocab size: {tokenizer.vocab_size:,}")
print(f"   Special tokens: BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")
print(f"\n📥 DOWNLOAD: You can download tokenizer from data/gpt2_tokenizer/")
print(f"\n💡 GPT-2 tokenizer advantages:")
print(f"   • 50K vocab (vs 8K) = better compression")
print(f"   • Production-tested by millions")
print(f"   • Perfect for English web text")
print(f"   • Zero RAM issues!")
