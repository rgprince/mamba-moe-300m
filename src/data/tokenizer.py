"""
SentencePiece tokenizer wrapper for Mamba-MoE 300M

Provides a clean interface for:
- Encoding text to token IDs
- Decoding token IDs to text  
- Managing special tokens
- Loading pretrained or training new tokenizers
"""

import os
from typing import List, Optional, Union
from pathlib import Path
import sentencepiece as spm


class SPTokenizer:
    """
    SentencePiece tokenizer wrapper
    
    Handles text <-> token ID conversion with special tokens.
    Compatible with Llama/Gemma tokenizer formats.
    """
    
    # Special token definitions
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize tokenizer
        
        Args:
            model_path: Path to .model file (if None, needs training)
        """
        self.sp = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load pretrained tokenizer"""
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path
        
        # Cache special token IDs
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        
        print(f"✓ Loaded tokenizer from {model_path}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Special tokens: BOS={self.bos_id}, EOS={self.eos_id}, PAD={self.pad_id}")
    
    @classmethod
    def train(
        cls,
        input_files: List[str],
        vocab_size: int = 32000,
        model_prefix: str = "tokenizer",
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        num_threads: int = 16
    ) -> "SPTokenizer":
        """
        Train a new SentencePiece tokenizer
        
        Args:
            input_files: List of text files for training
            vocab_size: Target vocabulary size
            model_prefix: Output model name (will create .model and .vocab)
            model_type: "bpe" or "unigram"
            character_coverage: Coverage for rare characters
            num_threads: Parallel threads
        
        Returns:
            Trained tokenizer instance
        """
        # Train command
        train_args = [
            f"--input={','.join(input_files)}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            f"--model_type={model_type}",
            f"--character_coverage={character_coverage}",
            f"--num_threads={num_threads}",
            f"--bos_id=1",  # <bos>
            f"--eos_id=2",  # <eos>
            f"--pad_id=0",  # <pad>
            f"--unk_id=3",  # <unk>
            "--byte_fallback=true",  # Handle any UTF-8 byte
            "--split_digits=true",  # Split numbers
            "--normalization_rule_name=nmt_nfkc",  # Normalize unicode
        ]
        
        print(f"Training tokenizer on {len(input_files)} file(s)...")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Model type: {model_type}")
        
        spm.SentencePieceTrainer.Train(" ".join(train_args))
        
        print(f"✓ Tokenizer trained: {model_prefix}.model")
        
        # Load and return
        return cls(f"{model_prefix}.model")
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs
        
        Args:
            text: Single text or list of texts
            add_bos: Prepend BOS token
            add_eos: Append EOS token
            max_length: Truncate to max length
        
        Returns:
            Token IDs (single list or list of lists)
        """
        if self.sp is None:
            raise ValueError("Tokenizer not loaded. Call load() or train() first.")
        
        # Handle batch
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        # Encode each text
        all_ids = []
        for t in texts:
            # Get IDs
            ids = self.sp.EncodeAsIds(t)
            
            # Add special tokens
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]
            
            # Truncate if needed
            if max_length and len(ids) > max_length:
                ids = ids[:max_length]
                # Ensure EOS at end if it was added
                if add_eos:
                    ids[-1] = self.eos_id
            
            all_ids.append(ids)
        
        return all_ids if is_batch else all_ids[0]
    
    def decode(
        self,
        ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text
        
        Args:
            ids: Single list of IDs or batch of lists
            skip_special_tokens: Remove BOS/EOS/PAD from output
        
        Returns:
            Decoded text (single string or list of strings)
        """
        if self.sp is None:
            raise ValueError("Tokenizer not loaded. Call load() or train() first.")
        
        # Handle batch
        is_batch = isinstance(ids[0], list) if ids else False
        id_lists = ids if is_batch else [ids]
        
        # Decode each sequence
        texts = []
        for id_list in id_lists:
            # Remove special tokens if requested
            if skip_special_tokens:
                id_list = [
                    i for i in id_list 
                    if i not in (self.bos_id, self.eos_id, self.pad_id, self.unk_id)
                ]
            
            # Decode
            text = self.sp.DecodeIds(id_list)
            texts.append(text)
        
        return texts if is_batch else texts[0]
    
    def encode_batch(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "np"
    ):
        """
        Encode batch with padding
        
        Args:
            texts: List of texts
            padding: Pad to same length
            max_length: Maximum sequence length
            return_tensors: "np" for numpy, "jax" for jax array
        
        Returns:
            Dict with input_ids and attention_mask
        """
        import numpy as np
        
        # Encode all
        all_ids = self.encode(texts, add_bos=True, add_eos=True, max_length=max_length)
        
        if padding:
            # Find max length
            max_len = max(len(ids) for ids in all_ids)
            if max_length:
                max_len = min(max_len, max_length)
            
            # Pad all to max length
            padded_ids = []
            attention_mask = []
            
            for ids in all_ids:
                # Truncate if needed
                if len(ids) > max_len:
                    ids = ids[:max_len]
                
                # Create mask (1 for real tokens, 0 for padding)
                mask = [1] * len(ids) + [0] * (max_len - len(ids))
                
                # Pad
                ids = ids + [self.pad_id] * (max_len - len(ids))
                
                padded_ids.append(ids)
                attention_mask.append(mask)
            
            # Convert to arrays
            input_ids = np.array(padded_ids, dtype=np.int32)
            attention_mask = np.array(attention_mask, dtype=np.int32)
            
            if return_tensors == "jax":
                import jax.numpy as jnp
                input_ids = jnp.array(input_ids)
                attention_mask = jnp.array(attention_mask)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            # Return ragged lists
            return {'input_ids': all_ids}
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.sp) if self.sp else 0
    
    def get_vocab(self) -> dict:
        """Get vocabulary as dict {token: id}"""
        if self.sp is None:
            return {}
        
        vocab = {}
        for i in range(self.vocab_size):
            token = self.sp.IdToPiece(i)
            vocab[token] = i
        return vocab
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def __repr__(self) -> str:
        if self.sp:
            return f"SPTokenizer(vocab_size={self.vocab_size}, model={self.model_path})"
        else:
            return "SPTokenizer(not loaded)"
