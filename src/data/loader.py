"""
Data loader for streaming training data efficiently
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Iterator, Dict, List, Optional, Callable
from pathlib import Path
import json


class StreamingDataLoader:
    """
    Efficient streaming data loader for TPU training
    
    Features:
    - Streaming from disk (no full load into memory)
    - Automatic batching
    - Shuffling with buffer
    - Preprocessing pipeline
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        batch_size: int = 32,
        seq_len: int = 8192,
        shuffle_buffer: int = 10000,
        num_epochs: int = 1,
        preprocess_fn: Optional[Callable] = None
    ):
        """
        Initialize data loader
        
        Args:
            data_path: Path to data file (jsonl format)
            tokenizer: Tokenizer instance
            batch_size: Batch size
            seq_len: Sequence length (context window)
            shuffle_buffer: Shuffle buffer size
            num_epochs: Number of epochs to loop
            preprocess_fn: Optional preprocessing function
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.num_epochs = num_epochs
        self.preprocess_fn = preprocess_fn or self._default_preprocess
    
    def _default_preprocess(self, sample: Dict) -> Dict:
        """Default preprocessing: tokenize and create labels"""
        text = sample.get('text', '')
        
        # Tokenize
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True, max_length=self.seq_len)
        
        # Pad if needed
        if len(ids) < self.seq_len:
            ids = ids + [self.tokenizer.pad_id] * (self.seq_len - len(ids))
        
        # Create input_ids and labels (shifted by 1)
        input_ids = ids[:-1]
        labels = ids[1:]
        
        return {
            'input_ids': np.array(input_ids, dtype=np.int32),
            'labels': np.array(labels, dtype=np.int32)
        }
    
    def _read_samples(self) -> Iterator[Dict]:
        """Read samples from file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    
    def _create_batches(self) -> Iterator[Dict]:
        """Create batches from samples"""
        batch = []
        
        for epoch in range(self.num_epochs):
            for sample in self._read_samples():
                # Preprocess
                processed = self.preprocess_fn(sample)
                batch.append(processed)
                
                # Yield batch when full
                if len(batch) == self.batch_size:
                    yield self._collate(batch)
                    batch = []
        
        # Yield remaining samples
        if batch:
            yield self._collate(batch)
    
    def _collate(self, batch: List[Dict]) -> Dict:
        """Collate batch into arrays"""
        input_ids = np.stack([b['input_ids'] for b in batch])
        labels = np.stack([b['labels'] for b in batch])
        
        return {
            'input_ids': jnp.array(input_ids),
            'labels': jnp.array(labels)
        }
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate over batches"""
        return self._create_batches()


class DataMixer:
    """
    Mix multiple data sources with specified weights
    
    Example:
        mixer = DataMixer({
            'fineweb': ('fineweb.jsonl', 0.50),
            'stack': ('stack.jsonl', 0.25),
            'math': ('math.jsonl', 0.15),
            'wiki': ('wiki.jsonl', 0.10)
        })
    """
    
    def __init__(
        self,
        sources: Dict[str, tuple],  # {name: (path, weight)}
        tokenizer,
        batch_size: int = 32,
        seq_len: int = 8192
    ):
        """
        Initialize data mixer
        
        Args:
            sources: Dict mapping source name to (path, weight) tuple
            tokenizer: Tokenizer instance
            batch_size: Batch size
            seq_len: Sequence length
        """
        self.sources = sources
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Normalize weights
        total_weight = sum(w for _, w in sources.values())
        self.weights = {
            name: weight / total_weight
            for name, (_, weight) in sources.items()
        }
        
        print(f"Data mixer initialized with {len(sources)} sources:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.2%}")
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate over mixed batches"""
        # Create loaders for each source
        loaders = {
            name: StreamingDataLoader(
                path,
                self.tokenizer,
                batch_size=int(self.batch_size * weight),
                seq_len=self.seq_len
            )
            for name, (path, weight) in self.sources.items()
        }
        
        # Interleave batches from all sources
        iterators = {name: iter(loader) for name, loader in loaders.items()}
        
        while iterators:
            # Try to get batch from each source
            batch_parts = []
            finished_sources = []
            
            for name, it in iterators.items():
                try:
                    batch = next(it)
                    batch_parts.append(batch)
                except StopIteration:
                    finished_sources.append(name)
            
            # Remove finished sources
            for name in finished_sources:
                del iterators[name]
            
            # Combine batch parts
            if batch_parts:
                combined = {
                    'input_ids': jnp.concatenate([b['input_ids'] for b in batch_parts]),
                    'labels': jnp.concatenate([b['labels'] for b in batch_parts])
                }
                yield combined
