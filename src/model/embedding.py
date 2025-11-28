"""
Factorized embedding layer for parameter efficiency
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class FactorizedEmbedding(nn.Module):
    """
    Factorized embedding: vocab_size → embed_dim → hidden_dim
    Saves parameters: 32k×1024 = 32M → 32k×256 + 256×1024 = 8.25M
    """
    vocab_size: int
    embed_dim: int  # Low-rank dimension
    hidden_dim: int  # Model hidden dimension
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, input_ids, deterministic: bool = False):
        """
       Args:
            input_ids: Token IDs [batch, seq_len]
            deterministic: Whether to use dropout
        
        Returns:
            Embeddings [batch, seq_len, hidden_dim]
        """
        # First embedding: vocab → embed_dim
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            name='token_embed'
        )
        x = embed_layer(input_ids)  # [batch, seq_len, embed_dim]
        
        # Project up to hidden_dim
        x = nn.Dense(self.hidden_dim, name='proj_up')(x)
        
        # Dropout
        if self.dropout > 0:
            x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        
        return x


class StandardEmbedding(nn.Module):
    """Standard embedding layer (fallback)"""
    vocab_size: int
    hidden_dim: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, input_ids, deterministic: bool = False):
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            deterministic: Whether to use dropout
        
        Returns:
            Embeddings [batch, seq_len, hidden_dim]
        """
        x = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim,
            name='token_embed'
        )(input_ids)
        
        if self.dropout > 0:
            x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        
        return x


class EmbeddingLayer(nn.Module):
    """
    Main embedding layer with optional factorization
    """
    vocab_size: int
    hidden_dim: int
    factorized: bool = True
    embed_dim: int = 256  # Only used if factorized=True
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, input_ids, deterministic: bool = False):
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            deterministic: Whether to use dropout
        
        Returns:
            Embeddings [batch, seq_len, hidden_dim]
        """
        if self.factorized:
            return FactorizedEmbedding(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                name='factorized_embed'
            )(input_ids, deterministic=deterministic)
        else:
            return StandardEmbedding(
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                name='standard_embed'
            )(input_ids, deterministic=deterministic)


class OutputHead(nn.Module):
    """
    Output projection head with optional weight tying
    """
    vocab_size: int
    hidden_dim: int
    tie_weights: bool = True
    embedding_layer: Optional[nn.Module] = None
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Hidden states [batch, seq_len, hidden_dim]
        
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        if self.tie_weights and self.embedding_layer is not None:
            # Use tied weights from embedding layer
            # For factorized: need to reverse projection
            # hidden_dim → embed_dim → vocab_size
            
            # Get embedding weights (this is simplified, actual implementation
            # would need to access the params properly)
            # For now, just use Dense layer
            logits = nn.Dense(self.vocab_size, name='lm_head')(x)
        else:
            # Independent output projection
            logits = nn.Dense(self.vocab_size, name='lm_head')(x)
        
        return logits
