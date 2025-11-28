"""
Differential Multi-Query Attention

Combines:
- Multi-Query Attention (shared KV heads for efficiency)
- Differential Attention (Q1 - λQ2 for noise cancellation)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

from .utils import RMSNorm, apply_rotary_emb, get_causal_mask


class DifferentialMultiQueryAttention(nn.Module):
    """
    Differential Multi-Query Attention
    
    Improvements over standard attention:
    1. Multi-Query: Single KV head shared across all Q heads (4-8× faster)
    2. Differential: Q1@K1 - λ(Q2@K2) cancels attention noise
    """
    hidden_dim: int
    num_heads: int = 4
    num_kv_heads: int = 1  # Multi-query
    head_dim: Optional[int] = None
    differential: bool = True
    diff_lambda: float = 0.5
    dropout: float = 0.1
    
    def setup(self):
        if self.head_dim is None:
            assert self.hidden_dim % self.num_heads == 0
            self.head_dim = self.hidden_dim // self.num_heads
    
    @nn.compact
    def __call__(
        self,
        x,
        freqs_cos: Optional[jnp.ndarray] = None,
        freqs_sin: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ):
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            freqs_cos: RoPE cos frequencies [seq_len, head_dim]
            freqs_sin: RoPE sin frequencies [seq_len, head_dim]
            mask: Attention mask [seq_len, seq_len] (True = attend)
            deterministic: Whether to apply dropout
        
        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        batch, seq_len, dim = x.shape
        
        if self.differential:
            # Differential attention: need two sets of Q projections
            # Q1, Q2: [batch, seq_len, num_heads, head_dim]
            q1 = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name='q1_proj')(x)
            q2 = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name='q2_proj')(x)
            q1 = q1.reshape(batch, seq_len, self.num_heads, self.head_dim)
            q2 = q2.reshape(batch, seq_len, self.num_heads, self.head_dim)
        else:
            # Standard attention
            q = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name='q_proj')(x)
            q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        
        # Multi-query: single KV head
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, name='k_proj')(x)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, name='v_proj')(x)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE if provided
        if freqs_cos is not None and freqs_sin is not None:
            if self.differential:
                q1, k = apply_rotary_emb(q1, k, freqs_cos, freqs_sin)
                q2, _ = apply_rotary_emb(q2, k, freqs_cos, freqs_sin)
            else:
                q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        
        # Expand KV to match number of query heads (for multi-query)
        # k, v: [batch, seq_len, 1, head_dim] -> [batch, seq_len, num_heads, head_dim]
        k = jnp.repeat(k, self.num_heads // self.num_kv_heads, axis=2)
        v = jnp.repeat(v, self.num_heads // self.num_kv_heads, axis=2)
        
        # Compute attention
        if self.differential:
            # Differential attention: attn1 - λ * attn2
            attn_output = self._differential_attention(q1, q2, k, v, mask, deterministic)
        else:
            # Standard attention
            attn_output = self._standard_attention(q, k, v, mask, deterministic)
        
        # Reshape and project output
        # [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_dim]
        attn_output = attn_output.reshape(batch, seq_len, self.hidden_dim)
        output = nn.Dense(self.hidden_dim, use_bias=False, name='out_proj')(attn_output)
        
        return output
    
    def _standard_attention(self, q, k, v, mask, deterministic):
        """Standard scaled dot-product attention"""
        # q, k, v: [batch, seq_len, num_heads, head_dim]
        
        # Compute attention scores
        # [batch, num_heads, seq_len, seq_len]
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        
        # Apply causal mask
        if mask is not None:
            # mask: [seq_len, seq_len]
            # Expand to [1, 1, seq_len, seq_len]
            mask = mask[None, None, :, :]
            scores = jnp.where(mask, scores, -1e10)
        
        # Softmax
        attn_weights = nn.softmax(scores, axis=-1)
        
        # Dropout on attention weights
        if self.dropout > 0:
            attn_weights = nn.Dropout(self.dropout, deterministic=deterministic)(attn_weights)
        
        # Apply attention to values
        # [batch, num_heads, seq_len, seq_len] @ [batch, seq_len, num_heads, head_dim]
        # -> [batch, seq_len, num_heads, head_dim]
        output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        return output
    
    def _differential_attention(self, q1, q2, k, v, mask, deterministic):
        """
        Differential attention: attn(Q1, K, V) - λ * attn(Q2, K, V)
        
        This cancels out noise patterns in long contexts, improving accuracy.
        """
        # Compute two attention patterns
        # scores1: using Q1
        scores1 = jnp.einsum('bqhd,bkhd->bhqk', q1, k) / jnp.sqrt(self.head_dim)
        
        # scores2: using Q2
        scores2 = jnp.einsum('bqhd,bkhd->bhqk', q2, k) / jnp.sqrt(self.head_dim)
        
        # Apply mask to both
        if mask is not None:
            mask = mask[None, None, :, :]
            scores1 = jnp.where(mask, scores1, -1e10)
            scores2 = jnp.where(mask, scores2, -1e10)
        
        # Softmax for both
        attn_weights1 = nn.softmax(scores1, axis=-1)
        attn_weights2 = nn.softmax(scores2, axis=-1)
        
        # Differential: subtract weighted second pattern
        attn_weights = attn_weights1 - self.diff_lambda * attn_weights2
        
        # Renormalize (since we subtracted)
        attn_weights = attn_weights / (jnp.sum(attn_weights, axis=-1, keepdims=True) + 1e-8)
        
        # Ensure non-negative
        attn_weights = jnp.maximum(attn_weights, 0.0)
        
        # Dropout
        if self.dropout > 0:
            attn_weights = nn.Dropout(self.dropout, deterministic=deterministic)(attn_weights)
        
        # Apply to values
        output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        return output


class AttentionLayer(nn.Module):
    """
    Full attention layer with normalization and residual
    
    Architecture:
        x -> RMSNorm -> DifferentialMQA -> Residual
    """
    hidden_dim: int
    num_heads: int = 4
    num_kv_heads: int = 1
    head_dim: Optional[int] = None
    differential: bool = True
    diff_lambda: float = 0.5
    dropout: float = 0.1
    norm_eps: float = 1e-6
    
    @nn.compact
    def __call__(
        self,
        x,
        freqs_cos: Optional[jnp.ndarray] = None,
        freqs_sin: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ):
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            freqs_cos: RoPE cos frequencies
            freqs_sin: RoPE sin frequencies
            mask: Attention mask
            deterministic: Whether to apply dropout
        
        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        residual = x
        x = RMSNorm(eps=self.norm_eps, name='norm')(x)
        
        x = DifferentialMultiQueryAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            differential=self.differential,
            diff_lambda=self.diff_lambda,
            dropout=self.dropout,
            name='attn'
        )(x, freqs_cos, freqs_sin, mask, deterministic)
        
        x = x + residual
        
        return x
