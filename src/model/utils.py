"""
Common utilities and helper functions
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional
import flax.linen as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    eps: float = 1e-6
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input tensor [..., dim]
        Returns:
            Normalized tensor with same shape
        """
        # Get feature dimension
        dim = x.shape[-1]
        
        # Learnable scale parameter
        scale = self.param('scale', nn.initializers.ones, (dim,))
        
        # RMS normalization
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        normalized = x / rms
        
        return scale * normalized


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation"""
    dim: int
    hidden_dim: Optional[int] = None
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        """
        Args:
            x: Input tensor [..., dim]
            deterministic: Whether to use dropout
        Returns:
            Output tensor [..., dim]
        """
        hidden_dim = self.hidden_dim or self.dim * 4
        
        # Two projection paths
        gate = nn.Dense(hidden_dim, name='gate')(x)
        value = nn.Dense(hidden_dim, name='value')(x)
        
        # SwiGLU: gate(swish activation) * value
        x = nn.swish(gate) * value
        
        # Dropout
        if self.dropout > 0:
            x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        
        # Project back to original dimension
        x = nn.Dense(self.dim, name='proj_out')(x)
        
        return x


def apply_rotary_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    freqs_cos: jnp.ndarray,
    freqs_sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply rotary position embeddings to queries and keys
    
    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        freqs_cos: Cosine frequencies [seq_len, head_dim]
        freqs_sin: Sine frequencies [seq_len, head_dim]
    
    Returns:
        Rotated (q, k) tensors
    """
    # Reshape q and k for rotation
    # [..., head_dim] -> [..., head_dim//2, 2]
    q_r = q.reshape(*q.shape[:-1], -1, 2)
    k_r = k.reshape(*k.shape[:-1], -1, 2)
    
    # Split into real and imaginary parts
    q_real, q_imag = q_r[..., 0], q_r[..., 1]
    k_real, k_imag = k_r[..., 0], k_r[..., 1]
    
    # Expand freqs to match shape [1, seq_len, 1, head_dim//2]
    freqs_cos = freqs_cos[None, :, None, :]
    freqs_sin = freqs_sin[None, :, None, :]
    
    # Apply rotation
    q_rotated_real = q_real * freqs_cos - q_imag * freqs_sin
    q_rotated_imag = q_real * freqs_sin + q_imag * freqs_cos
    
    k_rotated_real = k_real * freqs_cos - k_imag * freqs_sin
    k_rotated_imag = k_real * freqs_sin + k_imag * freqs_cos
    
    # Concatenate back
    q_out = jnp.stack([q_rotated_real, q_rotated_imag], axis=-1)
    k_out = jnp.stack([k_rotated_real, k_rotated_imag], axis=-1)
    
    # Reshape back to original
    q_out = q_out.reshape(q.shape)
    k_out = k_out.reshape(k.shape)
    
    return q_out, k_out


def precompute_freqs_rope(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    yarn_scale: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute RoPE frequencies with YaRN scaling
    
    Args:
        dim: Head dimension
        max_seq_len: Maximum sequence length
        theta: RoPE theta parameter
        yarn_scale: YaRN extrapolation scale
    
    Returns:
        (freqs_cos, freqs_sin) each [max_seq_len, dim//2]
    """
    # Frequency for each dimension pair
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
    
    # Apply YaRN scaling
    if yarn_scale > 1.0:
        freqs = freqs / yarn_scale
    
    # Position indices
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    
    # Outer product: [max_seq_len, dim//2]
    freqs = jnp.outer(t, freqs)
    
    # Compute cos and sin
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    
    return freqs_cos, freqs_sin


def get_causal_mask(seq_len: int) -> jnp.ndarray:
    """
    Create causal attention mask
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Boolean mask [seq_len, seq_len] where True means attend
    """
    # Lower triangular matrix (including diagonal)
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask


def stochastic_depth(
    x: jnp.ndarray,
    residual: jnp.ndarray,
    drop_rate: float,
    deterministic: bool,
    rng: Optional[random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Stochastic depth (layer dropout) for regularization
    
    Args:
        x: Layer output
        residual: Residual connection input
        drop_rate: Probability of dropping layer
        deterministic: Whether in eval mode
        rng: Random key
    
    Returns:
        Output with stochastic depth applied
    """
    if deterministic or drop_rate == 0.0:
        return x + residual
    
    # Sample dropout
    keep_prob = 1.0 - drop_rate
    if rng is not None:
        mask = random.bernoulli(rng, keep_prob, shape=(x.shape[0],) + (1,) * (x.ndim - 1))
        mask = mask.astype(x.dtype)
    else:
        mask = keep_prob
    
    return (x * mask / keep_prob) + residual
