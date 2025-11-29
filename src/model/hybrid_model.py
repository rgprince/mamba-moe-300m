"""
Full Mamba-MoE Hybrid Model Assembly

Combines all components:
- Mamba 2 SSM (every layer)
- Soft MoE (every 4th layer)
- Differential Attention (every 6th layer)
- Memory System
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict, Tuple

from .config import ModelConfig
from .embedding import EmbeddingLayer, OutputHead
from .utils import RMSNorm, precompute_freqs_rope, get_causal_mask, stochastic_depth
from .mamba import MambaLayer
from .moe import MoELayer
from .attention import AttentionLayer
from .memory import MemorySystem


class HybridBlock(nn.Module):
    """
    Single hybrid block that may contain:
    - Mamba SSM (always)
    - MoE layer (conditional, based on layer index)
    - Attention layer (conditional, based on layer index)
    """
    config: ModelConfig
    layer_idx: int
    
    @nn.compact
    def __call__(
        self,
        x,
        freqs_cos: Optional[jnp.ndarray] = None,
        freqs_sin: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            freqs_cos: RoPE cosine frequencies
            freqs_sin: RoPE sine frequencies
            mask: Causal attention mask
            deterministic: Whether in eval mode
        
        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_losses: Dictionary with auxiliary losses (MoE load balancing, etc.)
        """
        aux_losses = {}
        
        # Mamba SSM (always present)
        x = MambaLayer(
            hidden_dim=self.config.hidden_dim,
            state_dim=self.config.mamba.state_dim,
            conv_kernel=self.config.mamba.conv_kernel,
            expand_factor=self.config.mamba.expand_factor,
            dt_rank=self.config.mamba.dt_rank,
            dropout=self.config.dropout,
            norm_eps=self.config.norm_eps,
            name='mamba'
        )(x, deterministic=deterministic)
        
        # MoE layer (conditional)
        if self.layer_idx in self.config.moe.layer_indices:
            x, moe_aux = MoELayer(
                hidden_dim=self.config.hidden_dim,
                intermediate_dim=self.config.intermediate_dim,
                num_experts=self.config.moe.num_experts,
                shared_expert=self.config.moe.shared_expert,
                temperature=self.config.moe.temperature,
                expert_dropout=self.config.moe.expert_dropout,
                dropout=self.config.dropout,
                norm_eps=self.config.norm_eps,
                name='moe'
            )(x, deterministic=deterministic)
            
            aux_losses.update({f'moe_layer_{self.layer_idx}_{k}': v for k, v in moe_aux.items()})
        
        # Attention layer (conditional)
        if self.layer_idx in self.config.attention.layer_indices:
            x = AttentionLayer(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.attention.num_heads,
                num_kv_heads=self.config.attention.num_kv_heads,
                head_dim=self.config.attention.head_dim,
                differential=self.config.attention.differential,
                diff_lambda=self.config.attention.diff_lambda,
                dropout=self.config.attention_dropout,
                norm_eps=self.config.norm_eps,
                name='attn'
            )(x, freqs_cos, freqs_sin, mask, deterministic)
        
        return x, aux_losses


class MambaMoEModel(nn.Module):
    """
    Full Mamba-MoE 300M Hybrid Model
    
    Architecture:
        Embedding -> [24x HybridBlocks] -> Memory -> Norm -> LM Head
    """
    config: ModelConfig
    
    @nn.compact
    def __call__(
        self,
        input_ids,
        prev_memory: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
        return_aux: bool = False
    ):
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            prev_memory: Previous memory state (for conversation continuity)
            deterministic: Whether in eval mode (no dropout)
            return_aux: Whether to return auxiliary info (losses, memory, etc.)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            (optional) aux: Dictionary with memory, allocation, losses
        """
        batch, seq_len = input_ids.shape
        
        # Embeddings
        x = EmbeddingLayer(
            vocab_size=self.config.embedding.vocab_size,
            hidden_dim=self.config.hidden_dim,
            factorized=self.config.embedding.factorized,
            embed_dim=self.config.embedding.embed_dim,
            dropout=self.config.embedding.dropout,
            name='embedding'
        )(input_ids, deterministic=deterministic)
        
        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_rope(
            dim=self.config.attention.head_dim,
            max_seq_len=seq_len,
            theta=self.config.position_encoding.rope_theta,
            yarn_scale=self.config.position_encoding.yarn_scale
        )
        
        # Causal mask
        mask = get_causal_mask(seq_len)
        
        # Process through hybrid blocks
        all_aux_losses = {}
        for i in range(self.config.num_layers):
            x, aux_losses = HybridBlock(
                config=self.config,
                layer_idx=i,
                name=f'layer_{i}'
            )(x, freqs_cos, freqs_sin, mask, deterministic)
            
            # Accumulate auxiliary losses
            all_aux_losses.update(aux_losses)
        
        # Memory system
        memory_output, updated_memory = MemorySystem(
            hidden_dim=self.config.hidden_dim,
            num_recurrent_tokens=self.config.memory.num_recurrent_tokens,
            memory_dim=self.config.memory.memory_dim,
            controller_hidden_dim=self.config.memory.controller_hidden_dim,
            num_tiers=self.config.memory.num_tiers,
            name='memory'
        )(x, prev_memory)
        
        # Add memory to final representation
        x = x + memory_output  # Broadcast [batch, 1, hidden_dim]
        
        # Final normalization
        x = RMSNorm(eps=self.config.norm_eps, name='final_norm')(x)
        
        # LM head
        logits = OutputHead(
            vocab_size=self.config.embedding.vocab_size,
            hidden_dim=self.config.hidden_dim,
            tie_weights=self.config.embedding.tie_weights,
            name='lm_head'
        )(x)
        
        if return_aux:
            aux = {
                'memory': updated_memory,
                'aux_losses': all_aux_losses,
            }
            return logits, aux
        else:
            return logits
    
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        prev_memory: Optional[jnp.ndarray] = None,
        rng: jax.random.PRNGKey = None
    ):
        """
        Autoregressive generation
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            prev_memory: Previous conversation memory
            rng: Random key
        
        Returns:
            generated_ids: [batch, seq_len + max_new_tokens]
        """
        current_ids = input_ids
        current_memory = prev_memory
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits, aux = self(
                current_ids,
                prev_memory=current_memory,
                deterministic=True,
                return_aux=True
            )
            
            # Update memory
            current_memory = aux['memory']
            
            # Get last token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token (simplified - use top-p later)
            if rng is not None:
                rng, sample_rng = jax.random.split(rng)
                next_token = jax.random.categorical(sample_rng, next_token_logits, axis=-1)
            else:
                next_token = jnp.argmax(next_token_logits, axis=-1)
            
            # Append to sequence
            current_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=1)
        
        return current_ids


def create_model_from_config(config_path: str) -> MambaMoEModel:
    """
    Create model from YAML config file
    
    Args:
        config_path: Path to model config YAML
    
    Returns:
        Initialized model
    """
    config = ModelConfig.from_yaml(config_path)
    return MambaMoEModel(config=config)
