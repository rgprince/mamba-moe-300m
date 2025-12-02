"""
Soft Mixture of Experts (MoE) implementation

Implements soft routing where all experts are blended with learned weights,
avoiding the collapse and load balancing issues of hard routing.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

from .utils import SwiGLU


class Expert(nn.Module):
    """Single expert network (SwiGLU FFN)"""
    hidden_dim: int
    intermediate_dim: int
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            deterministic: Whether to apply dropout
        
        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        return SwiGLU(
            dim=self.hidden_dim,
            hidden_dim=self.intermediate_dim,
            dropout=self.dropout,
            name='swiglu'
        )(x, deterministic=deterministic)


class SoftMoE(nn.Module):
    """
    Soft Mixture of Experts with shared expert
    
    Instead of hard routing (top-k experts), we do soft routing:
    - All experts process the input
    - Outputs are weighted by routing probabilities
    - More stable, no expert collapse
    """
    hidden_dim: int
    intermediate_dim: int
    num_experts: int = 4
    shared_expert: bool = True
    temperature: float = 1.0
    expert_dropout: float = 0.1
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x, deterministic: bool = False) -> Tuple[jnp.ndarray, dict]:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            deterministic: Whether to apply dropout
        
        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_loss_dict: Dictionary with load balancing loss
        """
        batch, seq_len, dim = x.shape
        
        # Router network: computes expert weights
        # [batch, seq_len, hidden_dim] -> [batch, seq_len, num_experts]
        router_logits = nn.Dense(self.num_experts, name='router')(x)
        
        # Soft routing: weighted blend of all experts
        # Apply temperature for controlling sharpness
        router_probs = nn.softmax(router_logits / self.temperature, axis=-1)
        
        # Expert dropout during training (regularization)
        if not deterministic and self.expert_dropout > 0:
            keep_prob = 1.0 - self.expert_dropout
            expert_mask = jax.random.bernoulli(
                self.make_rng('dropout'),
                keep_prob,
                shape=(batch, seq_len, self.num_experts)
            )
            router_probs = router_probs * expert_mask
            # Renormalize
            router_probs = router_probs / (jnp.sum(router_probs, axis=-1, keepdims=True) + 1e-8)
        
        # Process input through all experts
        expert_outputs = []
        for i in range(self.num_experts):
            expert_out = Expert(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                dropout=self.dropout,
                name=f'expert_{i}'
            )(x, deterministic=deterministic)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs: [num_experts, batch, seq_len, hidden_dim]
        expert_outputs = jnp.stack(expert_outputs, axis=0)
        
        # Weighted combination
        # router_probs: [batch, seq_len, num_experts]
        # expert_outputs: [num_experts, batch, seq_len, hidden_dim]
        
        # Reshape for matmul: [batch, seq_len, 1, num_experts] @ [batch, seq_len, num_experts, hidden_dim]
        router_probs_expanded = router_probs[:, :, None, :]  # [batch, seq_len, 1, num_experts]
        expert_outputs_transposed = expert_outputs.transpose(1, 2, 0, 3)  # [batch, seq_len, num_experts, hidden_dim]
        
        # Weighted sum
        moe_output = jnp.matmul(router_probs_expanded, expert_outputs_transposed).squeeze(2)
        # Result: [batch, seq_len, hidden_dim]
        
        # Shared expert (always active)
        if self.shared_expert:
            shared_output = Expert(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                dropout=self.dropout,
                name='shared_expert'
            )(x, deterministic=deterministic)
            
            # Add shared expert output
            moe_output = moe_output + shared_output
        
        # Compute auxiliary losses
        aux_losses = self._compute_load_balancing_loss(router_probs)
        
        return moe_output, aux_losses
    
    def _compute_load_balancing_loss(self, router_probs):
        """
        Compute load balancing loss to encourage uniform expert utilization
        
        Args:
            router_probs: [batch, seq_len, num_experts]
        
        Returns:
            Dictionary with loss components
        """
        # Average routing probability per expert (over batch and sequence)
        expert_usage = jnp.mean(router_probs, axis=(0, 1))  # [num_experts]
        
        # Ideal uniform distribution
        uniform_prob = 1.0 / self.num_experts
        
        # L2 loss from uniform distribution
        load_balancing_loss = jnp.mean((expert_usage - uniform_prob) ** 2)
        
        
        # Entropy regularization (encourage diverse routing)
        # 🔧 FIX: Use maximum instead of addition to prevent log(very small number)
        entropy = -jnp.sum(router_probs * jnp.log(jnp.maximum(router_probs, 1e-7)), axis=-1)
        entropy_loss = -jnp.mean(entropy)  # Negative because we want to maximize entropy

        
        return {
            'load_balancing_loss': load_balancing_loss,
            'entropy_loss': entropy_loss,
            'expert_usage': expert_usage,  # For logging
            'router_probs_mean': jnp.mean(router_probs, axis=(0, 1))  # For logging
        }


class MoELayer(nn.Module):
    """
    Full MoE layer with normalization and residual
    
    Architecture:
        x -> RMSNorm -> SoftMoE -> Residual
    """
    hidden_dim: int
    intermediate_dim: int
    num_experts: int = 4
    shared_expert: bool = True
    temperature: float = 1.0
    expert_dropout: float = 0.1
    dropout: float = 0.0
    norm_eps: float = 1e-6
    
    @nn.compact
    def __call__(self, x, deterministic: bool = False) -> Tuple[jnp.ndarray, dict]:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            deterministic: Whether to apply dropout
        
        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_losses: Dictionary with MoE auxiliary losses
        """
        from .utils import RMSNorm
        
        residual = x
        x = RMSNorm(eps=self.norm_eps, name='norm')(x)
        
        x, aux_losses = SoftMoE(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_experts=self.num_experts,
            shared_expert=self.shared_expert,
            temperature=self.temperature,
            expert_dropout=self.expert_dropout,
            dropout=self.dropout,
            name='moe'
        )(x, deterministic=deterministic)
        
        x = x + residual
        
        return x, aux_losses
