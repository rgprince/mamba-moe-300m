"""
Hierarchical Memory System for exceptional context retention

Implements:
- L2: Recurrent memory tokens (learnable compression)
- Memory controller (dynamic allocation)
- Memory read/write operations
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

from .utils import RMSNorm


class RecurrentMemoryTokens(nn.Module):
    """
    Learnable memory tokens that compress context across long sequences
    
    These tokens are updated recurrently and store compressed information
    from the conversation history.
    """
    num_tokens: int = 8
    memory_dim: int = 1024
    
    @nn.compact
    def __call__(self, x, prev_memory: Optional[jnp.ndarray] = None):
        """
        Args:
            x: Current context [batch, seq_len, hidden_dim]
            prev_memory: Previous memory state [batch, num_tokens, memory_dim]
        
        Returns:
            updated_memory: [batch, num_tokens, memory_dim]
        """
        batch, seq_len, hidden_dim = x.shape
        
        # Initialize memory if not provided
        if prev_memory is None:
            prev_memory = jnp.zeros((batch, self.num_tokens, self.memory_dim))
        
        # Compute summary of current context (mean pooling)
        context_summary = jnp.mean(x, axis=1, keepdims=True)  # [batch, 1, hidden_dim]
        
        # Project context to memory dimension
        context_proj = nn.Dense(self.memory_dim, name='context_proj')(context_summary)
        
        # Update memory with gating mechanism
        # Gate decides how much to update vs keep old memory
        update_gate = nn.Dense(self.num_tokens * self.memory_dim, name='update_gate')(context_proj)
        update_gate = nn.sigmoid(update_gate.reshape(batch, self.num_tokens, self.memory_dim))
        
        # New memory content
        new_content = nn.Dense(self.num_tokens * self.memory_dim, name='new_content')(context_proj)
        new_content = nn.tanh(new_content.reshape(batch, self.num_tokens, self.memory_dim))
        
        # Gated update: memory = gate * prev + (1-gate) * new
        updated_memory = update_gate * prev_memory + (1 - update_gate) * new_content
        
        return updated_memory


class MemoryController(nn.Module):
    """
    Memory allocation controller
    
    Learns to dynamically allocate attention between different memory tiers
    based on the query type.
    """
    hidden_dim: int
    controller_hidden_dim: int = 512
    num_tiers: int = 4  # L1, L2, L3, L4
    
    @nn.compact
    def __call__(self, query):
        """
        Args:
            query: Query representation [batch, hidden_dim]
        
        Returns:
            allocation_weights: [batch, num_tiers] - weights for each memory tier
        """
        # MLP to predict allocation
        x = nn.Dense(self.controller_hidden_dim, name='fc1')(query)
        x = nn.gelu(x)
        x = nn.Dense(self.controller_hidden_dim // 2, name='fc2')(x)
        x = nn.gelu(x)
        
        # Output allocation logits
        allocation_logits = nn.Dense(self.num_tiers, name='allocation_head')(x)
        
        # Softmax to get valid probability distribution
        allocation_weights = nn.softmax(allocation_logits, axis=-1)
        
        return allocation_weights


class MemorySystem(nn.Module):
    """
    Full hierarchical memory system
    
    Manages:
    - Recurrent memory tokens for L2 (conversation memory)
    - Memory controller for dynamic allocation
    """
    hidden_dim: int
    num_recurrent_tokens: int = 8
    memory_dim: int = 1024
    controller_hidden_dim: int = 512
    num_tiers: int = 4
    
    @nn.compact
    def __call__(
        self,
        x,
        prev_memory: Optional[jnp.ndarray] = None,
        return_allocation: bool = False
    ):
        """
        Args:
            x: Current input [batch, seq_len, hidden_dim]
            prev_memory: Previous memory state [batch, num_tokens, memory_dim]
            return_allocation: Whether to return allocation weights
        
        Returns:
            memory_output: Memory-augmented representation
            updated_memory: Updated memory state for next iteration
            (optional) allocation_weights: Memory tier allocation
        """
        batch, seq_len, dim = x.shape
        
        # Update recurrent memory
        updated_memory = RecurrentMemoryTokens(
            num_tokens=self.num_recurrent_tokens,
            memory_dim=self.memory_dim,
            name='recurrent_memory'
        )(x, prev_memory)
        
        # Query representation for controller (use last token)
        query_rep = x[:, -1, :]  # [batch, hidden_dim]
        
        # Get allocation weights from controller
        allocation_weights = MemoryController(
            hidden_dim=self.hidden_dim,
            controller_hidden_dim=self.controller_hidden_dim,
            num_tiers=self.num_tiers,
            name='controller'
        )(query_rep)
        
        # For now, just use recurrent memory (L1 + L2)
        # L3 (long-term) and L4 (external) would be added during inference
        
        # Project memory back to hidden dimension
        memory_proj = nn.Dense(self.hidden_dim, name='memory_proj')(
            updated_memory.reshape(batch, -1)  # Flatten memory tokens
        )
        
        # Add memory to input as additional context
        memory_output = memory_proj[:, None, :]  # [batch, 1, hidden_dim]
        
        if return_allocation:
            return memory_output, updated_memory, allocation_weights
        else:
            return memory_output, updated_memory
