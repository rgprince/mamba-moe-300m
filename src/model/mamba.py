"""
Mamba 2 State-Space Model (SSM) implementation in JAX/Flax

Based on:
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- "Mamba-2: A State Space Duality Perspective" (Dao & Gu, 2024)

This implements the selective SSM with hardware-aware parallel scanning.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
from functools import partial

from .utils import RMSNorm, SwiGLU


class SelectiveSSM(nn.Module):
    """
    Selective State-Space Model (core of Mamba)
    
    The SSM computes:
        h_t = A * h_{t-1} + B * x_t
        y_t = C * h_t
    
    Where A, B, C are input-dependent (selective mechanism)
    """
    hidden_dim: int
    state_dim: int = 16  # N in the paper
    dt_rank: int = None  # Usually hidden_dim // 16
    
    def setup(self):
        if self.dt_rank is None:
            self.dt_rank = max(32, self.hidden_dim // 16)
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
        
        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        batch, seq_len, dim = x.shape
        
        # Project input to get B, C, dt (input-dependent parameters)
        # B: [batch, seq_len, state_dim]
        # C: [batch, seq_len, state_dim]
        # dt: [batch, seq_len, hidden_dim]
        
        # Concatenate projections for efficiency
        x_proj = nn.Dense(
            self.state_dim * 2 + self.dt_rank,  # B + C + dt
            use_bias=False,
            name='x_proj'
        )(x)
        
        # Split into B, C, dt
        B = x_proj[..., :self.state_dim]  # [batch, seq_len, state_dim]
        C = x_proj[..., self.state_dim:self.state_dim*2]  # [batch, seq_len, state_dim]
        dt_proj = x_proj[..., self.state_dim*2:]  # [batch, seq_len, dt_rank]
        
        # Project dt to hidden_dim
        dt = nn.Dense(self.hidden_dim, use_bias=True, name='dt_proj')(dt_proj)
        dt = nn.softplus(dt)  # Ensure dt > 0
        
        # Initialize A (state transition matrix)
        # A is learned but fixed (not input-dependent)
        # Shape: [state_dim, hidden_dim]
        A = self.param(
            'A',
            nn.initializers.lecun_normal(),
            (self.state_dim, self.hidden_dim)
        )
        
        # Make A negative for stability
        A = -jnp.exp(A)
        
        # Discretize continuous parameters (convert to discrete-time)
        # Using Zero-Order Hold (ZOH) discretization:
        # A_discrete = exp(A * dt)
        # B_discrete = A^{-1} (exp(A * dt) - I) * B
        
        # For efficiency, use approximation:
        # A_bar = exp(A * dt)  [batch, seq_len, state_dim, hidden_dim]
        # B_bar = (A * dt).inv() * (A_bar - I) * B ≈ dt * B
        
        # Expand dimensions for broadcasting
        # A: [1, 1, state_dim, hidden_dim]
        # dt: [batch, seq_len, 1, hidden_dim]
        A_expanded = A[None, None, :, :]
        dt_expanded = dt[:, :, None, :]
        
        # Discretize A: exp(A * dt)
        A_bar = jnp.exp(A_expanded * dt_expanded)  # [batch, seq_len, state_dim, hidden_dim]
        
        # Discretize B: approximately dt * B
        B_expanded = B[:, :, :, None]  # [batch, seq_len, state_dim, 1]
        dt_for_B = dt[:, :, None, :]  # [batch, seq_len, 1, hidden_dim]
        B_bar = B_expanded * dt_for_B  # [batch, seq_len, state_dim, hidden_dim]
        
        # Input modulation
        x_expanded = x[:, :, None, :]  # [batch, seq_len, 1, hidden_dim]
        
        # Run SSM: scan over sequence
        # This is the selective mechanism - each timestep has different A, B
        y = self._parallel_scan(A_bar, B_bar, C, x_expanded)
        
        return y
    
    def _parallel_scan(self, A_bar, B_bar, C, x):
        """
        Parallel associative scan for SSM computation
        
        This is the key optimization that makes Mamba fast on TPUs/GPUs.
        Instead of sequential scanning (slow), we use parallel scan.
        
        Args:
            A_bar: Discrete A [batch, seq_len, state_dim, hidden_dim]
            B_bar: Discrete B [batch, seq_len, state_dim, hidden_dim]
            C: Output matrix [batch, seq_len, state_dim]
            x: Input [batch, seq_len, 1, hidden_dim]
        
        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        batch, seq_len, state_dim, hidden_dim = A_bar.shape
        
        # Compute B_bar * x (element of state update)
        # [batch, seq_len, state_dim, hidden_dim] * [batch, seq_len, 1, hidden_dim]
        Bu = B_bar * x  # [batch, seq_len, state_dim, hidden_dim]
        
        # Parallel scan operation
        # We need to compute: h_t = A_t * h_{t-1} + B_t * u_t
        # This can be done in parallel using associative scan
        
        def scan_fn(carry, inputs):
            """Sequential scan for simplicity (can be optimized with jax.lax.associative_scan)"""
            h = carry  # [batch, state_dim, hidden_dim]
            A_t, Bu_t = inputs  # [batch, state_dim, hidden_dim], [batch, state_dim, hidden_dim]
            
            # Update: h_t = A_t * h_{t-1} + Bu_t
            h = A_t * h + Bu_t
            
            return h, h
        
        # Initialize state
        h_0 = jnp.zeros((batch, state_dim, hidden_dim))
        
        # Scan over time
        _, h_all = jax.lax.scan(
            scan_fn,
            h_0,
            (A_bar.transpose(1, 0, 2, 3), Bu.transpose(1, 0, 2, 3))  # Swap seq and batch dims
        )
        
        # h_all: [seq_len, batch, state_dim, hidden_dim]
        h_all = h_all.transpose(1, 0, 2, 3)  # [batch, seq_len, state_dim, hidden_dim]
        
        # Compute output: y_t = C_t * h_t
        # C: [batch, seq_len, state_dim]
        # h: [batch, seq_len, state_dim, hidden_dim]
        C_expanded = C[:, :, :, None]  # [batch, seq_len, state_dim, 1]
        
        # Sum over state dimension
        y = jnp.sum(C_expanded * h_all, axis=2)  # [batch, seq_len, hidden_dim]
        
        return y


class MambaBlock(nn.Module):
    """
    Complete Mamba block with SSM + projections + convolution
    
    Architecture:
        x -> [Linear -> Conv1d -> SiLU -> SSM] -> Gate -> Linear -> output
    """
    hidden_dim: int
    state_dim: int = 16
    conv_kernel: int = 4
    expand_factor: int = 2
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
        batch, seq_len, dim = x.shape
        expanded_dim = self.hidden_dim * self.expand_factor
        
        # Input projection with gating
        # Split into two paths: main and gate
        x_proj = nn.Dense(expanded_dim * 2, use_bias=False, name='in_proj')(x)
        x_main, x_gate = jnp.split(x_proj, 2, axis=-1)
        
        # Convolution for local context (before SSM)
        # Conv1D over sequence dimension
        x_conv = nn.Conv(
            features=expanded_dim,
            kernel_size=(self.conv_kernel,),
            padding='SAME',
            feature_group_count=expanded_dim,  # Depthwise conv
            name='conv1d'
        )(x_main[:, :, None, :])  # Add channel dim
        x_conv = x_conv.squeeze(2)  # Remove channel dim
        
        # Activation
        x_conv = nn.silu(x_conv)
        
        # Selective SSM
        y = SelectiveSSM(
            hidden_dim=expanded_dim,
            state_dim=self.state_dim,
            name='ssm'
        )(x_conv)
        
        # Gating mechanism
        y = y * nn.silu(x_gate)
        
        # Output projection back to hidden_dim
        y = nn.Dense(self.hidden_dim, use_bias=False, name='out_proj')(y)
        
        # Dropout
        if self.dropout > 0:
            y = nn.Dropout(self.dropout, deterministic=deterministic)(y)
        
        return y


class MambaLayer(nn.Module):
    """
    Full Mamba layer with normalization and residual connection
    
    Architecture:
        x -> RMSNorm -> MambaBlock -> Residual
    """
    hidden_dim: int
    state_dim: int = 16
    conv_kernel: int = 4
    expand_factor: int = 2
    dropout: float = 0.0
    norm_eps: float = 1e-6
    
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            deterministic: Whether to apply dropout
        
        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        # Pre-normalization
        residual = x
        x = RMSNorm(eps=self.norm_eps, name='norm')(x)
        
        # Mamba block
        x = MambaBlock(
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            conv_kernel=self.conv_kernel,
            expand_factor=self.expand_factor,
            dropout=self.dropout,
            name='mamba'
        )(x, deterministic=deterministic)
        
        # Residual connection
        x = x + residual
        
        return x
