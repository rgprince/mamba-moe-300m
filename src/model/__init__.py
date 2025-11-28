"""
Model package - complete export
"""

from .config import ModelConfig, count_parameters
from .utils import RMSNorm, SwiGLU, apply_rotary_emb, precompute_freqs_rope, get_causal_mask
from .embedding import EmbeddingLayer, OutputHead
from .mamba import MambaBlock, MambaLayer
from .moe import SoftMoE, MoELayer
from .attention import DifferentialMultiQueryAttention, AttentionLayer
from .memory import MemorySystem, RecurrentMemoryTokens, MemoryController
from .hybrid_model import HybridBlock, MambaMoEModel, create_model_from_config

__all__ = [
    # Config
    "ModelConfig",
    "count_parameters",
    
    # Utils
    "RMSNorm",
    "SwiGLU",
    "apply_rotary_emb",
    "precompute_freqs_rope",
    "get_causal_mask",
    
    # Embedding
    "EmbeddingLayer",
    "OutputHead",
    
    # Core components
    "MambaBlock",
    "MambaLayer",
    "SoftMoE",
    "MoELayer",
    "DifferentialMultiQueryAttention",
    "AttentionLayer",
    "MemorySystem",
    "RecurrentMemoryTokens",
    "MemoryController",
    
    # Full model
    "HybridBlock",
    "MambaMoEModel",
    "create_model_from_config",
]
