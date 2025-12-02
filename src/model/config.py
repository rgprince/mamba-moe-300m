"""
Model configuration and utilities
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Embedding layer configuration"""
    vocab_size: Optional[int] = None  # Must be set from config, no hardcoded default!
    factorized: bool = True
    embed_dim: int = 256  # Factorized dimension
    hidden_dim: int = 1024
    dropout: float = 0.1
    tie_weights: bool = True



@dataclass
class PositionEncodingConfig:
    """Position encoding configuration"""
    type: str = "yarn_rope"
    max_seq_len_train: int = 8192
    max_seq_len_infer: int = 32768
    rope_theta: float = 10000.0
    yarn_scale: float = 1.0


@dataclass
class MambaConfig:
    """Mamba 2 SSM configuration"""
    state_dim: int = 16
    conv_kernel: int = 4
    expand_factor: int = 2
    use_fast_path: bool = True
    dt_rank: int = 128  # Changed from "auto" to concrete int value


@dataclass
class MoEConfig:
    """Mixture of Experts configuration"""
    num_layers: int = 6
    layer_indices: list[int] = field(default_factory=lambda: [3, 7, 11, 15, 19, 23])
    num_experts: int = 4
    shared_expert: bool = True
    routing_type: str = "soft"
    temperature: float = 1.0
    expert_dropout: float = 0.1
    load_balancing_weight: float = 0.01
    target_entropy: float = 0.95


@dataclass
class AttentionConfig:
    """Attention layer configuration"""
    num_layers: int = 4
    layer_indices: list[int] = field(default_factory=lambda: [5, 11, 17, 23])
    num_heads: int = 4
    num_kv_heads: int = 1  # Multi-query
    head_dim: int = 256
    differential: bool = True
    diff_lambda: float = 0.5
    dropout: float = 0.1
    use_flash_attention: bool = True


@dataclass
class MemoryConfig:
    """Memory system configuration"""
    num_recurrent_tokens: int = 8
    memory_dim: int = 1024
    controller_hidden_dim: int = 512
    controller_num_layers: int = 2
    allocation_enabled: bool = True
    num_tiers: int = 4


@dataclass
class ModelConfig:
    """Complete model configuration"""
    name: str = "mamba-moe-300m-v1"
    version: str = "1.0.0"
    
    # Architecture
    num_layers: int = 24
    hidden_dim: int = 1024
    intermediate_dim: int = 2816
    
    # Components
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    position_encoding: PositionEncodingConfig = field(default_factory=PositionEncodingConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Normalization & Activation
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    activation: str = "swiglu"
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    layer_drop_rate: float = 0.0
    
    # Initialization
    init_type: str = "normal"
    init_std: float = 0.02
    rescale_prenorm_residual: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load config from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse nested configs
        model_dict = config_dict.get('model', {})
        
        # Get vocab_size from model level (NOT embedding level!)
        vocab_size = model_dict.get('vocab_size')
        if vocab_size is None:
            raise ValueError("vocab_size must be specified in model config!")
        
        # Create embedding config with proper vocab_size
        embedding_dict = model_dict.get('embedding', {})
        embedding_dict['vocab_size'] = vocab_size  # Override with model-level vocab
        embedding_cfg = EmbeddingConfig(**embedding_dict)
        
        pos_cfg = PositionEncodingConfig(**model_dict.get('position_encoding', {}))
        
        # Parse mamba config with type casting
        mamba_dict = model_dict.get('mamba', {})
        mamba_cfg = MambaConfig(
            state_dim=int(mamba_dict.get('state_dim', 16)),
            conv_kernel=int(mamba_dict.get('conv_kernel', 4)),
            expand_factor=int(mamba_dict.get('expand_factor', 2)),
            use_fast_path=mamba_dict.get('use_fast_path', True),
            dt_rank=int(mamba_dict.get('dt_rank', 128))  # Explicit int cast
        )
        moe_cfg = MoEConfig(
            num_layers=model_dict.get('moe', {}).get('num_layers', 6),
            layer_indices=model_dict.get('moe', {}).get('layer_indices', [3, 7, 11, 15, 19, 23]),
            num_experts=model_dict.get('moe', {}).get('num_experts', 4),
            shared_expert=model_dict.get('moe', {}).get('shared_expert', True),
            routing_type=model_dict.get('moe', {}).get('routing', {}).get('type', 'soft'),
            temperature=model_dict.get('moe', {}).get('routing', {}).get('temperature', 1.0),
            expert_dropout=model_dict.get('moe', {}).get('routing', {}).get('expert_dropout', 0.1),
            load_balancing_weight=model_dict.get('moe', {}).get('load_balancing', {}).get('loss_weight', 0.01),
            target_entropy=model_dict.get('moe', {}).get('load_balancing', {}).get('target_entropy', 0.95),
        )
        attn_cfg = AttentionConfig(**model_dict.get('attention', {}))
        
        # Parse memory config with nested controller
        memory_dict = model_dict.get('memory', {})
        controller_dict = memory_dict.pop('controller', {})
        
        mem_cfg = MemoryConfig(
            num_recurrent_tokens=memory_dict.get('num_recurrent_tokens', 8),
            memory_dim=memory_dict.get('memory_dim', 1024),
            controller_hidden_dim=controller_dict.get('hidden_dim', 512),
            controller_num_layers=controller_dict.get('num_layers', 2),
            allocation_enabled=memory_dict.get('allocation_enabled', True),
            num_tiers=memory_dict.get('num_tiers', 4)
        )
        
        return cls(
            name=model_dict.get('name', 'mamba-moe-300m-v1'),
            version=model_dict.get('version', '1.0.0'),
            num_layers=model_dict.get('num_layers', 24),
            hidden_dim=model_dict.get('hidden_dim', 1024),
            intermediate_dim=model_dict.get('intermediate_dim', 2816),
            embedding=embedding_cfg,
            position_encoding=pos_cfg,
            mamba=mamba_cfg,
            moe=moe_cfg,
            attention=attn_cfg,
            memory=mem_cfg,
            norm_type=model_dict.get('norm', {}).get('type', 'rmsnorm'),
            norm_eps=float(model_dict.get('norm', {}).get('eps', 1e-6)),  # Explicit float cast
            activation=model_dict.get('activation', 'swiglu'),
            dropout=model_dict.get('dropout', 0.1),
            attention_dropout=model_dict.get('attention_dropout', 0.1),
            residual_dropout=model_dict.get('residual_dropout', 0.1),
            layer_drop_rate=model_dict.get('layer_drop_rate', 0.0),
            init_type=model_dict.get('init', {}).get('type', 'normal'),
            init_std=model_dict.get('init', {}).get('std', 0.02),
            rescale_prenorm_residual=model_dict.get('init', {}).get('rescale_prenorm_residual', True),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'intermediate_dim': self.intermediate_dim,
            'norm_type': self.norm_type,
            'activation': self.activation,
        }
    
    def __repr__(self) -> str:
        return f"ModelConfig(name='{self.name}', layers={self.num_layers}, hidden_dim={self.hidden_dim})"


def count_parameters(config: ModelConfig) -> int:
    """Estimate parameter count from config"""
    params = 0
    
    # Embeddings (factorized)
    if config.embedding.factorized:
        params += config.embedding.vocab_size * config.embedding.embed_dim  # Input
        params += config.embedding.embed_dim * config.hidden_dim  # Projection
    else:
        params += config.embedding.vocab_size * config.hidden_dim
    
    # Mamba layers (rough estimate: ~7M per layer at 1024 dim)
    params += config.num_layers * 7_000_000
    
    # MoE layers
    expert_size = config.hidden_dim * config.intermediate_dim * 2  # Up + down projection
    params += config.moe.num_layers * (
        expert_size * (config.moe.num_experts + (1 if config.moe.shared_expert else 0))
    )
    
    # Attention layers (rough estimate: ~3M per layer)
    params += config.attention.num_layers * 3_000_000
    
    # Memory system
    params += config.memory.num_recurrent_tokens * config.memory.memory_dim
    params += config.memory.controller_num_layers * config.memory.controller_hidden_dim ** 2
    
    # Output head (tied with embeddings, so no extra params)
    
    return params
