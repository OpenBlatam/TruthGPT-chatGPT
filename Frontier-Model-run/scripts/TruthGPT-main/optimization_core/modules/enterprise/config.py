"""
Enterprise Configuration and Typed Models
"""
from enum import Enum
from pydantic import BaseModel, ConfigDict


class AdapterMode(str, Enum):
    """Adapter mode enum."""
    OPTIMIZATION = "optimization"
    TRAINING = "training"
    INFERENCE = "inference"
    ENTERPRISE = "enterprise"


class AdapterConfig(BaseModel):
    """Adapter configuration."""
    model_config = ConfigDict(extra="allow")

    mode: AdapterMode = AdapterMode.ENTERPRISE
    attention_heads: int = 16
    hidden_size: int = 512
    num_layers: int = 12
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    activation_dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    
    # Enterprise features
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    use_data_parallel: bool = True
    use_quantization: bool = True


class EnterpriseModelInfo(BaseModel):
    """Typed response for enterprise model info."""
    total_parameters: int
    trainable_parameters: int
    attention_heads: int
    hidden_size: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int
    use_flash_attention: bool
    use_gradient_checkpointing: bool
    use_mixed_precision: bool
    use_data_parallel: bool
    use_quantization: bool

