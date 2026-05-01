"""
Federated Learning Configuration
================================

Configuration dataclasses for federated learning systems.
"""
from dataclasses import dataclass, field
from .enums import AggregationMethod, ClientSelectionStrategy, PrivacyLevel

@dataclass
class FederatedLearningConfig:
    """Configuration for federated learning system"""
    # Basic settings
    aggregation_method: AggregationMethod = AggregationMethod.FEDAVG
    client_selection_strategy: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY
    
    # Training settings
    num_rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    
    # Aggregation settings
    aggregation_frequency: int = 1
    enable_weighted_aggregation: bool = True
    enable_momentum: bool = True
    momentum_factor: float = 0.9
    
    # Privacy settings
    noise_multiplier: float = 1.0
    l2_norm_clip: float = 1.0
    delta: float = 1e-5
    epsilon: float = 1.0
    
    # Communication settings
    communication_rounds: int = 10
    enable_compression: bool = True
    compression_ratio: float = 0.1
    enable_quantization: bool = True
    quantization_bits: int = 8
    
    # Advanced features
    enable_byzantine_robustness: bool = True
    enable_asynchronous_updates: bool = True
    enable_personalization: bool = True
    enable_meta_learning: bool = True
    
    def __post_init__(self):
        """Validate federated learning configuration"""
        if self.num_rounds <= 0:
            raise ValueError("Number of rounds must be positive")
        if self.clients_per_round <= 0:
            raise ValueError("Clients per round must be positive")
        if self.local_epochs <= 0:
            raise ValueError("Local epochs must be positive")
        if not (0 < self.learning_rate <= 1):
            raise ValueError("Learning rate must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not (0 < self.momentum_factor < 1):
            raise ValueError("Momentum factor must be between 0 and 1")
        if self.noise_multiplier < 0:
            raise ValueError("Noise multiplier must be non-negative")
        if self.l2_norm_clip <= 0:
            raise ValueError("L2 norm clip must be positive")
        if self.delta <= 0:
            raise ValueError("Delta must be positive")
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")

