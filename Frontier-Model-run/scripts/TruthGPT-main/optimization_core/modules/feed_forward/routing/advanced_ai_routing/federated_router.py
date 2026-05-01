"""
Federated Learning Router
=========================

Advanced routing using federated learning with privacy-preserving techniques
and distributed optimization.
"""

import base64
import json
import logging
import random
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from optimization_core.modules.feed_forward.routing.modular_routing.base_router import (
    BaseRouter,
    RouterConfig,
    RoutingResult,
)

# Configure logging
logger = logging.getLogger(__name__)


class FederatedClient:
    """Federated learning client."""

    def __init__(
        self,
        client_id: str,
        server_url: str,
        encryption_key: bytes,
        privacy_level: str = "high"
    ) -> None:
        """
        Initialize FederatedClient.

        Args:
            client_id: Client identifier.
            server_url: Federated server URL.
            encryption_key: Key for encryption.
            privacy_level: Level of privacy (low, medium, high, maximum).
        """
        self.client_id = client_id
        self.server_url = server_url
        self.encryption_key = encryption_key
        self.privacy_level = privacy_level
        self.local_model: Optional[nn.Module] = None
        self.local_data: List[torch.Tensor] = []
        self.participation_rate = 1.0
        self.is_active = True

        # Privacy-preserving techniques
        self.differential_privacy = privacy_level in ["high", "maximum"]
        self.secure_aggregation = privacy_level == "maximum"
        self.homomorphic_encryption = privacy_level == "maximum"

        # Communication
        self.session = requests.Session()

    def set_local_model(self, model: nn.Module) -> None:
        """Set local model."""
        self.local_model = model

    def add_local_data(self, data: torch.Tensor) -> None:
        """Add local training data."""
        self.local_data.append(data)

    def train_local_model(self, epochs: int = 1) -> Dict[str, Any]:
        """
        Train local model on local data.

        Args:
            epochs: Number of training epochs.

        Returns:
            Update dictionary containing model params and metrics.
        """
        if not self.local_model or not self.local_data:
            return {"error": "No model or data available"}

        # Local training
        optimizer = torch.optim.Adam(self.local_model.parameters())
        losses = []

        for epoch in range(epochs):
            for data in self.local_data:
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = F.mse_loss(output, data)  # Simplified loss (Auto-encoder style?)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Get model parameters
        model_params = [param.data.clone() for param in self.local_model.parameters()]

        # Apply privacy-preserving techniques
        if self.differential_privacy:
            model_params = self._apply_differential_privacy(model_params)

        if self.secure_aggregation:
            model_params = self._apply_secure_aggregation(model_params)

        return {
            "client_id": self.client_id,
            "model_params": model_params,
            "losses": losses,
            "data_size": len(self.local_data),
            "participation_rate": self.participation_rate
        }

    def _apply_differential_privacy(
        self,
        model_params: List[torch.Tensor],
        epsilon: float = 1.0
    ) -> List[torch.Tensor]:
        """Apply differential privacy to model parameters."""
        noisy_params = []
        for param in model_params:
            # Add Gaussian noise
            noise = torch.randn_like(param) * (1.0 / epsilon)
            noisy_param = param + noise
            noisy_params.append(noisy_param)
        return noisy_params

    def _apply_secure_aggregation(self, model_params: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply secure aggregation techniques."""
        # Simplified secure aggregation
        return model_params

    def send_update(self, update_data: Dict[str, Any]) -> bool:
        """
        Send model update to server.

        Args:
            update_data: Data to send.

        Returns:
            True if successful.
        """
        try:
            # Encrypt update data
            encrypted_data = self._encrypt_data(update_data)

            # Send to server
            # Note: In a real scenario, requests should have timeouts.
            response = self.session.post(
                f"{self.server_url}/federated/update",
                json=encrypted_data,
                headers={"Client-ID": self.client_id},
                timeout=10
            )

            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
            return False

    def receive_global_model(self) -> Optional[Dict[str, Any]]:
        """
        Receive global model from server.

        Returns:
            Decrypted global model data or None.
        """
        try:
            response = self.session.get(
                f"{self.server_url}/federated/model",
                headers={"Client-ID": self.client_id},
                timeout=10
            )

            if response.status_code == 200:
                encrypted_data = response.json()
                return self._decrypt_data(encrypted_data)
            return None
        except Exception as e:
            logger.error(f"Failed to receive global model: {e}")
            return None

    def _encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt data before transmission."""
        # Serialize data (Note: tensors need to be converted to lists/bytes usually,
        # here relying on json default=str which doesn't handle tensors well.
        # Assuming tensors are lists in `data` at this point or handled by custom encoder.
        # For this refactor, I will use a helper for tensors if they exist.)

        # Basic json dump might fail on tensors.
        # But `train_local_model` returns `model_params` as tensors.
        # We need to convert them to list/numpy for JSON serialization.

        # Helper to convert tensors to list (recursive)
        def _convert_tensors(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            elif isinstance(obj, list):
                return [_convert_tensors(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _convert_tensors(v) for k, v in obj.items()}
            return obj

        serializable_data = _convert_tensors(data)
        data_str = json.dumps(serializable_data)

        # Encrypt
        f = Fernet(self.encryption_key)
        encrypted_data = f.encrypt(data_str.encode())

        return {
            "encrypted_data": base64.b64encode(encrypted_data).decode(),
            "client_id": self.client_id
        }

    def _decrypt_data(self, encrypted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Decrypt received data."""
        try:
            # Decode and decrypt
            encrypted_bytes = base64.b64decode(encrypted_data["encrypted_data"])
            f = Fernet(self.encryption_key)
            decrypted_data = f.decrypt(encrypted_bytes)

            # Deserialize
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return None


class FederatedServer:
    """Federated learning server."""

    def __init__(self, server_url: str, encryption_key: bytes) -> None:
        """
        Initialize FederatedServer.

        Args:
            server_url: Server URL.
            encryption_key: Encryption key.
        """
        self.server_url = server_url
        self.encryption_key = encryption_key
        self.clients: Dict[str, FederatedClient] = {}
        self.global_model: Optional[nn.Module] = None
        self.aggregation_round = 0
        self.client_updates: Dict[str, Dict[str, Any]] = {}
        self.aggregation_threshold = 0.5  # Minimum participation rate

    def register_client(self, client: FederatedClient) -> bool:
        """Register a new client."""
        self.clients[client.client_id] = client
        return True

    def receive_client_update(self, client_id: str, update_data: Dict[str, Any]) -> bool:
        """Receive update from client."""
        self.client_updates[client_id] = update_data
        return True

    def aggregate_updates(self) -> Dict[str, Any]:
        """
        Aggregate client updates into global model.

        Returns:
            Aggregation result dictionary.
        """
        if not self.client_updates:
            return {"error": "No client updates available"}

        # Check participation rate
        participation_rate = len(self.client_updates) / len(self.clients)
        if participation_rate < self.aggregation_threshold:
            return {"error": f"Low participation rate: {participation_rate:.2f}"}

        # Aggregate model parameters
        aggregated_params = self._federated_averaging()

        # Update global model
        if self.global_model and aggregated_params:
            self._update_global_model(aggregated_params)

        # Clear client updates
        self.client_updates.clear()
        self.aggregation_round += 1

        return {
            "aggregation_round": self.aggregation_round,
            "participation_rate": participation_rate,
            "aggregated_params": aggregated_params
        }

    def _federated_averaging(self) -> List[torch.Tensor]:
        """Perform federated averaging of model parameters."""
        if not self.client_updates:
            return []

        # Get all model parameters
        all_params = []
        weights = []

        for _, update_data in self.client_updates.items():
            if "model_params" in update_data:
                # Convert list back to tensor if needed (serialized as list)
                params_list = update_data["model_params"]
                params_tensors = [torch.tensor(p) for p in params_list]
                all_params.append(params_tensors)
                # Weight by data size
                weight = update_data.get("data_size", 1)
                weights.append(weight)

        if not all_params:
            return []

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Average parameters
        aggregated_params = []
        for i in range(len(all_params[0])):
            weighted_sum = torch.zeros_like(all_params[0][i])
            for j, params in enumerate(all_params):
                weighted_sum += weights[j] * params[i]
            aggregated_params.append(weighted_sum)

        return aggregated_params

    def _update_global_model(self, aggregated_params: List[torch.Tensor]) -> None:
        """Update global model with aggregated parameters."""
        if self.global_model:
            for param, aggregated_param in zip(self.global_model.parameters(), aggregated_params):
                param.data = aggregated_param.clone()

    def get_global_model(self) -> Optional[Dict[str, Any]]:
        """Get current global model."""
        if self.global_model:
            return {
                "model_params": [param.data.cpu().tolist() for param in self.global_model.parameters()],
                "aggregation_round": self.aggregation_round
            }
        return None


class PrivacyPreservingTechniques:
    """Privacy-preserving techniques for federated learning."""

    @staticmethod
    def add_differential_privacy_noise(
        model_params: List[torch.Tensor],
        epsilon: float = 1.0,
        delta: float = 1e-5
    ) -> List[torch.Tensor]:
        """Add differential privacy noise to model parameters."""
        # Unused arg delta, kept for interface compatibility
        _ = delta
        
        noisy_params = []
        for param in model_params:
            # Calculate sensitivity
            sensitivity = torch.norm(param).item()

            # Calculate noise scale
            if epsilon > 0:
                noise_scale = sensitivity / epsilon
                # Add Gaussian noise
                noise = torch.randn_like(param) * noise_scale
            else:
                noise = torch.zeros_like(param)
                
            noisy_param = param + noise
            noisy_params.append(noisy_param)

        return noisy_params

    @staticmethod
    def secure_aggregation(
        client_updates: Dict[str, List[torch.Tensor]],
        threshold: int = 2
    ) -> List[torch.Tensor]:
        """Perform secure aggregation of client updates."""
        if len(client_updates) < threshold:
            return []

        all_params = list(client_updates.values())

        # Average parameters
        aggregated_params = []
        for i in range(len(all_params[0])):
            param_sum = torch.zeros_like(all_params[0][i])
            for params in all_params:
                param_sum += params[i]
            aggregated_params.append(param_sum / len(all_params))

        return aggregated_params

    @staticmethod
    def homomorphic_encryption(
        model_params: List[torch.Tensor],
        public_key: bytes
    ) -> List[bytes]:
        """Apply homomorphic encryption to model parameters."""
        # Unused public_key in simplified version
        _ = public_key
        
        # Simplified homomorphic encryption
        encrypted_params = []
        for param in model_params:
            # Serialize and encrypt
            param_bytes = param.numpy().tobytes()
            encrypted_bytes = base64.b64encode(param_bytes)
            encrypted_params.append(encrypted_bytes)

        return encrypted_params


@dataclass
class FederatedRouterConfig(RouterConfig):
    """Configuration for federated learning router."""
    server_url: str = "http://localhost:8000"
    client_id: str = "client_001"
    encryption_key: str = "default_key"
    privacy_level: str = "high"  # low, medium, high, maximum
    participation_rate: float = 1.0
    aggregation_threshold: float = 0.5
    local_epochs: int = 1
    global_rounds: int = 10
    enable_differential_privacy: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    enable_secure_aggregation: bool = True
    enable_homomorphic_encryption: bool = False
    communication_interval: float = 60.0  # seconds
    enable_asynchronous: bool = True
    max_clients: int = 100
    client_timeout: float = 300.0  # seconds


class FederatedRouter(BaseRouter):
    """
    Federated learning-based router that learns from distributed clients while preserving privacy.
    """

    def __init__(self, config: FederatedRouterConfig) -> None:
        """
        Initialize FederatedRouter.

        Args:
            config: Federated router configuration.
        """
        super().__init__(config)
        self.config = config
        self.federated_client: Optional[FederatedClient] = None
        self.federated_server: Optional[FederatedServer] = None
        self.global_model: Optional[nn.Module] = None
        self.local_model: Optional[nn.Module] = None
        self.client_updates: Dict[str, Any] = {}
        self.aggregation_round = 0
        self.participation_history: List[Any] = []
        self.privacy_metrics: Dict[str, Any] = {}

        # Generate encryption key
        self.encryption_key = self._generate_encryption_key()

    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for secure communication."""
        if self.config.encryption_key == "default_key":
            # Generate random key
            key = secrets.token_bytes(32)
        else:
            # Use provided key
            key = self.config.encryption_key.encode()

        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'federated_learning_salt',
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(key))

    def initialize(self) -> None:
        """Initialize the federated learning router."""
        # Create federated client
        self.federated_client = FederatedClient(
            client_id=self.config.client_id,
            server_url=self.config.server_url,
            encryption_key=self.encryption_key,
            privacy_level=self.config.privacy_level
        )

        # Create local model
        self.local_model = self._create_local_model()
        self.federated_client.set_local_model(self.local_model)

        # Initialize global model
        self.global_model = self._create_global_model()

        # Start federated learning process
        if self.config.enable_asynchronous:
            self._start_federated_learning_async()

        self._initialized = True
        logger.info(f"Federated learning router initialized with privacy level: {self.config.privacy_level}")

    def _create_local_model(self) -> nn.Module:
        """Create local model for federated learning."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.num_experts)
        )

    def _create_global_model(self) -> nn.Module:
        """Create global model for federated learning."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.num_experts)
        )

    def _start_federated_learning_async(self) -> None:
        """Start asynchronous federated learning process."""
        def federated_learning_loop():
            while True:
                try:
                    if not self.federated_client:
                         time.sleep(1)
                         continue
                         
                    # Receive global model
                    global_model_data = self.federated_client.receive_global_model()
                    if global_model_data:
                        self._update_global_model(global_model_data)

                    # Train local model
                    local_update = self.federated_client.train_local_model(self.config.local_epochs)

                    # Send update to server
                    self.federated_client.send_update(local_update)

                    # Wait for next communication
                    time.sleep(self.config.communication_interval)

                except Exception as e:
                    logger.error(f"Federated learning error: {e}")
                    time.sleep(60)  # Wait before retry

        # Start in background thread
        thread = threading.Thread(target=federated_learning_loop, daemon=True)
        thread.start()

    def route_tokens(
        self,
        input_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingResult:
        """
        Route tokens using federated learning model.

        Args:
            input_tokens: Input tensor.
            attention_mask: Mask.
            context: Context.

        Returns:
            RoutingResult.
        """
        start_time = time.time()

        # Validate input
        self.validate_input(input_tokens)

        # Check cache
        cache_key = self.get_cache_key(input_tokens, context)
        if cache_key:
            cached_result = self.get_cached_result(cache_key)
            if cached_result:
                return cached_result

        # Add to local data for training
        if self.federated_client:
            self.federated_client.add_local_data(input_tokens)

        # Use global model for routing
        expert_indices, expert_weights, confidence = self._federated_routing(input_tokens)

        # Create routing result
        result = RoutingResult(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            routing_confidence=confidence,
            routing_time=time.time() - start_time,
            strategy_used="federated_learning",
            metadata={
                'client_id': self.config.client_id,
                'aggregation_round': self.aggregation_round,
                'privacy_level': self.config.privacy_level,
                'participation_rate': self.config.participation_rate,
                'local_data_size': len(self.federated_client.local_data) if self.federated_client else 0
            }
        )

        # Cache result
        if cache_key:
            self.cache_result(cache_key, result)

        # Record metrics and log
        self.record_metrics(result)
        self.log_routing(result, input_tokens.shape)

        return result

    def _federated_routing(
        self,
        input_tokens: torch.Tensor
    ) -> Tuple[List[int], List[float], float]:
        """Perform federated learning-based routing."""
        if self.global_model is None:
            # Fallback to random routing
            expert_indices = [random.randint(0, self.config.num_experts - 1)]
            expert_weights = [1.0]
            confidence = 0.5
        else:
            # Use global model for routing
            with torch.no_grad():
                # Prepare input
                batch_size, seq_len, hidden_size = input_tokens.shape
                # Flatten input for simple MLP router or take Mean?
                # The _create_global_model uses Linear layers expecting (..., hidden_size)
                # If we flatten (batch_size, -1), input dim becomes seq_len * hidden_size.
                # But Linear layer is initialized with (config.hidden_size, 256)...
                # So we should probably route per token or pool?
                # The original code did: input_flat = input_tokens.view(batch_size, -1)
                # This seems incompatible with the model def: nn.Linear(self.config.hidden_size, ...)
                # unless seq_len is 1.
                # I will assume we should average over sequence aka embedding pooling.
                
                input_pooled = input_tokens.mean(dim=1) # [batch_size, hidden_size]

                # Get model output
                model_output = self.global_model(input_pooled)
                expert_probs = F.softmax(model_output, dim=-1)

                # Select experts
                expert_indices = []
                expert_weights = []

                # expert_probs [batch_size, num_experts]
                # Assuming batch_size 1 for single routing decision context or taking first
                probs = expert_probs[0]

                for i in range(self.config.num_experts):
                    if probs[i] > 0.1:  # Threshold for expert selection
                        expert_indices.append(i)
                        expert_weights.append(probs[i].item())
                
                if not expert_indices: # Fallback if no expert above threshold
                     max_idx = torch.argmax(probs).item()
                     expert_indices.append(max_idx)
                     expert_weights.append(probs[max_idx].item())

                confidence = probs.max().item()

        return expert_indices, expert_weights, confidence

    def _update_global_model(self, global_model_data: Dict[str, Any]) -> None:
        """Update global model with received parameters."""
        if self.global_model and "model_params" in global_model_data:
            model_params = global_model_data["model_params"]
            for param, new_param in zip(self.global_model.parameters(), model_params):
                param.data = torch.tensor(new_param)

            self.aggregation_round = global_model_data.get("aggregation_round", self.aggregation_round)
            logger.info(f"Updated global model from aggregation round {self.aggregation_round}")

    def get_privacy_metrics(self) -> Dict[str, Any]:
        """Get privacy-preserving metrics."""
        return {
            'privacy_level': self.config.privacy_level,
            'differential_privacy': self.config.enable_differential_privacy,
            'epsilon': self.config.epsilon,
            'delta': self.config.delta,
            'secure_aggregation': self.config.enable_secure_aggregation,
            'homomorphic_encryption': self.config.enable_homomorphic_encryption,
            'participation_rate': self.config.participation_rate,
            'aggregation_round': self.aggregation_round,
            'local_data_size': len(self.federated_client.local_data) if self.federated_client else 0
        }

    def get_router_info(self) -> Dict[str, Any]:
        """Get router information and statistics."""
        base_info = super().get_router_info()
        base_info.update({
            'router_type': 'federated_learning',
            'client_id': self.config.client_id,
            'server_url': self.config.server_url,
            'privacy_level': self.config.privacy_level,
            'participation_rate': self.config.participation_rate,
            'aggregation_round': self.aggregation_round,
            'privacy_metrics': self.get_privacy_metrics()
        })
        return base_info

