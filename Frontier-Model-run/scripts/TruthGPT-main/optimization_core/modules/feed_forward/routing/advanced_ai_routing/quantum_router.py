"""
Quantum-Inspired Router
=======================

Advanced routing using quantum-inspired algorithms, quantum annealing,
and quantum machine learning.
"""

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimization_core.modules.feed_forward.routing.modular_routing.base_router import (
    BaseRouter,
    RouterConfig,
    RoutingResult,
)

# Configure logging
logger = logging.getLogger(__name__)


class QuantumState:
    """Quantum state representation."""

    def __init__(self, num_qubits: int) -> None:
        """
        Initialize QuantumState.

        Args:
            num_qubits: Number of qubits.
        """
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1.0  # Initialize to |0⟩ state

    def apply_gate(self, gate: np.ndarray, qubits: List[int]) -> None:
        """
        Apply quantum gate to specified qubits.

        Args:
            gate: Gate matrix.
            qubits: List of qubit indices.
        """
        # Simplified gate application
        if len(qubits) == 1:
            qubit = qubits[0]
            # Apply single-qubit gate
            for i in range(2**self.num_qubits):
                if (i >> qubit) & 1:  # If qubit is |1⟩
                    self.state_vector[i] *= gate[1, 1]
                else:  # If qubit is |0⟩
                    self.state_vector[i] *= gate[0, 0]

    def measure(self) -> int:
        """
        Measure the quantum state and return classical result.

        Returns:
            Measured state index.
        """
        probabilities = np.abs(self.state_vector)**2
        # Normalize in case of numerical drift
        probabilities /= np.sum(probabilities)
        return np.random.choice(len(probabilities), p=probabilities)

    def get_amplitude(self, state: int) -> complex:
        """Get amplitude for specific state."""
        return self.state_vector[state]


class QuantumGate:
    """Quantum gate implementations."""

    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)

    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli-X gate."""
        return np.array([[0, 1], [1, 0]], dtype=complex)

    @staticmethod
    def pauli_y() -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

    @staticmethod
    def pauli_z() -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)

    @staticmethod
    def rotation_x(angle: float) -> np.ndarray:
        """Rotation around X-axis."""
        cos_angle = math.cos(angle / 2)
        sin_angle = math.sin(angle / 2)
        return np.array([
            [cos_angle, -1j * sin_angle],
            [-1j * sin_angle, cos_angle]
        ], dtype=complex)

    @staticmethod
    def rotation_y(angle: float) -> np.ndarray:
        """Rotation around Y-axis."""
        cos_angle = math.cos(angle / 2)
        sin_angle = math.sin(angle / 2)
        return np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ], dtype=complex)

    @staticmethod
    def rotation_z(angle: float) -> np.ndarray:
        """Rotation around Z-axis."""
        cos_angle = math.cos(angle / 2)
        sin_angle = math.sin(angle / 2)
        return np.array([
            [cos_angle - 1j * sin_angle, 0],
            [0, cos_angle + 1j * sin_angle]
        ], dtype=complex)


class QuantumCircuit:
    """Quantum circuit for routing decisions."""

    def __init__(self, num_qubits: int, num_experts: int) -> None:
        """
        Initialize QuantumCircuit.

        Args:
            num_qubits: Number of qubits.
            num_experts: Number of experts.
        """
        self.num_qubits = num_qubits
        self.num_experts = num_experts
        self.quantum_state = QuantumState(num_qubits)
        self.gates: List[Tuple[Any, ...]] = []

    def add_hadamard(self, qubit: int) -> None:
        """Add Hadamard gate."""
        self.gates.append(('hadamard', qubit))

    def add_rotation(self, qubit: int, angle: float, axis: str = 'z') -> None:
        """Add rotation gate."""
        self.gates.append(('rotation', qubit, angle, axis))

    def add_entanglement(self, qubit1: int, qubit2: int) -> None:
        """Add entanglement between qubits."""
        self.gates.append(('entanglement', qubit1, qubit2))

    def execute(self) -> int:
        """
        Execute the quantum circuit.

        Returns:
            Selected expert index.
        """
        # Apply gates in sequence
        for gate_info in self.gates:
            if gate_info[0] == 'hadamard':
                qubit = gate_info[1]
                gate = QuantumGate.hadamard()
                self.quantum_state.apply_gate(gate, [qubit])

            elif gate_info[0] == 'rotation':
                qubit, angle, axis = gate_info[1], gate_info[2], gate_info[3]
                if axis == 'x':
                    gate = QuantumGate.rotation_x(angle)
                elif axis == 'y':
                    gate = QuantumGate.rotation_y(angle)
                else:  # z
                    gate = QuantumGate.rotation_z(angle)
                self.quantum_state.apply_gate(gate, [qubit])

            elif gate_info[0] == 'entanglement':
                # Simplified entanglement simulation (Placeholder)
                pass

        # Measure and return expert index
        measurement = self.quantum_state.measure()
        return measurement % self.num_experts


class QuantumActivation(nn.Module):
    """Quantum-inspired activation function."""

    def __init__(self) -> None:
        """Initialize QuantumActivation."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum-inspired activation."""
        # Quantum superposition of activation functions
        sigmoid_x = torch.sigmoid(x)
        tanh_x = torch.tanh(x)
        relu_x = F.relu(x)

        # Superposition
        return 0.4 * sigmoid_x + 0.3 * tanh_x + 0.3 * relu_x


class QuantumEntanglement(nn.Module):
    """Quantum entanglement simulation."""

    def __init__(self, hidden_size: int, num_qubits: int) -> None:
        """
        Initialize QuantumEntanglement.

        Args:
            hidden_size: Hidden dimension.
            num_qubits: Number of qubits (unused in this simplified implementation but kept for API).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_qubits = num_qubits

        # Entanglement weights
        self.entanglement_matrix = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum entanglement."""
        # Entanglement operation
        entangled = torch.matmul(x, self.entanglement_matrix)

        # Normalize to maintain quantum state properties
        norm = torch.norm(entangled, dim=-1, keepdim=True)
        return entangled / (norm + 1e-8)


class QuantumEmbedding(nn.Module):
    """Quantum-inspired embedding layer."""

    def __init__(self, input_size: int, num_qubits: int) -> None:
        """
        Initialize QuantumEmbedding.

        Args:
            input_size: Input dimension.
            num_qubits: Number of qubits.
        """
        super().__init__()
        self.input_size = input_size
        self.num_qubits = num_qubits
        self.embedding_size = 2**num_qubits

        # Learnable quantum parameters
        self.quantum_weights = nn.Parameter(torch.randn(input_size, self.embedding_size))
        self.quantum_phases = nn.Parameter(torch.randn(input_size, num_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum-inspired embedding."""
        batch_size = x.size(0)

        # Apply quantum-inspired transformation
        # Note: This loop is slow for large batches.
        # Vectorized implementation would be better but keeping structure.
        quantum_features = []
        for i in range(batch_size):
            sample = x[i]
            quantum_state = torch.zeros(self.embedding_size, device=x.device, dtype=torch.complex64)

            # Complex calculation needs care with types
            # Using separate real/imag arithmetic if needed, or specialized complex tensors.
            # Simplified version using basic torch complex support if available or manual.
            # Assuming torch >= 1.7 supports complex.

            # x shape [input_size]
            # phases shape [input_size, num_qubits] -> we need broadcating?
            # logic: amplitude = exp(1j * phase * input_feature)
            # state += amplitude * weight
            # Wait, quantum_weights is [input_size, embedding_size]
            # This loop logic seems custom.

            # Re-implementing strictly as per original logic but ensuring types.
            # Warning: Original logic `sample = x[i]` iterates over batch.
            # `for j, feature in enumerate(sample)` iterates over input_size.
            
            # Vectorized approx:
            # amplitudes = exp(1j * phases * input)
            # This is complex.
            
            # Fallback to original loop for fidelity to original algorithm
            current_state = torch.zeros(self.embedding_size, dtype=torch.cfloat, device=x.device)
            
            # Using loop for clarity (optimization candidate later)
            for j in range(x.size(1)): # input_size
                feature = sample[j]
                # Broadcasting phase
                # phase is vector of size num_qubits.
                # feature is scalar.
                # amplitude is vector of size num_qubits?
                # But we add to `quantum_state` which is `embedding_size`.
                # The original code:
                # `amplitude = torch.exp(1j * self.quantum_phases[j] * feature)`
                # `quantum_state += amplitude * self.quantum_weights[j]` 
                # This implies multiplication broadcasting.
                # quantum_weights[j] is [embedding_size].
                # amplitude is [num_qubits].
                # They don't match unless logic was intended differently.
                # Assuming `amplitude` is scaler or same size.
                
                # Correction: The original code likely assumed some broadcasting or was pseudo-code.
                # To make it runnable:
                # We will map phase interaction to embedding size via weights.
                
                phase_val = self.quantum_phases[j].mean() # Simplify to scalar phase for this feature
                amp = torch.exp(1j * phase_val * feature)
                current_state += amp * self.quantum_weights[j]

            quantum_features.append(current_state.real)

        return torch.stack(quantum_features)


class QuantumLayer(nn.Module):
    """Quantum-inspired layer."""

    def __init__(self, hidden_size: int, num_qubits: int) -> None:
        """
        Initialize QuantumLayer.

        Args:
            hidden_size: Hidden dimension.
            num_qubits: Number of qubits.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_qubits = num_qubits

        # Quantum-inspired operations
        self.quantum_linear = nn.Linear(hidden_size, hidden_size)
        self.quantum_activation = QuantumActivation()
        self.quantum_entanglement = QuantumEntanglement(hidden_size, num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum-inspired forward pass."""
        # Linear transformation
        x = self.quantum_linear(x)

        # Quantum activation
        x = self.quantum_activation(x)

        # Quantum entanglement
        x = self.quantum_entanglement(x)

        return x


class QuantumOutput(nn.Module):
    """Quantum-inspired output layer."""

    def __init__(self, hidden_size: int, output_size: int, num_qubits: int) -> None:
        """
        Initialize QuantumOutput.

        Args:
            hidden_size: Hidden dimension.
            output_size: Output dimension.
            num_qubits: Number of qubits.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_qubits = num_qubits

        # Quantum measurement
        self.measurement_weights = nn.Parameter(torch.randn(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum-inspired output."""
        # Quantum measurement
        measurements = torch.matmul(x, self.measurement_weights)

        # Apply quantum probability
        probabilities = F.softmax(measurements, dim=-1)

        return probabilities


class QuantumNeuralNetwork(nn.Module):
    """Quantum-inspired neural network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_qubits: int = 4
    ) -> None:
        """
        Initialize QuantumNeuralNetwork.

        Args:
            input_size: Input dimension.
            hidden_size: Hidden dimension.
            output_size: Output dimension.
            num_qubits: Number of qubits.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_qubits = num_qubits

        # Quantum-inspired layers
        self.quantum_embedding = QuantumEmbedding(input_size, num_qubits)
        self.quantum_layers = nn.ModuleList([
            QuantumLayer(hidden_size, num_qubits) for _ in range(2)
        ])
        self.quantum_output = QuantumOutput(hidden_size, output_size, num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum-inspired network."""
        # Quantum embedding
        x = self.quantum_embedding(x)

        # Quantum layers
        for layer in self.quantum_layers:
            x = layer(x)

        # Quantum output
        x = self.quantum_output(x)

        return x


class QuantumAnnealing:
    """Quantum annealing for optimization."""

    def __init__(self, num_variables: int, num_experts: int) -> None:
        """
        Initialize QuantumAnnealing.

        Args:
            num_variables: Number of variables.
            num_experts: Number of possible values (experts).
        """
        self.num_variables = num_variables
        self.num_experts = num_experts
        self.temperature = 1.0
        self.temperature_decay = 0.99
        self.min_temperature = 0.01

    def anneal(
        self,
        objective_function: Callable[[List[int]], float],
        max_iterations: int = 1000
    ) -> Tuple[List[int], float]:
        """
        Perform quantum annealing optimization.

        Args:
            objective_function: Function to minimize.
            max_iterations: Maximum iterations.

        Returns:
            Tuple of (best solution, best energy).
        """
        # Initialize random solution
        current_solution = [random.randint(0, self.num_experts - 1) for _ in range(self.num_variables)]
        current_energy = objective_function(current_solution)

        best_solution = current_solution.copy()
        best_energy = current_energy

        for _ in range(max_iterations):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution)
            neighbor_energy = objective_function(neighbor)

            # Quantum tunneling probability
            delta_energy = neighbor_energy - current_energy
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / self.temperature):
                current_solution = neighbor
                current_energy = neighbor_energy

                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy

            # Cool down
            self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)

        return best_solution, best_energy

    def _generate_neighbor(self, solution: List[int]) -> List[int]:
        """Generate neighbor solution."""
        neighbor = solution.copy()
        # Randomly change one variable
        index = random.randint(0, len(neighbor) - 1)
        neighbor[index] = random.randint(0, self.num_experts - 1)
        return neighbor


@dataclass
class QuantumRouterConfig(RouterConfig):
    """Configuration for quantum-inspired router."""
    num_qubits: int = 4
    quantum_circuit_depth: int = 3
    quantum_entanglement: bool = True
    quantum_superposition: bool = True
    quantum_measurement: str = "probabilistic"  # probabilistic, deterministic
    quantum_annealing: bool = True
    annealing_temperature: float = 1.0
    annealing_decay: float = 0.99
    quantum_neural_network: bool = True
    quantum_embedding_dim: int = 64
    quantum_layers: int = 2
    enable_quantum_noise: bool = True
    noise_level: float = 0.1
    quantum_optimization: bool = True
    optimization_iterations: int = 1000


class QuantumRouter(BaseRouter):
    """
    Quantum-inspired router using quantum algorithms and quantum machine learning.
    """

    def __init__(self, config: QuantumRouterConfig) -> None:
        """
        Initialize QuantumRouter.

        Args:
            config: Quantum router configuration.
        """
        super().__init__(config)
        self.config = config
        self.quantum_circuit: Optional[QuantumCircuit] = None
        self.quantum_neural_network: Optional[QuantumNeuralNetwork] = None
        self.quantum_annealing: Optional[QuantumAnnealing] = None
        self.quantum_state_history: List[Any] = []

    def initialize(self) -> None:
        """Initialize the quantum router."""
        # Create quantum circuit
        self.quantum_circuit = QuantumCircuit(
            num_qubits=self.config.num_qubits,
            num_experts=self.config.num_experts
        )

        # Create quantum neural network
        if self.config.quantum_neural_network:
            self.quantum_neural_network = QuantumNeuralNetwork(
                input_size=self.config.hidden_size,
                hidden_size=self.config.hidden_size,
                output_size=self.config.num_experts,
                num_qubits=self.config.num_qubits
            )

        # Create quantum annealing
        if self.config.quantum_annealing:
            self.quantum_annealing = QuantumAnnealing(
                num_variables=self.config.num_experts,
                num_experts=self.config.num_experts
            )

        self._initialized = True
        logger.info(f"Quantum router initialized with {self.config.num_qubits} qubits")

    def route_tokens(
        self,
        input_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingResult:
        """
        Route tokens using quantum-inspired algorithms.

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

        # Extract quantum features
        quantum_features = self._extract_quantum_features(input_tokens, attention_mask, context)

        # Apply quantum routing
        expert_indices, expert_weights, confidence = self._quantum_routing(quantum_features)

        # Create routing result
        result = RoutingResult(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            routing_confidence=confidence,
            routing_time=time.time() - start_time,
            strategy_used="quantum_inspired",
            metadata={
                'quantum_features': quantum_features.cpu().numpy().tolist(),
                'num_qubits': self.config.num_qubits,
                'quantum_circuit_depth': self.config.quantum_circuit_depth,
                'quantum_entanglement': self.config.quantum_entanglement,
                'quantum_superposition': self.config.quantum_superposition
            }
        )

        # Cache result
        if cache_key:
            self.cache_result(cache_key, result)

        # Record metrics and log
        self.record_metrics(result)
        self.log_routing(result, input_tokens.shape)

        return result

    def _extract_quantum_features(
        self,
        input_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Extract quantum-inspired features."""
        # Unused args
        _ = attention_mask
        _ = context

        batch_size, seq_len, hidden_size = input_tokens.shape

        features: List[float] = []

        # Quantum superposition of token statistics
        token_mean = input_tokens.mean(dim=(0, 1))
        token_std = input_tokens.std(dim=(0, 1))

        # Quantum superposition
        quantum_mean = torch.exp(1j * token_mean).real
        quantum_std = torch.exp(1j * token_std).real

        features.extend(quantum_mean.tolist())
        features.extend(quantum_std.tolist())

        # Quantum entanglement features
        if self.config.quantum_entanglement:
            entanglement_features = self._calculate_entanglement_features(input_tokens)
            features.extend(entanglement_features)

        # Quantum noise
        if self.config.enable_quantum_noise:
            noise = torch.randn_like(input_tokens) * self.config.noise_level
            noisy_features = (input_tokens + noise).mean(dim=(0, 1))
            features.extend(noisy_features.tolist())

        # Pad or truncate to hidden_size
        while len(features) < self.config.hidden_size:
            features.append(0.0)
        features = features[:self.config.hidden_size]

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _calculate_entanglement_features(self, input_tokens: torch.Tensor) -> List[float]:
        """Calculate quantum entanglement features."""
        batch_size, seq_len, hidden_size = input_tokens.shape

        # Simplified entanglement calculation
        entanglement_features = []

        # Calculate correlations between different dimensions
        # Limit iteration for performance
        limit = min(10, hidden_size)
        for i in range(limit):
            for j in range(i + 1, limit):
                if input_tokens.size(0) > 1 or input_tokens.size(1) > 1:
                     vec_i = input_tokens[:, :, i].flatten()
                     vec_j = input_tokens[:, :, j].flatten()
                     if vec_i.numel() > 1:
                        # Stack for corrcoef
                        matrix = torch.stack([vec_i, vec_j])
                        corr = torch.corrcoef(matrix)[0, 1]
                        val = corr.item() if not torch.isnan(corr) else 0.0
                     else:
                        val = 0.0
                else:
                    val = 0.0
                entanglement_features.append(val)

        return entanglement_features

    def _quantum_routing(self, quantum_features: torch.Tensor) -> Tuple[List[int], List[float], float]:
        """Perform quantum-inspired routing."""
        if self.config.quantum_neural_network and self.quantum_neural_network is not None:
            # Use quantum neural network
            with torch.no_grad():
                quantum_output = self.quantum_neural_network(quantum_features)
                expert_probs = F.softmax(quantum_output, dim=-1)

                # Select experts based on quantum probabilities
                expert_indices = []
                expert_weights = []
                
                # Assuming batch size 1 for routing decision
                probs = expert_probs[0]

                for i in range(self.config.num_experts):
                    if probs[i] > 0.1:  # Threshold for expert selection
                        expert_indices.append(i)
                        expert_weights.append(probs[i].item())
                
                if not expert_indices:
                     max_idx = torch.argmax(probs).item()
                     expert_indices.append(max_idx)
                     expert_weights.append(probs[max_idx].item())

                confidence = probs.max().item()

        elif self.config.quantum_annealing and self.quantum_annealing is not None:
            # Use quantum annealing
            def objective_function(solution: List[int]) -> float:
                # Simplified objective function
                return -float(sum(solution))  # Minimize negative sum (maximize sum)

            best_solution, best_energy = self.quantum_annealing.anneal(
                objective_function,
                self.config.optimization_iterations
            )

            expert_indices = [i for i, val in enumerate(best_solution) if val > 0]
            if not expert_indices:
                 expert_indices = [0] # Fallback
            
            expert_weights = [1.0 / len(expert_indices)] * len(expert_indices)
            confidence = 1.0 - abs(best_energy) / self.config.num_experts

        else:
            # Use quantum circuit
            if self.quantum_circuit is None:
                 # Should not happen if initialized
                 return [0], [1.0], 0.0
                 
            self._build_quantum_circuit(quantum_features)
            expert_index = self.quantum_circuit.execute()

            expert_indices = [expert_index]
            expert_weights = [1.0]
            confidence = 0.8  # Default quantum confidence

        return expert_indices, expert_weights, confidence

    def _build_quantum_circuit(self, quantum_features: torch.Tensor) -> None:
        """Build quantum circuit for routing."""
        # Reset circuit
        self.quantum_circuit = QuantumCircuit(
            num_qubits=self.config.num_qubits,
            num_experts=self.config.num_experts
        )

        # Add quantum gates based on features
        # Flatten features for indexing
        flat_features = quantum_features.flatten()
        
        for i in range(self.config.quantum_circuit_depth):
            for qubit in range(self.config.num_qubits):
                # Add Hadamard gates for superposition
                if self.config.quantum_superposition:
                    self.quantum_circuit.add_hadamard(qubit)

                # Add rotations based on features
                # modulo indexing
                feature_idx = qubit % flat_features.size(0)
                feature_value = flat_features[feature_idx].item()
                angle = feature_value * math.pi
                self.quantum_circuit.add_rotation(qubit, angle, 'z')

            # Add entanglement
            if self.config.quantum_entanglement and i < self.config.quantum_circuit_depth - 1:
                for qubit in range(self.config.num_qubits - 1):
                    self.quantum_circuit.add_entanglement(qubit, qubit + 1)

    def get_router_info(self) -> Dict[str, Any]:
        """Get router information and statistics."""
        base_info = super().get_router_info()
        base_info.update({
            'router_type': 'quantum_inspired',
            'num_qubits': self.config.num_qubits,
            'quantum_circuit_depth': self.config.quantum_circuit_depth,
            'quantum_entanglement': self.config.quantum_entanglement,
            'quantum_superposition': self.config.quantum_superposition,
            'quantum_measurement': self.config.quantum_measurement,
            'quantum_annealing': self.config.quantum_annealing is not None,
            'quantum_neural_network': self.config.quantum_neural_network,
            'quantum_optimization': self.config.quantum_optimization,
            'quantum_state_history': len(self.quantum_state_history)
        })
        return base_info

