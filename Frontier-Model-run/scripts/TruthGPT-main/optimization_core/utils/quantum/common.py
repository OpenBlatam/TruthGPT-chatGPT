"""
Common Quantum Utilities for TruthGPT
Shared logic for state initialization, gate application, and normalization.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Callable

logger = logging.getLogger(__name__)

def initialize_quantum_state(num_qubits: int) -> np.ndarray:
    """Initialize a normalized quantum state (superposition)."""
    size = 2 ** num_qubits
    # Use complex128 to ensure phase information is preserved
    state = np.random.random(size).astype(np.complex128) + 1j * np.random.random(size).astype(np.complex128)
    return state / np.linalg.norm(state)

def normalize_state(state: np.ndarray) -> np.ndarray:
    """Ensure the quantum state is normalized to 1."""
    norm = np.linalg.norm(state)
    if norm < 1e-15:
        return state
    return state / norm

def apply_single_qubit_gate(state: np.ndarray, gate_matrix: np.ndarray, qubit_index: int) -> np.ndarray:
    """
    Apply a single-qubit gate properly using full matrix multiplication
    on the relevant subspaces.
    """
    num_amplitudes = len(state)
    new_state = state.copy().astype(np.complex128)
    
    # Iterate through all pairs of amplitudes that differ only at qubit_index
    for i in range(num_amplitudes):
        if not (i & (1 << qubit_index)):
            i0 = i
            i1 = i | (1 << qubit_index)
            
            # Extract the 2-vector for this subspace
            v = np.array([state[i0], state[i1]])
            # Apply the gate matrix
            v_new = gate_matrix @ v
            
            new_state[i0] = v_new[0]
            new_state[i1] = v_new[1]
            
    return normalize_state(new_state)

def apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    """Apply a CNOT gate correctly."""
    new_state = state.copy().astype(np.complex128)
    for i in range(len(state)):
        if (i >> control) & 1:
            # Flip target bit
            j = i ^ (1 << target)
            # In a CNOT, we just swap amplitudes i and j if control is 1?
            # No, CNOT on state |10> becomes |11>, so we swap the logic.
            # Actually, CNOT maps |10> to |11> and |11> to |10>.
            # So if we are at index i where control is 1, and target is 0, we move amplitude to target 1.
            # But we only do this once per pair.
            if i < j:
                new_state[i], new_state[j] = state[j], state[i]
                
    return normalize_state(new_state)

def calculate_quantum_advantage(num_qubits: int, fidelity: float, base: float = 1.0) -> float:
    """Standardized quantum advantage calculation."""
    return base * (1.0 + num_qubits * 0.1) * fidelity

def apply_elementwise_quantum_op(x: np.ndarray, func: Callable, noise_level: float = 0.01) -> np.ndarray:
    """
    Apply an element-wise operation with simulated quantum noise/tunneling.
    Used for activation functions.
    """
    # Ensure result can hold complex if needed, but usually stays real for activations
    # unless specified. 
    result = func(x).astype(np.complex128)
    
    if noise_level > 0:
        noise = (np.random.normal(0, noise_level, x.shape) + 
                 1j * np.random.normal(0, noise_level, x.shape))
        result += noise
        
    return result
