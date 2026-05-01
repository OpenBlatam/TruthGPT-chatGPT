import sys
import os
import numpy as np
import torch
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

from TruthGPT_main.optimization_core.utils.quantum.common import initialize_quantum_state, normalize_state, apply_single_qubit_gate, apply_cnot
from TruthGPT_main.optimization_core.utils.quantum.revolutionary_quantum_deep_learning_system import RevolutionaryQuantumNeuralLayer, RevolutionaryQuantumConfig

def test_quantum_utilities():
    print("Testing Quantum Utilities...")
    # Test state initialization
    state = initialize_quantum_state(2)
    assert len(state) == 4
    assert np.isclose(np.linalg.norm(state), 1.0)
    print("✅ State initialization and normalization passed")

    # Test single qubit gate (Hadamard-like)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    state_00 = np.array([1, 0, 0, 0], dtype=complex)
    new_state = apply_single_qubit_gate(state_00, h, 0) # Apply H to qubit 0
    # Should be (|00> + |10>) / sqrt(2) -> index 0 and 2
    assert np.isclose(new_state[0], 1/np.sqrt(2))
    assert np.isclose(new_state[2], 1/np.sqrt(2))
    assert np.isclose(new_state[1], 0)
    assert np.isclose(new_state[3], 0)
    print("✅ Single qubit gate application (Hadamard test) passed")

    # Test CNOT (control 0, target 1)
    # Start with |10> (index 1 if LSB is qubit 0, or index 2 if MSB is qubit 0)
    # My apply_cnot uses: if (i >> control) & 1: j = i ^ (1 << target)
    # So if control is 0, target is 1:
    # index 1 (binary 01 -> qubit0=1, qubit1=0) -> j = 01 ^ 10 = 11 (index 3)
    state_10 = np.array([0, 1, 0, 0], dtype=complex) # qubit0=1, qubit1=0
    cnot_state = apply_cnot(state_10, 0, 1)
    assert np.isclose(cnot_state[3], 1.0)
    print("✅ CNOT gate application passed")

def test_neural_layer():
    print("Testing Revolutionary Quantum Neural Layer...")
    config = RevolutionaryQuantumConfig(num_qubits=2, num_layers=1)
    layer = RevolutionaryQuantumNeuralLayer(config, "test_layer", 4)
    
    # Input should be a quantum state of size 2^num_qubits = 4
    x = initialize_quantum_state(2)
    output = layer.forward(x)
    assert len(output) == 4
    print("✅ Neural layer forward pass passed")

if __name__ == "__main__":
    try:
        test_quantum_utilities()
        test_neural_layer()
        print("\n✨ ALL ENTERPRISE REFACTOR VERIFICATIONS PASSED ✨")
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
