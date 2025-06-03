import numpy as np


def attention(Q, K, V, dk):
    """
    Compute attention mechanism.
    
    Args:
        Q: Query matrix of shape (seq_len, d_model)
        K: Key matrix of shape (seq_len, d_model)
        V: Value matrix of shape (seq_len, d_model)
        dk: Dimension for scaling (typically d_model)
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Compute attention scores
    QKT = np.dot(Q, K.T)
    scaled_scores = QKT / np.sqrt(dk)
    
    # Apply softmax with numerical stability
    def stable_softmax(x):
        """Numerically stable softmax."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    # Apply softmax to each row (query)
    attention_weights = stable_softmax(scaled_scores)
    
    # Apply attention weights to values
    weighted_values = np.dot(attention_weights, V)
    return weighted_values


Q = np.random.rand(2, 2)
K = np.random.rand(2, 2)
V = np.random.rand(2, 2)
dk = np.shape(Q)[1]

outputs = attention(Q, K, V, dk)
print(outputs)