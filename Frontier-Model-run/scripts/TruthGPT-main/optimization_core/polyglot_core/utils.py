"""
Utility functions for polyglot_core.

Common utilities, helpers, and convenience functions.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import warnings


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes_value: Bytes to format
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_time(seconds: float) -> str:
    """
    Format time to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1.5 ms")
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60.0:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def validate_shape(tensor: np.ndarray, expected_shape: tuple, name: str = "tensor"):
    """
    Validate tensor shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape
        name: Tensor name for error messages
        
    Raises:
        ValueError: If shape doesn't match
    """
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {tensor.shape}, expected {expected_shape}"
        )


def ensure_contiguous(tensor: np.ndarray) -> np.ndarray:
    """
    Ensure tensor is C-contiguous.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Contiguous tensor
    """
    if not tensor.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(tensor)
    return tensor


def pad_sequence(
    sequences: List[np.ndarray],
    max_length: Optional[int] = None,
    pad_value: float = 0.0,
    pad_side: str = "right"
) -> np.ndarray:
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of sequences
        max_length: Maximum length (default: max of sequences)
        pad_value: Padding value
        pad_side: "left" or "right"
        
    Returns:
        Padded array [batch, max_length, ...]
    """
    if not sequences:
        raise ValueError("sequences cannot be empty")
    
    if max_length is None:
        max_length = max(len(s) for s in sequences)
    
    # Get shape of first sequence (excluding length dimension)
    first_shape = sequences[0].shape[1:] if sequences[0].ndim > 1 else ()
    
    # Create padded array
    batch_size = len(sequences)
    padded_shape = (batch_size, max_length) + first_shape
    padded = np.full(padded_shape, pad_value, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len > max_length:
            seq = seq[:max_length]
            seq_len = max_length
        
        if pad_side == "right":
            padded[i, :seq_len] = seq
        else:
            padded[i, -seq_len:] = seq
    
    return padded


def truncate_sequence(
    sequence: np.ndarray,
    max_length: int,
    side: str = "right"
) -> np.ndarray:
    """
    Truncate sequence to max length.
    
    Args:
        sequence: Input sequence
        max_length: Maximum length
        side: "left" or "right"
        
    Returns:
        Truncated sequence
    """
    if len(sequence) <= max_length:
        return sequence
    
    if side == "right":
        return sequence[:max_length]
    else:
        return sequence[-max_length:]


def batch_tensors(tensors: List[np.ndarray], pad: bool = True) -> np.ndarray:
    """
    Batch multiple tensors into single array.
    
    Args:
        tensors: List of tensors
        pad: Pad to same length if True
        
    Returns:
        Batched tensor
    """
    if not tensors:
        raise ValueError("tensors cannot be empty")
    
    if pad:
        return pad_sequence(tensors)
    else:
        # Stack if same shape
        shapes = [t.shape for t in tensors]
        if len(set(shapes)) == 1:
            return np.stack(tensors)
        else:
            raise ValueError("Tensors have different shapes and pad=False")


def get_device_info() -> Dict[str, Any]:
    """
    Get device information (CPU, GPU, etc.).
    
    Returns:
        Dict with device information
    """
    info = {
        'cpu_count': None,
        'cpu_freq': None,
        'memory_total_gb': None,
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_names': []
    }
    
    try:
        import psutil
        if psutil:
            info['cpu_count'] = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            info['cpu_freq'] = cpu_freq.current if cpu_freq else None
            
            mem = psutil.virtual_memory()
            info['memory_total_gb'] = mem.total / (1024**3)
    except (ImportError, AttributeError):
        pass
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(info['gpu_count'])]
    except ImportError:
        pass
    
    return info


def print_device_info():
    """Print device information."""
    info = get_device_info()
    
    print("\n" + "=" * 60)
    print("Device Information")
    print("=" * 60)
    
    if info['cpu_count']:
        print(f"CPU Cores: {info['cpu_count']}")
        if info['cpu_freq']:
            print(f"CPU Frequency: {info['cpu_freq']:.0f} MHz")
    
    if info['memory_total_gb']:
        print(f"Total Memory: {info['memory_total_gb']:.2f} GB")
    
    if info['gpu_available']:
        print(f"GPU Available: Yes ({info['gpu_count']} device(s))")
        for i, name in enumerate(info['gpu_names']):
            print(f"  GPU {i}: {name}")
    else:
        print("GPU Available: No")
    
    print("=" * 60 + "\n")


def suppress_warnings(func):
    """Decorator to suppress warnings."""
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


def estimate_memory_usage(
    shape: tuple,
    dtype: np.dtype = np.float32,
    num_copies: int = 1
) -> int:
    """
    Estimate memory usage for tensor.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        num_copies: Number of copies
        
    Returns:
        Estimated bytes
    """
    element_size = dtype.itemsize
    num_elements = np.prod(shape)
    return num_elements * element_size * num_copies


def create_random_tensor(
    shape: tuple,
    dtype: np.dtype = np.float32,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create random tensor with reproducible seed.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        seed: Random seed
        
    Returns:
        Random tensor
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(*shape).astype(dtype)


def check_backend_compatibility(backend: str, feature: str) -> bool:
    """
    Check if backend supports a feature.
    
    Args:
        backend: Backend name
        feature: Feature name
        
    Returns:
        True if compatible
    """
    compatibility = {
        'rust': ['kv_cache', 'compression', 'tokenization', 'data_loading'],
        'cpp': ['attention', 'flash_attention', 'cuda', 'inference', 'quantization'],
        'go': ['http', 'grpc', 'distributed', 'messaging'],
        'python': ['all']
    }
    
    backend_lower = backend.lower()
    if backend_lower not in compatibility:
        return False
    
    features = compatibility[backend_lower]
    return feature in features or 'all' in features


